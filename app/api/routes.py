import json
import os
import asyncio
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import jwt

from app.models.schemas import UserAuth, ChatRequest, ChatResponse, SavePlanRequest, ThreadCreateRequest
from app.services.planner_service import save_user_plan, get_user_history
from app.core.security import SecurityGuard
from app.agents.trip_planner_agent import TripPlannerAgent
from app.db.repository import create_thread, get_threads_by_user, insert_message, get_messages_by_thread
from app.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM


router = APIRouter()
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")


planner_agent = TripPlannerAgent(mongodb_uri=MONGODB_URI)


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin (Token không hợp lệ)",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        return username
    except jwt.PyJWTError:
        raise credentials_exception

class UserCreate(BaseModel):
    username: str
    password: str


@router.post("/register")
async def register_user(user: UserCreate):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    users_collection = db.get_collection("users")
    
    if users_collection.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
        
    hashed_password = get_password_hash(user.password)
    users_collection.insert_one({
        "username": user.username,
        "password_hash": hashed_password
    })
    return {"message": "Đăng ký thành công!"}


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    users_collection = db.get_collection("users")
    
    user_in_db = users_collection.find_one({"username": form_data.username})
    
    if not user_in_db or not verify_password(form_data.password, user_in_db["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}



@router.post("/threads")
def api_create_thread(req: ThreadCreateRequest):
    return create_thread(req.username, req.title)

@router.get("/threads/{username}")
def api_get_threads(username: str):
    return get_threads_by_user(username)

@router.get("/messages/{thread_id}")
def api_get_messages(thread_id: str):
    return get_messages_by_thread(thread_id)

@router.post("/chat/stream")
async def api_chat_stream(request: ChatRequest):
    if not SecurityGuard.is_input_safe(request.message):
        raise HTTPException(status_code=400, detail="Vi phạm Guardrails")
    
    insert_message(request.thread_id, "user", request.message)
    
    async def event_generator():
        full_text = ""
        async for event in planner_agent.achat_stream(request.thread_id, request.message):
            kind = event["event"]
            if kind == "on_tool_start":
                tool_name = event["name"]
                query_data = event["data"].get("input", {})
                query = query_data.get("query", str(query_data))
                yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'query': query})}\n\n"
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if getattr(chunk, "content", None):
                    content = chunk.content
                    text_piece = ""
                    
                    if isinstance(content, list):
                        text_piece = "".join([item.get("text", "") for item in content if isinstance(item, dict)])
                    elif isinstance(content, str):
                        text_piece = content
                        
                    if text_piece:
                        full_text += text_piece
                        yield f"data: {json.dumps({'type': 'content', 'data': text_piece})}\n\n"
    
        insert_message(request.thread_id, "ai", full_text)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@router.post("/save_plan")
def save_plan(req: SavePlanRequest):
    return save_user_plan(req.username, req.plan_text)

@router.get("/history/{username}")
def get_history(username: str):
    return get_user_history(username)

class PaymentVerificationRequest(BaseModel):
    username: str
    card_name: str
    card_number: str
    cvv: str

@router.post("/verify-payment")
async def verify_payment(req: PaymentVerificationRequest):
    await asyncio.sleep(1.5)
    
    if req.card_number.startswith("0000"):
        raise HTTPException(status_code=400, detail="Thẻ bị từ chối: Tài khoản không hợp lệ hoặc bị khóa!")
        
    if req.card_name.strip() == "":
        raise HTTPException(status_code=400, detail="Tên chủ tài khoản không được để trống!")
        
    if len(req.cvv) != 3:
        raise HTTPException(status_code=400, detail="Mã bảo mật CVV sai định dạng!")
        
    return {"status": "success", "message": "Thông tin tài khoản hợp lệ, thanh toán thành công!"}

@router.get("/transactions/{username}")
async def get_user_transactions(username: str):
    
    return [
        {"service_name": "Khách sạn Mường Thanh", "amount": "1,500,000 VND", "created_at": "19/03/2026"},
        {"service_name": "Vé máy bay VNA", "amount": "2,300,000 VND", "created_at": "19/03/2026"}
    ]