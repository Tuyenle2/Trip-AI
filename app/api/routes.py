import json
import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import UserAuth, ChatRequest, ChatResponse, SavePlanRequest, ThreadCreateRequest
from app.services.auth_service import register_user_service, login_user_service
from app.services.planner_service import save_user_plan, get_user_history
from app.core.security import SecurityGuard
from app.agents.trip_planner_agent import TripPlannerAgent
from app.db.repository import create_thread, get_threads_by_user, insert_message, get_messages_by_thread

router = APIRouter()
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
planner_agent = TripPlannerAgent(mongodb_uri=MONGODB_URI)

@router.post("/register")
def register_user(user: UserAuth):
    return register_user_service(user.username, user.password)

@router.post("/login")
def login_user(user: UserAuth):
    return login_user_service(user.username, user.password)
@router.post("/threads")
def api_create_thread(req: ThreadCreateRequest):
    return create_thread(req.username, req.title)

@router.get("/threads/{username}")
def api_get_threads(username: str):
    return get_threads_by_user(username)

@router.get("/messages/{thread_id}")
def api_get_messages(thread_id: str):
    return get_messages_by_thread(thread_id)

@router.post("/chat", response_model=ChatResponse)
def api_chat(request: ChatRequest):
    if not SecurityGuard.is_input_safe(request.message):
        return ChatResponse(reply="🛑 Hệ thống từ chối yêu cầu do vi phạm Guardrails.", is_safe=False)
    insert_message(request.thread_id, "user", request.message)
    try:
        reply_text = planner_agent.chat(thread_id=request.thread_id, user_input=request.message)
        
        insert_message(request.thread_id, "ai", reply_text)
        
        return ChatResponse(reply=reply_text, is_safe=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save_plan")
def save_plan(req: SavePlanRequest):
    return save_user_plan(req.username, req.plan_text)

@router.get("/history/{username}")
def get_history(username: str):
    return get_user_history(username)
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


import asyncio
from fastapi import HTTPException
from pydantic import BaseModel

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


from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from app.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM

router = APIRouter()


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


@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    
    fake_db_password_hash = get_password_hash("123456") 
    
    if not verify_password(form_data.password, fake_db_password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sai tên đăng nhập hoặc mật khẩu",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
  
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/transactions/me")
async def get_my_transactions(current_user: str = Depends(get_current_user)):
    return {"message": f"Xin chào {current_user}, đây là lịch sử giao dịch của bạn."}


from pydantic import BaseModel
from app.auth import get_password_hash 
from pydantic import BaseModel
from app.auth import get_password_hash # Import hàm băm mật khẩu từ file auth.py của bạn

class UserCreate(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register_user(user: UserCreate):
    # 1. Băm mật khẩu để bảo mật
    hashed_password = get_password_hash(user.password)
    
    # 2. LƯU VÀO DATABASE (PostgreSQL hoặc MongoDB)
    # Ví dụ với MongoDB (bạn cần import client DB của bạn vào đây):
    # existing_user = db.users.find_one({"username": user.username})
    # if existing_user:
    #     raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
    # db.users.insert_one({"username": user.username, "password_hash": hashed_password})
    
    return {"message": "Đăng ký thành công!"}