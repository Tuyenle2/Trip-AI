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
#from app.db.repository import create_thread, get_threads_by_user, insert_message, get_messages_by_thread
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



import uuid

@router.get("/threads/{username}")
def api_get_threads(username: str):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    threads_col = db.get_collection("threads")
    threads = list(threads_col.find({"username": username}, {"_id": 0}).sort("created_at", -1))
    for t in threads:
        t["id"] = t.get("thread_id") 
    return threads

@router.post("/save_plan")
def save_plan(req: SavePlanRequest):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    history_col = db.get_collection("history")
    
    history_col.insert_one({
        "username": req.username,
        "plan": req.plan_text,
        "date": datetime.now().strftime("%d/%m/%Y %H:%M")
    })
    return {"status": "success"}

@router.get("/history/{username}")
def get_history(username: str):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    history_col = db.get_collection("history")
    history = list(history_col.find({"username": username}, {"_id": 0}))
    return history[::-1]


@router.post("/threads")
def api_create_thread(req: ThreadCreateRequest):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    threads_col = db.get_collection("threads")
    
    thread_id = str(uuid.uuid4())
    new_thread = {
        "thread_id": thread_id,
        "username": req.username,
        "title": req.title,
        "created_at": datetime.now()
    }
    threads_col.insert_one(new_thread)
    return {"id": thread_id, "thread_id": thread_id, "title": req.title}

@router.get("/messages/{thread_id}")
def api_get_messages(thread_id: str):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    msg_col = db.get_collection("messages")
    messages = list(msg_col.find({"thread_id": thread_id}, {"_id": 0}).sort("created_at", 1))
    return messages

@router.post("/chat/stream")
async def api_chat_stream(request: ChatRequest):
    if not SecurityGuard.is_input_safe(request.message):
        raise HTTPException(status_code=400, detail="Vi phạm Guardrails")
    
    db = planner_agent.client.get_database("ai_trip_planner_db")
    msg_col = db.get_collection("messages")
    
    # Lưu tin nhắn của User
    msg_col.insert_one({
        "thread_id": request.thread_id,
        "role": "user",
        "content": request.message,
        "created_at": datetime.now()
    })
    
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
    
        # Sau khi AI nói xong, lưu tin nhắn của AI vào DB
        if full_text:
            msg_col.insert_one({
                "thread_id": request.thread_id,
                "role": "ai",
                "content": full_text,
                "created_at": datetime.now()
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

from datetime import datetime
from pydantic import BaseModel

class TransactionCreate(BaseModel):
    username: str
    service_name: str
    amount: str

@router.post("/transactions")
async def save_transaction(transaction: TransactionCreate):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    trans_col = db.get_collection("transactions")
    
    new_trans = {
        "username": transaction.username,
        "service_name": transaction.service_name,
        "amount": transaction.amount,
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M")
    }
    trans_col.insert_one(new_trans)
    return {"status": "success"}

@router.get("/transactions/{username}")
async def get_user_transactions(username: str):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    trans_col = db.get_collection("transactions")
    
    transactions = list(trans_col.find({"username": username}, {"_id": 0}))
    return transactions