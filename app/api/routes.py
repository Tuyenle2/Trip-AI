import json
import os
import uuid
import asyncio
from datetime import datetime
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import jwt

from app.models.schemas import UserAuth, ChatRequest, ChatResponse, SavePlanRequest, ThreadCreateRequest
from app.core.security import SecurityGuard
from app.agents.trip_planner_agent import TripPlannerAgent
from app.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM

router = APIRouter()
load_dotenv()
MONGODB_URI = os.getenv("MONGODB_URI")
planner_agent = TripPlannerAgent(mongodb_uri=MONGODB_URI)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

class UserCreate(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register_user(user: UserCreate):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    users_col = db.get_collection("users")
    if users_col.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
    hashed_password = get_password_hash(user.password)
    users_col.insert_one({"username": user.username, "password_hash": hashed_password})
    return {"message": "Đăng ký thành công!"}

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    users_col = db.get_collection("users")
    user_in_db = users_col.find_one({"username": form_data.username})
    
    if not user_in_db or not verify_password(form_data.password, user_in_db["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sai mật khẩu")
    
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/threads")
def api_create_thread(req: ThreadCreateRequest):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    threads_col = db.get_collection("threads")
    thread_id = str(uuid.uuid4())
    threads_col.insert_one({
        "thread_id": thread_id,
        "username": req.username,
        "title": req.title,
        "created_at": datetime.now()
    })
    return {"id": thread_id, "thread_id": thread_id, "title": req.title}

@router.get("/threads/{username}")
def api_get_threads(username: str):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    threads_col = db.get_collection("threads")
    threads = list(threads_col.find({"username": username}, {"_id": 0}).sort("created_at", -1))
    for t in threads:
        t["id"] = t.get("thread_id")
    return threads

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
    
        if full_text:
            msg_col.insert_one({
                "thread_id": request.thread_id,
                "role": "ai",
                "content": full_text,
                "created_at": datetime.now()
            })

    return StreamingResponse(event_generator(), media_type="text/event-stream")

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

class TransactionCreate(BaseModel):
    username: str
    service_name: str
    amount: str

@router.post("/transactions")
async def save_transaction(transaction: TransactionCreate):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    trans_col = db.get_collection("transactions")
    trans_col.insert_one({
        "username": transaction.username,
        "service_name": transaction.service_name,
        "amount": transaction.amount,
        "created_at": datetime.now().strftime("%d/%m/%Y %H:%M")
    })
    return {"status": "success"}

@router.get("/transactions/{username}")
async def get_user_transactions(username: str):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    trans_col = db.get_collection("transactions")
    transactions = list(trans_col.find({"username": username}, {"_id": 0}))
    return transactions

class PaymentVerificationRequest(BaseModel):
    username: str
    card_name: str
    card_number: str
    cvv: str

@router.post("/verify-payment")
async def verify_payment(req: PaymentVerificationRequest):
    await asyncio.sleep(1.5)
    if req.card_number.startswith("0000"):
        raise HTTPException(status_code=400, detail="Thẻ bị từ chối!")
    if req.card_name.strip() == "":
        raise HTTPException(status_code=400, detail="Tên không được trống!")
    if len(req.cvv) != 3:
        raise HTTPException(status_code=400, detail="CVV sai!")
    return {"status": "success", "message": "Thanh toán thành công!"}

class RoomAuthRequest(BaseModel):
    room_id: str
    password: str

@router.post("/room/join")
def join_room(req: RoomAuthRequest):
    db = planner_agent.client.get_database("ai_trip_planner_db")
    rooms_col = db.get_collection("rooms")
    room_id_upper = req.room_id.upper()
    
    room = rooms_col.find_one({"room_id": room_id_upper})
    if room:
        if room["password"] != req.password:
            raise HTTPException(status_code=400, detail="Sai mật khẩu phòng!")
        return {"status": "success", "message": "Đã vào phòng"}
    else:
        rooms_col.insert_one({"room_id": room_id_upper, "password": req.password, "created_at": datetime.now()})
        return {"status": "success", "message": "Đã tạo phòng mới"}

class ConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            if websocket in self.active_connections[room_id]:
                self.active_connections[room_id].remove(websocket)

    async def broadcast(self, message: str, room_id: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws/room/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    await manager.connect(websocket, room_id)
    db = planner_agent.client.get_database("ai_trip_planner_db")
    room_col = db.get_collection("room_messages")
    
    history = list(room_col.find({"room_id": room_id}, {"_id": 0}).sort("created_at", 1).limit(50))
    for msg in history:
        await websocket.send_text(json.dumps(msg))
        
    try:
        while True:
            data = await websocket.receive_text()
            
            
            msg_doc = {
                "room_id": room_id,
                "username": username,
                "message": data,
                "created_at": datetime.now().strftime("%H:%M")
            }
            room_col.insert_one(msg_doc)
            msg_doc.pop("_id", None)  
            await manager.broadcast(json.dumps(msg_doc), room_id)
            
            
            if "@AI" in data.upper() or "@BOT" in data.upper():
                think_doc = {
                    "room_id": room_id, 
                    "username": "AI Bot 🤖", 
                    "message": "⏳ <i>Đang phân tích dữ liệu để gợi ý...</i>", 
                    "created_at": datetime.now().strftime("%H:%M")
                }
                await manager.broadcast(json.dumps(think_doc), room_id)
                
                try:
                    
                    ai_reply = await asyncio.to_thread(planner_agent.chat, thread_id=f"room_{room_id}", user_input=data)
                    ai_doc = {
                        "room_id": room_id,
                        "username": "AI Bot 🤖",
                        "message": ai_reply,
                        "created_at": datetime.now().strftime("%H:%M")
                    }
                    room_col.insert_one(ai_doc)
                    ai_doc.pop("_id", None)
                    await manager.broadcast(json.dumps(ai_doc), room_id)
                except Exception as e:
                    err_doc = {"room_id": room_id, "username": "AI Bot 🤖", "message": "❌ Lỗi kết nối AI.", "created_at": datetime.now().strftime("%H:%M")}
                    await manager.broadcast(json.dumps(err_doc), room_id)

    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)