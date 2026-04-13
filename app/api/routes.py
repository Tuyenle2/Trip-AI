import json
import os
import uuid
import asyncio
from datetime import datetime, timedelta
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Depends, status, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
import traceback
from langchain_core.messages import HumanMessage
from app.models.schemas import UserAuth, ChatRequest, ChatResponse, SavePlanRequest, ThreadCreateRequest
from app.core.security import SecurityGuard
from app.agents.trip_planner_agent import TripPlannerAgent
from app.auth import verify_password, get_password_hash, create_access_token, SECRET_KEY, ALGORITHM
import redis.asyncio as aioredis 

router = APIRouter()
load_dotenv()

# --- CẤU HÌNH REDIS ---
REDIS_URL = os.getenv("REDIS_URL", "")
redis_client = aioredis.from_url(
    REDIS_URL,
    decode_responses=True,
    ssl_cert_reqs=None, 
    ssl_check_hostname=False,
    socket_timeout=10,
    socket_connect_timeout=10,
    retry_on_timeout=True
)

# --- SINGLETON LAZY LOAD AGENT ---
planner_agent_instance = None
def get_agent():
    global planner_agent_instance
    if planner_agent_instance is None:
        print("🚀 Khởi tạo Agent khi có người dùng đầu tiên nhắn tin...")
        uri = os.getenv("DATABASE_URL") or os.getenv("MONGODB_URI") 
        planner_agent_instance = TripPlannerAgent(mongodb_uri=uri)
    return planner_agent_instance

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/token")

# --- MODELS ---
class UserCreate(BaseModel):
    username: str
    password: str

class RoomAuthRequest(BaseModel):
    room_id: str
    password: str

class TransactionCreate(BaseModel):
    username: str
    service_name: str
    amount: str

class PaymentVerificationRequest(BaseModel):
    username: str
    card_name: str
    card_number: str
    cvv: str

# --- AUTH & THREAD ROUTES ---
@router.post("/register")
async def register_user(user: UserCreate):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    users_col = db.get_collection("users")
    if users_col.find_one({"username": user.username}):
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
    hashed_password = get_password_hash(user.password)
    users_col.insert_one({"username": user.username, "password_hash": hashed_password})
    return {"message": "Đăng ký thành công!"}

@router.post("/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    users_col = db.get_collection("users")
    user_in_db = users_col.find_one({"username": form_data.username})
    if not user_in_db or not verify_password(form_data.password, user_in_db["password_hash"]):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Sai mật khẩu")
    access_token = create_access_token(data={"sub": form_data.username})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/threads")
def api_create_thread(req: ThreadCreateRequest):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    threads_col = db.get_collection("threads")
    thread_id = str(uuid.uuid4())
    threads_col.insert_one({"thread_id": thread_id, "username": req.username, "title": req.title, "created_at": datetime.now()})
    return {"id": thread_id, "thread_id": thread_id, "title": req.title}

@router.get("/threads/{username}")
def api_get_threads(username: str):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    threads = list(db.get_collection("threads").find({"username": username}, {"_id": 0}).sort("created_at", -1))
    for t in threads: t["id"] = t.get("thread_id")
    return threads

@router.get("/messages/{thread_id}")
def api_get_messages(thread_id: str):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    return list(db.get_collection("messages").find({"thread_id": thread_id}, {"_id": 0}).sort("created_at", 1))

# --- CHAT STREAMING CÁ NHÂN ---
@router.post("/chat/stream")
async def api_chat_stream(request: ChatRequest):
    if not SecurityGuard.is_input_safe(request.message):
        raise HTTPException(status_code=400, detail="Vi phạm Guardrails")

    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    msg_col = db.get_collection("messages")

    msg_col.insert_one({
        "thread_id": request.thread_id,
        "role": "user",
        "content": request.message,
        "created_at": datetime.now()
    })

    async def event_generator():
        try:
            full_text = ""
            # QUAN TRỌNG: Dùng agent.app_graph (Đã tích hợp đủ Tools và Checkpointer)
            async for event in agent.app_graph.astream_events(
                {"messages": [HumanMessage(content=request.message)]},
                version="v2",
                config={"configurable": {"thread_id": request.thread_id}}
            ):
                kind = event["event"]

                if kind == "on_tool_start":
                    yield f"data: {json.dumps({'type': 'tool', 'name': event['name'], 'query': event['data'].get('input', {})})}\n\n"

                elif kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        text = "".join([i.get("text", "") for i in content if isinstance(i, dict)]) if isinstance(content, list) else content
                        if text:
                            full_text += text
                            yield f"data: {json.dumps({'type': 'content', 'data': text})}\n\n"

            if full_text:
                msg_col.insert_one({
                    "thread_id": request.thread_id,
                    "role": "ai",
                    "content": full_text,
                    "created_at": datetime.now()
                })
                yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            print("=== LỖI CRASH TRONG MULTI-AGENT GRAPH ===")
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

# --- GROUP CHAT & REDIS PUB/SUB ---
class RedisConnectionManager:
    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, room_id: str):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)
        await websocket.send_text(json.dumps({
            "username": "HỆ THỐNG ⚙️", 
            "message": "✅ Đã thông suốt đường truyền Redis!", 
            "created_at": (datetime.now() + timedelta(hours=7)).strftime("%H:%M")
        }))

    def disconnect(self, websocket: WebSocket, room_id: str):
        if room_id in self.active_connections:
            self.active_connections[room_id] = [w for w in self.active_connections[room_id] if w != websocket]

manager = RedisConnectionManager()

@router.websocket("/ws/room/{room_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, username: str):
    await manager.connect(websocket, room_id)
    
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    room_col = db.get_collection("room_messages") 

    try:
        history = list(room_col.find({"room_id": room_id}, {"_id": 0}).sort("created_at", 1).limit(50))
        for msg in history:
            await websocket.send_text(json.dumps(msg))
    except Exception as e:
        print(f"History Error: {e}")
        
    async def redis_listener():
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(f"chat_{room_id}")
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    await websocket.send_text(message["data"])
        except Exception: pass
        finally:
            await pubsub.unsubscribe(f"chat_{room_id}")
            await pubsub.close()

    listener_task = asyncio.create_task(redis_listener())

    try:
        while True:
            data = await websocket.receive_text()
            vn_now = (datetime.now() + timedelta(hours=7)).strftime("%H:%M")

            msg_doc = {"room_id": room_id, "username": username, "message": data, "created_at": vn_now}
            room_col.insert_one(msg_doc.copy())
            msg_doc.pop("_id", None)
            await redis_client.publish(f"chat_{room_id}", json.dumps(msg_doc))
            
            if "@AI" in data.upper() or "@BOT" in data.upper():
                await redis_client.publish(f"chat_{room_id}", json.dumps({
                    "room_id": room_id, "username": "AI Bot 🤖", "message": "⏳ Đang phân tích...", "created_at": vn_now
                }))
                
                try:
                    recent_msgs = list(room_col.find({"room_id": room_id}, {"_id": 0}).sort("created_at", -1).limit(10))
                    recent_msgs.reverse()
                    context = "\n".join([f"{m['username']}: {m['message']}" for m in recent_msgs])
                    
                    full_text = ""
                    # Gọi Agent AI 
                    async for event in agent.app_graph.astream_events(
                        {"messages": [HumanMessage(content=f"Bối cảnh: {context}\n\nCâu hỏi: {data}")]},
                        version="v2",
                        config={"configurable": {"thread_id": f"room_{room_id}"}}
                    ):
                        if event["event"] == "on_chat_model_stream":
                            chunk_content = event["data"]["chunk"].content
                            text = "".join([c.get("text", "") for c in chunk_content if isinstance(c, dict)]) if isinstance(chunk_content, list) else chunk_content
                            if text: full_text += text
                    
                    if full_text:
                        ai_doc = {"room_id": room_id, "username": "AI Bot 🤖", "message": full_text, "created_at": vn_now}
                        room_col.insert_one(ai_doc.copy())
                        ai_doc.pop("_id", None)
                        await redis_client.publish(f"chat_{room_id}", json.dumps(ai_doc))
                except Exception as e:
                    await redis_client.publish(f"chat_{room_id}", json.dumps({
                        "room_id": room_id, "username": "AI Bot 🤖", "message": f"❌ Lỗi: {str(e)}", "created_at": vn_now
                    }))
    except WebSocketDisconnect:
        manager.disconnect(websocket, room_id)
        listener_task.cancel()

# --- OTHER ROUTES ---
@router.post("/save_plan")
def save_plan(req: SavePlanRequest):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    db.get_collection("history").insert_one({"username": req.username, "plan": req.plan_text, "date": datetime.now().strftime("%d/%m/%Y %H:%M")})
    return {"status": "success"}

@router.get("/history/{username}")
def get_history(username: str):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    return list(db.get_collection("history").find({"username": username}, {"_id": 0}))[::-1]

@router.post("/transactions")
async def save_transaction(transaction: TransactionCreate):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    db.get_collection("transactions").insert_one({"username": transaction.username, "service_name": transaction.service_name, "amount": transaction.amount, "created_at": datetime.now().strftime("%d/%m/%Y %H:%M")})
    return {"status": "success"}

@router.get("/transactions/{username}")
async def get_user_transactions(username: str):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    return list(db.get_collection("transactions").find({"username": username}, {"_id": 0}))

@router.post("/verify-payment")
async def verify_payment(req: PaymentVerificationRequest):
    await asyncio.sleep(1.5)
    if req.card_number.startswith("0000"): raise HTTPException(status_code=400, detail="Thẻ bị từ chối!")
    return {"status": "success", "message": "Thành công!"}

@router.post("/room/join")
def join_room(req: RoomAuthRequest):
    agent = get_agent()
    db = agent.client.get_database("ai_trip_planner_db")
    rooms_col = db.get_collection("rooms")
    room_id_upper = req.room_id.upper()
    room = rooms_col.find_one({"room_id": room_id_upper})
    if room:
        if room["password"] != req.password: raise HTTPException(status_code=400, detail="Sai mật khẩu!")
        return {"status": "success"}
    rooms_col.insert_one({"room_id": room_id_upper, "password": req.password, "created_at": datetime.now()})
    return {"status": "success"}