import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import UserAuth, ChatRequest, ChatResponse, SavePlanRequest, ThreadCreateRequest
from app.services.auth_service import register_user_service, login_user_service
from app.services.planner_service import save_user_plan, get_user_history
from app.core.security import SecurityGuard
from app.agents.trip_planner_agent import TripPlannerAgent
from app.db.repository import create_thread, get_threads_by_user, insert_message, get_messages_by_thread

router = APIRouter()
planner_agent = TripPlannerAgent()

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
    
    # Lưu tin nhắn của user vào DB
    insert_message(request.thread_id, "user", request.message)
    
    async def event_generator():
        full_text = ""
        # Duyệt qua từng hành động của AI
        async for event in planner_agent.achat_stream(request.thread_id, request.message):
            kind = event["event"]
            
            # Nếu AI quyết định DÙNG TOOL (Google Search)
            if kind == "on_tool_start":
                tool_name = event["name"]
                query_data = event["data"].get("input", {})
                query = query_data.get("query", str(query_data))
                yield f"data: {json.dumps({'type': 'tool', 'name': tool_name, 'query': query})}\n\n"
            
            # Nếu AI đang TRẢ VỀ CHỮ (Typewriter effect)
            elif kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if getattr(chunk, "content", None):
                    content = chunk.content
                    
                    # --- BẢN VÁ LỖI NẰM Ở ĐÂY ---
                    text_piece = ""
                    # Kiểm tra xem AI đang trả về List hay String
                    if isinstance(content, list):
                        # Lọc lấy phần text từ trong list các dictionary
                        text_piece = "".join([item.get("text", "") for item in content if isinstance(item, dict)])
                    elif isinstance(content, str):
                        text_piece = content
                        
                    # Chỉ cộng vào và gửi lên UI nếu thực sự có chữ
                    if text_piece:
                        full_text += text_piece
                        yield f"data: {json.dumps({'type': 'content', 'data': text_piece})}\n\n"
                    # -----------------------------
        
        # Sau khi dòng chảy kết thúc, lưu trọn bộ câu trả lời vào Database
        insert_message(request.thread_id, "ai", full_text)

    # Trả về kết nối mở liên tục (SSE)
    return StreamingResponse(event_generator(), media_type="text/event-stream")