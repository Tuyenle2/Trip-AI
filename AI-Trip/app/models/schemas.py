from pydantic import BaseModel

class UserAuth(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    username: str
    thread_id: str  
    message: str

class ChatResponse(BaseModel):
    reply: str
    is_safe: bool

class SavePlanRequest(BaseModel):
    username: str
    plan_text: str

# MỚI: Schema để tạo luồng chat
class ThreadCreateRequest(BaseModel):
    username: str
    title: str = "Chuyến đi mới"