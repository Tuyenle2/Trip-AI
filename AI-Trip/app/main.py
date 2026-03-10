from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import setup_env

# 1. BẮT BUỘC PHẢI GỌI HÀM NÀY ĐẦU TIÊN (Để nạp file .env)
setup_env()

# 2. Sau khi đã nạp .env xong, bây giờ mới được import các file khác!
from app.db.database import init_db
from app.api.routes import router

# 3. Khởi tạo Database SQLite
init_db()

# 4. Khởi tạo ứng dụng FastAPI
app = FastAPI(title="AI Trip Planner API Pro", version="3.0")

# 5. Mở khóa CORS cho Frontend gọi
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 6. Đăng ký các Routes (API endpoints)
app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "AI Trip Planner API is running", "docs": "/docs"}