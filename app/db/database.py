import os
import psycopg2

def get_db_connection():
    """Lấy kết nối từ biến môi trường DATABASE_URL"""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("Chưa cấu hình DATABASE_URL trong file .env")
    return psycopg2.connect(db_url)

def init_db():
    """Khởi tạo các bảng bằng cú pháp PostgreSQL"""
    conn = get_db_connection()
    c = conn.cursor()
    
   
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR(255) PRIMARY KEY, 
                    password VARCHAR(255))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS saved_plans (
                    id SERIAL PRIMARY KEY, 
                    username VARCHAR(255), 
                    plan_text TEXT, 
                    created_at VARCHAR(50))''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS threads (
                    id VARCHAR(255) PRIMARY KEY,
                    username VARCHAR(255),
                    title VARCHAR(255),
                    created_at VARCHAR(50))''')
                    
   
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id SERIAL PRIMARY KEY,
                    thread_id VARCHAR(255),
                    role VARCHAR(50),
                    content TEXT,
                    created_at VARCHAR(50))''')
    
    conn.commit()
    conn.close()