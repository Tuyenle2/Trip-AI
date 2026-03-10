import sqlite3

DB_PATH = 'ai_planner.db'

def get_db_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # Bảng người dùng và lịch trình đã lưu
    c.execute('''CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS saved_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, 
                    username TEXT, 
                    plan_text TEXT, 
                    created_at TEXT)''')
    
    # --- 2 BẢNG MỚI CHO TÍNH NĂNG NHIỀU LUỒNG CHAT ---
    c.execute('''CREATE TABLE IF NOT EXISTS threads (
                    id TEXT PRIMARY KEY,
                    username TEXT,
                    title TEXT,
                    created_at TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    thread_id TEXT,
                    role TEXT,
                    content TEXT,
                    created_at TEXT)''')
    conn.commit()
    conn.close()