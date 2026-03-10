from app.db.database import get_db_connection
import uuid
from datetime import datetime, timezone, timedelta

def get_user_password(username: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    conn.close()
    return result

def create_user(username: str, hashed_password: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hashed_password))
    conn.commit()
    conn.close()

def insert_plan(username: str, plan_text: str, created_at: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO saved_plans (username, plan_text, created_at) VALUES (?, ?, ?)", 
              (username, plan_text, created_at))
    conn.commit()
    conn.close()

def get_plans_by_user(username: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT created_at, plan_text FROM saved_plans WHERE username=? ORDER BY id DESC", (username,))
    results = c.fetchall()
    conn.close()
    return results

def create_thread(username: str, title: str):
    thread_id = str(uuid.uuid4())
    vn_tz = timezone(timedelta(hours=7))
    created_at = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S")
    
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO threads (id, username, title, created_at) VALUES (?, ?, ?, ?)", 
              (thread_id, username, title, created_at))
    conn.commit()
    conn.close()
    return {"id": thread_id, "title": title, "created_at": created_at}

def get_threads_by_user(username: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, title, created_at FROM threads WHERE username=? ORDER BY created_at DESC", (username,))
    results = c.fetchall()
    conn.close()
    return [{"id": r[0], "title": r[1], "created_at": r[2]} for r in results]

def insert_message(thread_id: str, role: str, content: str):
    vn_tz = timezone(timedelta(hours=7))
    created_at = datetime.now(vn_tz).strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO messages (thread_id, role, content, created_at) VALUES (?, ?, ?, ?)", 
              (thread_id, role, content, created_at))
    conn.commit()
    conn.close()

def get_messages_by_thread(thread_id: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT role, content FROM messages WHERE thread_id=? ORDER BY id ASC", (thread_id,))
    results = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1]} for r in results]