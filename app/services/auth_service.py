import hashlib
import sqlite3
from fastapi import HTTPException
from app.db.repository import create_user, get_user_password

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user_service(username, password):
    if len(username) < 3 or len(password) < 3:
        raise HTTPException(status_code=400, detail="Tài khoản và mật khẩu phải >= 3 ký tự.")
    try:
        create_user(username, hash_password(password))
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
    return {"message": "Đăng ký thành công"}

def login_user_service(username, password):
    result = get_user_password(username)
    if result and result[0] == hash_password(password):
        return {"message": "Đăng nhập thành công", "username": username}
    raise HTTPException(status_code=401, detail="Sai tài khoản hoặc mật khẩu")