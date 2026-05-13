import hashlib
import psycopg2
from fastapi import HTTPException
from app.db.repository import create_user, get_user_password

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def register_user_service(username, password):
    if len(username) < 3 or len(password) < 3:
        raise HTTPException(status_code=400, detail="The username and password must be at least 3 characters long.")
    try:
        create_user(username, hash_password(password))
    except psycopg2.IntegrityError: 
        raise HTTPException(status_code=400, detail="Username already exists!")
    return {"message": "Registration successful"}

def login_user_service(username, password):
    result = get_user_password(username)
    if result and result[0] == hash_password(password):
        return {"message": "Login successful", "username": username}
    raise HTTPException(status_code=401, detail="Invalid username or password")