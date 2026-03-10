from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import setup_env

setup_env()
from app.db.database import init_db
from app.api.routes import router
init_db()
app = FastAPI(title="AI Trip Planner API Pro", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")

@app.get("/")
def root():
    return {"message": "AI Trip Planner API is running", "docs": "/docs"}