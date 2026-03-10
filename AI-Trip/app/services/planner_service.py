from datetime import datetime, timezone, timedelta
from app.db.repository import insert_plan, get_plans_by_user

def save_user_plan(username: str, plan_text: str):
    vn_tz = timezone(timedelta(hours=7))
    created_at = datetime.now(vn_tz).strftime("%H:%M %d/%m/%Y")
    insert_plan(username, plan_text, created_at)
    return {"message": "Đã lưu lịch trình thành công"}

def get_user_history(username: str):
    results = get_plans_by_user(username)
    return [{"date": r[0], "plan": r[1]} for r in results]