import os
import certifi
from pymongo import MongoClient
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.mongodb import MongoDBSaver
from app.core.state import AgentState
from app.agents.researcher import call_researcher
from app.agents.planner import call_planner
from app.core.logger import get_logger
logger = get_logger(__name__)
class TripPlannerSystem:
    def __init__(self):
        logger.info("🚀 Khởi tạo hệ thống Multi-Agent & Database...")
        uri = os.getenv("MONGODB_URI")
        if not uri:
            raise ValueError("Chưa cấu hình MONGODB_URI!")
            
        self.client = MongoClient(uri, tls=True, tlsCAFile=certifi.where())
        self.memory = MongoDBSaver(self.client)
        
        
        workflow = StateGraph(AgentState)
        workflow.add_node("researcher", call_researcher)
        workflow.add_node("planner", call_planner)
        
     
        def supervisor_router(state: AgentState):
            last_msg = state["messages"][-1].content.lower()
            quick_words = ["chào", "hello", "hi", "ok", "yes", "đồng ý", "thanh toán", "book", "tuyệt", "cảm ơn"]
            if any(word in last_msg for word in quick_words) and len(last_msg) < 50:
                logger.info("🔀 [Router]: Lệnh cơ bản -> Đi thẳng đến Planner")
                return "planner"
            logger.info("🔀 [Router]: Yêu cầu phức tạp -> Qua Researcher tìm dữ liệu")
            return "researcher"

        workflow.add_conditional_edges(START, supervisor_router, {"researcher": "researcher", "planner": "planner"})
        workflow.add_edge("researcher", "planner")
        workflow.add_edge("planner", END)
        
        self.app_graph = workflow.compile(checkpointer=self.memory)

_system_instance = None

def get_agent():
    global _system_instance
    if _system_instance is None:
        _system_instance = TripPlannerSystem()
    return _system_instance