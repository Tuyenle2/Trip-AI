# app/core/graph.py
from langgraph.graph import StateGraph, START, END, MessagesState
from app.agents.researcher_agent import researcher_node
from app.agents.planner_agent import planner_node

# Hàm định tuyến thông minh (Router)
def route_request(state: MessagesState):
    last_msg = state["messages"][-1].content.lower()
    
    # Những câu giao tiếp cơ bản không cần tìm kiếm
    if any(word in last_msg for word in ["cảm ơn", "chào", "hello", "thanh toán", "đồng ý", "ok"]):
        return "planner"
    
    # Các câu hỏi về du lịch, cần tra cứu
    return "researcher"

# Khởi tạo đồ thị
workflow = StateGraph(MessagesState)

# Thêm 2 Agent vào đồ thị
workflow.add_node("researcher", researcher_node)
workflow.add_node("planner", planner_node)

# Bắt đầu đồ thị -> Gọi Router để phân luồng
workflow.add_conditional_edges(
    START,
    route_request,
    {
        "researcher": "researcher",
        "planner": "planner"
    }
)

# Nếu đi qua Researcher xong thì bắt buộc chuyển cho Planner viết lịch
workflow.add_edge("researcher", "planner")

# Xong Planner thì kết thúc trả kết quả cho người dùng
workflow.add_edge("planner", END)

# Đóng gói và biên dịch (kèm Checkpointer nếu có)
multi_agent_app = workflow.compile(checkpointer=memory)