# app/core/graph.py
from langgraph.graph import StateGraph, START, END, MessagesState
from app.agents.researcher_agent import researcher_node
from app.agents.planner_agent import planner_node

def route_request(state: MessagesState):
    last_msg = state["messages"][-1].content.lower()
    if any(word in last_msg for word in ["cảm ơn", "chào", "hello", "thanh toán", "đồng ý", "ok"]):
        return "planner"
    return "researcher"

workflow = StateGraph(MessagesState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("planner", planner_node)
workflow.add_conditional_edges(
    START,
    route_request,
    {
        "researcher": "researcher",
        "planner": "planner"
    }
)
workflow.add_edge("researcher", "planner")
workflow.add_edge("planner", END)
multi_agent_app = workflow.compile(checkpointer=memory)