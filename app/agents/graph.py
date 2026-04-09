from langgraph.graph import StateGraph, END
from .state import AgentState
from .researcher import researcher_node
from .planner import planner_node # Bạn tạo tương tự researcher

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("planner", planner_node)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "planner")
workflow.add_edge("planner", END)

# Export cái này để routes.py gọi
multi_agent_app = workflow.compile()