from langgraph.graph import StateGraph, END
from .state import AgentState
from .researcher import researcher_node
from .trip_planner_agent import planner_nod

workflow = StateGraph(AgentState)

workflow.add_node("researcher", researcher_node)
workflow.add_node("planner", planner_nod)

workflow.set_entry_point("researcher")
workflow.add_edge("researcher", "planner")
workflow.add_edge("planner", END)

multi_agent_app = workflow.compile()