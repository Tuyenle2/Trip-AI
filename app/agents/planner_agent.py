# app/agents/planner_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from app.tools.time_tool import get_current_time

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
llm_with_tools = llm.bind_tools([get_current_time])

planner_prompt = """You are Navia's 'Schedule Design Manager'.

Your task is to receive data reports from the Researcher and design a complete itinerary.

Rules:
1. Present the itinerary in the standard format (Day 1, Day 2...).

2. Include the tag [MAP_PLACES: ...] at the end.

3. If the client confirms the itinerary, ask if they want to pay. If the client says "Ok/Agree", generate [PAYMENT_FORM: Service Name | Price].

4. DO NOT invent locations that are not included in the Researcher's report.."""


async def planner_node(state: dict):
    print(" [Planner Agent] Designing the schedule...")
    messages = [SystemMessage(content=planner_prompt)] + state["messages"]
    response = await llm_with_tools.ainvoke(messages)
    return {"messages": [response]}