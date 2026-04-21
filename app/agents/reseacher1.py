# app/agents/researcher_agent.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import SystemMessage
from app.tools.search_tool import google_search_tool
from app.tools.rag_tool import rag_tool

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
researcher_prompt = """You are Navia's 'Tourism Data Collection Specialist'.
Your responsibilities:
1. Read customer requirements.

2. Use tools (Google Search, Internal RAG) to search for:

- Popular tourist attractions, opening hours, and ticket prices.

- Good restaurants and their ratings.

- Estimated hotel or flight prices.

3. COMPILE this into a raw data report. DO NOT create Day 1 or Day 2 itineraries.
Only return factual data."""

researcher_react_agent = create_react_agent(
    model=llm, 
    tools=[google_search_tool, rag_tool], 
    state_modifier=SystemMessage(content=researcher_prompt)
)
async def researcher_node(state: dict):
    print("🔍 [Researcher Agent] Start searching for information...")
    response = await researcher_react_agent.ainvoke({"messages": state["messages"]})
    final_report = response["messages"][-1]
    return {"messages": [final_report]}