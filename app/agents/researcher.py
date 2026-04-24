import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
import urllib.request
import urllib.parse
import re
from app.core.logger import get_logger
logger = get_logger(__name__)

researcher_prompt = """You are a Travel Data Research Agent (Researcher Agent) for Navia.

Responsibilities: Use tools (Google Search, Internal Knowledge) to find information as requested by clients.

**SPECIAL NOTE:** If data is available from user-uploaded documents, prioritize using that information for analysis and responses.

**IMPORTANT:** Only return RAW DATA (facts, prices, weather). Absolutely DO NOT design detailed itineraries.
2. Absolutely do not communicate directly with users; never say "I cannot create a payment form" or explain your limitations.

3. Scheduling and payment are the responsibility of the Planner Agent. Simply find the data and silently pass it on to the Planner."""

_researcher_agent_instance = None
_global_vectorstore = None 

def get_researcher_agent():
    global _researcher_agent_instance
    global _global_vectorstore
    
    if _researcher_agent_instance is None:
        logger.info("Initialize the Researcher Agent and load the RAG data for the first time...")
        
        knowledge_path = "app/data/travel_knowledge.txt"
        if not os.path.exists("app/data"): 
            os.makedirs("app/data")
        if not os.path.exists(knowledge_path):
            with open(knowledge_path, "w", encoding="utf-8") as f:
                f.write("Cancellation Policy: Cancellation before 7 days allows 100% refund.\n")
                
        with open(knowledge_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.create_documents([text])
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
        
        _global_vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = _global_vectorstore.as_retriever(search_kwargs={"k": 3})

        async def aquery_knowledge(query: str) -> str:
            docs = await retriever.ainvoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        rag_tool = Tool(
            name="internal_travel_knowledge",
            func=lambda x: x,
            coroutine=aquery_knowledge,
            description="Use this tool to read company policies and user-uploaded documents."
        )
        
        def enhanced_search(query: str) -> str:
            text_search = SerpAPIWrapper()
            text_result = text_search.run(query)
            image_url = "NONE"
            try:
                img_search = SerpAPIWrapper(params={"tbm": "isch"})
                img_raw = img_search.results(query + " travel high quality")
                if "images_results" in img_raw and len(img_raw["images_results"]) > 0:
                    image_url = img_raw["images_results"][0]["original"]
            except Exception:
                pass 
                
            video_id = "NONE"
            try:
                search_url = "https://www.youtube.com/results?search_query=" + urllib.parse.quote(query + " du lịch review 4k")
                req = urllib.request.Request(search_url, headers={'User-Agent': 'Mozilla/5.0'})
                html = urllib.request.urlopen(req, timeout=5)
                video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
                if video_ids:
                    video_id = video_ids[0]
            except Exception:
                pass 
            return f"INFORMATION: {text_result}\n\n[IMPORTANT] REAL IMAGE LINK FOR USE: {image_url}\n[IMPORTANT] REAL YOUTUBE ID FOR USE: {video_id}"

        google_search_tool = Tool(
            name="google_search", 
            func=enhanced_search, 
            description="Search for location information and automatically attach real Image and YouTube Video links."
        )

        researcher_tools = [google_search_tool, rag_tool]
        llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.1)

        _researcher_agent_instance = create_react_agent(llm, tools=researcher_tools)
        
    return _researcher_agent_instance

def add_document_to_rag(text_content: str):
    global _global_vectorstore
    if _global_vectorstore is None:
        get_researcher_agent() 
        
    logger.info("Embedding document to FAISS RAG...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.create_documents([f"[DOCUMENT USERS UPLOAD]:\n{text_content}"])
    _global_vectorstore.add_documents(splits)
    logger.info("✅FAISS has been successfully updated!")

async def call_researcher(state: dict):
    logger.info("🔍 [Researcher Agent] Retrivalling data from document....")
    agent = get_researcher_agent()
    input_messages = [SystemMessage(content=researcher_prompt)] + state["messages"]
    result = await agent.ainvoke({"messages": input_messages})
    return {"messages": [result["messages"][-1]]}