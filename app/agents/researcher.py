import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from app.core.logger import get_logger
logger = get_logger(__name__)

researcher_prompt = """You are a Travel Data Research Agent (Researcher Agent) for Navia.

Responsibilities: Use tools (Google Search, Internal Knowledge) to find information as requested by clients.

**SPECIAL NOTE:** If data is available from user-uploaded documents, prioritize using that information for analysis and responses.

**IMPORTANT:** Only return RAW DATA (facts, prices, weather). Absolutely DO NOT design detailed itineraries."""

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
        
        # Sửa lại công cụ google_search_tool
        search = SerpAPIWrapper()

        def search_with_media(query: str):
            # Tìm kiếm thông tin tổng quát
            results = search.run(query)
            # Tìm kiếm link hình ảnh (sử dụng tính năng search_type của SerpAPI)
            image_results = search.results(f"{query} photos")
            images = [item.get("thumbnail") for item in image_results.get("images_results", [])[:3]]
            
            return f"Information: {results}\nImages: {', '.join(images)}"

        google_search_tool = Tool(
            name="google_search", 
            func=search_with_media, 
            description="Search for flight information, weather, and prices online. Returns both text and relevant images."
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