import os
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent

researcher_prompt = """Bạn là Tác nhân Nghiên cứu Dữ liệu Du lịch (Researcher Agent) của Navia.
Nhiệm vụ: Sử dụng công cụ (Google Search, Internal Knowledge) để tìm kiếm thông tin theo yêu cầu của khách hàng.
QUAN TRỌNG: Chỉ trả về DỮ LIỆU THÔ (facts, giá cả, thời tiết). Tuyệt đối KHÔNG thiết kế lịch trình chi tiết."""

# Biến toàn cục để lưu trữ Agent (Lazy Loading)
_researcher_agent_instance = None

def get_researcher_agent():
    global _researcher_agent_instance
    if _researcher_agent_instance is None:
        print("🚀 Khởi tạo Researcher Agent và nạp dữ liệu RAG lần đầu tiên...")
        
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
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        async def aquery_knowledge(query: str) -> str:
            docs = await retriever.ainvoke(query)
            return "\n\n".join([doc.page_content for doc in docs])

        rag_tool = Tool(
            name="internal_travel_knowledge",
            func=lambda x: x,
            coroutine=aquery_knowledge,
            description="Use this tool to look up company policies, cancellation rules, and exclusive travel guides."
        )

        search = SerpAPIWrapper()
        google_search_tool = Tool(
            name="google_search", 
            func=search.run, 
            coroutine=search.arun,
            description="Search for flight information, weather, and prices online."
        )

        researcher_tools = [google_search_tool, rag_tool]
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)

        # Đã loại bỏ hoàn toàn tham số messages_modifier/prompt để tránh lỗi phiên bản thư viện
        _researcher_agent_instance = create_react_agent(llm, tools=researcher_tools)
        
    return _researcher_agent_instance

async def call_researcher(state: dict):
    print("🔍 [Researcher Agent] Đang tra cứu dữ liệu...")
    
    # Chỉ gọi khởi tạo khi có khách nhắn tin
    agent = get_researcher_agent()
    
    # Chèn System Prompt thủ công vào lịch sử tin nhắn
    input_messages = [SystemMessage(content=researcher_prompt)] + state["messages"]
    
    result = await agent.ainvoke({"messages": input_messages})
    return {"messages": [result["messages"][-1]]}