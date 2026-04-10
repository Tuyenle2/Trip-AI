import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import BaseMessage
from .state import AgentState

def init_retriever():
    knowledge_path = "app/data/travel_knowledge.txt"
    
    # Tạo thư mục và file mẫu nếu chưa tồn tại
    if not os.path.exists("app/data"): 
        os.makedirs("app/data")
    if not os.path.exists(knowledge_path):
        with open(knowledge_path, "w", encoding="utf-8") as f:
            f.write("Chính sách hoàn hủy: Hủy trước 7 ngày hoàn 100%.\n")

    # Đọc dữ liệu từ cẩm nang
    with open(knowledge_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Cắt nhỏ văn bản (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.create_documents([text])

    # Nhúng (Embedding) và tạo Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    
    # Trả về công cụ tìm kiếm, lấy 3 kết quả liên quan nhất (k=3)
    return vectorstore.as_retriever(search_kwargs={"k": 3})

retriever_instance = None

def researcher_node(state: AgentState):
    global retriever_instance
    # Kỹ thuật Lazy Load: Khởi tạo FAISS khi có khách chat lần đầu
    if retriever_instance is None:
        print("🚀 Đang khởi tạo RAG Database lần đầu tiên...")
        retriever_instance = init_retriever()
        
    last_user_message = state["messages"][-1].content
    print(f"🔍 [Researcher] Đang tìm kiếm thông tin cho: '{last_user_message}'")
    
    # Sử dụng retriever_instance thay vì retriever
    docs = retriever_instance.invoke(last_user_message)
    
    search_results = "\n\n".join([doc.page_content for doc in docs])
    if not search_results.strip():
        search_results = "Không có dữ liệu nội bộ liên quan đến yêu cầu này."
        
    print(f"✅ [Researcher] Đã tìm thấy: {search_results[:50]}...")
    
    return {
        "context_data": search_results,
    }