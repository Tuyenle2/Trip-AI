def researcher_node(state: AgentState):
    query = state["messages"][-1].content
    # Giả sử bạn dùng hàm search đã có của mình
    search_results = your_mongodb_vector_search(query) 
    
    return {
        "context_data": search_results,
        "messages": [BaseMessage(content="Đang tra cứu dữ liệu du lịch...", role="assistant")]
    }