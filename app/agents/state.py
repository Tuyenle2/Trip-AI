from typing import Annotated, TypedDict, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # 'add_messages' giúp cộng dồn lịch sử chat
    messages: Annotated[List[BaseMessage], add_messages]
    # Nơi chứa thông tin khách sạn/địa điểm tìm được từ RAG hoặc MCP
    context_data: str