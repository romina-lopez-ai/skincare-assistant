from __future__ import annotations

from langgraph_orchestrator.services.rag_service import retrieve_medical_context
from langgraph_orchestrator.state import ChatState


def retrieve_context(state: ChatState) -> ChatState:
    """Search the medical knowledge base and store retrieved context .

    Orchestration responsibilities (kept here):
    - Extract the latest user query from message history.
    - Write the retrieved context string to state["rag_context"].

    Model behavior (delegated to rag_service):
    - Vector-store retrieval is handled by retrieve_medical_context().
    """
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    context = retrieve_medical_context(query)

    print(f"[retrieve_context] Query: '{query[:60]}' — {len(context)} chars retrieved")

    return {**state, "rag_context": context}
