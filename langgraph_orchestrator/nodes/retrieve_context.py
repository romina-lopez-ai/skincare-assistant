"""retrieve_context.py — LangGraph node for context retrieval."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#from langgraph_orchestrator.services.rag_service import retrieve_medical_context
from langgraph_orchestrator.state import ChatState
from rag.retriever import retriever


def retrieve_context(state: ChatState) -> ChatState:
    """
    LangGraph node: retrieve medical context for the last user message.

    This node calls the RAG retriever and stores the result in state["rag_context"].
    """
    # Extract last user query
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    if not query:
        return {**state, "rag_context": ""}

    # RAG logic
    result = retriever({"question": query})
    evidences = result.get("evidences", [])

    # Format context for the LLM
    if evidences:
        context_parts = [ev.get("text", "") for ev in evidences]
        rag_context = "[CONTEXT]\n\n" + "\n\n".join(context_parts) + "\n\n[/CONTEXT]"
    else:
        rag_context = ""

    print(f"[retrieve_context] Query: '{query[:60]}' — {len(evidences)} chunks")

    return {**state, "rag_context": rag_context}

# Mock implementation of the retrieve_context node.
#def retrieve_context(state: ChatState) -> ChatState:
"""Search the medical knowledge base and store retrieved context.

    Orchestration responsibilities (kept here):
    - Extract the latest user query from message history.
    - Write the retrieved context string to state["rag_context"].

    Model behavior (delegated to rag_service):
    - Vector-store retrieval is handled by retrieve_medical_context().
    """
"""
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    context = retrieve_medical_context(query)

    print(f"[retrieve_context] Query: '{query[:60]}' — {len(context)} chars retrieved")

    return {**state, "rag_context": context}
"""