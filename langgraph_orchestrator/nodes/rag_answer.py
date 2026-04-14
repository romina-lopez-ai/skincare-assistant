"""rag_answer.py — LangGraph node for RAG-grounded answer generation."""
from __future__ import annotations
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

#from langgraph_orchestrator.services.llm_service import generate_followup_response
from langgraph_orchestrator.state import ChatState
from rag.answer_generator import answer_generator


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> block from response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def rag_answer(state: ChatState) -> ChatState:
    """
    LangGraph node: generate a grounded answer using retrieved context.
    """
    # Get last user query
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    # Get context from state
    context = state.get("rag_context", "")

    # Build evidences
    evidences = []
    if context and context != "":
        evidences = [{"text": context, "score": 1.0, "meta": {"source": "retrieved"}}]

    # Generate answer
    result = answer_generator({"question": query, "evidences": evidences})
    raw_answer = result.get("answer", "")

    # Optionally strip <think> tags for user-facing response
    """
    user_answer = _strip_think_tags(raw_answer)
    if not user_answer:
        user_answer = raw_answer
    """
    user_answer = raw_answer
    # Append to messages
    messages = state["messages"] + [{"role": "assistant", "content": user_answer}]

    print(f"[rag_answer] Query: '{query[:60]}' — Answer length: {len(user_answer)} chars")

    return {
        **state,
        "messages": messages,
        "rag_context": None,
    }


#Mock implementation of the rag_answer node.
#def rag_answer(state: ChatState) -> ChatState:
"""Generate a grounded follow-up answer using retrieved context.

    Orchestration responsibilities (kept here):
    - Call the LLM service (which has access to rag_context via state).
    - Append the reply to messages.
    - Clear rag_context after use to prevent stale context on the next turn.

    Model behavior (delegated to llm_service):
    - Generating the grounded answer is handled by generate_followup_response().
      The real implementation should inject state["rag_context"] into the
      LLM prompt and instruct the model to answer only from that context.
    """
"""
    reply    = generate_followup_response(state)
    messages = state["messages"] + [{"role": "assistant", "content": reply}]

    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    print(f"[rag_answer] Answered: '{query[:60]}'")

    return {
        **state,
        "messages":    messages,
        "rag_context": None,  # Clear after use — prevents bleed into next turn
    }
"""