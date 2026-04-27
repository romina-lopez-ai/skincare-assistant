from __future__ import annotations

from langgraph_orchestrator.services.llm_service import generate_followup_response
from langgraph_orchestrator.state import ChatState


def rag_answer(state: ChatState) -> ChatState:
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
