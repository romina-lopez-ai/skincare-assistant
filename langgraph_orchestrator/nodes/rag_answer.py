from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def rag_answer(state: ChatState) -> ChatState:
    """Generate a grounded answer by injecting retrieved context into the LLM.

    Responsibilities:
    - Take the [CONTEXT] block from state["rag_context"].
    - Build a prompt that instructs the LLM to answer using only the provided
      context (grounded generation).
    - Call the fine-tuned LLM and append its response to state["messages"].
    - Clear state["rag_context"] after use to avoid stale context bleeding into
      the next turn.

    Key rule:
        The LLM must be instructed not to answer beyond the retrieved context.
        If the context does not contain enough information, the model should say
        so rather than hallucinating.

    Real implementation will:
    - Construct a system-level instruction such as:
        "Answer the user's question using ONLY the information in [CONTEXT].
         If the context does not cover the question, say you don't have enough
         information."
    - Pass the full message history + context to the LLM API.

    Returns:
        Updated state with the grounded assistant reply appended to messages
        and rag_context reset to None.
    """
    context = state.get("rag_context") or "[CONTEXT]\nNo context retrieved.\n[/CONTEXT]"

    # Extract latest user query for display
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    # --- Placeholder grounded LLM call ---
    mock_reply = (
        "[PLACEHOLDER — RAG answer] Based on the available medical information: "
        "Eczema is typically managed with regular moisturising using emollients and, "
        "during flares, topical corticosteroids. Identifying and avoiding personal "
        "triggers (e.g. soaps, fabrics, allergens) is also recommended."
    )

    messages = state["messages"] + [{"role": "assistant", "content": mock_reply}]

    print(f"[rag_answer] Answered query: '{query[:60]}...'")

    return {
        **state,
        "messages": messages,
        "rag_context": None,  # Clear after use
    }
