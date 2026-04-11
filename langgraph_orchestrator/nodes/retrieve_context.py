from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def retrieve_context(state: ChatState) -> ChatState:
    """Search the medical knowledge base and store retrieved context.

    Responsibilities:
    - Extract a query from the latest user message.
    - Pass the query to the RAG retriever (vector store + embedding model).
    - Format the retrieved document chunks into a [CONTEXT] block.
    - Store the formatted context in state["rag_context"] so the rag_answer
      node can inject it into the LLM prompt.

    When this node is active:
        diagnosis_done = True AND the follow-up question is knowledge-type.

    Key rule:
        RAG is only used for knowledge / follow-up questions such as treatment,
        causes, triggers, prevention, and prognosis.  It is NOT used during
        symptom collection or for conversational replies.

    Real implementation will:
    - Encode the query with the same embedding model used to build the index.
    - Retrieve the top-k most relevant chunks from the FAISS / Chroma store.
    - Assemble chunks with source metadata into a structured [CONTEXT] block.

    Returns:
        Updated state with rag_context set to the retrieved [CONTEXT] string.
    """
    # Extract latest user query
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    # --- Placeholder retrieval ---
    mock_context = (
        "[CONTEXT]\n"
        "Source: Dermatology Handbook, Chapter 4 — Eczema Management\n"
        "Eczema (atopic dermatitis) is commonly managed with emollients applied "
        "regularly to moisturise the skin barrier. Mild-to-moderate flares are "
        "typically treated with low-to-medium potency topical corticosteroids. "
        "Trigger avoidance (soaps, certain fabrics, allergens) is an important "
        "part of long-term management.\n"
        "[/CONTEXT]"
    )

    print(f"[retrieve_context] Query: '{query[:60]}...'  — {len(mock_context)} chars retrieved")

    return {**state, "rag_context": mock_context}
