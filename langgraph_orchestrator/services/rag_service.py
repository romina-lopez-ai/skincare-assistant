"""rag_service.py — Retrieval backend abstraction for the dermatology chatbot.

The function is a **mock** that reproduces the placeholder behavior that
previously lived inside retrieve_context.py.

Teammate integration guide
--------------------------
Replace the function body with a real vector-store retrieval call.  The
function signature and return type must not change so the node file keeps
working without modification.

Expected real implementation pattern:
    1. Encode `query` with the same embedding model used to build the index.
    2. Search the FAISS / Chroma / Pinecone store for the top-k chunks.
    3. Assemble the retrieved chunks (with optional source metadata) into the
       [CONTEXT]...[/CONTEXT] block format shown in the mock.
    4. Return the formatted string.

The [CONTEXT] block is injected into the LLM prompt by rag_answer.py and
llm_service.generate_followup_response().
"""

from __future__ import annotations


def retrieve_medical_context(query: str) -> str:
    """Search the medical knowledge base for context relevant to query.

    MOCK: always returns a fixed eczema-management excerpt regardless of query.

    Real implementation should:
    - Embed the query and perform a vector similarity search.
    - Return the top-k most relevant chunks formatted as a [CONTEXT] block.
    - If no relevant chunks are found, return an empty or minimal context
      so the LLM can say it does not have enough information.

    Args:
        query: The user's follow-up question text.

    Returns:
        A formatted [CONTEXT]...[/CONTEXT] string to be injected into the
        LLM prompt.
    """
    return (
        "[CONTEXT]\n"
        "Source: Dermatology Handbook, Chapter 4 — Eczema Management\n"
        "Eczema (atopic dermatitis) is commonly managed with emollients applied "
        "regularly to moisturise the skin barrier. Mild-to-moderate flares are "
        "typically treated with low-to-medium potency topical corticosteroids. "
        "Trigger avoidance (soaps, certain fabrics, allergens) is an important "
        "part of long-term management.\n"
        "[/CONTEXT]"
    )
