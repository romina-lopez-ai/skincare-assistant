"""rag_service.py — Retrieval backend abstraction for the dermatology chatbot.

Public interface (called by node files)
----------------------------------------
- retrieve_medical_context(query) → formatted [CONTEXT]…[/CONTEXT] string

The public function reads config.USE_MOCK_RAG and delegates to either the
mock implementation (_mock_retrieve_medical_context) or the real
implementation (_real_retrieve_medical_context).

Teammate integration guide
--------------------------
1. Set USE_MOCK_RAG = False in config.py (or export USE_MOCK_RAG=0).
2. Fill in _real_retrieve_medical_context below with your vector-store call.
3. The public function and all node files stay unchanged.

The _mock_* function is preserved and continues to work exactly as before.
"""

from __future__ import annotations

from langgraph_orchestrator.config import USE_MOCK_RAG


# ===========================================================================
# Public interface — called by node files
# ===========================================================================

def retrieve_medical_context(query: str) -> str:
    """Search the medical knowledge base for context relevant to query.

    Args:
        query: The user's follow-up question text.

    Returns:
        A formatted [CONTEXT]...[/CONTEXT] string to be injected into the
        LLM prompt.  If no relevant chunks are found, the real implementation
        should return an empty or minimal context so the LLM can say it does
        not have enough information.
    """
    if USE_MOCK_RAG:
        return _mock_retrieve_medical_context(query)
    return _real_retrieve_medical_context(query)


# ===========================================================================
# Mock implementation — fixed response used during development
# ===========================================================================

def _mock_retrieve_medical_context(_query: str) -> str:
    """MOCK: always returns a fixed eczema-management excerpt regardless of query."""
    _ = _query  # mock — query is not used; input intentionally ignored
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


# ===========================================================================
# Real implementation — replace this body with an actual vector-store call
# ===========================================================================

def _real_retrieve_medical_context(query: str) -> str:
    """REAL: embed the query and search the vector store for relevant chunks.

    Implementation guide:
    1. Encode query with the same embedding model used to build the index.
    2. Search the FAISS / Chroma / Pinecone store for the top-k chunks.
    3. Assemble the retrieved chunks (with optional source metadata) into a
       [CONTEXT]...[/CONTEXT] block — the same format as the mock above.
    4. Return the formatted string.  Return an empty string (or a minimal
       "[CONTEXT][/CONTEXT]" block) when no relevant chunks are found.
    """
    raise NotImplementedError(
        f"_real_retrieve_medical_context(query={query!r}): replace this with "
        "your vector-store retrieval call. "
        "Set USE_MOCK_RAG = True in config.py to use the mock instead."
    )
