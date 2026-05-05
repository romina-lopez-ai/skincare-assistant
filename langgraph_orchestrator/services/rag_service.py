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
# Real implementation — ChromaDB + BM25 + FlashRank retrieval
# ===========================================================================

def _real_retrieve_medical_context(query: str) -> str:
    """REAL: retrieve grounded clinical context from ChromaDB via rag/retriever.py.

    Pipeline (implemented in rag/retriever.py):
        1. Query expansion  — append clinical keywords
        2. Dense retrieval  — ChromaDB semantic search
        3. BM25 retrieval   — lexical search over same corpus
        4. RRF fusion       — merges + deduplicates both lists
        5. FlashRank rerank — cross-encoder refines top candidates
        6. Score filter     — drops chunks below RELEVANCE_THRESHOLD
        7. Top-2 returned   — matches Dataset-C 1-chunk / 2-chunk format

    NOTE: LLM generation is NOT done here.  The [CONTEXT] string returned by
    this function is injected into the last user message by
    llm_service._real_generate_followup_response(), which uses the model
    already loaded in llm_service._load_llm() — no second model is loaded.
    rag/answer_generator.py and rag/model_loader.py are NOT called.

    Returns:
        Formatted "[CONTEXT]\n...\n[/CONTEXT]" string, or empty string when
        no chunk passes RELEVANCE_THRESHOLD (the LLM answers from its own knowledge).
    """
    from rag.retriever import retriever  # noqa: PLC0415 — lazy import to avoid startup cost

    result    = retriever({"question": query})
    evidences = result.get("evidences", [])

    if not evidences:
        return ""

    blocks = []
    for i, ev in enumerate(evidences, start=1):
        text   = ev.get("text", "").strip()
        meta   = ev.get("meta", {})
        source = meta.get("title") or meta.get("source", "Unknown source")
        score  = ev.get("score", 0.0)
        if text:
            #blocks.append(f"[Evidence {i} | source: {source} | relevance: {score:.2f}]\n{text}")
            score_pct = int(score * 100)
            blocks.append(f"## PRIORITY {i}: {source}\n{text}")
    if not blocks:
        return ""

    return "[CONTEXT]\n" + "\n\n".join(blocks) + "\n[/CONTEXT]"
