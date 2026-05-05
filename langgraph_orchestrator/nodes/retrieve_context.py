from __future__ import annotations

import re

from langgraph_orchestrator.services.rag_service import retrieve_medical_context
from langgraph_orchestrator.state import ChatState

# ---------------------------------------------------------------------------
# Known dermatology conditions — used for history-based condition extraction
# and to protect medical terms from spell correction.
# ---------------------------------------------------------------------------
_KNOWN_CONDITIONS = {
    "eczema", "psoriasis", "acne", "dermatitis", "rosacea",
    "tinea", "seborrheic", "urticaria", "vitiligo", "melanoma",
}

# Vague pronouns / references that point to the diagnosed condition.
_GENERIC_REFS = re.compile(
    r"\b(it|this|that|the condition|my condition|this condition|the disease)\b",
    re.IGNORECASE,
)

# Spell checker — loaded once at import time.
try:
    from spellchecker import SpellChecker as _SpellChecker
    _spell = _SpellChecker()
    _spell.word_frequency.load_words(_KNOWN_CONDITIONS)  # protect medical terms
    _SPELLCHECK_AVAILABLE = True
except ImportError:
    _SPELLCHECK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _condition_from_history(state: ChatState) -> str:
    """Scan recent assistant messages for a known condition name.

    Used as fallback when densenet confidence is low or label is missing.
    """
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            text = msg["content"].lower()
            for cond in _KNOWN_CONDITIONS:
                if cond in text:
                    return cond
    return ""


def _correct_query(query: str) -> str:
    """Fix obvious typos using pyspellchecker (no-op if library not installed)."""
    if not _SPELLCHECK_AVAILABLE:
        return query
    words = query.split()
    corrected = [_spell.correction(w) or w for w in words]
    fixed = " ".join(corrected)
    if fixed != query:
        print(f"[retrieve_context] Spell-corrected: '{query}' → '{fixed}'")
    return fixed


def _reformulate_query(query: str, state: ChatState) -> str:
    """Replace vague references or complete short fragments with the diagnosis label.

    Priority order for condition source:
      1. densenet_output label (most direct)
      2. densenet_output primary field
      3. condition extracted from conversation history (fallback when densenet unreliable)
    """
    condition = (
        (state.get("densenet_output") or {}).get("label")
        or (state.get("densenet_output") or {}).get("primary")
        or _condition_from_history(state)
        or ""
    ).strip()

    if not condition:
        return query

    # Case 1: vague pronoun/reference → replace inline
    if _GENERIC_REFS.search(query):
        reformulated = _GENERIC_REFS.sub(condition, query)
        print(f"[retrieve_context] Reformulated: '{query}' → '{reformulated}'")
        return reformulated

    # Case 2: short fragment with no explicit subject → prefix with condition
    if len(query.split()) < 5:
        reformulated = f"{condition} {query}"
        print(f"[retrieve_context] Prefixed: '{query}' → '{reformulated}'")
        return reformulated

    return query


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def retrieve_context(state: ChatState) -> ChatState:
    """Search the medical knowledge base and store retrieved context.

    Orchestration responsibilities (kept here):
    - Extract the latest user query from message history.
    - Apply spell correction and query reformulation before retrieval.
    - Write the retrieved context string to state["rag_context"].

    Model behavior (delegated to rag_service):
    - Vector-store retrieval is handled by retrieve_medical_context().
    """
    query = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            query = msg.get("content", "")
            break

    query = _correct_query(query)
    query = _reformulate_query(query, state)

    context = retrieve_medical_context(query)

    print(f"[retrieve_context] Query: '{query[:60]}' — {len(context)} chars retrieved")

    return {**state, "rag_context": context}
