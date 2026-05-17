from __future__ import annotations

from langgraph_orchestrator.state import ChatState

# Route labels — used as edge keys in the graph definition.
ROUTE_RAG = "retrieve_context"
ROUTE_LLM = "llm_followup"

# Keywords that suggest the user wants grounded medical information.

_RAG_KEYWORDS = (
    "treat", "treatment", "medication", "cream", "medicine",
    "caus", "trigger", "prevent", "prognos", "spread",
    "contagious", "diet", "allerg", "chronic", "cure",
    "serious", "dangerous", "last", "long", "worse", "better",
    "infect", "heal", "scarr", "main", "risk", "improve",
    "protect", "appear", "recommend", "diagnos", "symptom", "condition",
)

def followup_router(state: ChatState) -> ChatState:
    """Decide how to handle a follow-up question after diagnosis.

    Routing logic:
    - If the latest user message contains any RAG keyword (treatment, causes,
      prevention, prognosis, medications, etc.) → route to retrieve_context
      so the answer is grounded in retrieved medical knowledge.
    - Otherwise → route to a standard LLM response (conversational follow-up).

    The chosen route is written to state["_route"] for the LangGraph
    conditional edge function to read.

    Real implementation will:
    - Replace the keyword heuristic with an LLM intent-classification call or
      a small trained classifier for more reliable routing.
    - Consider the conversation history (not just the last message) to handle
      multi-turn follow-ups correctly.

    Returns:
        State with "_route" set to ROUTE_RAG or ROUTE_LLM.
    """
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "").lower()
            break

    is_knowledge_question = any(kw in last_user_msg for kw in _RAG_KEYWORDS)
    route = ROUTE_RAG if is_knowledge_question else ROUTE_LLM

    print(f"[followup_router] → {route}  (keyword_match={is_knowledge_question})")
    return {**state, "_route": route}
