"""helpers.py — Lightweight utilities for workflow scenario tests.

These helpers keep the scenario tests readable by hiding boilerplate.
They are intentionally simple — no magic, no frameworks.

Public API
----------
make_state(**overrides)         — build a ChatState dict from defaults
run_turn(graph, state, message) — simulate one user turn through the graph
last_assistant_message(state)   — extract the most recent assistant reply
all_assistant_messages(state)   — list all assistant replies in order

Canonical test messages
-----------------------
ALL_SLOTS_MESSAGE   — one message that fills all 5 symptom slots
FOUR_SLOTS_MESSAGE  — fills exactly 4 slots (triggers left empty)
                      used to test the MIN_SLOTS_REQUIRED threshold
"""

from __future__ import annotations

from langgraph_orchestrator.state import create_initial_state

# ---------------------------------------------------------------------------
# Canonical test messages
#
# Slot coverage is verified against _SLOT_KEYWORDS in symptom_service.py.
# ---------------------------------------------------------------------------

# Fills all 5 slots in one message — lets tests skip past symptom collection.
ALL_SLOTS_MESSAGE: str = (
    "I have a dry, itchy red rash on my arm that started two weeks ago. "
    "It's moderate, and gets worse with soap and stress."
)
# location=arm · duration=two weeks · appearance=dry/itchy/red/rash
# severity=moderate · triggers=soap/stress

# Fills exactly 4 slots — triggers intentionally omitted.
# The mock LLM will ask about triggers (no [END_SYMPTOMS] emitted),
# but 4 filled >= MIN_SLOTS_REQUIRED so completion fires from slot count.
FOUR_SLOTS_MESSAGE: str = (
    "There is a scaly red patch on my face that appeared a month ago and "
    "it is moderate."
)
# location=face · duration=a month ago · appearance=scaly/red/patch
# severity=moderate · triggers=None


# ---------------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------------

def make_state(**overrides) -> dict:
    """Return a fresh base ChatState with any field overrides applied.

    The base is always a clean initial state so tests never share mutable
    data between runs.

    Usage::

        state = make_state(symptoms_complete=True, image_uploaded=True,
                           image_path="/fake/img.jpg")
    """
    base = dict(create_initial_state("test-session"))
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Turn simulator
# ---------------------------------------------------------------------------

def run_turn(graph, state: dict, user_message: str) -> dict:
    """Append user_message to state and invoke the graph for one full turn.

    The graph runs to END and the final state dict is returned.
    No in-memory session store is used — state flows through the graph
    directly, keeping tests isolated and reproducible.

    Args:
        graph:        Compiled LangGraph instance (from the conftest fixture).
        state:        Current ChatState dict.
        user_message: Text of the user's message for this turn.

    Returns:
        Updated state dict after the graph has completed.
    """
    state_with_msg = {
        **state,
        "messages": state["messages"] + [{"role": "user", "content": user_message}],
    }
    return dict(graph.invoke(state_with_msg))


# ---------------------------------------------------------------------------
# Message extractors
# ---------------------------------------------------------------------------

def last_assistant_message(state: dict) -> str:
    """Return the content of the most recent assistant message, or '' if none."""
    for msg in reversed(state.get("messages", [])):
        if msg.get("role") == "assistant":
            return msg.get("content", "")
    return ""


def all_assistant_messages(state: dict) -> list[str]:
    """Return the content of every assistant message in chronological order."""
    return [
        msg["content"]
        for msg in state.get("messages", [])
        if msg.get("role") == "assistant"
    ]
