from __future__ import annotations

from langgraph_orchestrator.services.llm_service import generate_symptom_response
from langgraph_orchestrator.state import ChatState

MAX_SYMPTOM_TURNS = 10
REQUIRED_SLOTS    = ("location", "duration", "appearance", "severity", "triggers")


def collect_symptoms(state: ChatState) -> ChatState:
    """Ask the user focused questions to fill the symptom slots.

    Orchestration responsibilities (kept here):
    - Increment symptom_turns.
    - Detect completion via [END_SYMPTOMS] signal, slot count, or max turns.
    - Update phase and symptoms_complete flag.
    - Append the assistant reply to messages.

    Model behavior (delegated to llm_service):
    - Generating the next question is handled by generate_symptom_response().
    """
    turns = state["symptom_turns"] + 1
    slots = state["symptom_slots"]

    reply = generate_symptom_response(state)

    messages = state["messages"] + [{"role": "assistant", "content": reply}]

    filled           = [k for k, v in slots.items() if v is not None]
    end_signal       = "[END_SYMPTOMS]" in reply
    all_slots_filled = len(filled) >= len(REQUIRED_SLOTS)
    max_turns_reached = turns >= MAX_SYMPTOM_TURNS

    symptoms_complete = end_signal or all_slots_filled or max_turns_reached

    if symptoms_complete:
        print(f"[collect_symptoms] Complete after {turns} turns "
              f"(end_signal={end_signal}, max_turns={max_turns_reached})")
    else:
        missing = [k for k in REQUIRED_SLOTS if not slots.get(k)]
        print(f"[collect_symptoms] Turn {turns} — missing: {missing}")

    return {
        **state,
        "messages":          messages,
        "symptom_turns":     turns,
        "symptoms_complete": symptoms_complete,
        "phase":             "await_image" if symptoms_complete else "symptom_collection",
    }
