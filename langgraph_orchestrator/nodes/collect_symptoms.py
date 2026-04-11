from __future__ import annotations

from langgraph_orchestrator.state import ChatState

# Maximum number of symptom-collection turns before forcing a transition.
MAX_SYMPTOM_TURNS = 10

# Slots that must be filled before symptoms are considered complete.
REQUIRED_SLOTS = ("location", "duration", "appearance", "severity", "triggers")


def collect_symptoms(state: ChatState) -> ChatState:
    """Ask the user focused questions to fill the symptom slots.

    Responsibilities:
    - Call the fine-tuned LLM to generate the next symptom question.
    - Parse the LLM response to detect whether any slots are now filled.
    - Detect the [END_SYMPTOMS] signal in the LLM output.
    - Apply fallback completion rules when the signal is absent:
        * all required slots are filled, or
        * symptom_turns has reached MAX_SYMPTOM_TURNS.
    - Append the assistant question to state["messages"].
    - Increment symptom_turns.
    - Set symptoms_complete = True when done.

    Real implementation will:
    - Format the current symptom_slots into the LLM prompt so the model knows
      which slots are still missing.
    - Extract slot values from the model's reply with a lightweight parser.
    - Handle the [END_SYMPTOMS] tag as the primary completion signal.

    Returns:
        Updated state with the assistant message appended, symptom_turns
        incremented, and symptoms_complete set if collection is finished.
    """
    turns = state["symptom_turns"] + 1
    slots = state["symptom_slots"]

    # --- Placeholder LLM call ---
    filled = [k for k, v in slots.items() if v is not None]
    missing = [k for k in REQUIRED_SLOTS if slots.get(k) is None]

    if missing:
        mock_reply = f"[PLACEHOLDER] Could you tell me more about the {missing[0]} of your skin concern?"
    else:
        mock_reply = "[PLACEHOLDER] Thank you. I have all the information I need. [END_SYMPTOMS]"

    messages = state["messages"] + [{"role": "assistant", "content": mock_reply}]

    # Completion detection
    end_signal = "[END_SYMPTOMS]" in mock_reply
    all_slots_filled = len(filled) + (1 if missing and mock_reply else 0) >= len(REQUIRED_SLOTS)
    max_turns_reached = turns >= MAX_SYMPTOM_TURNS

    symptoms_complete = end_signal or all_slots_filled or max_turns_reached

    if symptoms_complete:
        print(f"[collect_symptoms] Symptoms complete after {turns} turns "
              f"(end_signal={end_signal}, max_turns={max_turns_reached})")
    else:
        print(f"[collect_symptoms] Turn {turns} — still missing: {missing}")

    return {
        **state,
        "messages": messages,
        "symptom_turns": turns,
        "symptoms_complete": symptoms_complete,
        "phase": "await_image" if symptoms_complete else "symptom_collection",
    }
