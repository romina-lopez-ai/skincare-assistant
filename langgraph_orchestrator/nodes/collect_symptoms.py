from __future__ import annotations

from langgraph_orchestrator.services.llm_service import (
    generate_symptom_response,
    refine_symptom_slots,  # corrects slots via base LLM at end of symptom collection
)
from langgraph_orchestrator.services.symptom_service import (
    MIN_SYMPTOM_TURNS,
    REQUIRED_SLOTS,
    extract_slots_from_message,
    filled_slot_count,
    is_symptoms_complete,
)
from langgraph_orchestrator.state import ChatState


def collect_symptoms(state: ChatState) -> ChatState:
    """Ask the user focused questions to fill the symptom slots.

    Orchestration responsibilities (kept here):
    1. Extract slot values from the latest user message (heuristic).
    2. Ask the LLM for the next question (or completion signal).
    3. Apply deterministic completion rules — does not rely solely on
       the model emitting [END_SYMPTOMS].
    4. Update symptom_turns, symptoms_complete, phase, and messages.

    Model behavior (delegated):
    - Slot extraction  → symptom_service.extract_slots_from_message()
    - Completion rules → symptom_service.is_symptoms_complete()
    - Question text    → llm_service.generate_symptom_response()
    """
    # ------------------------------------------------------------------
    # 1. Extract slots from the latest user message (before calling LLM)
    #    so that generate_symptom_response() sees up-to-date slot values
    #    and skips slots that are already filled.
    # ------------------------------------------------------------------
    last_user_msg = ""
    for msg in reversed(state["messages"]):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    updated_slots = extract_slots_from_message(last_user_msg, state["symptom_slots"])

    # Pass updated slots forward so the LLM service sees current state
    state_with_slots = {**state, "symptom_slots": updated_slots}

    # ------------------------------------------------------------------
    # 2. Ask the LLM for the next question / completion reply
    # ------------------------------------------------------------------
    reply = generate_symptom_response(state_with_slots)

    # ------------------------------------------------------------------
    # 3. Apply deterministic completion rules
    # ------------------------------------------------------------------
    turns      = state["symptom_turns"] + 1
    end_signal = "[END_SYMPTOMS]" in reply

    # Strip premature [END_SYMPTOMS] — model sometimes ends too early.
    # Matches inference_example.py MIN_QUESTIONS_BEFORE_END guard.
    if end_signal and turns < MIN_SYMPTOM_TURNS:
        end_signal = False

    # Always strip [END_SYMPTOMS] from user-facing text — it's an internal
    # control tag, never shown to the user regardless of whether we honor it.
    if "[END_SYMPTOMS]" in reply:
        reply = reply.replace("[END_SYMPTOMS]", "").strip()

    complete   = is_symptoms_complete(updated_slots, turns, end_signal)

    # ------------------------------------------------------------------
    # 4. Log and update state
    # ------------------------------------------------------------------
    filled  = filled_slot_count(updated_slots)
    missing = [k for k in REQUIRED_SLOTS if not updated_slots.get(k)]

    if complete:
        reason = (
            "model signal" if end_signal
            else f"{filled}/{len(REQUIRED_SLOTS)} slots filled"
            if filled >= 4
            else "max turns"
        )
        print(f"[collect_symptoms] Complete after {turns} turns ({reason})")
        # Refine slots with base LLM: reads the full conversation once to correct
        # keyword extraction errors (wrong slot, cut text, missed context).
        # LLM values take priority; keyword values kept as fallback for nulls.
        _state_for_refine = {**state_with_slots, "messages": state["messages"]}
        _refined = refine_symptom_slots(_state_for_refine)
        updated_slots = {
            k: _refined.get(k) or updated_slots.get(k)
            for k in updated_slots
        }
        print(f"[collect_symptoms] Slots after LLM refinement: {updated_slots}")
    else:
        print(f"[collect_symptoms] Turn {turns} — filled {filled}/{len(REQUIRED_SLOTS)}, "
              f"missing: {missing}")

    messages = state["messages"] + [{"role": "assistant", "content": reply}]

    # Summary: update diagnosis_summary symptoms on every turn as slots fill in progressively
    _existing_summary = state.get("diagnosis_summary") or {}
    _updated_summary = {**_existing_summary, "symptoms": dict(updated_slots)}

    return {
        **state,
        "symptom_slots":     updated_slots,
        "messages":          messages,
        "symptom_turns":     turns,
        "symptoms_complete": complete,
        "phase":             "await_image" if complete else "symptom_collection",
        "diagnosis_summary": _updated_summary,  # Summary: progressive symptom tracking
    }
