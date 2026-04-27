from __future__ import annotations

import uuid
from typing import Any
from typing_extensions import TypedDict


class SymptomSlots(TypedDict):
    location: str | None
    duration: str | None
    appearance: str | None
    severity: str | None
    triggers: str | None


class ChatState(TypedDict):
    # Session identity
    session_id: str

    # Conversation phase
    # Allowed values: "start" | "symptom_collection" | "await_image" |
    #                 "joint_diagnosis" | "follow_up" | "end_session"
    phase: str

    # Full message history: each entry is {"role": "user"|"assistant", "content": str}
    messages: list[dict[str, Any]]

    # Symptom tracking
    symptom_slots: SymptomSlots
    symptom_turns: int
    symptoms_complete: bool

    # Image state
    image_uploaded: bool
    image_acknowledged: bool   # True after the early-upload acknowledgement is sent once
    image_path: str | None
    image_processed: bool
    densenet_output: dict[str, Any] | None

    # Diagnosis state
    diagnosis_done: bool
    diagnosis_summary: str | None

    # Follow-up / RAG state
    rag_context: str | None

    # Safety / fallback
    fallback_triggered: bool
    uncertainty_flag: bool

    # Internal routing signal — written by input_router and followup_router nodes,
    # read by route_from_followup_router.  Must be declared in the TypedDict so
    # LangGraph preserves it across node boundaries.  Never shown to the user.
    _route: str


def create_initial_state(session_id: str) -> ChatState:
    """Return a blank ChatState for a new session.

    Args:
        session_id: Unique identifier for this session.  Pass an explicit ID
                    (e.g. from a web request) or generate one with
                    ``str(uuid.uuid4())``.

    Returns:
        A fully-initialised ChatState with all optional fields set to their
        zero / None values and the phase set to ``"start"``.
    """
    return ChatState(
        session_id=session_id,
        phase="start",
        messages=[],
        symptom_slots=SymptomSlots(
            location=None,
            duration=None,
            appearance=None,
            severity=None,
            triggers=None,
        ),
        symptom_turns=0,
        symptoms_complete=False,
        image_uploaded=False,
        image_acknowledged=False,
        image_path=None,
        image_processed=False,
        densenet_output=None,
        diagnosis_done=False,
        diagnosis_summary=None,
        rag_context=None,
        fallback_triggered=False,
        uncertainty_flag=False,
        _route="",
    )
