from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def session_entry(state: ChatState) -> ChatState:
    """Entry point for every conversation turn.

    Responsibilities:
    - Receive the incoming user message (already appended to state["messages"]
      before this node is called by the graph runner).
    - Detect whether the input contains text, an image, or both and surface that
      information so the downstream input_router can make a clean routing decision.
    - Transition the phase from "start" to "symptom_collection" on the very first
      turn so the rest of the graph sees a consistent phase value.

    Real implementation will:
    - Validate / normalise the incoming message format.
    - Detect image attachments and set image_uploaded / image_path.
    - Optionally log session start time.

    Returns:
        Updated state with phase set to "symptom_collection" (if this is the
        first turn) and any image metadata populated.
    """
    # Placeholder: advance phase from "start" to "symptom_collection"
    if state["phase"] == "start":
        state = {**state, "phase": "symptom_collection"}

    print(f"[session_entry] session={state['session_id']}  phase={state['phase']}  "
          f"messages={len(state['messages'])}")

    return state
