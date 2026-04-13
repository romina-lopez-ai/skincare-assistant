from __future__ import annotations

from langgraph_orchestrator.state import ChatState

# Route labels returned by this node — used as edge keys in the graph definition.
ROUTE_COLLECT_SYMPTOMS = "collect_symptoms"
ROUTE_AWAIT_OR_STORE_IMAGE = "await_or_store_image"
ROUTE_RUN_VISION = "run_vision"
ROUTE_FOLLOWUP_ROUTER = "followup_router"


def input_router(state: ChatState) -> ChatState:
    """Decide which processing path to follow this turn.

    Routing logic (in priority order):
    1. If diagnosis is already done → followup_router (handle follow-up question).
    2. If symptoms are complete AND image is uploaded but not yet processed → run_vision.
    3. If symptoms are complete but no image yet → await_or_store_image (prompt for upload).
    4. If an image was uploaded early AND not yet acknowledged → await_or_store_image
       (send the one-time acknowledgement, then return to symptom collection).
    5. Default → collect_symptoms.

    Rule 4 uses image_acknowledged to ensure the acknowledgement message is only
    sent once.  After await_or_store_image sets image_acknowledged = True, the
    early image is transparent to the router and symptom collection continues
    normally through the default branch.

    The chosen route is written to state["_route"] so the LangGraph conditional
    edge function can read it without duplicating the logic.

    Returns:
        State with "_route" set to one of the ROUTE_* constants above.
    """
    route: str

    # MILESTONE GUARD (primary + secondary) — mirrors routing.py
    # diagnosis_done is the primary signal; phase == "follow_up" is the
    # secondary fallback in case the flag and phase are briefly out of sync.
    if state["diagnosis_done"] or state["phase"] == "follow_up":
        route = ROUTE_FOLLOWUP_ROUTER
    elif state["symptoms_complete"] and state["image_uploaded"] and not state["image_processed"]:
        route = ROUTE_RUN_VISION
    elif state["symptoms_complete"] and not state["image_uploaded"]:
        route = ROUTE_AWAIT_OR_STORE_IMAGE
    elif state["image_uploaded"] and not state["image_acknowledged"] and not state["symptoms_complete"]:
        # Early image — acknowledge it exactly once, then fall back to symptom collection
        route = ROUTE_AWAIT_OR_STORE_IMAGE
    else:
        route = ROUTE_COLLECT_SYMPTOMS

    print(f"[input_router] → {route}")
    return {**state, "_route": route}
