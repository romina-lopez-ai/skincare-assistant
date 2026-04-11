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
    3. If symptoms are complete but no image yet → await_or_store_image.
    4. If an image was uploaded during symptom collection → await_or_store_image
       (store it, then continue collecting symptoms).
    5. Default → collect_symptoms.

    The chosen route is written to state["_route"] so the LangGraph conditional
    edge function can read it without duplicating the logic.

    Real implementation will:
    - Apply the same priority rules but with stricter checks (e.g. inspect last
      user message for explicit knowledge questions before reaching diagnosis).

    Returns:
        State with "_route" set to one of the ROUTE_* constants above.
    """
    route: str

    if state["diagnosis_done"]:
        route = ROUTE_FOLLOWUP_ROUTER
    elif state["symptoms_complete"] and state["image_uploaded"] and not state["image_processed"]:
        route = ROUTE_RUN_VISION
    elif state["symptoms_complete"] and not state["image_uploaded"]:
        route = ROUTE_AWAIT_OR_STORE_IMAGE
    elif state["image_uploaded"] and not state["symptoms_complete"]:
        # Early image upload — store it, keep collecting symptoms
        route = ROUTE_AWAIT_OR_STORE_IMAGE
    else:
        route = ROUTE_COLLECT_SYMPTOMS

    print(f"[input_router] → {route}")
    return {**state, "_route": route}
