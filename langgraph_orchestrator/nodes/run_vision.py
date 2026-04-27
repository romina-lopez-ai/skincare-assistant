from __future__ import annotations

from langgraph_orchestrator.services.vision_service import (
    CONFIDENCE_FALLBACK_THRESHOLD,
    CONFIDENCE_THRESHOLD,
    run_vision_inference,
)
from langgraph_orchestrator.state import ChatState


def run_vision(state: ChatState) -> ChatState:
    """Run the vision classifier on the uploaded skin image.

    Orchestration responsibilities (kept here):
    - Read image_path from state.
    - Set image_processed = True.
    - Apply three-tier confidence assessment using both thresholds.
    - Update phase.

    Three-tier confidence assessment:
    - conf >= CONFIDENCE_THRESHOLD (0.60):
        uncertainty_flag=False, fallback_triggered=False  → confident diagnosis
    - CONFIDENCE_FALLBACK_THRESHOLD <= conf < CONFIDENCE_THRESHOLD (0.40–0.59):
        uncertainty_flag=True,  fallback_triggered=False  → cautious diagnosis
    - conf < CONFIDENCE_FALLBACK_THRESHOLD (0.40):
        uncertainty_flag=True,  fallback_triggered=True   → route_after_vision
        sends the turn back to collect_symptoms (ask for clearer image / more info)

    Model behavior (delegated to vision_service):
    - DenseNet inference is handled by run_vision_inference().
    """
    result     = run_vision_inference(state["image_path"])
    confidence = result["confidence"]

    if confidence >= CONFIDENCE_THRESHOLD:
        uncertainty_flag    = False
        fallback_triggered  = False
        tier                = "high"
    elif confidence >= CONFIDENCE_FALLBACK_THRESHOLD:
        uncertainty_flag    = True
        fallback_triggered  = False
        tier                = "medium"
    else:
        uncertainty_flag    = True
        fallback_triggered  = True
        tier                = "low (fallback)"

    print(
        f"[run_vision] label={result['label']}  "
        f"confidence={confidence:.2f}  "
        f"tier={tier}  "
        f"uncertainty={uncertainty_flag}  "
        f"fallback={fallback_triggered}"
    )

    return {
        **state,
        "densenet_output":   result,
        "image_processed":   True,
        "uncertainty_flag":  uncertainty_flag,
        "fallback_triggered": fallback_triggered,
        "phase":             "joint_diagnosis",
    }
