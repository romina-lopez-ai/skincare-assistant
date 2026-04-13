from __future__ import annotations

from langgraph_orchestrator.services.vision_service import (
    CONFIDENCE_THRESHOLD,
    run_vision_inference,
)
from langgraph_orchestrator.state import ChatState


def run_vision(state: ChatState) -> ChatState:
    """Run the vision classifier on the uploaded skin image.

    Orchestration responsibilities (kept here):
    - Read image_path from state.
    - Set image_processed = True.
    - Set uncertainty_flag if confidence is below threshold.
    - Update phase.

    Model behavior (delegated to vision_service):
    - DenseNet inference is handled by run_vision_inference().
    """
    result     = run_vision_inference(state["image_path"])
    uncertainty = result["confidence"] < CONFIDENCE_THRESHOLD

    print(f"[run_vision] label={result['label']}  "
          f"confidence={result['confidence']:.2f}  "
          f"uncertainty={uncertainty}")

    return {
        **state,
        "densenet_output":  result,
        "image_processed":  True,
        "uncertainty_flag": uncertainty,
        "phase":            "joint_diagnosis",
    }
