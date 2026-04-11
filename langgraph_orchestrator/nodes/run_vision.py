from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def run_vision(state: ChatState) -> ChatState:
    """Run the DenseNet vision model on the uploaded skin image.

    Responsibilities:
    - Load the image from state["image_path"].
    - Pass it through the DenseNet classifier.
    - Store the structured result in state["densenet_output"]:
        {
            "label":       str   — predicted condition (e.g. "acne", "eczema"),
            "confidence":  float — model confidence in [0, 1],
            "morphology":  str   — brief morphological description,
        }
    - Set image_processed = True so this node is not called again.
    - Set uncertainty_flag = True if confidence is below the threshold.

    Key rule:
        This node does NOT generate any user-facing message.  The LLM always
        produces the final response; DenseNet output is an intermediate signal
        only.

    Real implementation will:
    - Load and pre-process the image (resize, normalise) to match DenseNet input.
    - Run inference with the trained DenseNet weights.
    - Map the predicted class index to a human-readable label.

    Returns:
        Updated state with densenet_output populated, image_processed = True,
        and optionally uncertainty_flag = True.
    """
    CONFIDENCE_THRESHOLD = 0.6

    # --- Placeholder DenseNet inference ---
    mock_output = {
        "label": "eczema",
        "confidence": 0.82,
        "morphology": "dry, scaly patches with erythema",
    }

    uncertainty = mock_output["confidence"] < CONFIDENCE_THRESHOLD

    print(f"[run_vision] label={mock_output['label']}  "
          f"confidence={mock_output['confidence']:.2f}  "
          f"uncertainty={uncertainty}")

    return {
        **state,
        "densenet_output": mock_output,
        "image_processed": True,
        "uncertainty_flag": uncertainty,
        "phase": "joint_diagnosis",
    }
