"""vision_service.py — Vision backend abstraction for the dermatology chatbot.

Both functions are **mocks** that reproduce the placeholder behavior that
previously lived inside run_vision.py and project_image_features.py.

Teammate integration guide
--------------------------
Replace each function body with real model inference.  The function signatures
and return types must not change so the node files keep working without
modification.

Expected real implementation pattern:
    run_vision_inference:
        1. Load and pre-process the image (resize, normalise to DenseNet input).
        2. Run forward pass through the trained DenseNet weights.
        3. Map the top predicted class index to a label string.
        4. Extract confidence (softmax probability of the top class).
        5. Optionally derive morphology description from the prediction.
        6. Return the structured dict.

    project_vision_features:
        1. Accept the DenseNet penultimate-layer feature tensor (not just label).
        2. Pass it through the trained projection MLP / linear layer.
        3. Return the embedding vector in a format the LLM can consume
           (e.g. a list of floats, a tensor, or a special token string).
"""

from __future__ import annotations

# Confidence thresholds for three-tier uncertainty assessment.
#
# Tier 1 — conf >= CONFIDENCE_THRESHOLD (0.60):
#     High confidence. uncertainty_flag=False, fallback_triggered=False.
#     Normal confident diagnosis proceeds.
#
# Tier 2 — CONFIDENCE_FALLBACK_THRESHOLD <= conf < CONFIDENCE_THRESHOLD (0.40–0.59):
#     Medium confidence. uncertainty_flag=True, fallback_triggered=False.
#     Proceeds to joint_diagnosis with hedged/cautious language.
#
# Tier 3 — conf < CONFIDENCE_FALLBACK_THRESHOLD (0.40):
#     Low confidence. uncertainty_flag=True, fallback_triggered=True.
#     route_after_vision sends the turn back to collect_symptoms so the
#     user is asked for a clearer image or more symptom information.

CONFIDENCE_THRESHOLD          = 0.60  # minimum for a confident diagnosis
CONFIDENCE_FALLBACK_THRESHOLD = 0.40  # below this → trigger full fallback


def run_vision_inference(image_path: str) -> dict:
    """Run the DenseNet classifier on the uploaded skin image.

    MOCK: always returns a fixed eczema result regardless of the image path.

    Real implementation should:
    - Load the image from image_path.
    - Pre-process and run DenseNet inference.
    - Return label, confidence, and morphology derived from the model output.

    Args:
        image_path: Absolute path to the saved image file.

    Returns:
        dict with keys:
            label       (str)   — predicted skin condition
            confidence  (float) — model confidence in [0, 1]
            morphology  (str)   — brief morphological description
    """
    return {
        "label":      "eczema",
        "confidence": 0.82,
        "morphology": "dry, scaly patches with erythema",
    }


def project_vision_features(densenet_output: dict) -> dict:
    """Convert DenseNet output into LLM-compatible embeddings.

    MOCK: builds a human-readable string from the DenseNet dict fields and
    stores it under the key "projected_embedding".

    Real implementation should:
    - Accept the raw DenseNet feature tensor from the penultimate layer.
    - Pass it through the trained projection layer (MLP or linear map).
    - Store the resulting fixed-size embedding (e.g. 4096-dim vector) under
      "projected_embedding" so joint_diagnosis can attach it to the LLM input.

    Args:
        densenet_output: The dict returned by run_vision_inference, possibly
                         augmented with a raw feature tensor by the real model.

    Returns:
        The same dict with a new "projected_embedding" key added.
    """
    projected = (
        f"[PROJECTED] condition={densenet_output.get('label', 'unknown')}  "
        f"confidence={densenet_output.get('confidence', 0.0):.2f}  "
        f"morphology='{densenet_output.get('morphology', '')}'"
    )
    return {**densenet_output, "projected_embedding": projected}
