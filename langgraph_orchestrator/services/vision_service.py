"""vision_service.py — Vision backend abstraction for the dermatology chatbot.

Public interface (called by node files)
----------------------------------------
- run_vision_inference(image_path)      → DenseNet classification result
- project_vision_features(densenet_out) → result dict with projected_embedding added

Each public function reads config.USE_MOCK_VISION and delegates to either the
mock implementation (_mock_*) or the real implementation (_real_*).

Teammate integration guide
--------------------------
1. Set USE_MOCK_VISION = False in config.py (or export USE_MOCK_VISION=0).
2. Fill in each _real_* function below with your model calls.
3. The public functions and all node files stay unchanged.

The _mock_* functions are preserved and continue to work exactly as before.

Confidence thresholds
---------------------
Both thresholds are defined here and imported by run_vision.py.
Adjust them here to tune the three-tier uncertainty behaviour globally.

Tier 1 — conf >= CONFIDENCE_THRESHOLD (0.60):
    High confidence. uncertainty_flag=False, fallback_triggered=False.
    Normal confident diagnosis proceeds.

Tier 2 — CONFIDENCE_FALLBACK_THRESHOLD <= conf < CONFIDENCE_THRESHOLD (0.40–0.59):
    Medium confidence. uncertainty_flag=True, fallback_triggered=False.
    Proceeds to joint_diagnosis with hedged/cautious language.

Tier 3 — conf < CONFIDENCE_FALLBACK_THRESHOLD (0.40):
    Low confidence. uncertainty_flag=True, fallback_triggered=True.
    await_or_store_image sends a re-upload prompt and resets image flags.
"""

from __future__ import annotations

from langgraph_orchestrator.config import USE_MOCK_VISION

# ---------------------------------------------------------------------------
# Confidence thresholds — imported by run_vision.py
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD          = 0.60  # minimum for a confident diagnosis
CONFIDENCE_FALLBACK_THRESHOLD = 0.40  # below this → trigger full fallback


# ===========================================================================
# Public interface — called by node files
# ===========================================================================

def run_vision_inference(image_path: str) -> dict:
    """Run the DenseNet classifier on the uploaded skin image.

    Args:
        image_path: Absolute path to the saved image file.

    Returns:
        dict with keys:
            label       (str)   — predicted skin condition
            confidence  (float) — model confidence in [0, 1]
            morphology  (str)   — brief morphological description
    """
    if USE_MOCK_VISION:
        return _mock_run_vision_inference(image_path)
    return _real_run_vision_inference(image_path)


def project_vision_features(densenet_output: dict) -> dict:
    """Convert DenseNet output into LLM-compatible embeddings.

    Args:
        densenet_output: The dict returned by run_vision_inference, possibly
                         augmented with a raw feature tensor by the real model.

    Returns:
        The same dict with a new "projected_embedding" key added.
    """
    if USE_MOCK_VISION:
        return _mock_project_vision_features(densenet_output)
    return _real_project_vision_features(densenet_output)


# ===========================================================================
# Mock implementations — fixed responses used during development
# ===========================================================================

def _mock_run_vision_inference(_image_path: str) -> dict:
    """MOCK: always returns a fixed eczema result regardless of image path.

    Change the confidence value here to exercise the three confidence tiers
    during manual testing:
        0.82 → Tier 1 (confident diagnosis)
        0.50 → Tier 2 (cautious diagnosis)
        0.25 → Tier 3 (re-upload fallback)
    """
    _ = _image_path  # mock — image is not read; input intentionally ignored
    return {
        "label":      "eczema",
        "confidence": 0.82,
        "morphology": "dry, scaly patches with erythema",
    }


def _mock_project_vision_features(densenet_output: dict) -> dict:
    """MOCK: builds a human-readable embedding string from the DenseNet dict."""
    projected = (
        f"[PROJECTED] condition={densenet_output.get('label', 'unknown')}  "
        f"confidence={densenet_output.get('confidence', 0.0):.2f}  "
        f"morphology='{densenet_output.get('morphology', '')}'"
    )
    return {**densenet_output, "projected_embedding": projected}


# ===========================================================================
# Real implementations — replace these bodies with actual model calls
# ===========================================================================

def _real_run_vision_inference(image_path: str) -> dict:
    """REAL: load and run the trained DenseNet classifier.

    Implementation guide:
    1. Load and pre-process the image (resize, normalise to DenseNet input).
    2. Run forward pass through the trained DenseNet weights.
    3. Map the top predicted class index to a label string.
    4. Extract confidence (softmax probability of the top class).
    5. Optionally derive morphology description from the prediction.
    6. Return the structured dict — keys must match the mock output above.
    """
    raise NotImplementedError(
        f"_real_run_vision_inference({image_path!r}): replace this with your "
        "DenseNet call. Set USE_MOCK_VISION = True in config.py to use the mock instead."
    )


def _real_project_vision_features(densenet_output: dict) -> dict:
    """REAL: project DenseNet features through the trained projection layer.

    Implementation guide:
    1. Accept the raw DenseNet penultimate-layer feature tensor.
       (Add it to the densenet_output dict in _real_run_vision_inference.)
    2. Pass the tensor through the trained projection MLP / linear layer.
    3. Store the resulting embedding under "projected_embedding".
       The joint_diagnosis node reads this key to attach visual context to
       the LLM prompt.
    4. Return the augmented dict.
    """
    raise NotImplementedError(
        f"_real_project_vision_features(label={densenet_output.get('label')!r}): "
        "replace this with your projection call. "
        "Set USE_MOCK_VISION = True in config.py to use the mock instead."
    )
