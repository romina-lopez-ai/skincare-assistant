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
2. Verify _DENSENET_CHECKPOINT and _PROJECTION_CHECKPOINT paths below are
   correct for your environment.
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

Real-mode implementation notes
-------------------------------
STATUS — DenseNetFeatureExtractor: FULLY WIRED
STATUS — ProjectionMLP:            FULLY WIRED
STATUS — _load_vision_models:      FULLY WIRED (raises FileNotFoundError if
                                   checkpoints are missing)
STATUS — _real_run_vision_inference:    FULLY WIRED
STATUS — _real_project_vision_features: FULLY WIRED

Key design decision: DenseNetFeatureExtractor.forward() returns BOTH
  (pooled_features, logits) in a single forward pass:
  - pooled_features (B, 1024): backbone global-avg-pool output.
    projection.pth was trained with densenet_dim=1024 — this is what it expects.
  - logits (B, 3): output of the custom 3-class head → softmax → label/conf.

This avoids running two separate forward passes for classification and
projection.  The inference_example.py DenseNetFeatureExtractor only stored
self.features (backbone), discarding the classifier — it cannot produce
logits.  Our version stores both self.features and self.classifier.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from langgraph_orchestrator.config import USE_MOCK_VISION

# ---------------------------------------------------------------------------
# Confidence thresholds — imported by run_vision.py
# ---------------------------------------------------------------------------

CONFIDENCE_THRESHOLD          = 0.60  # minimum for a confident diagnosis
CONFIDENCE_FALLBACK_THRESHOLD = 0.40  # below this → trigger full fallback


# ===========================================================================
# Image preprocessing — must match DenseNet-121 training pipeline exactly
# ===========================================================================

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Class names must match the order used during DenseNet training.
# Verify against densenet_model/outputs/class_names.txt if training was re-run.
_CLASS_NAMES = ["acne", "eczema", "psoriasis"]

# Short morphology descriptions used as a fallback when the LLM is in mock mode
# or when no richer description is available from the model.
_MORPHOLOGY_MAP: dict[str, str] = {
    "acne":      "comedones, papules, or pustules on oily skin",
    "eczema":    "dry, scaly patches with erythema and possible lichenification",
    "psoriasis": "silvery-white scaly plaques on an erythematous base",
}

# ---------------------------------------------------------------------------
# Checkpoint paths — resolved relative to repo root
# ---------------------------------------------------------------------------

# Repo root = skincare_model/ (two levels above this file)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

_DENSENET_CHECKPOINT = os.path.join(
    _REPO_ROOT, "densenet_model", "outputs", "model.pth"
)

# Primary projection checkpoint (placed at repo root after training)
_PROJECTION_CHECKPOINT = os.path.join(_REPO_ROOT, "projection.pth")

# Fallback search order — checked when _PROJECTION_CHECKPOINT is missing
_PROJECTION_FALLBACKS = [
    os.path.join(_REPO_ROOT, "unified_training", "outputs", "merged_model", "projection.pth"),
    os.path.join(_REPO_ROOT, "unified_training", "outputs", "projection.pth"),
]


# ===========================================================================
# Visual model components
# ===========================================================================

class ProjectionMLP(nn.Module):
    """Projects DenseNet backbone features (1024-dim) into LLM token embedding space.

    Architecture: Linear(1024 → 2048) → GELU → Linear(2048 → num_tokens * llm_dim)
    Output is reshaped to (batch, num_tokens, llm_dim).

    Matches visual_alignment/projection.py and the weights in projection.pth.
    Default hyperparameters (densenet_dim=1024, llm_dim=3072, num_tokens=32,
    hidden_dim=2048) must match those used during training.

    STATUS: FULLY WIRED — architecture and weight loading confirmed against
    visual_alignment/projection.py.
    """

    def __init__(
        self,
        densenet_dim: int = 1024,
        llm_dim: int = 3072,
        num_tokens: int = 32,
        hidden_dim: int = 2048,
    ) -> None:
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.mlp = nn.Sequential(
            nn.Linear(densenet_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, densenet_dim) -> (B, num_tokens, llm_dim)"""
        return self.mlp(x).view(-1, self.num_tokens, self.llm_dim)


class DenseNetFeatureExtractor(nn.Module):
    """Wraps the trained DenseNet-121 to extract backbone features AND logits.

    Two outputs are returned from a single forward pass:
      pooled  (B, 1024): backbone global-avg-pool vector — fed into ProjectionMLP.
      logits  (B, 3):    custom 3-class head output — used for label/confidence.

    WHY both outputs?
    -----------------
    projection.pth was trained with densenet_dim=1024, so it expects the
    raw backbone output BEFORE the custom classifier head.  At the same time
    the service needs class probabilities for the confidence tiers.  Running
    one forward pass and splitting the result is cheaper than two passes.

    DIFFERENCE from inference_example.py
    --------------------------------------
    inference_example.py's DenseNetFeatureExtractor only stores self.features
    (backbone), discarding the classifier — it can only produce the 1024-dim
    vector.  This version additionally stores self.classifier so it can also
    produce logits.

    STATUS: FULLY WIRED — split confirmed against densenet_model/src/model.py
    which defines: classifier = Sequential(Linear(1024→512), ReLU,
    Dropout(0.5), Linear(512→3)).
    """

    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()

        # densenet_model/src must be on sys.path for `from model import build_model`
        _densenet_src = os.path.join(_REPO_ROOT, "densenet_model", "src")
        if _densenet_src not in sys.path:
            sys.path.insert(0, _densenet_src)
        from model import build_model  # type: ignore[import]

        full = build_model(pretrained=False, fine_tune=False, num_classes=3)
        full.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        full.eval()

        # DenseNet backbone — produces feature maps before global avg pool
        self.features = full.features

        # Global average pool (matches DenseNet-121 standard forward pass)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom 3-class head: Linear(1024→512) → ReLU → Dropout(0.5) → Linear(512→3)
        # Stored separately so we can expose the penultimate 1024-dim vector
        # independently from the final logits.
        self.classifier = full.classifier

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass returning (pooled_features, logits).

        Returns:
            pooled:  (B, 1024) — backbone output after global avg pool.
                     Pass this to ProjectionMLP for visual token generation.
            logits:  (B, 3)   — classifier head output before softmax.
                     Apply F.softmax(logits, dim=1) to get class probabilities.
        """
        feat   = F.relu(self.features(x))
        pooled = self.pool(feat).view(feat.size(0), -1)  # (B, 1024)
        logits = self.classifier(pooled)                  # (B, 3)
        return pooled, logits


# ---------------------------------------------------------------------------
# Lazy model loader — loads once per process and caches in module globals
# ---------------------------------------------------------------------------

_feat_extractor: DenseNetFeatureExtractor | None = None
_projection: ProjectionMLP | None = None
_vision_device: torch.device | None = None


def _load_vision_models() -> tuple[DenseNetFeatureExtractor, ProjectionMLP, torch.device]:
    """Load and cache DenseNet + ProjectionMLP on first call.

    Subsequent calls return the cached instances immediately.

    STATUS: FULLY WIRED — raises FileNotFoundError with a clear message if
    either checkpoint is absent so teammates see a useful error instead of
    an obscure torch exception.

    TODO: If using a different base LLM (not LLaMA-3.2-3B), update `llm_dim`
    below to match that model's hidden_size, or read it from the LLM config.
    """
    global _feat_extractor, _projection, _vision_device

    if _feat_extractor is not None and _projection is not None:
        assert _vision_device is not None
        return _feat_extractor, _projection, _vision_device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    # -- DenseNet checkpoint --------------------------------------------------
    if not os.path.exists(_DENSENET_CHECKPOINT):
        raise FileNotFoundError(
            f"[vision_service] DenseNet checkpoint not found:\n  {_DENSENET_CHECKPOINT}\n"
            "Train the model (densenet_model/src/train.py) or update "
            "_DENSENET_CHECKPOINT in vision_service.py."
        )

    # -- ProjectionMLP checkpoint ---------------------------------------------
    proj_path = _PROJECTION_CHECKPOINT
    if not os.path.exists(proj_path):
        for fb in _PROJECTION_FALLBACKS:
            if os.path.exists(fb):
                proj_path = fb
                break
        else:
            raise FileNotFoundError(
                f"[vision_service] projection.pth not found at:\n  {_PROJECTION_CHECKPOINT}\n"
                f"  (also checked {_PROJECTION_FALLBACKS})\n"
                "Run unified_training to produce projection.pth, or update "
                "_PROJECTION_CHECKPOINT in vision_service.py."
            )

    print(f"[vision_service] Loading DenseNet extractor: {_DENSENET_CHECKPOINT}")
    feat_extractor = DenseNetFeatureExtractor(_DENSENET_CHECKPOINT).eval().to(device)

    # llm_dim=3072 matches LLaMA-3.2-3B hidden_size.
    # TODO: read this from the LLM model config if the base model changes.
    llm_dim = 3072
    print(f"[vision_service] Loading ProjectionMLP: {proj_path}")
    projection = ProjectionMLP(
        densenet_dim=1024, llm_dim=llm_dim, num_tokens=32, hidden_dim=2048
    )
    projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
    projection.eval().to(device).to(dtype)

    _feat_extractor = feat_extractor
    _projection     = projection
    _vision_device  = device

    print("[vision_service] Visual models ready.")
    return _feat_extractor, _projection, _vision_device


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
        Real mode additionally includes:
            features_tensor (torch.Tensor, shape (1, 1024)) — backbone features
                            consumed by project_vision_features.
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
        In real mode, projected_embedding is a torch.Tensor of shape
        (1, 32, llm_dim).  In mock mode it is a human-readable string.
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
# Real implementations
# ===========================================================================

def _real_run_vision_inference(image_path: str) -> dict:
    """REAL: load and run the trained DenseNet classifier.

    STATUS: FULLY WIRED.

    A single forward pass through DenseNetFeatureExtractor produces both:
      - pooled backbone features (1024-dim) for downstream projection
      - 3-class logits → softmax → label and confidence

    The raw `features_tensor` is stored in the return dict so that
    _real_project_vision_features can consume it without re-running the model.

    Returns dict with keys:
        label           (str)          — top predicted class name
        confidence      (float)        — softmax probability of top class
        morphology      (str)          — static description for the predicted class
        features_tensor (torch.Tensor) — (1, 1024) CPU tensor for projection
    """
    feat_extractor, _, device = _load_vision_models()

    image      = Image.open(image_path).convert("RGB")
    img_tensor = _IMAGE_TRANSFORM(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        pooled_features, logits = feat_extractor(img_tensor)   # (1,1024), (1,3)
        probs = F.softmax(logits, dim=1).squeeze(0).cpu()      # (3,)

    top_idx    = int(probs.argmax().item())
    label      = _CLASS_NAMES[top_idx]
    confidence = float(probs[top_idx].item())
    morphology = _MORPHOLOGY_MAP.get(label, "skin lesion")

    return {
        "label":           label,
        "confidence":      confidence,
        "morphology":      morphology,
        # Detach and move to CPU so the tensor is safe to store in state
        # and later consumed by _real_project_vision_features.
        # Shape (1, 1024) — batch dim preserved for the projection MLP.
        "features_tensor": pooled_features.detach().cpu(),
    }


def _real_project_vision_features(densenet_output: dict) -> dict:
    """REAL: project 1024-dim backbone features through the trained ProjectionMLP.

    STATUS: FULLY WIRED.

    Reads `features_tensor` (shape (1, 1024)) from densenet_output — written
    by _real_run_vision_inference — and passes it through the cached
    ProjectionMLP to produce 32 visual tokens of size llm_dim.

    The resulting tensor (shape (1, 32, llm_dim)) is stored under the key
    "projected_embedding".  joint_diagnosis / _real_generate_diagnosis_response
    reads this key to inject visual context into the LLM's input embedding space.

    Returns:
        Augmented densenet_output dict with "projected_embedding" added as a
        torch.Tensor of shape (1, 32, llm_dim).
    """
    _, projection, device = _load_vision_models()
    dtype = next(projection.parameters()).dtype

    features_tensor: torch.Tensor | None = densenet_output.get("features_tensor")
    if features_tensor is None:
        raise ValueError(
            "[vision_service] _real_project_vision_features: 'features_tensor' key "
            "is missing from densenet_output.  This key is written by "
            "_real_run_vision_inference — make sure USE_MOCK_VISION=False when "
            "USE_MOCK_VISION=False is set for the projection step."
        )

    vis_feat = features_tensor.to(device).to(dtype)  # (1, 1024)

    with torch.no_grad():
        projected = projection(vis_feat)              # (1, 32, llm_dim)

    return {**densenet_output, "projected_embedding": projected}
