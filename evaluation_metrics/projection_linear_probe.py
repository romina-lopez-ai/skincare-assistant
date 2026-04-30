"""Projection layer evaluation via linear probe.

Measures whether the ProjectionMLP encodes enough visual information
for downstream condition discrimination. A logistic regression is fit
on the mean visual token from training images and evaluated on
validation images.

Metrics:
  - probe_accuracy           : overall classification accuracy
  - per_condition_precision   : precision per condition
  - per_condition_recall      : recall per condition
  - per_condition_f1          : F1 per condition
  - confusion_matrix          : 3x3 matrix (acne/eczema/psoriasis)

Supports evaluating Phase 2 and Phase 4 projections side by side.

Usage (from project root):
    python evaluation_metrics/projection_cosine_similarity.py
    python evaluation_metrics/projection_cosine_similarity.py --max-samples 50
    python evaluation_metrics/projection_cosine_similarity.py --phase2-only
"""

import argparse
import json
import logging
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torchvision import transforms
from tqdm import tqdm

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_this_dir, "..")
for _p in (_this_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visual_alignment.projection import ProjectionMLP
from evaluation_metrics.config import ProjectionEvalConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# DenseNet feature extractor (frozen)
# ---------------------------------------------------------------------------

class DenseNetFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        sys.path.insert(0, os.path.join(_project_root, "densenet_model", "src"))
        from model import build_model
        full = build_model(pretrained=False, fine_tune=False, num_classes=3)
        full.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.features = full.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = F.relu(self.features(x))
        return self.pool(feat).view(feat.size(0), -1)


# ---------------------------------------------------------------------------
# Visual token extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_visual_tokens(image_path, feat_extractor, projection, device):
    image = Image.open(image_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    vis_feat = feat_extractor(img_tensor)
    visual_tokens = projection(vis_feat.to(torch.bfloat16))
    return visual_tokens.squeeze(0).float()


@torch.no_grad()
def extract_features_from_samples(
    samples, feat_extractor, projection, images_root, device, condition_labels,
):
    """Extract mean visual token vectors and labels for a set of samples."""
    features = []
    labels = []

    for item in tqdm(samples, desc="extracting", unit="sample"):
        img_rel = item["image_path"].replace("images/", "", 1)
        image_path = os.path.join(images_root, img_rel)
        ground_truth = item["DenseNet_output"]["primary"]

        if ground_truth not in condition_labels:
            continue

        visual_tokens = extract_visual_tokens(
            image_path, feat_extractor, projection, device,
        )
        mean_visual = visual_tokens.mean(dim=0).cpu().numpy()

        features.append(mean_visual)
        labels.append(ground_truth)

    return np.array(features), np.array(labels)


# ---------------------------------------------------------------------------
# Linear probe evaluation
# ---------------------------------------------------------------------------

def evaluate_projection_probe(
    projection, feat_extractor,
    train_samples, val_samples,
    images_root, device, config,
):
    """Fit logistic regression on train visual tokens, evaluate on val."""
    logger.info("Extracting train features …")
    X_train, y_train = extract_features_from_samples(
        train_samples, feat_extractor, projection,
        images_root, device, config.condition_labels,
    )

    logger.info("Extracting validation features …")
    X_val, y_val = extract_features_from_samples(
        val_samples, feat_extractor, projection,
        images_root, device, config.condition_labels,
    )

    logger.info(f"Train: {len(X_train)} samples, Val: {len(X_val)} samples")

    probe = LogisticRegression(max_iter=1000, random_state=42)
    probe.fit(X_train, y_train)

    y_pred = probe.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)

    labels = list(config.condition_labels)
    report = classification_report(
        y_val, y_pred, labels=labels, output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    per_condition = {}
    for condition in labels:
        per_condition[condition] = {
            "precision": round(report[condition]["precision"], 4),
            "recall": round(report[condition]["recall"], 4),
            "f1": round(report[condition]["f1-score"], 4),
            "support": report[condition]["support"],
        }

    summary = {
        "n_train": len(X_train),
        "n_val": len(X_val),
        "probe_accuracy": round(accuracy, 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report["weighted avg"]["f1-score"], 4),
        "per_condition": per_condition,
        "confusion_matrix": {
            "labels": labels,
            "matrix": cm.tolist(),
        },
    }

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path, max_samples=None):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def load_projection(checkpoint_path, config, device):
    from transformers import AutoConfig
    llm_config = AutoConfig.from_pretrained(
        config.slm_model_name, token=config.hf_token or None,
    )
    projection = ProjectionMLP(
        densenet_dim=config.densenet_dim,
        llm_dim=llm_config.hidden_size,
        num_tokens=config.num_visual_tokens,
        hidden_dim=config.projection_hidden_dim,
    )
    projection.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    projection.eval().to(device).to(torch.bfloat16)
    return projection


def print_summary(tag, summary):
    logger.info(f"\n{'=' * 60}")
    logger.info(f"  LINEAR PROBE — {tag}")
    logger.info(f"{'=' * 60}")
    logger.info(f"  Train samples:    {summary['n_train']}")
    logger.info(f"  Val samples:      {summary['n_val']}")
    logger.info(f"  Probe accuracy:   {summary['probe_accuracy']:.1%}")
    logger.info(f"  Macro F1:         {summary['macro_f1']:.4f}")
    logger.info(f"  Weighted F1:      {summary['weighted_f1']:.4f}")
    logger.info(f"")
    logger.info(f"  {'Condition':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>9}")
    logger.info(f"  {'-' * 49}")
    for c, m in summary["per_condition"].items():
        logger.info(
            f"  {c:<12} {m['precision']:>10.4f} {m['recall']:>8.4f} "
            f"{m['f1']:>8.4f} {m['support']:>9}"
        )

    cm = summary["confusion_matrix"]
    labels = cm["labels"]
    matrix = cm["matrix"]
    logger.info(f"")
    logger.info(f"  Confusion matrix (rows=true, cols=predicted):")
    logger.info(f"  {'':>12} " + " ".join(f"{l:>10}" for l in labels))
    for i, label in enumerate(labels):
        row = " ".join(f"{v:>10}" for v in matrix[i])
        logger.info(f"  {label:>12} {row}")
    logger.info(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Projection layer linear probe evaluation",
    )
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap samples per split (default: all)")
    parser.add_argument("--phase2-only", action="store_true")
    parser.add_argument("--phase4-only", action="store_true")
    args = parser.parse_args()

    config = ProjectionEvalConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    train_path = os.path.join(_project_root, config.dataset_train)
    val_path = os.path.join(_project_root, config.dataset_val)
    train_samples = load_jsonl(train_path, args.max_samples)
    val_samples = load_jsonl(val_path, args.max_samples)
    logger.info(f"Loaded {len(train_samples)} train, {len(val_samples)} val samples")

    images_root = os.path.join(_project_root, config.images_dir)

    feat_extractor = DenseNetFeatureExtractor(
        os.path.join(_project_root, config.densenet_checkpoint)
    ).eval().to(device)

    all_summaries = {}

    checkpoints = []
    if not args.phase4_only:
        phase2_path = os.path.join(_project_root, config.phase2_projection)
        if os.path.exists(phase2_path):
            checkpoints.append(("phase2", phase2_path))
        else:
            logger.warning(f"Phase 2 projection not found: {phase2_path}")

    if not args.phase2_only:
        phase4_path = os.path.join(_project_root, config.phase4_projection)
        if os.path.exists(phase4_path):
            checkpoints.append(("phase4", phase4_path))
        else:
            logger.warning(f"Phase 4 projection not found: {phase4_path}")

    for tag, ckpt_path in checkpoints:
        logger.info(f"\nEvaluating {tag} projection: {ckpt_path}")
        projection = load_projection(ckpt_path, config, device)

        summary = evaluate_projection_probe(
            projection, feat_extractor,
            train_samples, val_samples,
            images_root, device, config,
        )

        all_summaries[tag] = summary
        print_summary(tag, summary)

        del projection
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "phase2" in all_summaries and "phase4" in all_summaries:
        p2 = all_summaries["phase2"]
        p4 = all_summaries["phase4"]
        logger.info(f"\n{'=' * 60}")
        logger.info("  Phase 2 → Phase 4 comparison")
        logger.info(f"{'=' * 60}")
        for key in ("probe_accuracy", "macro_f1", "weighted_f1"):
            delta = p4[key] - p2[key]
            logger.info(
                f"  {key:<20} {p2[key]:>8.4f} → {p4[key]:>8.4f}  "
                f"({'+' if delta >= 0 else ''}{delta:.4f})"
            )
        logger.info(f"{'=' * 60}")

    os.makedirs(os.path.join(_project_root, config.output_dir), exist_ok=True)
    summary_path = os.path.join(
        _project_root, config.output_dir, "projection_probe_summary.json",
    )
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)

    logger.info(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
