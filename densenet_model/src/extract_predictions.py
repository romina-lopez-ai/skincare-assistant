"""Extract DenseNet predictions for all train + validation images.

Outputs a JSON file: datasets/densenet_predictions.json
{
  "images/train/acne/acne_00000.jpg": {
    "primary": "acne", "confidence": 0.94,
    "secondary": "eczema", "confidence_2": 0.04,
    "tertiary": "psoriasis", "confidence_3": 0.02
  },
  ...
}
"""

import json
import os
import sys

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────────
ROOT         = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHECKPOINT   = os.path.join(ROOT, "densenet_model", "outputs", "model.pth")
IMAGES_ROOT  = os.path.join(ROOT, "datasets", "images_dataset")
OUTPUT_JSON  = os.path.join(ROOT, "datasets", "densenet_predictions.json")

CLASSES = ["acne", "eczema", "psoriasis"]
SPLITS  = ["train", "validation"]

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ── load model ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(ROOT, "densenet_model", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = build_model(pretrained=False, fine_tune=False, num_classes=3)
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval().to(device)
print(f"Loaded checkpoint: {CHECKPOINT}")

# ── run inference ──────────────────────────────────────────────────────────
predictions = {}

for split in SPLITS:
    for cls in CLASSES:
        folder = os.path.join(IMAGES_ROOT, split, cls)
        if not os.path.isdir(folder):
            print(f"[WARN] folder not found: {folder}")
            continue

        image_files = [f for f in os.listdir(folder)
                       if f.lower().endswith((".jpg", ".jpeg", ".png"))]

        print(f"\n{split}/{cls}: {len(image_files)} images")

        for fname in tqdm(image_files, desc=f"{split}/{cls}"):
            img_path = os.path.join(folder, fname)
            rel_path = f"images/{split}/{cls}/{fname}"

            try:
                image = Image.open(img_path).convert("RGB")
                tensor = TRANSFORM(image).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs  = F.softmax(logits, dim=1).squeeze().cpu()

                sorted_idx  = probs.argsort(descending=True)
                predictions[rel_path] = {
                    "primary":      CLASSES[sorted_idx[0]],
                    "confidence":   round(probs[sorted_idx[0]].item(), 4),
                    "secondary":    CLASSES[sorted_idx[1]],
                    "confidence_2": round(probs[sorted_idx[1]].item(), 4),
                    "tertiary":     CLASSES[sorted_idx[2]],
                    "confidence_3": round(probs[sorted_idx[2]].item(), 4),
                }
            except Exception as e:
                print(f"[ERROR] {rel_path}: {e}")

# ── save ───────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\nDone. {len(predictions)} predictions saved to {OUTPUT_JSON}")
