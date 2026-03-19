"""
Step 2 — Build GPT-4o batch request file.

Walks images/train/ and images/validation/, encodes each image to base64,
and writes a JSONL file ready for the OpenAI Batch API.

Usage:
    python scripts/prepare_batch.py
"""

import base64
import json
from pathlib import Path

from prompts import DISEASE_LABELS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

IMAGES_DIR = Path("images")
CHUNKS_DIR = Path("batch_chunks")
MODEL = "gpt-4o"
MAX_TOKENS = 700
SPLITS = ["train", "validation"]
CHUNK_SIZE = 50  # ~54,000 enqueued tokens per batch (safe under 90k limit)


def encode_image(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_request(image_path: Path, split: str, label: str) -> dict:
    disease_label = DISEASE_LABELS[label]
    user_text = USER_PROMPT_TEMPLATE.format(disease_label=disease_label)
    b64 = encode_image(image_path)

    return {
        "custom_id": f"{split}__{label}__{image_path.name}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": MODEL,
            "max_tokens": MAX_TOKENS,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "low",
                            },
                        },
                        {"type": "text", "text": user_text},
                    ],
                },
            ],
        },
    }


def iter_images(split: str) -> list[tuple[Path, str]]:
    split_dir = IMAGES_DIR / split
    pairs: list[tuple[Path, str]] = []
    for label_dir in sorted(split_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        if label not in DISEASE_LABELS:
            print(f"  [skip] unknown label folder: {label_dir}")
            continue
        for img in sorted(label_dir.glob("*.jpg")):
            pairs.append((img, label))
    return pairs


def main() -> None:
    all_pairs: list[tuple[Path, str, str]] = []
    for split in SPLITS:
        pairs = iter_images(split)
        print(f"{split}: {len(pairs)} images")
        all_pairs.extend((img, split, label) for img, label in pairs)

    CHUNKS_DIR.mkdir(exist_ok=True)
    chunks = [all_pairs[i:i + CHUNK_SIZE] for i in range(0, len(all_pairs), CHUNK_SIZE)]

    for idx, chunk in enumerate(chunks):
        chunk_file = CHUNKS_DIR / f"batch_chunk_{idx:02d}.jsonl"
        with open(chunk_file, "w") as out:
            for image_path, split, label in chunk:
                request = build_request(image_path, split, label)
                out.write(json.dumps(request) + "\n")
        print(f"  chunk {idx:02d}: {len(chunk)} requests → {chunk_file}")

    print(f"\nWrote {len(all_pairs)} requests across {len(chunks)} chunks → {CHUNKS_DIR}/")


if __name__ == "__main__":
    main()
