"""
Step 4 — Download results from all completed batch chunks and save dataset.

Reads batch_ids.txt (written by submit_batch.py), downloads each batch output,
and writes:
  dataset/train.jsonl
  dataset/validation.jsonl

Each line:
  {"image_path": "...", "split": "...", "label": "...", "note": "..."}

Usage:
    python scripts/parse_results.py
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BATCH_IDS_FILE = Path("batch_ids.txt")
DATASET_DIR = Path("dataset")

load_dotenv()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)


def load_batch_ids() -> list[str]:
    if not BATCH_IDS_FILE.exists():
        sys.exit("Error: batch_ids.txt not found. Run submit_batch.py first.")
    return [line.strip() for line in BATCH_IDS_FILE.read_text().splitlines() if line.strip()]


def download_results(client: OpenAI, batch_id: str) -> list[dict]:
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        print(f"  [skip] batch {batch_id} has no output (status: {batch.status})")
        return []
    content = client.files.content(batch.output_file_id).text
    return [json.loads(line) for line in content.strip().splitlines()]


def parse_custom_id(custom_id: str) -> tuple[str, str, str]:
    """Returns (split, label, filename) from 'split__label__filename'."""
    parts = custom_id.split("__", 2)
    if len(parts) != 3:
        raise ValueError(f"Unexpected custom_id format: {custom_id}")
    return parts[0], parts[1], parts[2]


def extract_note(result: dict) -> str | None:
    try:
        return result["response"]["body"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        return None


def save_dataset(records: list[dict]) -> None:
    DATASET_DIR.mkdir(exist_ok=True)
    split_files: dict[str, list[str]] = {"train": [], "validation": []}

    for result in records:
        custom_id = result.get("custom_id", "")
        note = extract_note(result)

        if note is None:
            print(f"  [skip] no note for {custom_id}")
            continue

        split, label, filename = parse_custom_id(custom_id)
        image_path = f"images/{split}/{label}/{filename}"
        record = json.dumps(
            {"image_path": image_path, "split": split, "label": label, "note": note}
        )

        if split in split_files:
            split_files[split].append(record)
        else:
            print(f"  [skip] unknown split '{split}' for {custom_id}")

    for split, lines in split_files.items():
        out_path = DATASET_DIR / f"{split}.jsonl"
        with open(out_path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Saved {len(lines)} records → {out_path}")


def main() -> None:
    client = get_client()
    batch_ids = load_batch_ids()
    print(f"Found {len(batch_ids)} batch IDs in {BATCH_IDS_FILE}\n")

    all_results: list[dict] = []
    for i, batch_id in enumerate(batch_ids):
        print(f"[{i + 1}/{len(batch_ids)}] Downloading {batch_id}...")
        results = download_results(client, batch_id)
        print(f"  {len(results)} results")
        all_results.extend(results)

    print(f"\nTotal results: {len(all_results)}")
    save_dataset(all_results)
    print("\nDone. Dataset ready in dataset/")


if __name__ == "__main__":
    main()
