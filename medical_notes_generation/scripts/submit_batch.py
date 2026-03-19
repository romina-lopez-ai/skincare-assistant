"""
Step 3 — Upload and submit one batch chunk at a time.

Submits chunks sequentially, waiting for each to complete before the next.
Saves all batch IDs to batch_ids.txt.

Usage:
    python scripts/submit_batch.py
"""

import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

CHUNKS_DIR = Path("batch_chunks")
BATCH_IDS_FILE = Path("batch_ids.txt")
POLL_INTERVAL = 30  # seconds

load_dotenv()


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY not set in .env")
    return OpenAI(api_key=api_key)


def upload_and_submit(client: OpenAI, chunk_file: Path) -> str:
    print(f"  Uploading {chunk_file.name} ({chunk_file.stat().st_size / 1_000:.0f} KB)...")
    with open(chunk_file, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"  Submitted → batch_id: {batch.id}")
    return batch.id


def wait_for_completion(client: OpenAI, batch_id: str) -> bool:
    """Returns True if completed, False if failed/cancelled."""
    while True:
        batch = client.batches.retrieve(batch_id)
        counts = batch.request_counts
        print(
            f"  status: {batch.status} | "
            f"completed: {counts.completed} / {counts.total} | "
            f"failed: {counts.failed}"
        )
        match batch.status:
            case "completed":
                return True
            case "failed" | "expired" | "cancelled":
                print(f"  WARNING: ended with {batch.status}. Retrying in 60s...")
                return False
            case _:
                time.sleep(POLL_INTERVAL)


def count_completed() -> int:
    if not BATCH_IDS_FILE.exists():
        return 0
    return len([line for line in BATCH_IDS_FILE.read_text().splitlines() if line.strip()])


def main() -> None:
    client = get_client()
    chunks = sorted(CHUNKS_DIR.glob("batch_chunk_*.jsonl"))

    if not chunks:
        sys.exit(f"No chunk files found in {CHUNKS_DIR}/. Run prepare_batch.py first.")

    already_done = count_completed()
    remaining = chunks[already_done:]
    print(f"Found {len(chunks)} chunks. {already_done} already completed. {len(remaining)} remaining.\n")

    if not remaining:
        print("All chunks already submitted. Run parse_results.py.")
        return

    with open(BATCH_IDS_FILE, "a") as ids_file:
        for i, chunk_file in enumerate(remaining):
            label = f"[{already_done + i + 1}/{len(chunks)}] {chunk_file.name}"

            for attempt in range(2):  # try twice per chunk
                if attempt > 0:
                    print(f"  Retry attempt {attempt + 1}...")
                    time.sleep(60)

                print(label)
                batch_id = upload_and_submit(client, chunk_file)
                success = wait_for_completion(client, batch_id)

                if success:
                    ids_file.write(batch_id + "\n")
                    ids_file.flush()
                    print(f"  Done.\n")
                    break
            else:
                print(f"  SKIPPED {chunk_file.name} after 2 attempts.\n")

    print(f"All chunks completed. Batch IDs saved → {BATCH_IDS_FILE}")
    print("Run parse_results.py to download and save the dataset.")


if __name__ == "__main__":
    main()
