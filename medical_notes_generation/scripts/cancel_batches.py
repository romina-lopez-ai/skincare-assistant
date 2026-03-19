"""Cancel all in-progress or pending batches to free up enqueued token quota."""

import os
import sys

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
batches = client.batches.list(limit=20)

for batch in batches.data:
    if batch.status in ("validating", "in_progress", "finalizing"):
        print(f"Cancelling {batch.id} (status: {batch.status})...")
        client.batches.cancel(batch.id)
    else:
        print(f"Skipping  {batch.id} (status: {batch.status})")

print("\nDone.")
