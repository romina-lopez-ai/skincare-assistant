"""Quick diagnostic — prints full batch details including error info."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
batch_id = sys.argv[1] if len(sys.argv) > 1 else Path("batch_ids.txt").read_text().splitlines()[-1].strip()
batch = client.batches.retrieve(batch_id)

print(f"Status:       {batch.status}")
print(f"Counts:       {batch.request_counts}")
print(f"Input file:   {batch.input_file_id}")
print(f"Output file:  {batch.output_file_id}")
print(f"Error file:   {batch.error_file_id}")
print(f"Errors:       {batch.errors}")

if batch.error_file_id:
    print("\n--- Error file contents ---")
    print(client.files.content(batch.error_file_id).text[:2000])
