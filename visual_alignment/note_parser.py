"""Parse and truncate medical notes to keep only disease label + MORPHOLOGY + LESION CHARACTERISTICS."""

import re
import logging

logger = logging.getLogger(__name__)

# Matches section 3 (SYMPTOMS) across all formatting variants:
#   "3. **SYMPTOMS**"  "**3. SYMPTOMS:**"  "### 3. Symptoms"  "3. **Symptoms:**"
#   "### SYMPTOMS"     "**SYMPTOMS:**"     "## Symptoms"  (no number prefix)
_SYMPTOMS_PATTERN = re.compile(
    r'\n+\s*(?:#{2,4}\s*)?(?:\*\*)?(?:3[\.\)]\s*)?(?:\*\*)?\s*(?:SYMPTOMS|Symptoms)\b',
    re.IGNORECASE,
)


def truncate_note(note: str) -> str:
    """Keep disease label + MORPHOLOGY + LESION CHARACTERISTICS, strip the rest.

    Finds where section 3 (SYMPTOMS) starts and returns everything before it.
    Falls back to the full note if the pattern is not found.
    """
    match = _SYMPTOMS_PATTERN.search(note)
    if match:
        return note[:match.start()].strip()

    logger.warning("Could not find SYMPTOMS section — returning full note")
    return note.strip()


if __name__ == "__main__":
    import json
    import sys
    from collections import defaultdict

    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else "datasets/medical_notes_dataset/train.jsonl"
    counts = defaultdict(int)
    failures = []

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            original = item["note"]
            truncated = truncate_note(original)
            label = item["label"]
            counts[label] += 1

            if truncated == original.strip():
                failures.append((i, label))

            # Print 3 samples per class
            if counts[label] <= 3:
                print(f"\n{'='*60}")
                print(f"[{label}] Sample {counts[label]} (line {i})")
                print(f"{'='*60}")
                print(truncated[:500])

    print(f"\n\nTotal samples: {sum(counts.values())}")
    print(f"Per class: {dict(counts)}")
    if failures:
        print(f"WARNING: {len(failures)} notes could not be truncated (returned full note)")
        for idx, label in failures[:5]:
            print(f"  line {idx} ({label})")
    else:
        print("All notes truncated successfully.")
