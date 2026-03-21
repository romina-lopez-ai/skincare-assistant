"""Prepare Dataset A — Visual Alignment dataset with <think> + [DIAGNOSIS] format.

Uses GPT-4o to generate natural <think> + [DIAGNOSIS] from DenseNet predictions
and clinical notes.

Inputs:
  - datasets/medical_notes_dataset/train.jsonl
  - datasets/medical_notes_dataset/validation.jsonl
  - datasets/densenet_predictions.json

Output:
  - datasets/dataset_a/train.jsonl
  - datasets/dataset_a/validation.jsonl
"""

import json
import os
import time
import logging

from openai import OpenAI
from dotenv import load_dotenv

from system_prompt import SYSTEM_UNIFIED

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "medical_notes_generation", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
client = OpenAI()

GENERATION_PROMPT = """You are a medical dataset formatter. Given a clinical note and DenseNet model predictions for a skin image, generate a structured response with two parts:

1. A <think> block containing internal clinical reasoning
2. A [DIAGNOSIS] block containing a conversational patient-facing response

## DenseNet Predictions
- Primary: {primary} (confidence: {confidence:.2f})
- Secondary: {secondary} (confidence: {confidence_2:.2f})

## Ground Truth Label
{label}

## Clinical Note
{note}

## Instructions

Generate the response in EXACTLY this format:

<think>
DenseNet primary: {primary} ({confidence:.2f}) — [high/moderate/low confidence based on value].
DenseNet secondary: {secondary} ({confidence_2:.2f}) — [high/moderate/low/negligible].
Morphology: [extract 3-5 short clinical keywords from the MORPHOLOGY section, e.g. "thick plaques", "silvery scales", "erythema"] — [body location].
Differential: [summarize DIFFERENTIAL DIAGNOSIS section concisely — which conditions ruled out and why].
Urgency: routine.
</think>

[DIAGNOSIS]
[Write a warm, empathetic 2-3 sentence observation of what you see in the image, using natural conversational language. Mention the condition name. Then add a disclaimer.]

[Add 3-4 practical recommendations from the PATIENT ADVICE section as bullet points.]
[/DIAGNOSIS]

IMPORTANT:
- Morphology keywords must be SHORT clinical terms (e.g. "papules", "erythema", "silvery scales"), NOT full sentences
- The [DIAGNOSIS] must sound conversational and empathetic, NOT like a clinical report
- Vary your opening — don't always start with "From the image"
- Always include the disclaimer about seeing a dermatologist
- Output ONLY the <think> and [DIAGNOSIS] blocks, nothing else"""


def generate_response(note: str, densenet: dict, label: str) -> str:
    """Call GPT-4o to generate <think> + [DIAGNOSIS] from note + predictions."""
    prompt = GENERATION_PROMPT.format(
        primary=densenet["primary"],
        confidence=densenet["confidence"],
        secondary=densenet["secondary"],
        confidence_2=densenet["confidence_2"],
        label=label,
        note=note,
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=600,
    )

    return response.choices[0].message.content.strip()


def validate_response(text: str) -> bool:
    """Check that the response has both required blocks."""
    has_think = "<think>" in text and "</think>" in text
    has_diagnosis = "[DIAGNOSIS]" in text and "[/DIAGNOSIS]" in text
    return has_think and has_diagnosis


def process_split(notes_path: str, predictions: dict, output_path: str):
    """Process one split (train or validation) into Dataset A format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing progress to allow resuming
    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                item = json.loads(line)
                existing.add(item["image_path"])
        logger.info(f"Resuming: {len(existing)} already processed in {output_path}")

    count = len(existing)
    skipped = 0
    errors = 0

    with open(notes_path) as fin, open(output_path, "a") as fout:
        lines = fin.readlines()
        total_lines = len(lines)

        for line_num, line in enumerate(lines):
            item = json.loads(line)
            image_path = item["image_path"]
            label = item["label"]
            note = item["note"]

            # Skip already processed
            if image_path in existing:
                continue

            # Look up DenseNet predictions
            if image_path not in predictions:
                logger.warning(f"Line {line_num}: no prediction for {image_path}, skipping")
                skipped += 1
                continue

            densenet = predictions[image_path]

            # API call with retry
            for attempt in range(3):
                try:
                    assistant_content = generate_response(note, densenet, label)

                    if not validate_response(assistant_content):
                        logger.warning(f"Line {line_num}: invalid format, retrying ({attempt+1}/3)")
                        continue

                    record = {
                        "image_path": image_path,
                        "image_msg_index": 1,
                        "DenseNet_output": {
                            "primary": densenet["primary"],
                            "confidence": densenet["confidence"],
                            "secondary": densenet["secondary"],
                            "confidence_2": densenet["confidence_2"],
                        },
                        "messages": [
                            {"role": "system", "content": SYSTEM_UNIFIED},
                            {"role": "user", "content": ""},
                            {"role": "assistant", "content": assistant_content},
                        ],
                    }

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    count += 1

                    if count % 25 == 0:
                        logger.info(f"Progress: {count}/{total_lines} processed")
                    break

                except Exception as e:
                    logger.error(f"Line {line_num}, attempt {attempt+1}: {e}")
                    if attempt < 2:
                        time.sleep(2 ** attempt)
                    else:
                        errors += 1

    logger.info(f"Wrote {count} examples to {output_path} (skipped {skipped}, errors {errors})")
    return count


def main():
    pred_path = os.path.join(ROOT, "datasets", "densenet_predictions.json")
    with open(pred_path) as f:
        predictions = json.load(f)
    logger.info(f"Loaded {len(predictions)} DenseNet predictions")

    splits = {
        "train": os.path.join(ROOT, "datasets", "medical_notes_dataset", "train.jsonl"),
        "validation": os.path.join(ROOT, "datasets", "medical_notes_dataset", "validation.jsonl"),
    }

    total = 0
    for split_name, notes_path in splits.items():
        out_path = os.path.join(ROOT, "datasets", "dataset_a", f"{split_name}.jsonl")
        total += process_split(notes_path, predictions, out_path)

    logger.info(f"Dataset A complete. Total examples: {total}")

    # Print a sample
    sample_path = os.path.join(ROOT, "datasets", "dataset_a", "train.jsonl")
    with open(sample_path) as f:
        sample = json.loads(f.readline())
    print("\n" + "=" * 60)
    print("SAMPLE — Dataset A")
    print("=" * 60)
    print(f"image_path: {sample['image_path']}")
    print(f"image_msg_index: {sample['image_msg_index']}")
    print(f"DenseNet_output: {json.dumps(sample['DenseNet_output'], indent=2)}")
    print(f"\nAssistant content:\n{sample['messages'][2]['content']}")


if __name__ == "__main__":
    main()
