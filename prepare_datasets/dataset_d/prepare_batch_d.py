"""Prepare Dataset D in Batch Mode — Part 1: Request Generation.

This script creates a .jsonl file for the OpenAI Batch API.
It combines dialogue generation and joint reasoning into a SINGLE request 
per record to minimize cost and latency.

Inputs:
  - datasets/medical_notes_dataset/train.jsonl
  - datasets/medical_notes_dataset/validation.jsonl
  - datasets/densenet_predictions.json

Output:
  - datasets/dataset_d/batch_input_d.jsonl
"""

import json
import os
import logging
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

COMBINED_PROMPT = """You are a medical data generator. Given a clinical note and DenseNet model predictions for a skin image, generate two components in a single JSON object.

## Input Data
- **Clinical Note**: {note}
- **DenseNet Predictions**: {densenet_info}
- **Ground Truth Label**: {label}

## Task 1: Symptom Dialogue ('dialogue' key)
Generate a natural 3-5 turn symptom collection dialogue between a user and an assistant.
- Assistant follows: "collect information on location, duration, morphology, severity, and triggers".
- Patient uses natural, non-clinical language (e.g. "red bumps" vs "papules").
- Assistant asks ONE question at a time.
- Assistant ends with: "Thank you. All details recorded.[END_SYMPTOMS]"

## Task 2: Joint Reasoning ('assistant_content' key)
Generate a response in EXACTLY this format:
<think>
DenseNet primary: [class] ([conf]) — [confidence level].
DenseNet secondary: [class] ([conf]) — [confidence level].
Morphology: [3-5 clinical keywords from note] — [location].
User reported: [summarize symptoms from the 'dialogue' Task 1 in 1-2 sentences].
Visual + symptoms: [reason how visual evidence and Task 1 reported symptoms align].
Differential: [summarize DIFFERENTIAL DIAGNOSIS section].
Urgency: routine.
</think>

[DIAGNOSIS]
[Empathetic 2-3 sentence diagnosis mentioning BOTH visual findings and the symptoms reported in Task 1. Mention the condition name. Include disclaimer.]
- [3-4 bullet points from PATIENT ADVICE]
[/DIAGNOSIS]

## Output Format
Output ONLY a valid JSON object with two keys: "dialogue" (list of messages) and "assistant_content" (string)."""

def prepare_batch():
    pred_path = os.path.join(PROJECT_ROOT, "datasets", "densenet_predictions.json")
    with open(pred_path) as f:
        predictions = json.load(f)

    splits = ["train", "validation"]
    batch_requests = []

    for split in splits:
        notes_path = os.path.join(PROJECT_ROOT, "datasets", "medical_notes_dataset", f"{split}.jsonl")
        if not os.path.exists(notes_path):
            continue
            
        with open(notes_path) as f:
            for line in f:
                item = json.loads(line)
                image_path = item["image_path"]
                
                if image_path not in predictions:
                    continue
                
                densenet = predictions[image_path]
                densenet_info = (f"Primary: {densenet['primary']} ({densenet['confidence']:.2f}), "
                                 f"Secondary: {densenet['secondary']} ({densenet['confidence_2']:.2f})")
                
                # Custom ID includes image path and split for easy parsing later
                custom_id = f"{split}|{image_path}"
                
                prompt = COMBINED_PROMPT.format(
                    note=item["note"],
                    densenet_info=densenet_info,
                    label=item["label"]
                )

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [{"role": "user", "content": prompt}],
                        "response_format": {"type": "json_object"},
                        "temperature": 0.7
                    }
                }
                batch_requests.append(request)

    output_dir = os.path.join(PROJECT_ROOT, "datasets", "dataset_d")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "batch_input_d.jsonl")

    with open(output_path, "w", encoding="utf-8") as f:
        for req in batch_requests:
            f.write(json.dumps(req) + "\n")

    logger.info(f"Created batch input with {len(batch_requests)} requests at {output_path}")

if __name__ == "__main__":
    prepare_batch()
