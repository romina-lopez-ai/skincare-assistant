"""Prepare Dataset D in Batch Mode — Part 2: Results Parsing.

This script takes the .jsonl output from the OpenAI Batch API and reconstructs 
 the final Dataset D (train.jsonl and validation.jsonl).

Inputs:
  - datasets/dataset_d/batch_output_d.jsonl
  - datasets/densenet_predictions.json

Output:
  - datasets/dataset_d/train.jsonl
  - datasets/dataset_d/validation.jsonl
"""

import json
import os
import logging
from system_prompt import SYSTEM_UNIFIED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(ROOT)

def parse_batch_results():
    pred_path = os.path.join(PROJECT_ROOT, "datasets", "densenet_predictions.json")
    with open(pred_path) as f:
        predictions = json.load(f)

    results_path = os.path.join(PROJECT_ROOT, "datasets", "dataset_d", "batch_output_d.jsonl")
    if not os.path.exists(results_path):
        logger.error(f"Batch results not found at {results_path}")
        return

    outputs = {"train": [], "validation": []}

    with open(results_path) as f:
        for line in f:
            batch_item = json.loads(line)
            custom_id = batch_item["custom_id"]
            split, image_path = custom_id.split("|", 1)
            
            if batch_item["response"]["status_code"] != 200:
                logger.warning(f"Error in {custom_id}: {batch_item['response']['body']['error']}")
                continue

            # Parse content
            content_str = batch_item["response"]["body"]["choices"][0]["message"]["content"]
            try:
                content = json.loads(content_str)
                dialogue = content["dialogue"]
                assistant_content = content["assistant_content"]
            except Exception as e:
                logger.error(f"Failed to parse JSON content for {custom_id}: {e}")
                continue

            # Construct message list
            messages = [{"role": "system", "content": SYSTEM_UNIFIED}]
            messages.extend(dialogue)
            
            # Calculate image index
            image_msg_index = len(messages)
            messages.append({"role": "user", "content": ""})
            messages.append({"role": "assistant", "content": assistant_content})

            densenet = predictions[image_path]
            
            record = {
                "image_path": image_path,
                "image_msg_index": image_msg_index,
                "DenseNet_output": {
                    "primary": densenet["primary"],
                    "confidence": densenet["confidence"],
                    "secondary": densenet["secondary"],
                    "confidence_2": densenet["confidence_2"],
                },
                "messages": messages,
            }
            outputs[split].append(record)

    # Save to splits
    for split, records in outputs.items():
        if not records:
            continue
        out_path = os.path.join(PROJECT_ROOT, "datasets", "dataset_d", f"{split}.jsonl")
        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info(f"Saved {len(records)} records to {out_path}")

if __name__ == "__main__":
    parse_batch_results()
