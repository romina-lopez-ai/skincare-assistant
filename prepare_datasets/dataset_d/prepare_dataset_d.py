"""Prepare Dataset D — Joint Consultation dataset.

Generates:
1. Symptom collection dialogue (from SYMPTOMS section)
2. <think> + [DIAGNOSIS] integrating DenseNet + note + user symptoms.

Inputs:
  - datasets/medical_notes_dataset/train.jsonl
  - datasets/medical_notes_dataset/validation.jsonl
  - datasets/densenet_predictions.json

Output:
  - datasets/dataset_d/train.jsonl
  - datasets/dataset_d/validation.jsonl
"""

import json
import os
import time
import logging
import re

from openai import OpenAI
from dotenv import load_dotenv

from system_prompt import SYSTEM_UNIFIED

load_dotenv()
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "medical_notes_generation", ".env"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))
# Navigate to project root
PROJECT_ROOT = os.path.dirname(ROOT)
client = OpenAI()

DIALOGUE_PROMPT = """You are a medical data generator. Given the SYMPTOMS section of a clinical note, generate a natural multi-turn dialogue (3-5 turns) between a patient and a dermatology assistant.

## SYMPTOMS section from Clinical Note
{symptoms_section}

## Instructions
- The dialogue must follow the assistant's instructions: "collect detailed information through focused one-at-a-time questions covering location, duration, morphology, severity, and triggers".
- The patient should use natural, non-clinical language (e.g., "red bumps", "itchy" instead of "papules", "pruritic").
- The assistant should ask only ONE question at a time.
- The assistant must end the collection with [END_SYMPTOMS] when all relevant symptoms from the note have been covered.
- Output ONLY a JSON object with a "dialogue" key containing the list of messages (role and content), starting with the user's first message and ending with the assistant's [END_SYMPTOMS] message.

Example Output:
{{
  "dialogue": [
    {{"role": "user", "content": "I have some red itchy spots on my elbow."}},
    {{"role": "assistant", "content": "How long have you had them?"}},
    {{"role": "user", "content": "About a week."}},
    {{"role": "assistant", "content": "Thank you. I have everything I need.[END_SYMPTOMS]"}}
  ]
}}"""

JOINT_REASONING_PROMPT = """You are a medical dataset formatter. Given a clinical note, DenseNet predictions, and a preceding symptom collection dialogue, generate a joint reasoning response.

## Preceding Dialogue
{dialogue_text}

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
DenseNet primary: {primary} ({confidence:.2f}) — [high/moderate/low confidence].
DenseNet secondary: {secondary} ({confidence_2:.2f}) — [high/moderate/low/negligible].
Morphology: [extract 3-5 short clinical keywords from MORPHOLOGY section, e.g. "thick plaques", "silvery scales"] — [location].
User reported: [summarize symptoms reported by the user in the dialogue turns above in 1-2 sentences].
Visual + symptoms: [reason how visual evidence and reported symptoms align or conflict].
Differential: [summarize DIFFERENTIAL DIAGNOSIS section concisely].
Urgency: routine.
</think>

[DIAGNOSIS]
[Write an empathetic 2-3 sentence diagnosis that explicitly mentions BOTH what was seen in the image AND what the user described in the dialogue. Mention the condition name. Include the disclaimer about seeing a dermatologist.]

[Add 3-4 practical recommendations from the PATIENT ADVICE section as bullet points.]
[/DIAGNOSIS]

IMPORTANT:
- The [DIAGNOSIS] must sound conversational and empathetic.
- Explicitly connect the user's reported symptoms with the visual findings.
- Output ONLY the <think> and [DIAGNOSIS] blocks."""


def extract_symptoms_section(note: str) -> str:
    """Extract the SYMPTOMS section from the clinical note."""
    match = re.search(r"3\. \*\*SYMPTOMS\*\*(.*?)(?=4\. \*\*DIFFERENTIAL|$)", note, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return "No explicit symptoms section found."


def generate_dialogue(note: str) -> list:
    """Generate the symptom collection dialogue turns."""
    symptoms_section = extract_symptoms_section(note)
    prompt = DIALOGUE_PROMPT.format(symptoms_section=symptoms_section)

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Mini is enough for simple dialogue generation
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        response_format={"type": "json_object"}
    )

    content = response.choices[0].message.content
    try:
        # Some models might wrap the list in a "dialogue" key or just return the list
        data = json.loads(content)
        if isinstance(data, dict) and "dialogue" in data:
            return data["dialogue"]
        if isinstance(data, list):
            return data
        # If it's a dict with keys like "0", "1"...
        if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
            return [v for k, v in sorted(data.items(), key=lambda x: int(x[0]))]
        # Just return values if it's a dict with one key that is a list
        for val in data.values():
            if isinstance(val, list):
                return val
        return []
    except Exception as e:
        logger.error(f"Failed to parse dialogue JSON: {e}")
        return []


def generate_joint_response(note: str, densenet: dict, label: str, dialogue: list) -> str:
    """Generate the final <think> + [DIAGNOSIS] block."""
    dialogue_text = "\n".join([f"{m['role']}: {m['content']}" for m in dialogue])
    prompt = JOINT_REASONING_PROMPT.format(
        dialogue_text=dialogue_text,
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
        max_tokens=800,
    )

    return response.choices[0].message.content.strip()


def validate_response(text: str) -> bool:
    """Check that the response has both required blocks."""
    has_think = "<think>" in text and "</think>" in text
    has_diagnosis = "[DIAGNOSIS]" in text and "[/DIAGNOSIS]" in text
    return has_think and has_diagnosis


def process_split(notes_path: str, predictions: dict, output_path: str):
    """Process one split (train or validation) into Dataset D format."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load existing progress to allow resuming
    existing = set()
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    existing.add(item["image_path"])
                extra_except:
                    continue
        logger.info(f"Resuming: {len(existing)} already processed in {output_path}")

    count = len(existing)
    skipped = 0
    errors = 0

    with open(notes_path) as fin:
        lines = fin.readlines()
        total_lines = len(lines)

    with open(output_path, "a", encoding="utf-8") as fout:
        for line_num, line in enumerate(lines):
            item = json.loads(line)
            image_path = item["image_path"]
            label = item["label"]
            note = item["note"]

            if image_path in existing:
                continue

            if image_path not in predictions:
                skipped += 1
                continue

            densenet = predictions[image_path]

            for attempt in range(2):
                try:
                    # 1. Generate Dialogue
                    dialogue = generate_dialogue(note)
                    if not dialogue or "[END_SYMPTOMS]" not in dialogue[-1]["content"]:
                        logger.warning(f"Line {line_num}: invalid dialogue, retrying")
                        continue

                    # 2. Generate Joint Response
                    assistant_content = generate_joint_response(note, densenet, label, dialogue)

                    if not validate_response(assistant_content):
                        logger.warning(f"Line {line_num}: invalid format, retrying")
                        continue

                    # Construct message list
                    messages = [{"role": "system", "content": SYSTEM_UNIFIED}]
                    messages.extend(dialogue)
                    
                    # image_msg_index is the index of the message where image tokens will be injected.
                    # In Dataset D, it's after [END_SYMPTOMS].
                    # The message list currently ends with the assistant saying [END_SYMPTOMS].
                    # We add an empty user message where the image would be.
                    image_msg_index = len(messages)
                    messages.append({"role": "user", "content": ""})
                    messages.append({"role": "assistant", "content": assistant_content})

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

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    count += 1

                    if count % 10 == 0:
                        logger.info(f"Progress: {count}/{total_lines} processed")
                    break

                except Exception as e:
                    logger.error(f"Line {line_num}, attempt {attempt+1}: {e}")
                    time.sleep(1)
                    if attempt == 1:
                        errors += 1

    logger.info(f"Wrote {count} examples to {output_path} (skipped {skipped}, errors {errors})")
    return count


def main():
    pred_path = os.path.join(PROJECT_ROOT, "datasets", "densenet_predictions.json")
    if not os.path.exists(pred_path):
        logger.error(f"Predictions not found at {pred_path}")
        return

    with open(pred_path) as f:
        predictions = json.load(f)
    logger.info(f"Loaded {len(predictions)} DenseNet predictions")

    splits = {
        "train": os.path.join(PROJECT_ROOT, "datasets", "medical_notes_dataset", "train.jsonl"),
        "validation": os.path.join(PROJECT_ROOT, "datasets", "medical_notes_dataset", "validation.jsonl"),
    }

    total = 0
    for split_name, notes_path in splits.items():
        if not os.path.exists(notes_path):
            logger.warning(f"Notes path not found: {notes_path}")
            continue
        out_path = os.path.join(PROJECT_ROOT, "datasets", "dataset_d", f"{split_name}.jsonl")
        total += process_split(notes_path, predictions, out_path)

    logger.info(f"Dataset D complete. Total examples: {total}")


if __name__ == "__main__":
    main()
