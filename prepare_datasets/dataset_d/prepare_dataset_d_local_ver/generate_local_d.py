"""Local Transformer-based Dataset D Generation.

This script uses a local Large Language Model (via the transformers library) 
to generate the Joint Consultation dataset (Dataset D).
It combines dialogue generation and joint reasoning into a SINGLE model call 
per record for efficiency.

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
import logging
import re
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Fix ModuleNotFoundError & NameError (__file__)
import sys
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments like Jupyter/REPL
    SCRIPT_DIR = os.getcwd()

# Add the parent directory to sys.path to find system_prompt.py
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from system_prompt import SYSTEM_UNIFIED

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Constants
ROOT = SCRIPT_DIR
# Go up TWO levels from 'prepare_datasets/dataset_d/' to reach the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(ROOT))
MODEL_NAME = "Qwen/Qwen3.5-4B" # As requested in the transformer.py example

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


FEW_SHOT_EXAMPLE = """
### Example Input
- **Clinical Note**: 3. **SYMPTOMS**: Red itchy bumps on the elbow for 2 days. 4. **DIFFERENTIAL**: Psoriasis, Eczema.
- **DenseNet Predictions**: Primary: psoriasis (0.80), Secondary: eczema (0.15)
- **Ground Truth Label**: psoriasis

### Example Output
{
  "dialogue": [
    {"role": "user", "content": "I have some red itchy bumps on my elbow."},
    {"role": "assistant", "content": "How long have you had these bumps?"},
    {"role": "user", "content": "Just for about two days now."},
    {"role": "assistant", "content": "Thank you. All details recorded.[END_SYMPTOMS]"}
  ],
  "assistant_content": "<think>\\nDenseNet primary: psoriasis (0.80) — high confidence.\\nDenseNet secondary: eczema (0.15) — low confidence.\\nMorphology: red itchy bumps — elbow.\\nUser reported: Red itchy bumps on the elbow for two days.\\nVisual + symptoms: Visual evidence of scaling and the reported itchiness align with early psoriasis.\\nDifferential: Eczema.\\nUrgency: routine.\\n</think>\\n\\n[DIAGNOSIS]\\nBased on the image and your description of itchy bumps, this appears to be psoriasis. You should consult a dermatologist for a formal assessment.\\n- Keep the area moisturized.\\n- Avoid scratching.\\n[/DIAGNOSIS]"
}
"""

def clean_json_string(text: str) -> str:
    """Extract the first valid JSON object from a string."""
    text = text.strip()
    # Ensure it starts with '{' if we pre-filled it
    if not text.startswith('{'):
        start_idx = text.find('{')
        if start_idx == -1:
            return text
        text = text[start_idx:]
        
    try:
        # JSONDecoder.raw_decode will parse the first valid JSON object and stop.
        # This is perfect for stripping trailing chatter.
        decoder = json.JSONDecoder()
        obj, index = decoder.raw_decode(text)
        return json.dumps(obj) # Re-serialize to ensure it's clean
    except json.JSONDecodeError:
        # Fallback to the previous rfind method if raw_decode fails
        end_idx = text.rfind('}')
        if end_idx != -1:
            return text[:end_idx + 1]
    return text


class DatasetDGenerator:
    def __init__(self, model_name: str):
        logger.info(f"Loading model and tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # VRAM Optimization: Use 4-bit quantization to fit on consumer GPUs
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        self.model.eval()

    def generate_record(self, note: str, densenet_info: str, label: str) -> dict:
        """Generate a single record using the local LLM."""
        prompt = f"{COMBINED_PROMPT}\n\n{FEW_SHOT_EXAMPLE}\n\n### Real Task\n- **Clinical Note**: {note}\n- **DenseNet Predictions**: {densenet_info}\n- **Ground Truth Label**: {label}\n\nOutput ONLY the JSON object. Use '\\n' for newlines inside strings. Do not use markdown. Start with '{{'."

        # 1. Prepare the prompt text
        messages = [
            {"role": "system", "content": "You are a medical data generator. You output ONLY valid JSON. No conversation. Use escaped newlines (\\n)."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            prompt_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompt_text += "{"
        except Exception:
            prompt_text = prompt + "\n\nAssistant: {"

        # 2. Tokenize
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.model.device)

        # 3. Generate
        with torch.no_grad():
            output_tokens = self.model.generate(
                **inputs,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
            )

        # 4. Decode
        input_length = inputs.input_ids.shape[1]
        new_tokens = output_tokens[0][input_length:]
        response_text = "{" + self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        try:
            cleaned_text = clean_json_string(response_text)
            data = json.loads(cleaned_text)
            return data
        except Exception as e:
            logger.error(f"Failed to parse JSON: {e}\nResponse text snippet: {response_text[:200]}...")
            return None

    def process_splits(self, predictions: dict):
        splits = ["train", "validation"]
        output_dir = os.path.join(PROJECT_ROOT, "datasets", "dataset_d")
        os.makedirs(output_dir, exist_ok=True)

        for split in splits:
            notes_path = os.path.join(PROJECT_ROOT, "datasets", "medical_notes_dataset", f"{split}.jsonl")
            output_path = os.path.join(output_dir, f"{split}.jsonl")
            
            if not os.path.exists(notes_path):
                logger.warning(f"Notes path not found: {notes_path}")
                continue

            # Load existing progress
            existing_images = set()
            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8") as f:
                    for line in f:
                        existing_images.add(json.loads(line)["image_path"])
                logger.info(f"Resuming {split}: {len(existing_images)} already processed.")

            with open(notes_path, "r", encoding="utf-8") as fin:
                lines = fin.readlines()

            with open(output_path, "a", encoding="utf-8") as fout:
                for line in tqdm(lines, desc=f"Processing {split}"):
                    item = json.loads(line)
                    image_path = item["image_path"]

                    if image_path in existing_images:
                        continue
                    if image_path not in predictions:
                        continue

                    densenet = predictions[image_path]
                    densenet_info = (f"Primary: {densenet['primary']} ({densenet['confidence']:.2f}), "
                                     f"Secondary: {densenet['secondary']} ({densenet['confidence_2']:.2f})")

                    # Generate with retry
                    record_data = None
                    for attempt in range(2):
                        record_data = self.generate_record(item["note"], densenet_info, item["label"])
                        if record_data and "dialogue" in record_data and "assistant_content" in record_data:
                            # Basic validation of dialogue end
                            if "[END_SYMPTOMS]" in record_data["dialogue"][-1]["content"]:
                                break
                        logger.warning(f"Retrying generation for {image_path} (Attempt {attempt+1})")
                    
                    if not record_data:
                        logger.error(f"Failed to generate valid data for {image_path} after 2 attempts.")
                        continue

                    # Construct final record
                    messages = [{"role": "system", "content": SYSTEM_UNIFIED}]
                    messages.extend(record_data["dialogue"])
                    
                    image_msg_index = len(messages)
                    messages.append({"role": "user", "content": ""})
                    messages.append({"role": "assistant", "content": record_data["assistant_content"]})

                    final_record = {
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

                    fout.write(json.dumps(final_record, ensure_ascii=False) + "\n")
                    fout.flush()


def main():
    pred_path = os.path.join(PROJECT_ROOT, "datasets", "densenet_predictions.json")
    if not os.path.exists(pred_path):
        logger.error(f"Predictions not found at {pred_path}")
        return

    with open(pred_path) as f:
        predictions = json.load(f)

    generator = DatasetDGenerator(MODEL_NAME)
    generator.process_splits(predictions)
    logger.info("Dataset D local generation complete.")


if __name__ == "__main__":
    main()
