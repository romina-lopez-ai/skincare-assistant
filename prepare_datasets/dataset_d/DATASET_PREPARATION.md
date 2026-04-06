# Dataset Preparation Guide

This directory contains the scripts required to generate the fine-tuning datasets for the Skincare Multimodal Model. These scripts transform raw clinical notes and vision model predictions into structured, multi-turn dialogues suitable for training a Small Language Model (SLM).

## Prerequisites

### 1. Environment & API Keys
The generation scripts use either OpenAI's GPT-4o models or local LLMs.
- **Python Libraries**: Install the required dependencies:
  ```bash
  pip install openai python-dotenv tqdm transformers torch accelerate bitsandbytes
  ```
- **API Key (Optional)**: If using OpenAI scripts, create a `.env` file with your `OPENAI_API_KEY`.

### 2. Required Input Files
Before running the scripts, ensure the following files are present in the `datasets/` directory:
- `datasets/densenet_predictions.json`: Pre-computed predictions from the DenseNet-121 vision backbone.
- `datasets/medical_notes_dataset/train.jsonl`: Ground truth clinical notes.
- `datasets/medical_notes_dataset/validation.jsonl`: Ground truth clinical notes.

## Dataset Scripts

### 1. `prepare_dataset_a.py` (Visual Alignment)
- **Task**: Teaches the model to "see" and interpret visual tokens using clinical notes as ground truth.
- **Output**: `datasets/dataset_a/train.jsonl` and `validation.jsonl`.
- **Run**: `python prepare_dataset_a.py`

### 2. `generate_local_d.py` (Joint Consultation - Local)
- **Task**: Generates Dataset D using a local Large Language Model (e.g., Qwen3.5-4B). This is the recommended method as it avoids API costs and leverages local GPUs.
- **Technical Details**: See [GENERATION_MECHANISM.md](./GENERATION_MECHANISM.md) for details on the prompting and parsing logic.
- **Output**: `datasets/dataset_d/train.jsonl` and `validation.jsonl`.
- **Run**: `python generate_local_d.py`

### 3. `prepare_dataset_d.py` (Joint Consultation - OpenAI)
- **Task**: Generates Dataset D using the OpenAI API (GPT-4o).
- **Run**: `python prepare_dataset_d.py`

### 4. Batch Mode (OpenAI Batch API)
For large-scale generation of Dataset D, use the Batch API to save 50% on costs:

1. **Prepare Batch File**:
   ```bash
   python prepare_batch_d.py
   ```
   This creates `datasets/dataset_d/batch_input_d.jsonl`.

2. **Submit to OpenAI**: Use your preferred method or the project's submission script to upload this file to the OpenAI Batch API.

3. **Download Results**: Once complete, save the results as `datasets/dataset_d/batch_output_d.jsonl`.

4. **Parse Results**:
   ```bash
   python parse_batch_d.py
   ```
   This script **de-multiplexes** the results back into their respective splits. It performs the following:
   - Identifies whether a record belongs to `train` or `validation`.
   - Reconstructs the full chat history with the `SYSTEM_UNIFIED` prompt and generated dialogues.
   - Dynamically calculates the `image_msg_index` for visual token injection.
   - Re-attaches metadata like `DenseNet_output`.
   - Outputs the final `datasets/dataset_d/train.jsonl` and `validation.jsonl` files.

## Output Format

Both scripts generate `.jsonl` files where each line is a JSON object containing:
- `image_path`: Path to the associated skin image.
- `image_msg_index`: The specific index in the `messages` list where visual tokens should be injected.
- `DenseNet_output`: The raw predictions used for internal reasoning.
- `messages`: A full chat history including system prompt, user messages, and assistant responses (with `<think>` and `[DIAGNOSIS]` blocks).

## Troubleshooting

- **Rate Limits**: If you encounter OpenAI rate limits, the scripts include basic retry logic with exponential backoff.
- **Missing Predictions**: If an image path in the clinical notes is not found in `densenet_predictions.json`, that example will be skipped automatically.
- **Invalid Formats**: If the LLM fails to generate the required `<think>` or `[DIAGNOSIS]` tags, the script will retry the generation up to 2 times before logging an error.
