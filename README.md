# Skincare Multimodal Model

This repository contains the codebase for developing a **Lightweight Multimodal Dermatology Chatbot**. The system is designed to perform symptom collection (anamnesis), interpret visual information from skin images, and provide grounded clinical observations and follow-up QA.

## Project Structure

The project is organized into several key modules, each handling a specific part of the multitask training pipeline:

### 1. `datasets/`
Contains all the data used for training and evaluation.
- `images_dataset/`: Raw skin images categorized into `acne`, `eczema`, and `psoriasis`.
- `dataset_a/`: Visual alignment dataset (A).
- `dataset_b/` & `dataset_c/`: Datasets for symptom collection and RAG QA.
- `dataset_d/`: Joint consultation dataset (D) for final multitask training.
- `medical_notes_dataset/`: Ground truth clinical notes and descriptions.
- `densenet_predictions.json`: Pre-computed predictions from the DenseNet model.

### 2. `densenet_model/`
Code for training and evaluating the **DenseNet-121** image classifier.
- `src/`: Core logic for model architecture, training, evaluation, and prediction extraction.
- `outputs/`: Training artifacts including metrics, confusion matrices, and the saved model.

### 3. `medical_notes_generation/`
Scripts for generating synthetic clinical notes and dialogues via OpenAI Batch API.
- `scripts/`: Batch submission, tracking, and parsing tools.
- `batch_chunks/`: Intermediate files for batch processing.

### 4. `prepare_datasets/`
Utility scripts to assemble the final multitask datasets.
- `prepare_dataset_a.py`: Generates the visual alignment dataset.
- `prepare_dataset_b/` & `prepare_dataset_c/`: Scripts for processing text-based tasks.
- `dataset_d/`: Scripts for Joint Consultation data (API and local generation).
- `temporal_datasets/`: Intermediate storage for processed datasets.

### 5. `visual_alignment/`
Code for multimodal integration and visual fine-tuning.
- `projection.py`: MLP mapping DenseNet features to the LLM space.
- `train.py`: Main training loop for Phase 2 (Visual Alignment).

### 6. `unified_training/`
Final multitask training and model merging.
- `train.py`: Full multitask fine-tuning script.
- `merge_model.py`: Merges LoRA weights back into the base model.
- `evaluate.py`: Evaluation logic for the unified model.

## Directory Structure

```text
skincare_model\
├───datasets\
│   ├───dataset_a\
│   ├───dataset_b\
│   ├───dataset_c\
│   ├───dataset_d\
│   ├───images_dataset\
│   ├───medical_notes_dataset\
│   └───densenet_predictions.json
├───densenet_model\
│   ├───outputs\
│   └───src\
│       ├───dataset.py
│       ├───evaluate.py
│       ├───evaluate_improved.py
│       ├───extract_predictions.py
│       ├───model.py
│       ├───predict.py
│       └───train.py
├───medical_notes_generation\
│   ├───batch_chunks\
│   ├───scripts\
│   │   ├───cancel_batches.py
│   │   ├───check_batch.py
│   │   ├───parse_results.py
│   │   ├───prepare_batch.py
│   │   ├───prompts.py
│   │   └───submit_batch.py
│   ├───batch_ids.txt
│   ├───plan.md
│   └───README.md
├───prepare_datasets\
│   ├───dataset_b\
│   ├───dataset_c\
│   ├───dataset_d\
│   ├───temporal_datasets\
│   ├───prepare_dataset_a.py
│   └───system_prompt.py
├───unified_training\
│   ├───config.py
│   ├───dataset.py
│   ├───evaluate.py
│   ├───merge_model.py
│   └───train.py
├───visual_alignment\
│   ├───config.py
│   ├───dataset.py
│   ├───generate.py
│   ├───note_parser.py
│   ├───projection.py
│   └───train.py
├───.gitignore
├───inference_example.py
├───multitask_training_plan.md
├───README.md
└───requirements.txt
```

## Training Pipeline

The project follows a curriculum learning strategy:
1. **Phase 0 (Vision):** Train DenseNet-121 on skin images.
2. **Phase 1 (Data Prep):** Generate synthetic clinical notes and dialogues.
3. **Phase 2 (Visual Alignment):** Train a projection layer for visual tokens (Dataset A).
4. **Phase 3 (Text Tasks):** Fine-tune the LLM on symptom collection and RAG QA (Datasets B & C).
5. **Phase 4 (Joint Reasoning):** Final multitask training for full consultations (Dataset D).

## Documentation
- `multitask_training_plan.md`: Full project strategy and specifications.
- `prepare_datasets/dataset_d/DATASET_PREPARATION.md`: Guide for generating Joint Consultation data.

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Refer to `multitask_training_plan.md` for the step-by-step execution guide.

## Using the Trained Model

The final merged model is available on Hugging Face: [santiagoboxiga/skincare-llama-3.2-3b](https://huggingface.co/santiagoboxiga/skincare-llama-3.2-3b)

### Requirements
```bash
pip install transformers torch accelerate torchvision pillow
```

### Interactive Chat
Run the included inference script for a multi-turn conversation with the assistant:
```bash
python inference_example.py
```

The model will be downloaded automatically from Hugging Face on first run (~6.4 GB).

To use a local copy of the merged model instead:
```bash
python inference_example.py --model-path unified_training/outputs/merged_model
```

**Options:**
```
--model-path       HF repo ID or local path (default: santiagoboxiga/skincare-llama-3.2-3b)
--max-new-tokens   Max tokens per response (default: 256)
--densenet-path    Path to DenseNet checkpoint (default: densenet_model/outputs/model.pth)
--projection-path  Path to projection.pth (auto-detected if omitted)
```

**Chat commands:**
- `reset` — start a new conversation without reloading the model
- `quit` — exit

### Consultation Flow

The script supports the full consultation pipeline: symptom collection → image upload → diagnosis.

```
You: My skin is red and itchy
Assistant: How long have you been experiencing this?
You: About 3 weeks
Assistant: Where on your body is the rash located?
...
Assistant: [END_SYMPTOMS]
------------------------------------------------------------
Symptom collection complete.
Provide a skin image for visual diagnosis (just paste the path):
------------------------------------------------------------
You: datasets/images_dataset/test/psoriasis/psoriasis_00108.jpg

Processing image ...
Assistant: <think>The image shows... </think> [DIAGNOSIS]...[/DIAGNOSIS]
------------------------------------------------------------
Diagnosis complete. Type 'reset' to start a new consultation.
------------------------------------------------------------
```

### Capabilities

The model supports three clinical modes depending on the input:

| Mode | Trigger | Behavior |
|---|---|---|
| Symptom Collection | User describes skin symptoms | Asks one focused question at a time, ends with `[END_SYMPTOMS]` |
| Follow-up QA | User message contains a `[CONTEXT]` block | Answers using only the provided context |
| Visual Diagnosis | Skin image provided after symptom collection | Reasons with `<think>`, wraps output in `[DIAGNOSIS]...[/DIAGNOSIS]` |

### Visual Pipeline

For visual diagnosis, the script requires two additional components beyond the merged LLM:

| Component | File | Purpose |
|---|---|---|
| DenseNet-121 | `densenet_model/outputs/model.pth` | Extracts 1024-dim features from skin images |
| ProjectionMLP | `projection.pth` (in merged model dir) | Maps DenseNet features → 32 visual tokens in LLM embedding space |

Both are loaded automatically at startup if present. Without them, the script runs in text-only mode (symptom collection and QA only).

> **Note:** The system prompt is required for correct model behavior. The model will not produce `[END_SYMPTOMS]`, `<think>` blocks, or disclaimers without it. The prompt is baked into `inference_example.py`.
