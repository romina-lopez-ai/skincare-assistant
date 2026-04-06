# Skincare Multimodal Model

This repository contains the codebase for developing a **Lightweight Multimodal Dermatology Chatbot**. The system is designed to perform symptom collection (anamnesis), interpret visual information from skin images, and provide grounded clinical observations and follow-up QA.

## Project Structure

The project is organized into several key modules, each handling a specific part of the multitask training pipeline:

### 1. `datasets/`
Contains all the data used for training and evaluation.
- `images_dataset/`: Raw skin images categorized into `acne`, `eczema`, and `psoriasis`.
- `dataset_a/`: Visual alignment dataset (A) for mapping images to clinical reasoning.
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
- `system_prompt.py`: Unified system prompt.
- `dataset_d/`: Scripts for Joint Consultation data, featuring both API-based and local generation methods (LLM-based).

### 5. `visual_alignment/`
Code for multimodal integration and joint training of the Small Language Model (SLM).
- `projection.py`: MLP mapping DenseNet features to the LLM space.
- `train.py`: Main training loop for multimodal fine-tuning.
- `config.py` & `generate.py`: Configuration and inference utilities.

## Directory Structure

```text
C:\Users\lsant\Projects\skincare_model\
├───datasets\
│   ├───dataset_a\
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
│   ├───prepare_dataset_a.py
│   ├───system_prompt.py
│   └───dataset_d\
│       ├───DATASET_PREPARATION.md
│       ├───prepare_dataset_d_api_ver\
│       │   ├───parse_batch_d.py
│       │   ├───prepare_batch_d.py
│       │   └───prepare_dataset_d.py
│       └───prepare_dataset_d_local_ver\
│           ├───generate_local_d.ipynb
│           ├───generate_local_d.py
│           ├───GENERATION_MECHANISM.md
│           └───local_generation_plan.md
├───visual_alignment\
│   ├───config.py
│   ├───dataset.py
│   ├───generate.py
│   ├───note_parser.py
│   ├───projection.py
│   └───train.py
├───.gitignore
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
