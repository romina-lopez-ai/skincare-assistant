# Medical Notes Generation

An end-to-end pipeline for fine-tuning a vision-language model to automatically generate structured clinical dermatology notes from skin condition images.

## Overview

This project builds a dermatology AI assistant that takes skin condition images as input and produces clinical notes covering:

- Morphology and lesion characteristics
- Symptom description
- Differential diagnosis considerations
- General patient advice

The system includes **medical safeguards** that prevent the model from recommending specific medications or treatments.

**Conditions covered:** Acne, Eczema, Psoriasis

## Project Status

| Step | Description | Status |
|------|-------------|--------|
| 1 | Environment setup | Done |
| 2 | Prepare batch annotation requests | Done |
| 3 | Submit batches to OpenAI | Done |
| 4 | Parse results & build dataset | Done |
| 5 | Fine-tune vision-language model | Pending |
| 6 | Evaluate on test split | Pending |
| 7 | Inference script | Pending |

## Dataset

| Split | Images | Annotated |
|-------|--------|-----------|
| Train | 640 | Yes (dataset/train.jsonl) |
| Validation | 137 | Yes (dataset/validation.jsonl) |
| Test | 138 | No (reserved for evaluation) |

Each split is balanced across three labels: acne, eczema, and psoriasis.

## Directory Structure

```
medical_notes_generation/
├── images/
│   ├── train/          # 640 training images (acne, eczema, psoriasis)
│   ├── validation/     # 137 validation images
│   └── test/           # 138 test images (unannotated)
├── dataset/
│   ├── train.jsonl     # 640 annotated training records
│   └── validation.jsonl # 137 annotated validation records
├── scripts/
│   ├── prompts.py          # GPT-4o prompt templates with medical safeguards
│   ├── prepare_batch.py    # Step 2: Build batch request files
│   ├── submit_batch.py     # Step 3: Upload and submit batches to OpenAI
│   ├── parse_results.py    # Step 4: Download results and save dataset
│   ├── check_batch.py      # Diagnostic: inspect batch status
│   └── cancel_batches.py   # Utility: cancel pending batches
├── batch_chunks/       # 16 chunked batch request files
├── batch_ids.txt       # OpenAI batch IDs (16 total)
├── plan.md             # Full project specification
└── requirements.txt    # Python dependencies
```

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Dependencies:**

| Package | Purpose |
|---------|---------|
| `openai>=1.30.0` | OpenAI Batch API and Files API |
| `python-dotenv>=1.0.0` | Load API key from `.env` |
| `Pillow>=10.0.0` | Image processing |

## Configuration

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your_key_here
```

## Annotation Pipeline (Steps 2–4)

These steps have already been run to produce the dataset. Re-run only if regenerating annotations.

### Step 2 — Prepare batch request files

```bash
python scripts/prepare_batch.py
```

Walks `images/train/` and `images/validation/`, encodes images to base64, and generates 16 JSONL chunk files in `batch_chunks/` (50 requests per chunk).

### Step 3 — Submit batches to OpenAI

```bash
python scripts/submit_batch.py
```

Uploads each chunk to the OpenAI Files API, submits batch jobs with a 24-hour completion window, polls for completion, and saves batch IDs to `batch_ids.txt`.

### Step 4 — Parse results and save dataset

```bash
python scripts/parse_results.py
```

Reads `batch_ids.txt`, downloads completed results, parses GPT-4o responses, and writes:
- `dataset/train.jsonl` — 640 records
- `dataset/validation.jsonl` — 137 records

### Utility scripts

```bash
python scripts/check_batch.py     # Inspect batch status and errors
python scripts/cancel_batches.py  # Cancel pending or in-progress batches
```

## Dataset Format

Each record in the JSONL files has the following structure:

```json
{
  "image_path": "images/train/acne/image_001.jpg",
  "split": "train",
  "label": "acne",
  "clinical_note": "..."
}
```

## Medical Safeguards

The system applies a two-layer approach to prevent medication advice:

1. **Annotation prompt** — The GPT-4o prompt explicitly forbids drug, medication, and prescription recommendations during dataset generation. All advice is general (hygiene, lifestyle, triggers) and closes with a recommendation to consult a dermatologist.

2. **Inference system prompt** — Applied at runtime regardless of fine-tuning. Reinforces no specific medication recommendations and encourages consulting healthcare professionals for severe cases.

## Cost Summary

| Task | Estimated Cost |
|------|---------------|
| Annotation (train + validation) | ~$1.70 (completed) |
| Fine-tuning (Google Colab) | ~$5–15 |
| **Total** | **< $20** |
