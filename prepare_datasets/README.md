# Dataset Preparation Guide

This guide outlines the creation, validation, and management of datasets for training a dermatology-focused AI assistant. Datasets are generated using various scripts and validated before final use.

## Overview of Datasets

- **Dataset A**: [To be defined - Placeholder for future dataset]
- **Dataset B**: Symptom-collection dialogues for gathering dermatological symptoms without diagnosis.
- **Dataset C**: RAG (Retrieval Augmented Generation) examples with reasoning and source prioritization.
- **Dataset D**: [To be defined - Placeholder for future dataset]

Each dataset follows a similar workflow: generation → temporal storage → validation → final location.

## Dataset A: [Placeholder]

[Add details when implemented]

## Dataset B: Symptom-Collection Dialogues

### Purpose
Teaches the model to conduct structured dialogues for collecting dermatological symptoms, ensuring comprehensive clinical information gathering without providing diagnoses or treatments.

### Generation Script
Located at `prepare_datasets/prepare_dataset_b/prepare_dataset_b.py`.

Run from project root:
```bash
python prepare_datasets/prepare_dataset_b/prepare_dataset_b.py --num_examples 400 --edge_ratio 0.3
```

**Arguments:**
- `--num_examples`: Number of examples (default: varies).
- `--edge_ratio`: Proportion of edge cases.

### Temporal Storage
Generated files saved to `prepare_datasets/temporal_datasets/dataset_b/`:
- `temp_train.jsonl`
- `temp_val.jsonl`
- `temp_test.jsonl`

### Validation
Use `prepare_datasets/prepare_dataset_b/eval_dataset_b.py` for automated evaluation across 7 dimensions (structural integrity, message count, symptom closure, etc.).

Run:
```bash
python prepare_datasets/prepare_dataset_b/eval_dataset_b.py --input prepare_datasets/temporal_datasets/dataset_b/
```

Outputs `eval_report_b.json` with quality metrics. Perform manual review as needed.

### Final Location
After validation, copy to `datasets/dataset_b/`:
```bash
copy prepare_datasets\temporal_datasets\dataset_b\temp_train.jsonl datasets\dataset_b\train.jsonl
copy prepare_datasets\temporal_datasets\dataset_b\temp_val.jsonl datasets\dataset_b\val.jsonl
copy prepare_datasets\temporal_datasets\dataset_b\temp_test.jsonl datasets\dataset_b\test.jsonl
```

## Dataset C: RAG with Reasoning

### Purpose
Trains the model for RAG tasks: receiving prioritized context, reasoning in `<think></think>` tags, comparing sources, citing references, and handling edge cases without medical prescriptions.

### Generation Script
Located at `prepare_datasets/prepare_dataset_c/prepare_dataset_c.py`.

Run from project root:
```bash
python prepare_datasets/prepare_dataset_c/prepare_dataset_c.py --num_examples 300
```

**Arguments:**
- `--num_examples`: Total examples (default: 100, recommend ≥100).
- `--output`: Custom output dir (default: `prepare_datasets/temporal_datasets/dataset_c`).

**Distribution:**
- 35% Single-source examples
- 35% Multi-source comparisons
- 30% Edge cases (8 types)

### Temporal Storage
Files in `prepare_datasets/temporal_datasets/dataset_c/`:
- `temp_train.jsonl`
- `temp_val.jsonl`
- `temp_test.jsonl`

### Validation
Use `prepare_datasets/prepare_dataset_c/eval_dataset_c.py` for automated review and cleaning.

Run:
```bash
python prepare_datasets/prepare_dataset_c/eval_dataset_c.py --input prepare_datasets/temporal_datasets/dataset_c/train.jsonl --output prepare_datasets/temporal_datasets/dataset_c_validated_temp/
```

Follow with manual review for quality assurance.

### Final Location
Copy validated files to `datasets/dataset_c/`:
```bash
mkdir -p datasets/dataset_c
cp prepare_datasets/temporal_datasets/dataset_c/train.jsonl datasets/dataset_c/
cp prepare_datasets/temporal_datasets/dataset_c/val.jsonl datasets/dataset_c/
cp prepare_datasets/temporal_datasets/dataset_c/test.jsonl datasets/dataset_c/
```

## Dataset D: [Placeholder]

[Add details when implemented]

## Final Dataset Structure

Validated datasets in `datasets/`:
```
datasets/
├── dataset_a/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── dataset_b/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
├── dataset_c/
│   ├── train.jsonl
│   ├── val.jsonl
│   └── test.jsonl
└── dataset_d/
    ├── train.jsonl
    ├── val.jsonl
    └── test.jsonl
```

All files are in JSONL format, ready for training.