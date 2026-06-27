# Skincare Multimodal Model

This repository contains the codebase for a **Lightweight Multimodal Dermatology Chatbot**. The system performs symptom collection (anamnesis), interprets visual information from skin images via a DenseNet-121 vision encoder, provides grounded clinical observations, and answers follow-up questions using a RAG pipeline backed by PubMed and local medical documents. A LangGraph orchestrator ties all components together behind a web UI.

## Demo

- ▶️ **[Demo video — full walkthrough](https://echo360.net.au/media/3cfa225d-fc8e-449c-9089-fdde13b4e1e3/public)**
- 🔗 **[Live app](https://terrst.fun/)** — *temporary deployment, may be offline*

## Team & My Contribution

This was a **Master's capstone built by a 6-person team**, published here with my teammates' permission. This repository is the **backend**; the front-end and full system integration live in a separate teammate's repository.

**My contribution** — [@romina-lopez-ai](https://github.com/romina-lopez-ai):

- Designed and built the **RAG pipeline** end-to-end — document store, retrieval, and grounded medical QA over **PubMed and MedlinePlus**.
- Implemented **multi-LoRA adapter switching** across the core services.
- **Integrated my components into the LangGraph workflow** — the symptom-collection and RAG retrieval nodes — and produced the consultation summary sent to the front-end.
- Generated the **symptom-collection and RAG-QA datasets** (datasets B & C) and the LLM fine-tuning data for symptoms + RAG.
- Wrote **tests** for these components.
- Contributed to **system-level design decisions**: LoRA adapters vs. full fine-tuning, and using a LangGraph workflow instead of relying on the model alone.

Architecture and project decisions were made collaboratively; individual contributions are visible in the commit history (`git log`).

> *Original shared team repository: [santiagoboxiga/skincare_model](https://github.com/santiagoboxiga/skincare_model)*

## Project Structure

### Training Pipeline

#### `datasets/`
All data used for training and evaluation.
- `images_dataset/`: Skin images categorized into `acne`, `eczema`, and `psoriasis`.
- `dataset_a/`: Visual alignment dataset.
- `dataset_b/` & `dataset_c/`: Symptom collection and RAG QA datasets.
- `dataset_d/`: Joint consultation dataset for final multitask training.
- `medical_notes_dataset/`: Ground truth clinical notes and descriptions.
- `densenet_predictions.json`: Pre-computed DenseNet predictions.

#### `densenet_model/`
**DenseNet-121** image classifier for skin conditions.
- `src/`: Model architecture, training, evaluation, and prediction extraction.
- `outputs/`: Training artifacts (metrics, confusion matrices, saved model).

#### `medical_notes_generation/`
Synthetic clinical note and dialogue generation via OpenAI Batch API.
- `scripts/`: Batch submission, tracking, and result parsing.

#### `prepare_datasets/`
Utility scripts to assemble the final multitask datasets (A through D).

#### `visual_alignment/`
Multimodal integration and visual fine-tuning.
- `projection.py`: MLP mapping DenseNet features to the LLM embedding space.
- `train.py`: Training loop for Phase 2 (Visual Alignment).

#### `unified_training/`
Final multitask training and model merging.
- `train.py`: Full multitask fine-tuning script.
- `merge_model.py`: Merges LoRA weights back into the base model.
- `evaluate.py`: Evaluation logic for the unified model.

### Inference & Orchestration

#### `langgraph_orchestrator/`
A **LangGraph**-based workflow that orchestrates the full consultation pipeline as a stateful graph.
- `build_graph.py`: Assembles and compiles the graph (session entry → symptom collection → image upload → vision → diagnosis → RAG follow-up).
- `nodes/`: Individual graph nodes — `collect_symptoms`, `run_vision`, `project_image_features`, `joint_diagnosis`, `retrieve_context`, `rag_answer`, `followup_router`, etc.
- `services/`: Shared services — `llm_service`, `rag_service`, `symptom_service`, `vision_service`.
- `routing.py`: Edge routing logic between nodes.
- `state.py`: Graph state definition.
- `web/`: Web application (FastAPI + HTML/JS frontend) exposing the chatbot via browser.

#### `rag/`
Retrieval Augmented Generation pipeline for grounded follow-up QA.
- `vector_store.py`: ChromaDB integration with `ParentDocumentRetriever` for parent/child chunk management.
- `query_processor.py`: Query expansion, Reciprocal Rank Fusion, FlashRank re-ranking, and score filtering.
- `retriever.py`: End-to-end retrieval orchestration.
- `answer_generator.py`: Final answer generation from retrieved context.
- `prompt_factory.py`: Prompt construction consistent with model training format.
- `model_loader.py`: Singleton access to embedding, re-ranker, and LLM models.

#### `scripts/`
Indexing and utility scripts.
- `rag_index_pubmed.py` / `rag_index_medlineplus.py` / `rag_index_local.py`: Populate the ChromaDB vector store from PubMed, MedlinePlus, and local PDF documents.
- `check_chroma.py`: Inspect ChromaDB collection contents.

### Evaluation & Benchmarking

#### `evaluation_metrics/`
Model evaluation and analysis tools.
- `projection_linear_probe.py`: Linear probe to measure whether the ProjectionMLP preserves enough visual information for condition discrimination.
- `rag/`: RAG evaluation — hit-rate metrics, RAGAS evaluation, visualization, and dataset generation.

#### `benchmark.py` & `benchmark_outputs/`
Comparative benchmark of the skincare model against LLaVA-1.5-7B and LLaMA-3.2-3B-Instruct (base) on visual diagnosis using Dataset D validation split.

### Data & Model Artifacts

- `data/`: Persisted ChromaDB vector store and indexed documents.
- `model/`: LoRA adapters and model artifacts.

## Directory Structure

```text
skincare_model/
├── datasets/
│   ├── dataset_a/
│   ├── dataset_b/
│   ├── dataset_c/
│   ├── dataset_d/
│   ├── images_dataset/
│   ├── medical_notes_dataset/
│   └── densenet_predictions.json
├── densenet_model/
│   ├── outputs/
│   └── src/
├── medical_notes_generation/
│   └── scripts/
├── prepare_datasets/
│   ├── dataset_b/
│   ├── dataset_c/
│   ├── dataset_d/
│   └── prepare_dataset_a.py
├── visual_alignment/
│   ├── projection.py
│   └── train.py
├── unified_training/
│   ├── train.py
│   ├── merge_model.py
│   └── evaluate.py
├── langgraph_orchestrator/
│   ├── build_graph.py
│   ├── nodes/
│   ├── services/
│   ├── routing.py
│   ├── state.py
│   └── web/
│       ├── app.py
│       ├── api/
│       ├── templates/
│       └── static/
├── rag/
│   ├── vector_store.py
│   ├── query_processor.py
│   ├── retriever.py
│   ├── answer_generator.py
│   ├── prompt_factory.py
│   └── model_loader.py
├── evaluation_metrics/
│   ├── projection_linear_probe.py
│   └── rag/
├── benchmark.py
├── benchmark_outputs/
├── scripts/
│   ├── rag_index_pubmed.py
│   ├── rag_index_medlineplus.py
│   └── rag_index_local.py
├── data/
├── model/
├── inference_example.py
├── multitask_training_plan.md
├── README.md
└── requirements.txt
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
- `rag/README.md`: RAG module architecture and component documentation.

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

For the LangGraph orchestrator (web UI):
```bash
pip install -r langgraph_orchestrator/requirements_langgraph.txt
```

For RAG-specific dependencies:
```bash
pip install -r rag/requirements.txt
```

### Training
Refer to `multitask_training_plan.md` for the full step-by-step execution guide covering all training phases.

### Running the Web Application
Launch the LangGraph-powered web UI for interactive consultations:
```bash
python -m langgraph_orchestrator.web.app
```

### RAG Indexing
Populate the vector store before using RAG-based follow-up QA:
```bash
python scripts/rag_index_pubmed.py
python scripts/rag_index_medlineplus.py
python scripts/rag_index_local.py
```

## Using the Trained Model

The final merged model is available on Hugging Face: [santiagoboxiga/skincare-llama-3.2-3b](https://huggingface.co/santiagoboxiga/skincare-llama-3.2-3b)

### Interactive CLI Chat
Run the included inference script for a multi-turn conversation:
```bash
pip install transformers torch accelerate torchvision pillow
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

The system supports the full consultation pipeline: symptom collection → image upload → diagnosis → follow-up QA.

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
| Follow-up QA | User message contains a `[CONTEXT]` block | Answers using only the provided context (RAG-grounded) |
| Visual Diagnosis | Skin image provided after symptom collection | Reasons with `<think>`, wraps output in `[DIAGNOSIS]...[/DIAGNOSIS]` |

### Visual Pipeline

For visual diagnosis, the system requires two additional components beyond the merged LLM:

| Component | File | Purpose |
|---|---|---|
| DenseNet-121 | `densenet_model/outputs/model.pth` | Extracts 1024-dim features from skin images |
| ProjectionMLP | `projection.pth` (in merged model dir) | Maps DenseNet features → 32 visual tokens in LLM embedding space |

Both are loaded automatically at startup if present. Without them, the system runs in text-only mode (symptom collection and QA only).

> **Note:** The system prompt is required for correct model behavior. The model will not produce `[END_SYMPTOMS]`, `<think>` blocks, or disclaimers without it. The prompt is baked into `inference_example.py` and the LangGraph orchestrator.
