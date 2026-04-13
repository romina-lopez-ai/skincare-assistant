# Real Model Integration Guide

This guide explains how to replace the mock service backends with your real trained models.
The graph structure, routing logic, node files, and state schema do **not** need to change.

---

## 1. Prerequisites — what must exist before starting

| Artifact | Default path | Where it comes from |
|---|---|---|
| Fine-tuned LLM | `unified_training/outputs/merged_model/` | or HuggingFace `santiagoboxiga/skincare-llama-3.2-3b` |
| DenseNet weights | `densenet_model/outputs/model.pth` | your training pipeline |
| Projection weights | `unified_training/outputs/merged_model/projection.pth` | or `unified_training/outputs/projection.pth` |

Install dependencies if not already present:

```bash
pip install transformers torch accelerate torchvision pillow
```

---

## 2. Turn on real mode — one file, three lines

Open `langgraph_orchestrator/config.py` and change the three flags:

```python
# config.py
USE_MOCK_LLM    = False   # was True
USE_MOCK_VISION = False   # was True
USE_MOCK_RAG    = True    # leave True until your vector store is ready
```

You can also override at launch without editing the file:

```bash
USE_MOCK_LLM=0 USE_MOCK_VISION=0 uvicorn langgraph_orchestrator.web.app:app --reload
```

Each flag is independent. If only the vision model is ready, set `USE_MOCK_VISION = False`
and leave the other two as `True`.

---

## 3. `vision_service.py` — two functions to fill in

**File:** `langgraph_orchestrator/services/vision_service.py`

The two real stubs are `_real_run_vision_inference` and `_real_project_vision_features`.

### 3a. Copy visual components from `inference_example.py`

The `DenseNetFeatureExtractor` and `ProjectionMLP` classes already exist verbatim in
`inference_example.py` (lines 60–91). Copy them to the top of `vision_service.py`
(above the public interface section) and add a model loader block:

```python
# ── paste DenseNetFeatureExtractor and ProjectionMLP from inference_example.py ──

import torch
import torch.nn.functional as F
from torchvision import transforms

_DENSENET_PATH   = "densenet_model/outputs/model.pth"
_PROJECTION_PATH = "unified_training/outputs/merged_model/projection.pth"
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

_IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Loaded once on first use
_feat_extractor = None
_projection     = None

def _load_vision_models():
    global _feat_extractor, _projection
    if _feat_extractor is None:
        _feat_extractor = DenseNetFeatureExtractor(_DENSENET_PATH).eval().to(_DEVICE)
    if _projection is None:
        _projection = ProjectionMLP(densenet_dim=1024, llm_dim=3072,
                                    num_tokens=32, hidden_dim=2048)
        _projection.load_state_dict(
            torch.load(_PROJECTION_PATH, map_location="cpu")
        )
        _projection.eval().to(_DEVICE).to(_DTYPE)

# Class index → label string (must match your DenseNet training order)
_LABEL_MAP = {0: "eczema", 1: "psoriasis", 2: "acne"}   # adjust to your classes
```

### 3b. Fill in `_real_run_vision_inference`

```python
def _real_run_vision_inference(image_path: str) -> dict:
    from PIL import Image
    _load_vision_models()

    img = Image.open(image_path).convert("RGB")
    img_tensor = _IMAGE_TRANSFORM(img).unsqueeze(0).to(_DEVICE)

    with torch.no_grad():
        features   = _feat_extractor(img_tensor)          # (1, 1024)
        probs      = F.softmax(features, dim=-1)[0]       # adjust if you have a separate head
        top_idx    = int(probs.argmax())
        confidence = float(probs[top_idx])
        label      = _LABEL_MAP.get(top_idx, f"class_{top_idx}")

    return {
        "label":           label,
        "confidence":      confidence,
        "morphology":      "",           # fill from model metadata if available
        "features_tensor": features,    # raw DenseNet tensor — used by projection
    }
```

### 3c. Fill in `_real_project_vision_features`

```python
def _real_project_vision_features(densenet_output: dict) -> dict:
    _load_vision_models()

    features = densenet_output["features_tensor"].to(_DEVICE).to(_DTYPE)

    with torch.no_grad():
        visual_tokens = _projection(features)   # (1, 32, 3072)

    return {
        **densenet_output,
        "projected_embedding": visual_tokens,  # torch.Tensor, shape (1, 32, llm_dim)
    }
```

> **Note:** `projected_embedding` changes type from `str` (mock) to `torch.Tensor` (real).
> The downstream `_real_generate_diagnosis_response` reads it as a tensor — this is
> intentional and mirrors the pattern in `inference_example.py` lines 183–186.

---

## 4. `llm_service.py` — three functions to fill in

**File:** `langgraph_orchestrator/services/llm_service.py`

### 4a. Copy generation helpers from `inference_example.py`

Copy the following functions verbatim from `inference_example.py` into `llm_service.py`.
They are self-contained and have no external dependencies beyond model/tokenizer:

| Function | Lines in `inference_example.py` |
|---|---|
| `build_prompt(history)` | 97–105 |
| `_chat_ids(messages, tokenizer, prime_assistant)` | 129–149 |
| `generate_text(model, tokenizer, history, max_new_tokens, device)` | 112–126 |
| `generate_with_image(model, tokenizer, history, image_path, ...)` | 153–203 |
| `_extract_first_turn(text)` | 206–219 |

### 4b. Add a model loader block

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_MODEL_PATH = "unified_training/outputs/merged_model"
# Alternative: use the HuggingFace repo directly
# _MODEL_PATH = "santiagoboxiga/skincare-llama-3.2-3b"

_MAX_TOKENS = 256
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DTYPE  = torch.bfloat16 if torch.cuda.is_available() else torch.float32

_model     = None
_tokenizer = None

def _load_llm():
    global _model, _tokenizer
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(_MODEL_PATH)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(
            _MODEL_PATH, torch_dtype=_DTYPE,
            device_map="auto" if _DEVICE.type == "cuda" else None,
        )
        if _DEVICE.type != "cuda":
            _model = _model.to(_DEVICE)
        _model.eval()
```

### 4c. Fill in `_real_generate_symptom_response`

```python
def _real_generate_symptom_response(state: dict) -> str:
    _load_llm()
    # generate_text (copied from inference_example.py) handles prompt building
    # and returns the first complete assistant turn
    return generate_text(_model, _tokenizer, state["messages"], _MAX_TOKENS, _DEVICE)
```

### 4d. Fill in `_real_generate_diagnosis_response`

```python
def _real_generate_diagnosis_response(state: dict) -> str:
    _load_llm()
    # generate_with_image (copied from inference_example.py) injects visual tokens
    # at the correct position in the combined embeddings
    return generate_with_image(
        _model, _tokenizer,
        state["messages"],
        state["image_path"],
        _feat_extractor,    # imported from vision_service, or re-loaded here
        _projection,
        _MAX_TOKENS,
        _DEVICE,
    )
```

> **Note on visual token sharing:**
> `generate_with_image` runs DenseNet + ProjectionMLP internally, which duplicates
> the work already done by `run_vision` and `project_image_features` nodes.
>
> - **Option A (recommended to start):** Let `generate_with_image` re-run the visual
>   pipeline using `_feat_extractor` and `_projection` from `vision_service`. No other
>   changes needed.
> - **Option B (no duplication):** Refactor `generate_with_image` to accept a
>   pre-computed `visual_tokens` tensor from
>   `state["densenet_output"]["projected_embedding"]` instead of `feat_extractor`
>   and `projection`. Replace lines 181–183 of `inference_example.py` with
>   `vis_tokens = visual_tokens`.

### 4e. Fill in `_real_generate_followup_response`

```python
def _real_generate_followup_response(state: dict) -> str:
    _load_llm()

    messages = list(state["messages"])
    rag_ctx  = state.get("rag_context")

    # If RAG context was retrieved, inject it as a system note before generating
    if rag_ctx:
        messages = messages + [{
            "role":    "system",
            "content": f"Use only the following context to answer:\n{rag_ctx}",
        }]

    return generate_text(_model, _tokenizer, messages, _MAX_TOKENS, _DEVICE)
```

---

## 5. `rag_service.py` — one function (independent of LLM/vision)

**File:** `langgraph_orchestrator/services/rag_service.py`

The real RAG backend is independent of the model weights. Fill in when your
vector store is ready:

```python
def _real_retrieve_medical_context(query: str) -> str:
    # 1. Encode the query with your embedding model
    # 2. Search FAISS / Chroma / Pinecone for top-k chunks
    # 3. Format the results as a [CONTEXT]...[/CONTEXT] block
    #    matching the structure the mock returns
    # 4. Return an empty string if nothing relevant is found
    raise NotImplementedError("Fill in your vector-store retrieval here")
```

Until this is implemented, leave `USE_MOCK_RAG = True` in `config.py`.

---

## 6. What must NOT change

These files are model-agnostic by design. Do not edit them during integration:

| File | Reason |
|---|---|
| `build_graph.py` | Graph topology and edge wiring |
| `routing.py` | All routing functions |
| `nodes/*.py` | All node orchestration logic |
| `state.py` | State schema |
| `utils/response_parser.py` | Output cleaning pipeline — strips `<think>`, `[DIAGNOSIS]` tags etc. |
| `web/` | FastAPI layer and frontend |

---

## 7. Integration checklist

```
[ ] Set USE_MOCK_LLM = False, USE_MOCK_VISION = False in config.py
[ ] Confirm model weights exist at the configured paths
[ ] Copy DenseNetFeatureExtractor, ProjectionMLP from inference_example.py
    into vision_service.py
[ ] Copy build_prompt, _chat_ids, generate_text, generate_with_image,
    _extract_first_turn from inference_example.py into llm_service.py
[ ] Add model loader blocks (_load_vision_models, _load_llm) to each service file
[ ] Fill in _real_run_vision_inference        (vision_service.py)
[ ] Fill in _real_project_vision_features     (vision_service.py)
[ ] Fill in _real_generate_symptom_response   (llm_service.py)
[ ] Fill in _real_generate_diagnosis_response (llm_service.py)
[ ] Fill in _real_generate_followup_response  (llm_service.py)
[ ] Start the server and run the test suite:
        pytest langgraph_orchestrator/tests/ -v
    All 15 scenario tests should still pass — they test orchestration, not model output
[ ] Do a manual end-to-end test in the browser UI
[ ] (Later) Fill in _real_retrieve_medical_context and set USE_MOCK_RAG = False
```

---

## 8. Confidence threshold reference

The two thresholds in `vision_service.py` control diagnosis behaviour.
Adjust them after evaluating your model's calibration:

| Constant | Default | Effect |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.60` | Below this → hedged diagnosis language |
| `CONFIDENCE_FALLBACK_THRESHOLD` | `0.40` | Below this → re-upload prompt, no diagnosis |

```python
# vision_service.py — adjust these to match your model's confidence distribution
CONFIDENCE_THRESHOLD          = 0.60
CONFIDENCE_FALLBACK_THRESHOLD = 0.40
```

---

## 9. Testing each tier manually

After integrating the real models, verify the three confidence tiers by temporarily
overriding the return value of `_real_run_vision_inference` to return a fixed
confidence, or by using the mock with the following values:

| Set `confidence` to | Expected behaviour |
|---|---|
| `0.82` | Tier 1 — confident diagnosis, direct language |
| `0.50` | Tier 2 — cautious diagnosis, hedged language |
| `0.25` | Tier 3 — re-upload prompt, `image_uploaded` reset to `False` |
