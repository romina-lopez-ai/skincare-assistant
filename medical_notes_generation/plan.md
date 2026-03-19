# Medical Notes Generation — Project Plan

## Goal
Fine-tune a vision-language model that takes a dermatology image as input and outputs a
structured clinical note, covering morphology, lesion characteristics, differentials, and
clinical recommendations.

---

## Dataset

```
images/
  train/       → 640 images (acne: 214, eczema: 213, psoriasis: 213)
  test/        → 138 images (acne: 46, eczema: 46, psoriasis: 46)
  validation/  → 137 images (acne: 45, eczema: 46, psoriasis: 46)
```

Labels are encoded in folder names — no manual annotation needed.

---

## Annotation Model: GPT-4o Batch API

**Why not LLaMA 3.2 Vision:**
- Quality gap in medical descriptions is significant
- 915 images via GPT-4o Batch costs ~$1.60 total — negligible
- No GPU setup required for annotation phase

---

## Step-by-Step Plan

### Step 1 — Environment Setup
- [ ] Create virtual environment (`uv` or `venv`)
- [ ] Install dependencies: `openai`, `Pillow`, `datasets`, `tqdm`
- [ ] Set `OPENAI_API_KEY` in `.env`

### Step 2 — Build Batch Request File (`prepare_batch.py`)
- [ ] Walk `images/train/` folder recursively
- [ ] For each image:
  - Read label from parent folder name
  - Encode image to base64
  - Build GPT-4o chat request with diagnosis-guided prompt (see below)
- [ ] Write all requests to `batch_requests.jsonl`

**Prompt template (disease-specific, injected at runtime):**

```
System:
You are a senior dermatologist. Write structured, clinically accurate notes
based on visual examination. Be specific about what you observe in the image.

User:
You are examining a confirmed case of [DISEASE NAME].

Describe this image in a clinical note format including:

1. MORPHOLOGY — describe asymmetry, borders, color variation, distribution,
   elevation, and surface texture visible in the image.
2. LESION CHARACTERISTICS — detail features specific to [DISEASE NAME]
   (e.g., comedones for acne, lichenification for eczema, silvery scales
   for psoriasis).
3. SYMPTOMS — list the typical symptoms this patient is likely experiencing
   based on the visual presentation (e.g., pruritus, pain, burning,
   tenderness).
4. DIFFERENTIAL DIAGNOSIS — briefly note 1–2 conditions to rule out and why.
5. PATIENT ADVICE — provide exactly 3 bullet points of general,
   non-prescription advice (skincare habits, lifestyle, triggers to avoid).
   Do NOT recommend any medication, drug, or treatment by name.
   Always end with: "If symptoms do not improve with general care,
   please consult a dermatologist or specialist."
```

**Disease-specific context injected per label:**

| Label | Key morphology cues | Typical symptoms |
|---|---|---|
| `acne` | comedones, papules, pustules, nodules; facial/truncal distribution | pain, tenderness, oiliness, scarring concern |
| `eczema` | erythema, lichenification, excoriations, weeping/crusting; flexural areas | intense pruritus, burning, dry/tight skin, sleep disruption |
| `psoriasis` | well-demarcated plaques, silvery-white scales, erythematous base; extensor surfaces | itching, flaking, joint pain (if psoriatic arthritis), nail changes |

> Note: injecting the known label + disease context produces richer, more
> consistent notes than asking the model to diagnose blindly. The symptom
> section bridges visual findings with patient experience, making the
> fine-tuned model more clinically useful.

### Step 3 — Submit Batch Job (`submit_batch.py`)
- [ ] Upload `batch_requests.jsonl` to OpenAI Files API
- [ ] Create batch job with `endpoint: /v1/chat/completions`
- [ ] Save `batch_id` to `batch_id.txt`
- [ ] Poll for completion (batches complete within 24h, usually faster)

### Step 4 — Parse Results (`parse_results.py`)
- [ ] Download completed batch output file
- [ ] Match each result to its `custom_id` (image path + label)
- [ ] Save structured dataset to `dataset/train.jsonl`:

```jsonl
{"image_path": "images/train/acne/img_001.jpg", "label": "acne", "note": "..."}
{"image_path": "images/train/eczema/img_002.jpg", "label": "eczema", "note": "..."}
```

- [ ] Repeat annotation for `validation/` split if needed
- [ ] Keep `test/` split unannotated — used only for final evaluation

### Step 5 — Fine-Tuning
- [ ] Choose base model: **LLaVA-1.6** or **Qwen2-VL-7B** (both open-source vision-language)
- [ ] Use **Unsloth + LoRA** for efficient fine-tuning (fits on 1x RTX 3090 / Colab A100)
- [ ] Training config:
  - Epochs: 3–5
  - LoRA rank: 16
  - Batch size: 4 with gradient accumulation
  - Checkpoint on `validation/` loss
- [ ] Save adapter weights

### Step 6 — Evaluation
- [ ] Run inference on `test/` split (138 images, no notes seen during training)
- [ ] Metrics:
  - **ROUGE-L** — text similarity to GPT-4o reference notes
  - **BERTScore** — semantic similarity
  - **Manual review** — sample 20 outputs and score clinical accuracy 1–5

### Step 7 — Inference Script (`generate_note.py`)
- [ ] Load fine-tuned model + adapter
- [ ] Accept image path as input
- [ ] Output structured clinical note

---

## Cost Summary

| Phase | Tool | Estimated Cost |
|---|---|---|
| Annotation (train 640 images) | GPT-4o Batch API | ~$1.40 |
| Annotation (validation 137 images) | GPT-4o Batch API | ~$0.30 |
| Fine-tuning | Colab A100 / local GPU | ~$5–15 (Colab) or free (local) |
| **Total** | | **< $20** |

---

## File Structure (end state)

```
medical_notes_generation/
  images/                    # original images (do not modify)
  dataset/
    train.jsonl              # 640 annotated pairs
    validation.jsonl         # 137 annotated pairs
  scripts/
    prepare_batch.py
    submit_batch.py
    parse_results.py
    generate_note.py
  batch_requests.jsonl       # intermediate — GPT-4o batch input
  batch_id.txt               # intermediate — for polling
  plan.md                    # this file
  .env                       # OPENAI_API_KEY (never commit)
  requirements.txt
```

---

## Safeguards

Two-layer approach to prevent the model from giving medical/medication advice.

### Layer 1 — GPT-4o Annotation Prompt (training data)
Enforced in the system prompt during synthetic data generation so the training
data itself never contains medication references:

```
You are a senior dermatologist writing clinical notes for a patient-facing
assistant. You must follow these rules strictly:
- Do NOT recommend any medication, drug, cream, or prescription by name.
- Do NOT suggest dosages or treatment regimens.
- General advice only: hygiene, lifestyle, triggers, skincare habits.
- Always close the advice section with:
  "If your symptoms do not improve with general care, please consult
   a dermatologist or specialist."
```

### Layer 2 — Inference System Prompt (deployed model)
Applied at runtime regardless of what the model learned during fine-tuning:

```
You are a dermatology assistant. Your role is to describe skin conditions
and provide general skincare guidance only.

Rules:
- Never recommend medications, prescriptions, or treatments by name.
- Never suggest dosages or medical procedures.
- If the condition appears severe or does not improve, always recommend
  consulting a licensed dermatologist or healthcare professional.
- You are not a substitute for professional medical advice.
```

> Both layers are necessary. Fine-tuning alone does not fully prevent a model
> from surfacing medication knowledge it acquired during pre-training.
> The inference system prompt acts as a runtime guardrail.

---

## Open Questions
- [ ] Fine-tune target model: LLaVA-1.6 vs Qwen2-VL vs LLaMA 3.2 Vision?
- [ ] Also annotate `test/` with GPT-4o to enable ROUGE/BERTScore eval?
- [ ] Deploy as API (FastAPI) or keep as local inference script?
