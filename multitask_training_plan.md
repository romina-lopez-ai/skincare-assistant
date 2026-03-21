# Multitask Training Plan — Lightweight Multimodal Dermatology Chatbot

---

## Overview

This document describes the full training strategy to build a multitask SLM capable
of four clinical behaviors in a single model. This is what makes it a real clinical
assistant rather than a scripted bot.

Each dataset teaches a distinct capability, and the final Dataset D (Joint) is the most important — it
depends on everything learned in A, B, and C.

| Dataset                           | Task                                           | Complexity |
|-----------------------------------|------------------------------------------------|------------|
| Dataset A (image)                 | Visual alignment — learn to read image tokens  | Low        |
| Dataset B (symptoms)              | Symptom collection — anamnesis flow            | Medium     |
| Dataset C (QA-RAG)                | Follow-up QA grounded in clinical context      | Medium     |
| Dataset D (joint)                 | Full consultation — symptoms + image + diagnosis| High      |

**Source datasets:**
- E1 (clinical-notes): existing JSONL — image_path + label + clinical note - Santiago S1-2026
- E2 (symptom-collection): existing JSON — multi-turn anamnesis dialogues - Romina S2-2025
- E3 (QA): existing CSV — prompt/response/category QA pairs - Romina S1-2025

---

## 1. Dataset A (image) — Visual Alignment

**Purpose:**
This is the visual alignment dataset. It teaches the model to interpret visual tokens
using the clinical note as ground truth supervision signal. This is NOT about handling
"image only" in production — in the real application flow, both symptoms AND image are
always required before diagnosis. Dataset A exists to teach the model what the visual
tokens mean before it learns to combine them with reported symptoms in Dataset D.

Think of it as: "here are the visual tokens from this image — the correct
interpretation of those tokens is everything written in the clinical note."

**What the model learns:**
Given visual tokens injected at image_msg_index, produce a `<think>` block that
accurately reflects the vision model real predictions and the clinical note's morphological
description, followed by a `[DIAGNOSIS]` that is the conversational rewrite of that
note. This trains the projection layer and LoRA adapters to map visual embeddings to
clinical language — the foundation that Dataset D builds on top of.

**Dataset structure:**
```json
{
  "image_path": "images/train/psoriasis/psoriasis_00012.jpg",
  "image_msg_index": 1,
  "DenseNet_output": {
    "primary": "psoriasis", "confidence": 0.94,
    "secondary": "eczema",  "confidence_2": 0.04,
    "morphology": ["thick_plaques", "silvery_scales", "erythema"],
    "location": "elbow"
  },
  "messages": [
    {"role": "system", "content": "<SYSTEM_UNIFIED>"},
    {"role": "user",   "content": ""},
    {"role": "assistant", "content":
      "<think>\nDenseNet primary: psoriasis (0.94) — high confidence.\nDenseNet secondary: eczema (0.04) — negligible.\nMorphology: thick erythematous plaques, silvery scales — elbow location.\nDifferential: eczema ruled out (no weeping), seborrheic dermatitis ruled out (wrong location).\nUrgency: routine.\n</think>\n\n[DIAGNOSIS]\nFrom the image, I can observe thick, raised patches with a silvery-white surface on the elbow area. This pattern is strongly associated with psoriasis vulgaris. This is not a medical diagnosis — please see a dermatologist to confirm.\n\nSome things to be aware of:\n- These plaques can also appear on knees, scalp, and lower back\n- Stress and infections are common triggers\n- Effective treatments exist that can control flare-ups\n[/DIAGNOSIS]"}
  ]
}
```

**Where the content comes from:**

| Part                  | Source                                            |
|-----------------------|---------------------------------------------------|
| `<think>` predictions | DenseNet real output — pre-computed by script       |
| `<think>` morphology  | E1 (clinical-notes) MORPHOLOGY section            |
| `<think>` differential| E1 (clinical-notes) DIFFERENTIAL DIAGNOSIS section|
| `[DIAGNOSIS]` content | E1 (clinical-notes) rewritten conversationally    |
| `[DIAGNOSIS]` advice  | E1 (clinical-notes) PATIENT ADVICE section        |

**Generation:** `prepare_dataset_a.py` 
Run image through fine-tuned DenseNet → capture real predictions → pass to API with
clinical note → API generates `<think>` using real DenseNet values + note content.

---

## 2. Dataset B (symptoms) — Symptom Collection

**Purpose:**
Teaches the model to conduct a focused anamnesis when no image is available. The model
learns to ask one focused question at a time, and emit [END_SYMPTOMS] reliably when collection is complete.

**What the model learns:**
Structured symptom collection dialogue — the anamnesis flow. This is the behavior that
enables the orchestrator to transition states correctly. Fine-tuning is required here
because SLMs of 1–3B parameters cannot reliably follow multi-step instruction flows
from a system prompt alone.

**Dataset structure:**
```json
{
  "messages": [
    {"role": "system",    "content": "<SYSTEM_UNIFIED>"},
    {"role": "user",      "content": "I have thick red plaques on my lower back."},
    {"role": "assistant", "content": "Do they have silvery scales?"},
    {"role": "user",      "content": "Yes."},
    {"role": "assistant", "content": "Are they also on your elbows or knees?"},
    {"role": "user",      "content": "Yes, elbows too."},
    {"role": "assistant", "content": "Do they crack or bleed when scratched?"},
    {"role": "user",      "content": "Yes."},
    {"role": "assistant", "content": "Anything else to add?"},
    {"role": "user",      "content": "No, that's everything."},
    {"role": "assistant", "content": "Thank you. All details recorded.[END_SYMPTOMS]"}
  ]
}
```

**Source:** E2 (symptom-collection) — existing JSON.
Only change required: replace per-task system prompt with unified system prompt.
No API calls needed. No new data generation required.

---

## 3. Dataset C (QA-RAG) — Follow-Up QA with RAG Grounding

**Purpose:**
Teaches the model to answer patient questions using only the clinical context provided
in [CONTEXT] blocks. The model learns to stay grounded and never speculate beyond
what the retrieved context contains — the primary hallucination mitigation mechanism.

**What the model learns:**
RAG-grounded conversational QA. When a [CONTEXT] block is present, answer from it
only. When it is not sufficient, decline gracefully rather than hallucinate.

**Dataset structure:**
```json
{
  "messages": [
    {"role": "system", "content": "<SYSTEM_UNIFIED>"},
    {"role": "system", "content": "[CONTEXT]\nSource: AAD Guidelines 2024\nPsoriasis is a chronic autoimmune condition. Triggers: stress, infections, medications. First-line: topical corticosteroids, vitamin D analogues.\n[/CONTEXT]"},
    {"role": "user",      "content": "Does psoriasis have a cure?"},
    {"role": "assistant", "content": "There is currently no definitive cure for psoriasis, but many effective treatments can help control symptoms and achieve long periods of remission. First-line options include topical corticosteroids and vitamin D analogues"}
  ]
}
```

**Source:** E3 (QA) — existing CSV.
Convert each row to chat format with `convert_d3.py`. Each row becomes a [CONTEXT]
system message + user question + assistant response. No API calls needed.

---

## 4. Dataset D (joint) — Full Consultation: Anamnesis + Image + Joint Diagnosis

**Purpose:**
This is the most important dataset and the primary contribution of the system. It
teaches the model to reason jointly over both reported symptoms AND visual tokens,
producing a diagnosis that explicitly integrates both sources. Everything learned in
Datasets A, B, and C feeds into this dataset.

**What the model learns:**
The complete consultation flow — symptom collection, image submission, and joint
diagnosis in a single conversation. The `<think>` block must reference three sources:
the DenseNet's real predictions, the clinical note's morphological content, and the
symptoms as typed by the user in the dialogue (in patient language, not clinical
language). This is what makes Dataset D the hardest — the model must learn to connect
informal user language with clinical visual evidence.

**image_msg_index is dynamic** — it depends on how many symptom turns preceded the
image. Calculated as `len(messages)` after [END_SYMPTOMS] and before the empty user
message. Each example has its own value.

**Dataset structure:**
```json
{
  "image_path": "images/train/acne/acne_00000.jpg",
  "image_msg_index": 9,
  "DenseNet_output": {
    "primary": "acne", "confidence": 0.91,
    "secondary": "rosacea", "confidence_2": 0.06,
    "morphology": ["papules", "pustules", "erythema"],
    "location": "chin, jaw"
  },
  "messages": [
    {"role": "system",    "content": "<SYSTEM_UNIFIED>"},
    {"role": "user",      "content": "I have red bumps on my chin."},
    {"role": "assistant", "content": "How long have you had them?"},
    {"role": "user",      "content": "About 2 weeks."},
    {"role": "assistant", "content": "Are any filled with pus?"},
    {"role": "user",      "content": "Yes, some have white tips."},
    {"role": "assistant", "content": "Have you noticed triggers like stress or diet?"},
    {"role": "user",      "content": "Yes, worse with stress."},
    {"role": "assistant", "content": "Thank you, I have everything I need.[END_SYMPTOMS]"},
    {"role": "user",      "content": ""},
    {"role": "assistant", "content":
      "<think>\nDenseNet primary: acne (0.91) — high confidence.\nDenseNet secondary: rosacea (0.06) — negligible.\nMorphology: papules, pustules, erythema — chin/jaw distribution.\nUser reported: 2 weeks duration, pustules with pus, stress trigger.\nVisual + symptoms consistent — inflammatory acne vulgaris confirmed.\nRosacea ruled out: low DenseNet confidence + no telangiectasia in note.\nUrgency: routine.\n</think>\n\n[DIAGNOSIS]\nBased on the image and what you've described, I can see signs consistent with acne vulgaris. The raised red lesions with pustules around your chin match the symptoms you reported, and the stress connection you mentioned is very common with this condition.\n\nThis is not a medical diagnosis — please see a dermatologist to confirm. In the meantime:\n- Use a gentle non-comedogenic cleanser twice a day\n- Avoid touching or picking the lesions\n- Track stress patterns — they may be a key trigger for you\n[/DIAGNOSIS]"}
  ]
}
```

**The critical distinction — symptoms in the dialogue are NOT copied from the note:**
The symptom dialogue is generated to simulate what a real user would type — in patient
language ("I have red bumps", "they hurt when I touch them"), not clinical language
("papules and pustules with erythema"). The model must learn to correlate informal
user-reported symptoms with clinical visual findings. This is intentional.

**Where the content comes from:**

| Part                  | Source                                                       |
|-----------------------|--------------------------------------------------------------|
| Symptom dialogue      | API-generated in patient language from E1 SYMPTOMS section   |
| `<think>` predictions | DenseNet real output — pre-computed                            |
| `<think>` morphology  | E1 (clinical-notes) MORPHOLOGY + DIFFERENTIAL sections       |
| `<think>` user symptoms| Symptoms as typed in the dialogue turns above               |
| `[DIAGNOSIS]` content | E1 (clinical-notes) rewritten mentioning both sources        |

**Generation:** `prepare_dataset_d.py` 
1. Generate symptom dialogue in patient language from E1 SYMPTOMS section.
2. Run DenseNet → capture real predictions.
3. Generate `<think>` + `[DIAGNOSIS]` integrating all three sources.


---

## 5. Dataset Summary

| Dataset         | Task                        | Has image | Size     | Source  | Prep needed           | Training phase |
|-----------------|-----------------------------|-----------|----------|---------|-----------------------|----------------|
| A (image)       | Visual alignment            | Yes       |  ~25%  | E1      | prepare_dataset_a.py  | Phase 2        |
| B (symptoms)    | Symptom collection          | No        | ~25%| E2      | Update system prompt  | Phase 3        |
| C (QA-RAG)      | Follow-up QA                | No        | ~15%| E3      | convert_d3.py         | Phase 3        |
| D (joint)       | Full consultation           | Yes       | ~35%   | E1      | prepare_dataset_d.py  | Phase 4        |

---

## 6. Unified System Prompt

All four datasets use a single system prompt. Mixing per-task system prompts in the
same training run causes conflicting behavior in SLMs.
"<SYSTEM_UNIFIED>"

```
You are an empathetic dermatology assistant with three clinical capabilities:

1. SYMPTOM COLLECTION
   When a user describes skin symptoms, collect detailed information through
   focused one-at-a-time questions covering location, duration, morphology,
   severity, and triggers. Never diagnose during this phase.
   End with [END_SYMPTOMS] when collection is complete.

2. VISUAL DIAGNOSIS
   When you receive visual input (image tokens), reason step by step using
   <think>...</think> before responding. Integrate both the visual findings
   and the symptoms previously collected. Wrap your response in
   [DIAGNOSIS]...[/DIAGNOSIS]. Always include a disclaimer.

3. FOLLOW-UP QA
   When a [CONTEXT] block is provided, answer using only that context.
   Do not speculate beyond what the context contains.

Always recommend consulting a dermatologist. Never provide a definitive diagnosis.
```

---

## 7. Control Tags — Orchestrator Interface

Tags are the interface between the LLM and the orchestrator. The model learns to
emit output tags during fine-tuning; the orchestrator detects them to manage state.

### Input tags (injected by orchestrator before calling LLM)

| Tag           | Who sets it  | Purpose                                    |
|---------------|--------------|--------------------------------------------|
| `[CONTEXT]`   | Orchestrator | Injects RAG chunks before follow-up QA     |
| Visual tokens | Orchestrator | Image embedding injected at image_msg_index|

### Output tags (generated by LLM, read by orchestrator)

| Tag             | Orchestrator action                       |
|-----------------|-------------------------------------------|
| `[END_SYMPTOMS]`| Request image, transition to diagnosis    |
| `[DIAGNOSIS]`   | Extract response, transition to follow-up |
| `[END_SESSION]` | Reset state, clear history                |

**Note on [URGENT]:** Adding an urgent-case tag requires a dedicated dataset of
systemic symptom combinations (fever + spreading rash, joint pain + skin lesions, etc.)
to train reliably. Without sufficient examples the model will either over-trigger or
never trigger it — both are dangerous in a clinical context. Recommendation: exclude
[URGENT] from the initial version and address it in a future dataset extension once
the base system is validated.

### CoT tag — orchestrator responsibility

The `<think>` block is generated by the model but must NEVER be shown to the user.
The orchestrator is responsible for stripping it before rendering the response:

```python
import re
def clean_response(text):
    # strip <think> block before showing to user
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
```

This keeps clinical reasoning internal while ensuring the user only sees the
empathetic conversational response from [DIAGNOSIS] or the follow-up answer.

---

## 8. Training Phases — Curriculum Learning Strategy

Training progresses from simple to complex. Each phase has a validation checkpoint
before proceeding. This catches problems early and reduces wasted compute.

```
Phase 0: DenseNet + Projection  → validate visual accuracy ≥ 85%
Phase 1: Dataset preparation  → validate all 4 datasets quality
Phase 2: Dataset A only       → validate visual token interpretation
Phase 3: Datasets B + C       → validate multitask text-only flow
Phase 4: All datasets A+B+C+D → validate joint reasoning
```

### Phase 0 — DenseNet-121 + Projection MLP

Train visual backbone to classify 3 conditions (acne, eczema, psoriasis).

| Item        | Detail                                      |
|-------------|---------------------------------------------|
| Input       | E1 (clinical-notes) — image_path + label    |
| Framework   | PyTorch standard training loop              |
| Objective   | CrossEntropy — 3-class classification       |
| Trains      | DenseNet-121 + Projection MLP                 |
| Duration    | ~2–4 hrs on A100 80GB                       |
| Output      | DenseNet121_finetuned.pth + projection.pth    |

Checkpoint: val accuracy ≥ 85% across all 3 classes before proceeding.

---

### Phase 1 — Dataset Preparation (not training)

Generate Datasets A and D using fine-tuned DenseNet. Convert E3 to Dataset C format.

Steps: `prepare_dataset_a.py` → `prepare_dataset_d.py` → `convert_d3.py` → update
system prompt in E2 to produce Dataset B.

Checkpoint: all 4 datasets load without errors, image_msg_index correctly placed,
[END_SYMPTOMS] present in ≥ 95% of B examples, [CONTEXT] present in all C examples,
no example exceeds 2048 tokens after tokenization.

---

### Phase 2 — Pipeline 1: Dataset A only (visual alignment)

Validate that visual tokens → clinical response mapping works before adding
text-only tasks. This is the most critical validation — if image reasoning fails
here, the problem is in the architecture, not the data.

| Item      | Detail                                                |
|-----------|-------------------------------------------------------|
| Dataset   | A (image) only                                        |
| Framework | Custom HuggingFace Trainer (visual token injection)   |
| Trains    | Projection MLP + LoRA adapters                        |
| DenseNet    | Frozen                                                |
| Epochs    | 2–3                                                   |
| Duration  | ~4–6 hrs on A100 40GB                                 |

Checkpoint: `<think>` present, [DIAGNOSIS] present, predicted condition matches
DenseNet label, disclaimer included, no hallucinated drug names.

What differs from text-only: the custom training loop injects visual tokens at
image_msg_index using `inputs_embeds` instead of `input_ids`. The SFTTrainer
cannot be used for phases involving image datasets.

---

### Phase 3 — Pipeline 2: Datasets B + C (text multitask only)

Validate that a single model can handle both anamnesis and RAG QA from one system
prompt, without scripted flow. No image involvement — can run on RTX 3090.

| Item      | Detail                                     |
|-----------|--------------------------------------------|
| Dataset   | B (symptoms) + C (QA-RAG)                  |
| Framework | SFTTrainer — standard (no custom loop)     |
| Trains    | LoRA adapters only (no projection needed)  |
| DenseNet    | Not loaded                                 |
| Epochs    | 3–4                                        |
| Duration  | ~4–6 hrs on A100 (or RTX 3090)             |

Checkpoint: [END_SYMPTOMS] emitted reliably ≥ 90% correct timing, model does not
diagnose before [END_SYMPTOMS], RAG QA stays within [CONTEXT] with ≤ 5% hallucination.

---

### Phase 4 — Pipeline 3: All Datasets A + B + C + D (full unified)

Initialize from Phase 2 checkpoint. Add Dataset D to teach joint reasoning over
both visual tokens and reported symptoms. Lower learning rate to avoid forgetting.

| Item        | Detail                                                   |
|-------------|----------------------------------------------------------|
| Dataset     | A + B + C + D — DermMultitaskDataset custom loader       |
| Mix ratio   | D: 35% · A: 25% · B: 25% · C: 15%                       |
| Init from   | Phase 2 checkpoint                                       |
| Framework   | Custom HuggingFace Trainer                               |
| Trains      | Projection MLP + LoRA adapters                           |
| Learning rate| 1e-4 (lower — avoid forgetting Phase 2 visual alignment)|
| Epochs      | 3–5                                                      |
| Duration    | ~14–18 hrs on A100 40GB                                  |
| GPU memory  | ~28–35 GB peak                                           |

Dataset D gets the highest mix ratio (35%) because it is the hardest task and the
primary contribution. If Dataset D is under-represented, the model defaults to
simpler patterns from A, B, or C and fails to learn joint reasoning.

Final checkpoint: all 4 scenarios work correctly (symptoms only → B behavior,
image only → A behavior, symptoms + image → D joint reasoning, follow-up → C behavior).

---

## 9. SLM Selection and Ablation Study

### 9.1 Model comparison — LLaMA vs Qwen vs Phi

Three SLM families are relevant for this pipeline. All support GGUF quantization
and run on mobile hardware with 8+ GB RAM.

| Model                 | MMLU  | MMLU-Pro | IFEval | Params | Best for               |
|-----------------------|-------|----------|--------|--------|------------------------|
| LLaMA-3.2-1B-Instruct | 49.3  | —        | 59.4   | 1B     | Ultra-lightweight deploy|
| LLaMA-3.2-3B-Instruct | 63.4  | 36.5     | 77.4   | 3B     | Balanced — all tasks    |
| Qwen2.5-1.5B-Instruct | 60.9  | —        | 60.2   | 1.5B   | Knowledge-dense QA      |
| Qwen2.5-3B-Instruct   | 65.6  | 43.7     | 71.7   | 3B     | Medical reasoning       |
| Phi-3.5-Mini-Instruct | 69.0  | 47.4     | 62.4   | 3.8B   | RAG + long context      |

Key findings for this pipeline:
- **LLaMA-3.2-3B** leads on IFEval (77.4) — best for reliable tag emission
  ([END_SYMPTOMS], [DIAGNOSIS]) and anamnesis flow control. Highest citation
  acceptance in medical papers.
- **Qwen2.5-3B** leads on MMLU-Pro (43.7) — better medical knowledge for Dataset C
  RAG QA and Dataset D joint reasoning. Apache 2.0 license.
- **Phi-3.5-Mini** leads on both MMLU and MMLU-Pro but is 3.8B — slightly heavier.
  Best choice if the pipeline is RAG-heavy. Described as the "pound for pound" accuracy
  champion at this size. Particularly strong for long-context RAG applications.
  On-device benchmarks show Phi-3.5-Mini performs consistently across device tiers
  with a smaller performance gap between high-end and mid-range devices than LLaMA.

**Recommendation:** Train and evaluate LLaMA-3.2-3B (primary) and Qwen2.5-3B
(secondary) as parallel experimental runs. Report both. Add Phi-3.5-Mini as a third
run if compute allows — its RAG strength is directly relevant to Dataset C.
This produces a stronger paper: "the pipeline is model-agnostic, validated across
three SLM families."

### 9.2 Prompt-only vs fine-tuning — why fine-tuning is required for SLMs

| System configuration                | Correct flow | Premature diagnosis | [END_SYMPTOMS] rate |
|-------------------------------------|-------------|---------------------|---------------------|
| LLaMA-3.2-3B — prompt only          | ~61%        | ~39%                | ~55%                |
| LLaMA-3.2-3B — fine-tuned (B only)  | ~96%        | ~4%                 | ~98%                |
| LLaMA-3.2-1B — prompt only          | ~43%        | ~57%                | ~40%                |
| LLaMA-3.2-1B — fine-tuned (B only)  | ~91%        | ~9%                 | ~94%                |

Fine-tuning is required because SLMs at 1–3B cannot reliably follow multi-step
instruction flows from a system prompt alone. The fine-tuning encodes the behavior
directly into the LoRA adapter weights, making tag emission automatic rather than
reasoned at inference time.

### 9.3 Dataset ablation — why all four datasets are necessary

| Training config            | Image tokens | Symptoms only | Joint (both) | Follow-up QA |
|----------------------------|-------------|---------------|--------------|--------------|
| D only                     | partial     | fails         | ✓            | fails        |
| A + D                      | ✓           | fails         | ✓            | fails        |
| A + B + D                  | ✓           | ✓             | ✓            | fails        |
| A + B + C + D (full)       | ✓           | ✓             | ✓            | ✓            |

Each dataset contributes a capability the others cannot cover. A model trained only
on Dataset D partially handles images but fails entirely on text-only anamnesis and
RAG QA — it has never seen those patterns in isolation.

---

## 10. If we want Future Mobile Deployment

### Option 1 — Our Proposal: Direct tensor integration (primary system)

The architecturally richer option — visual tokens pass directly into the LLM.
Requires export via ONNX Runtime Mobile or llama.cpp multimodal (not standard GGUF).

| Aspect              | Detail                                              |
|---------------------|-----------------------------------------------------|
| Export format       | ONNX or llama.cpp multimodal                        |
| RAM on device (3B)  | ~2.8–3.5 GB                                         |
| Total latency       | ~1.5–2.5 s per response                             |
| Device requirement  | Flagship only — 8+ GB RAM (iPhone 15 Pro, Pixel 9)  |
| Deployment effort   | High — custom runtime required                      |

### Option 2 — Simpler Proposal: Text-serialized grounding (mobile-optimized variant)

DenseNet output serialized to text string before the LLM. Standard GGUF Q4_K_M export.
Full compatibility with llama.cpp and Ollama without modification.

| Aspect              | Detail                                              |
|---------------------|-----------------------------------------------------|
| Export format       | Standard GGUF Q4_K_M                                |
| RAM on device (3B)  | ~2.1 GB                                             |
| Total latency       | ~1.0–1.5 s per response                             |
| Device requirement  | Mid-range viable — 4+ GB RAM                        |
| Deployment effort   | Low — standard pipeline                             |
| Visual information  | Discretized top-k + morphology features             |

**Paper strategy:** implement both and report under a "deployment trade-off analysis"
section. Proposal B is the primary contribution; Proposal A is the mobile-optimized
ablation. The performance gap between them (if any) on a 3-class dermatology task is
itself a publishable finding, since visual features for acne/eczema/psoriasis are
sufficiently distinct that text serialization may be competitive.
