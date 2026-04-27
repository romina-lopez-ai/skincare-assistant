# 🧠 Multimodal Dermatology Chatbot — LangGraph Orchestration Plan

## 🎯 Goal
Build a **single intelligent chatbot system** that:
- collects symptoms,
- analyzes skin images,
- provides diagnosis-style responses,
- answers follow-up questions using medical knowledge.

LangGraph will act as the **orchestrator (controller)**, while the **LLM performs reasoning**.

---

# 🧩 System Components

## 1. 🧠 Fine-tuned LLM (Main Brain)
- Base LLM + LoRA adapters  
- Handles:
  - symptom collection  
  - diagnosis reasoning  
  - follow-up answers  
- Always generates final response  

---

## 2. 👁️ DenseNet (Vision Model)
- Input: image  
- Output:
  - condition (acne, eczema, psoriasis)  
  - confidence  
  - morphology  

---

## 3. 🔄 Projection Layer
- Converts image features → LLM-compatible embeddings  
- Enables multimodal reasoning  

---

## 4. 📚 RAG (Retrieval System)
- Retrieves medical knowledge  
- Injects `[CONTEXT]` into LLM  
- Used only for knowledge / follow-up questions  

---

## 5. 🧭 LangGraph (Orchestrator )
Controls:
- conversation state  
- routing between components  
- when to use image / RAG  
- fallback logic  

---

# 🔁 High-Level Architecture
User
↓
LangGraph (State + Routing)
├── Text → LLM
├── Image → DenseNet → Projection → LLM
└── RAG → Retriever → LLM

Important: 
👉 LLM is ALWAYS the final response generator  
👉 DenseNet does not answer the user directly.
👉 Projection layer is only used when image input is present.
👉 RAG is only used when grounded medical/contextual answering is needed.
---

# 🧭 Conversation States

| State | Purpose |
|------|--------|
| `start` | begin conversation |
| `symptom_collection` | ask questions |
| `await_image` | wait for image |
| `joint_diagnosis` | image + symptoms reasoning |
| `follow_up` | answer questions |
| `end_session` | reset |

---

# 🔄 State Transitions
start
↓
symptom_collection
├─ image uploaded → store only
├─ [END_SYMPTOMS] → await_image
↓
await_image
├─ image received → run_vision
↓
run_vision → projection
↓
joint_diagnosis
├─ [DIAGNOSIS] → follow_up
↓
follow_up
├─ normal → LLM
├─ knowledge → RAG
├─ [END_SESSION] → end


---

# ⚙️ Key Design Rules

## ✅ Rule 1 — Diagnosis requires BOTH
Image + Symptoms → Diagnosis

---

## ✅ Rule 2 — Image alone is NOT enough
- store image if early  
- continue symptom collection  

---

## ✅ Rule 3 — LLM is always used
- never respond directly from DenseNet  
- always go through LLM  

---

## ✅ Rule 4 — RAG is conditional
Use only for:
- treatment  
- causes  
- medical knowledge  

---

## ✅ Rule 5 — Orchestrator ≠ reasoning
- LLM = reasoning  
- LangGraph = control flow  

---

# ⚠️ Fallback Logic (VERY IMPORTANT)

Because model may fail:

| Problem | Solution |
|--------|---------|
| no `[END_SYMPTOMS]` | use slot completion / max turns |
| repeated questions | force transition |
| early image | store + continue |
| weak/conflicting result | return uncertainty |

---

# 🧱 Proposed LangGraph Node Structure

## Node 1 — session_entry

### Purpose
- Start the session
- Initialize conversation state
- Store first user message
- Detect whether input contains text, image, or both

### Input
- user message
- optional image

### Output
- initialized state
- route to next node

## Node 2 — input_router

### Purpose
Decide which high-level path to follow based on:
- current state
- whether image is uploaded
- whether diagnosis has already been given
- whether the question looks like a knowledge/follow-up question

### Possible routes
- collect_symptoms
- await_or_store_image
- run_vision
- followup_router

## Node 3 — collect_symptoms

### Purpose
- Call the fine-tuned LLM for symptom collection
- Ask one focused question at a time
- Continue until symptom collection is complete

### Primary completion signal
[END_SYMPTOMS]

### Fallback completion
- enough symptom slots collected
- assistant semantically indicates completion
- max-turn limit reached

### Typical slots
- location
- duration
- appearance
- severity
- triggers

## Node 4 — await_or_store_image

### Purpose
- Request image if symptoms are complete but no image is uploaded
- Store image if user uploads it early
- Do not directly diagnose from image alone

### Important rule
If image is uploaded during symptom_collection, store it and continue symptom collection

## Node 5 — run_vision

### Purpose
- Run DenseNet on the uploaded image
- Produce:
  - predicted label
  - confidence
  - morphology-related info

### Output
- structured vision result saved in state

## Node 6 — project_image_features

### Purpose
- Pass DenseNet output / image features through projection layer
- Convert vision information into LLM-compatible form

### Used only when
- image is available
- symptoms are complete
- system is ready for multimodal diagnosis

## Node 7 — joint_diagnosis

### Purpose
- Call multimodal LLM path using:
  - symptom history
  - image features via projection layer
  - current conversation context

### Expected output
- <think>...</think>
- [DIAGNOSIS]...[/DIAGNOSIS]

### Notes
- Joint reasoning happens inside the model
- LangGraph only routes and manages state

## Node 8 — clean_response

### Purpose
- Strip <think> block before showing output to user
- Keep only diagnosis/final user-facing response

### Reason
- Internal reasoning should not be shown to the user

## Node 9 — followup_router

### Purpose
- Decide whether a follow-up question should be handled by:
  - standard LLM response
  - RAG-based grounded response

### Examples likely to use RAG
- treatment
- causes
- triggers
- prevention
- prognosis
- medication / cream questions

## Node 10 — retrieve_context

### Purpose
- Search relevant medical or project knowledge
- Return retrieved chunks/context for the user’s follow-up question

### Output
[CONTEXT] block

## Node 11 — rag_answer

### Purpose
- Inject [CONTEXT] into the LLM input
- Generate grounded answer using retrieved information only

## Node 12 — session_end

### Purpose
- Reset state
- End or restart session

---

# Suggested node flow
session_entry
   ↓
input_router
   ├── collect_symptoms
   │      ↓
   │   await_or_store_image
   │      ↓
   │   run_vision
   │      ↓
   │   project_image_features
   │      ↓
   │   joint_diagnosis
   │      ↓
   │   clean_response
   │      ↓
   │   followup_router
   │         ├── rag path → retrieve_context → rag_answer
   │         └── normal path → LLM
   ↓
session_end


# 🧪 Example Flows

## Flow 1 — Symptoms → Image → Follow-up Question
User describes symptoms
→ LLM asks symptom questions
→ [END_SYMPTOMS]
→ user uploads image
→ DenseNet
→ projection layer
→ LLM diagnosis
→ follow-up question
→ RAG
→ grounded answer

---

## Flow 2  — Hi → Symptoms → Early Image → Diagnosis
User says hi
→ LLM greets and begins intake
→ user describes symptoms
→ user uploads image during symptom collection
→ system stores image
→ continue questions
→ symptoms complete
→ DenseNet
→ projection layer
→ LLM diagnosis


---

## Flow 3 — Image First
User uploads image first
→ system stores image
→ asks symptom questions
→ symptoms complete
→ DenseNet
→ projection layer
→ LLM diagnosis


---

## Flow 4 — General Knowledge Question
User asks: "What causes psoriasis?"
→ route to RAG
→ retrieve context
→ LLM answers using context

# Session memory 
Temporary session memory as a conversation record + workflow state for one active user session.

## Identity / session info
- session_id
- user_id if needed
- started_at
- last_updated

## Conversation state
- phase
  - start
  - symptom_collection
  - await_image
  - joint_diagnosis
  - follow_up
  - end_session

## Message history
  - messages

## Symptom tracking
  - symptom_slots
    - location
    - duration
    - appearance
    - severity
    - triggers
  - symptom_turns
  - symptoms_complete

## Image state
- image_uploaded
- image_path
- image_processed
- densenet_output

## Diagnosis state
- diagnosis_done
- diagnosis_summary

## Follow-up / RAG state
- rag_context
- last_retrieved_docs
- followup_mode

## Safety / fallback
- fallback_triggered
- uncertainty_flag

# Session memory example
session_memory = {
    "session_id": "abc123",

    # conversation
    "messages": [],

    # state
    "phase": "start",

    # symptom tracking
    "symptom_slots": {
        "location": None,
        "duration": None,
        "appearance": None,
        "severity": None,
        "triggers": None
    },
    "symptom_turns": 0,
    "symptoms_complete": False,

    # image state
    "image_uploaded": False,
    "image_path": None,
    "image_processed": False,
    "densenet_output": None,

    # diagnosis
    "diagnosis_done": False,
    "diagnosis_summary": None,

    # follow-up / RAG
    "rag_context": None,

    # fallback / safety
    "fallback_triggered": False,
    "uncertainty_flag": False
}

---

# 🏆 Final Summary
ONE chatbot (LLM + LoRA)
+
Image modules (DenseNet, Projection, RAG)
+
LangGraph controls flow


---

# ✅ Next Step
- Confirm flow with team  
- Start implementing LangGraph nodes  
- Add fallback logic and testing scenarios  