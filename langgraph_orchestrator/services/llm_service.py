"""llm_service.py — LLM backend abstraction for the dermatology chatbot.

Public interface (called by node files)
----------------------------------------
- generate_symptom_response(state)   → next symptom-collection question
- generate_diagnosis_response(state) → raw multimodal diagnosis text
- generate_followup_response(state)  → follow-up answer (plain or RAG-grounded)

Each public function reads config.USE_MOCK_LLM and delegates to either the
mock implementation (_mock_*) or the real implementation (_real_*).

Teammate integration guide
--------------------------
1. Set USE_MOCK_LLM = False in config.py (or export USE_MOCK_LLM=0).
2. Optionally set the LLM_MODEL_PATH environment variable to a local
   merged-model directory; otherwise the default HuggingFace repo is used.
3. The public functions and all node files stay unchanged.

The _mock_* functions are preserved and continue to work exactly as before.

Real-mode implementation notes
-------------------------------
STATUS — _load_llm:                         FULLY WIRED
STATUS — _real_generate_symptom_response:   FULLY WIRED
STATUS — _real_generate_followup_response:  FULLY WIRED (RAG injection included)
STATUS — _real_generate_diagnosis_response:
    PRIMARY path (projected_embedding tensor from vision_service): FULLY WIRED
    FALLBACK path (re-run DenseNet+projection from image_path):    FULLY WIRED
    The primary path is used when USE_MOCK_VISION=False (vision_service
    places a tensor under densenet_output["projected_embedding"]).
    The fallback activates when that key is absent or not a tensor (e.g.
    when USE_MOCK_VISION=True but USE_MOCK_LLM=False — a mixed-mode run).

Helper functions
----------------
The following helpers are adapted directly from inference_example.py:
  build_prompt, _chat_ids, generate_text, generate_with_image,
  _extract_first_turn.
They are defined here (not imported from inference_example.py) so that
this service file is self-contained and inference_example.py can evolve
independently.
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import torch
import torch.nn as nn

from langgraph_orchestrator.config import USE_MOCK_LLM
from langgraph_orchestrator.services.symptom_service import REQUIRED_SLOTS


# ---------------------------------------------------------------------------
# Model / generation settings
# ---------------------------------------------------------------------------

# Base LLM — must be the unmerged base model when using LoRA adapters.
# Override with LLM_MODEL_PATH to point to a local directory.
# NOTE: image adapter was trained on meta-llama/Llama-3.2-3B;
#       symptoms adapter was trained on Llama-3.2-3B-Instruct (Kaggle).
#       Using 3B-Instruct as base gives best symptoms quality; image adapter
#       will still load but its performance may vary slightly.
_LLM_MODEL_PATH: str = os.environ.get(
    "LLM_MODEL_PATH",
    "meta-llama/Llama-3.2-3B-Instruct",  # base model for adapter loading
    #"RominaLopez/llama-3.2-3b-4bit-quantized",  # base model for adapter loading
)

# Adapter paths — one folder per task, relative to repo root.
# A folder is skipped automatically if adapter_model.safetensors is missing.
_ADAPTER_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "model", "adapters")
)
_ADAPTER_PATHS: dict[str, str] = {
    "symptoms": os.path.join(_ADAPTER_ROOT, "symptoms"),  # symptom collection
    "image":    os.path.join(_ADAPTER_ROOT, "image"),     # visual diagnosis
    "rag":      os.path.join(_ADAPTER_ROOT, "rag"),       # followup / RAG QA
}

# Maximum new tokens to generate per turn.  Increase for longer diagnoses.
_MAX_NEW_TOKENS: int = 1024

# separate prompts — replaced single _SYSTEM_PROMPT with three focused prompts,
# one per clinical phase, so each LLM call gets only the instructions it needs.

# separate prompts — symptom collection phase
_SYSTEM_PROMPT_SYMPTOMS = """You are an empathetic dermatology assistant in the symptom collection phase.

Collect detailed information about the user's skin concern through focused,
one-at-a-time questions covering location, duration, morphology, severity, and
triggers. Ask only one question per turn. Never diagnose during this phase.
When you have gathered all the necessary information, end your reply with [END_SYMPTOMS].

Always recommend consulting a dermatologist. Never provide a definitive diagnosis."""

# NEW dataset d — matches SYSTEM_VISUAL_DIAGNOSIS in prepare_datasets/system_prompt.py exactly.
# Must stay in sync: position of visual tokens depends on system prompt token length.
_SYSTEM_PROMPT_DIAGNOSIS = (
    "You are a dermatology assistant. Analyze the image tokens, then the patient "
    "symptoms ([PATIENT_SYMPTOMS]) and DenseNet statistics ([DENSENET_STATISTICS]). "
    "Reason inside <think>...</think> covering: visual morphology, symptom correlation, "
    "statistical validation, and differential diagnosis (1-2 sentences each). "
    "Respond inside [DIAGNOSIS]...[/DIAGNOSIS] as a doctor speaking to the patient: "
    "plain language, 3-4 short paragraphs, no headers, explain what you see and why. "
    "Never provide a definitive diagnosis. Always recommend consulting a dermatologist."
)

# separate prompts — follow-up QA phase
_SYSTEM_PROMPT_FOLLOWUP = """You are an empathetic dermatology assistant in the follow-up phase.

The patient has already received a diagnosis. Answer follow-up questions clearly
and concisely. When a [CONTEXT] block is provided, answer using only that context
and do not speculate beyond what it contains. If no context is provided, answer
from your general dermatology knowledge.

Always recommend consulting a dermatologist. Never provide a definitive diagnosis."""


# ---------------------------------------------------------------------------
# Lazy model loader — loads once per process and caches in module globals
# ---------------------------------------------------------------------------

_llm_model     = None   # PeftModel (if any adapter loaded) or AutoModelForCausalLM
_llm_tokenizer = None   # transformers AutoTokenizer
_llm_device: torch.device | None = None
_loaded_adapters: set[str] = set()  # names of adapters successfully loaded


def _adapter_is_ready(path: str) -> bool:
    """Return True if path contains the two files PEFT requires."""
    return (
        os.path.isfile(os.path.join(path, "adapter_config.json")) and
        os.path.isfile(os.path.join(path, "adapter_model.safetensors"))
    )


def _load_llm():
    """Load and cache the base LLM + available LoRA adapters on first call.

    Adapter loading (MultiLoRA):
        Each entry in _ADAPTER_PATHS is checked for adapter_config.json +
        adapter_model.safetensors.  Present adapters are loaded via PEFT with
        their folder name as the adapter_name.  Missing/empty folders are
        skipped — the base model weights are used for those tasks instead.

    Returns:
        (tokenizer, model, device)
    """
    global _llm_model, _llm_tokenizer, _llm_device, _loaded_adapters

    if _llm_model is not None:
        return _llm_tokenizer, _llm_model, _llm_device

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from peft import PeftModel  # type: ignore  # separate prompts — MultiLoRA

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[llm_service] Loading base LLM from: {_LLM_MODEL_PATH}")
    print("[llm_service] (First run may download the model — this can take a few minutes)")

    tokenizer = AutoTokenizer.from_pretrained(_LLM_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        _LLM_MODEL_PATH,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)

    # --- MultiLoRA: load each available adapter ---  # separate prompts
    first_loaded = True
    for adapter_name, adapter_path in _ADAPTER_PATHS.items():
        if not _adapter_is_ready(adapter_path):
            print(f"[llm_service] Adapter '{adapter_name}': not found — will use base model for this task.")
            continue
        if first_loaded:
            model = PeftModel.from_pretrained(  # separate prompts — wraps base model with PEFT
                model, adapter_path, adapter_name=adapter_name, is_trainable=False
            )
            first_loaded = False
        else:
            model.load_adapter(adapter_path, adapter_name=adapter_name)  # separate prompts
        _loaded_adapters.add(adapter_name)
        print(f"[llm_service] Adapter '{adapter_name}': loaded from {adapter_path}")

    model.eval()

    _llm_tokenizer = tokenizer
    _llm_model     = model
    _llm_device    = device

    print(f"[llm_service] LLM ready. Loaded adapters: {_loaded_adapters or {'(none — base model only)'}}")
    return _llm_tokenizer, _llm_model, _llm_device


# ===========================================================================
# Helper functions — adapted from inference_example.py
# ===========================================================================

def build_prompt(history: list[dict[str, str]], system_prompt: str = _SYSTEM_PROMPT_SYMPTOMS) -> str:  # separate prompts — added system_prompt param
    """Serialise message history into a LLaMA-3.2 chat prompt string.

    Prepends system_prompt automatically.  Skips messages with empty content.

    Args:
        history:       List of {"role": ..., "content": ...} dicts.  Should contain
                       only user/assistant turns (system is injected here).
        system_prompt: The phase-specific system prompt to prepend.  Defaults to
                       _SYSTEM_PROMPT_SYMPTOMS.

    Returns:
        Fully-formatted prompt string ending with the assistant header so the
        model generates the next assistant turn.
    """
    print(f"[llm_service] build_prompt → system_prompt: {system_prompt.splitlines()[0]!r}")  # separate prompts — debug
    text  = "<|begin_of_text|>"
    text += (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"  # separate prompts — was _SYSTEM_PROMPT
    )
    for msg in history:
        if msg["content"] == "":
            continue
        text += (
            f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
            f"{msg['content']}<|eot_id|>"
        )
    text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


def _chat_ids(messages: list[dict[str, str]], tokenizer, prime_assistant: bool = True) -> list[int]:
    """Build LLaMA-3.2 chat token IDs piece by piece.

    Avoids BPE boundary issues that arise from tokenizing the fully-assembled
    string (special tokens must be encoded separately from content).

    Args:
        messages:        Full message list including the system turn.
        tokenizer:       Loaded HuggingFace tokenizer.
        prime_assistant: If True, append the assistant header at the end so the
                         model starts generating the next assistant turn.

    Returns:
        List of integer token IDs.
    """
    bos    = tokenizer.bos_token_id
    ids    = [bos] if bos is not None else []
    start_h = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    end_h   = tokenizer.encode("<|end_header_id|>",   add_special_tokens=False)
    eot     = tokenizer.encode("<|eot_id|>",           add_special_tokens=False)
    nl2     = tokenizer.encode("\n\n",                 add_special_tokens=False)

    for msg in messages:
        if msg["content"] == "":
            continue
        role_ids    = tokenizer.encode(msg["role"],    add_special_tokens=False)
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        ids += start_h + role_ids + end_h + nl2 + content_ids + eot

    if prime_assistant:
        asst_ids = tokenizer.encode("assistant", add_special_tokens=False)
        ids += start_h + asst_ids + end_h + nl2

    return ids


@torch.no_grad()
def generate_text(model, tokenizer, history: list[dict[str, str]],
                  max_new_tokens: int, device: torch.device,
                  system_prompt: str = _SYSTEM_PROMPT_SYMPTOMS,
                  extract_first_turn: bool = True) -> str:  # separate prompts — added system_prompt param
    """Text-only generation: system prompt + history → next assistant turn.

    Args:
        model:          Loaded causal LM.
        tokenizer:      Matching tokenizer.
        history:        User/assistant turn list (system prompt is injected).
        max_new_tokens: Hard limit on generated tokens.
        device:         Torch device the model lives on.
        system_prompt:  The phase-specific system prompt to use.

    Returns:
        Cleaned assistant reply string (first turn only).
    """
    prompt = build_prompt(history, system_prompt)  # separate prompts — was build_prompt(history)
    ids    = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_id],
    )
    raw = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return _extract_first_turn(raw)


@torch.no_grad()
def generate_with_image(
    model,
    tokenizer,
    history: list[dict[str, str]],
    vis_tokens: torch.Tensor,
    max_new_tokens: int,
    device: torch.device,
    system_prompt: str = _SYSTEM_PROMPT_DIAGNOSIS,  # separate prompts — added system_prompt param
) -> str:
    """Multimodal generation: inject pre-computed visual tokens into input embeddings.

    Layout (matches the training-time injection in unified_training):
        [conversation context embeddings] + [32 visual token embeddings] + [assistant header]

    Args:
        model:          Loaded causal LM.
        tokenizer:      Matching tokenizer.
        history:        User/assistant turns (system prompt is prepended internally).
        vis_tokens:     Pre-computed visual tokens, shape (1, 32, llm_dim).
                        Produced by vision_service._real_project_vision_features.
        max_new_tokens: Hard limit on generated tokens.
        device:         Torch device the model lives on.
        system_prompt:  The phase-specific system prompt to use.

    Returns:
        Raw LLM output string.  May contain <think>…</think> and
        [DIAGNOSIS]…[/DIAGNOSIS] tags; clean_response.py strips these before
        the text reaches the user.

    NOTE: This function signature differs from inference_example.py's
    generate_with_image — it accepts vis_tokens directly instead of
    feat_extractor + projection + image_path, because the projection step
    already ran in vision_service.project_vision_features.
    """
    # NEW dataset d — layout: [sys] + [vis_tokens] + [user_context] + [asst_header]
    # Matches image_msg_index=1 training layout in unified_training/dataset.py.
    # _SYSTEM_PROMPT_DIAGNOSIS must stay identical to SYSTEM_VISUAL_DIAGNOSIS in
    # prepare_datasets/system_prompt.py — visual token position depends on sys length.

    # 1) Tokenize system turn only — marks the injection point for visual tokens
    sys_ids = _chat_ids(
        [{"role": "system", "content": system_prompt}], tokenizer, prime_assistant=False
    )

    # 2) Tokenize full conversation (sys + history) + assistant prime
    full_messages = [{"role": "system", "content": system_prompt}] + history
    all_ids = _chat_ids(full_messages, tokenizer, prime_assistant=True)

    asst_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )

    # 3) Split: sys | user_context | asst_header
    sys_end  = len(sys_ids)
    ctx_ids  = all_ids[sys_end : len(all_ids) - len(asst_header_ids)]

    sys_tensor  = torch.tensor([sys_ids],         dtype=torch.long, device=device)
    ctx_tensor  = torch.tensor([ctx_ids],         dtype=torch.long, device=device)
    asst_tensor = torch.tensor([asst_header_ids], dtype=torch.long, device=device)

    embed_fn    = model.get_input_embeddings()
    sys_embeds  = embed_fn(sys_tensor)   # (1, T_sys,  llm_dim)
    ctx_embeds  = embed_fn(ctx_tensor)   # (1, T_ctx,  llm_dim)
    asst_embeds = embed_fn(asst_tensor)  # (1, T_asst, llm_dim)

    vis_tokens = vis_tokens.to(device).to(sys_embeds.dtype)  # (1, 32, llm_dim)

    # 4) Assemble: [sys] + [vis_tokens] + [user_context] + [asst_header]
    combined  = torch.cat([sys_embeds, vis_tokens, ctx_embeds, asst_embeds], dim=1)
    attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

    # ── DEBUG PRINT ──────────────────────────────────────────────────────────
    _sep = "=" * 60
    print(f"\n{_sep}")
    print("[llm_service] generate_with_image — FULL PROMPT SENT TO MODEL")
    print(f"  Layout : [sys({len(sys_ids)} tok)] + [vis_tokens(32)] + "
          f"[ctx({len(ctx_ids)} tok)] + [asst_header({len(asst_header_ids)} tok)]")
    print(f"  Total  : {len(sys_ids) + 32 + len(ctx_ids) + len(asst_header_ids)} positions")
    print(f"\n── SYSTEM PROMPT ──\n{system_prompt}")
    if ctx_ids:
        print(f"\n── USER CONTEXT ──\n{tokenizer.decode(ctx_ids, skip_special_tokens=True)}")
    else:
        print("\n── USER CONTEXT ── (empty)")
    print(f"\n── VISUAL TOKENS ── shape={vis_tokens.shape} dtype={vis_tokens.dtype} "
          f"mean={vis_tokens.float().mean().item():.4f} std={vis_tokens.float().std().item():.4f}")
    print(_sep)
    # ─────────────────────────────────────────────────────────────────────────

    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    out = model.generate(
        inputs_embeds=combined,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_id],
    )
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    raw = re.sub(r'(\[\]\s*){3,}', '', raw).strip()

    # ── DEBUG PRINT ──────────────────────────────────────────────────────────
    print(f"\n{_sep}")
    print("[llm_service] RAW MODEL RESPONSE:")
    print(raw)
    print(f"{_sep}\n")
    # ─────────────────────────────────────────────────────────────────────────

    return raw


def _extract_first_turn(text: str) -> str:
    """Keep only the first assistant turn from generated text.

    The model sometimes continues generating additional fake turns.  This
    function trims everything after the first meaningful response ends.

    Adapted directly from inference_example.py.
    """
    for sep in ["\n\n\n", "\n\n"]:
        parts = text.split(sep)
        if len(parts) > 1 and len(parts[0].strip()) > 20:
            return parts[0].strip()

    q_pos = text.find("?")
    if q_pos != -1:
        after = text[q_pos + 1:].lstrip()
        if after:
            return text[: q_pos + 1].strip()

    return text


# ===========================================================================
# Diagnosis output helpers
# ===========================================================================

def _truncate_after_diagnosis_close(text: str) -> str:
    """Discard everything after the first [/DIAGNOSIS] closing tag.

    The model sometimes continues generating after [/DIAGNOSIS] (fake turns,
    repeated tokens, etc.).  Truncating at the tag boundary is the simplest
    and most reliable way to prevent that noise from reaching the parser.

    If no [/DIAGNOSIS] tag is present the full text is returned unchanged —
    the parser will still attempt to clean it.
    """
    close_tag = "[/DIAGNOSIS]"
    idx = text.upper().find(close_tag.upper())
    if idx != -1:
        return text[: idx + len(close_tag)]
    return text


# ===========================================================================
# Public interface — called by node files
# ===========================================================================

def generate_symptom_response(state: dict[str, Any]) -> str:
    """Generate the next symptom-collection question.

    Args:
        state: Current ChatState dict.

    Returns:
        Assistant reply string.  When all slots are filled the real model
        should include [END_SYMPTOMS] in its output.
    """
    if USE_MOCK_LLM:
        return _mock_generate_symptom_response(state)
    return _real_generate_symptom_response(state)


def generate_diagnosis_response(state: dict[str, Any]) -> str:
    """Generate a multimodal diagnosis using symptoms + image features.

    Args:
        state: Current ChatState dict (densenet_output must be populated).

    Returns:
        Raw LLM output string — may contain <think>…</think> and
        [DIAGNOSIS]…[/DIAGNOSIS] tags.  clean_response.py strips these
        before the text is shown to the user.
    """
    if USE_MOCK_LLM:
        return _mock_generate_diagnosis_response(state)
    return _real_generate_diagnosis_response(state)


def generate_followup_response(state: dict[str, Any]) -> str:
    """Generate a follow-up answer, optionally grounded in RAG context.

    Args:
        state: Current ChatState dict.  May contain rag_context.

    Returns:
        Plain assistant reply string (no special tags required).
    """
    if USE_MOCK_LLM:
        return _mock_generate_followup_response(state)
    return _real_generate_followup_response(state)


# ===========================================================================
# Mock implementations — hardcoded responses used during development
# ===========================================================================

def _mock_generate_symptom_response(state: dict[str, Any]) -> str:
    """MOCK: asks about the first unfilled slot.

    IMPORTANT: must only look at symptom_slots.  Must never branch on
    image_uploaded, image_acknowledged, or phase.  Image acknowledgement
    is the sole responsibility of await_or_store_image.  Low-confidence
    image fallback is also handled by await_or_store_image (Case C) —
    it never reaches collect_symptoms.
    """
    slots   = state.get("symptom_slots", {})
    missing = [k for k in REQUIRED_SLOTS if not slots.get(k)]

    if missing:
        return f"Could you tell me more about the {missing[0]} of your skin concern?"
    return "Thank you. I have all the information I need. [END_SYMPTOMS]"


def _mock_generate_diagnosis_response(state: dict[str, Any]) -> str:
    """MOCK: returns a hardcoded <think> + [DIAGNOSIS] block.

    Wording is adjusted based on uncertainty_flag:
    - uncertainty_flag=False (conf >= 0.60): confident language.
    - uncertainty_flag=True  (0.40 <= conf < 0.60): hedged language.
    """
    densenet    = state.get("densenet_output") or {}
    label       = densenet.get("label", "unknown")
    confidence  = densenet.get("confidence", 0.0)
    is_cautious = bool(state.get("uncertainty_flag"))

    if is_cautious:
        return (
            f"<think>DenseNet result: {label} at conf={confidence:.2f} — below "
            "the high-confidence threshold. Combining with symptom profile but "
            "hedging language accordingly.</think>\n"
            "[DIAGNOSIS]Based on your symptoms and the image, there are some "
            f"preliminary indications that this could be {label}. However, the "
            "image analysis returned a lower-than-usual confidence score, so this "
            "should be treated as a rough indication only. I strongly recommend "
            "consulting a dermatologist for an accurate diagnosis and an appropriate "
            "treatment plan tailored to your situation.[/DIAGNOSIS]"
        )

    return (
        f"<think>Combining symptom profile with DenseNet result "
        f"({label}, conf={confidence:.2f}). "
        "Patient reports dry, itchy patches. Consistent with eczema.</think>\n"
        "[DIAGNOSIS]Based on your symptoms and the image, this appears to be "
        f"{label} (atopic dermatitis). The affected area shows characteristic dry, "
        "scaly patches. I recommend consulting a dermatologist for a confirmed "
        "diagnosis and personalised treatment plan.[/DIAGNOSIS]"
    )


def _mock_generate_followup_response(state: dict[str, Any]) -> str:
    """MOCK: returns a hardcoded placeholder follow-up answer."""
    return (
        "Based on the available medical information: "
        "Eczema is typically managed with regular moisturising using emollients and, "
        "during flares, topical corticosteroids. Identifying and avoiding personal "
        "triggers (e.g. soaps, fabrics, allergens) is also recommended."
    )


# ===========================================================================
# MultiLoRA dispatch helper  # separate prompts
# ===========================================================================

def _generate_with_adapter(adapter_name: str, fn, **kwargs) -> str:
    """Call fn(**kwargs) with the correct adapter active.

    - adapter_name in _loaded_adapters → set_adapter then call fn.
    - adapter_name not loaded but model is PeftModel → disable all adapters
      (use base model weights) then call fn.
    - no adapters loaded at all (plain AutoModelForCausalLM) → call fn directly.
    """  # separate prompts — MultiLoRA dispatch
    model = kwargs["model"]
    if adapter_name in _loaded_adapters:
        model.set_adapter(adapter_name)
        return fn(**kwargs)
    if _loaded_adapters:  # PeftModel, but this specific adapter not trained yet
        with model.disable_adapter():
            return fn(**kwargs)
    return fn(**kwargs)  # plain base model, no PEFT wrapper


# ===========================================================================
# Real implementations
# ===========================================================================

def _real_generate_symptom_response(state: dict[str, Any]) -> str:
    """REAL: call the fine-tuned LLM to generate the next symptom question.

    STATUS: FULLY WIRED.

    Uses the full conversation history from state["messages"] as the chat
    context.  build_prompt() prepends _SYSTEM_PROMPT automatically so the
    model operates in symptom-collection mode.

    The model is expected to include [END_SYMPTOMS] in its reply when it
    has gathered enough information.  collect_symptoms.py detects this tag
    and advances the phase — no change to node code required.
    """
    tokenizer, model, device = _load_llm()
    history = state.get("messages", [])
    return _generate_with_adapter(  # separate prompts — MultiLoRA dispatch
        adapter_name="symptoms", fn=generate_text,
        model=model, tokenizer=tokenizer, history=history,
        max_new_tokens=_MAX_NEW_TOKENS, device=device,
        system_prompt=_SYSTEM_PROMPT_SYMPTOMS,
    )


def _build_dataset_d_user_context(state: dict[str, Any]) -> str:
    """NEW dataset d — build the single user message for visual diagnosis inference.

    Matches the user context format in unified_training Dataset D:
        [PATIENT_SYMPTOMS]      ← from symptom_slots (deterministic, no LLM call)
        [DENSENET_STATISTICS]   ← from densenet_output

    Order: symptoms first (clinical), DenseNet last (statistical validation).
    Matches the <think> reasoning order: visual → symptoms → DenseNet → differential.
    """
    slots   = state.get("symptom_slots") or {}
    densenet = state.get("densenet_output") or {}

    # ── [PATIENT_SYMPTOMS] ───────────────────────────────────────────────────
    slot_order = ("location", "duration", "appearance", "severity", "triggers")
    symptom_lines = [
        f"- {k.capitalize()}: {v}"
        for k in slot_order
        if (v := slots.get(k))
    ]
    symptoms_block = (
        "[PATIENT_SYMPTOMS]\n"
        + ("\n".join(symptom_lines) if symptom_lines else "- No symptom data collected")
        + "\n[/PATIENT_SYMPTOMS]"
    )

    # ── [DENSENET_STATISTICS] ────────────────────────────────────────────────
    primary   = (densenet.get("primary")   or densenet.get("label", "unknown")).capitalize()
    conf      = densenet.get("confidence", 0.0)
    secondary = (densenet.get("secondary") or densenet.get("label_2", "")).capitalize()
    conf2     = densenet.get("confidence_2", 0.0)

    densenet_lines = [
        "[DENSENET_STATISTICS]",
        f"- Primary: {primary} ({conf:.2f})",
    ]
    if secondary:
        densenet_lines.append(f"- Secondary: {secondary} ({conf2:.2f})")
    tertiary = (densenet.get("tertiary") or "").capitalize()
    if tertiary:
        conf3 = densenet.get("confidence_3", 0.0)
        densenet_lines.append(f"- Tertiary: {tertiary} ({conf3:.2f})")
    densenet_lines.append("[/DENSENET_STATISTICS]")
    densenet_block = "\n".join(densenet_lines)

    return f"{symptoms_block}\n\n{densenet_block}"


def _real_generate_diagnosis_response(state: dict[str, Any]) -> str:
    """REAL: multimodal diagnosis — NEW dataset d format.

    Layout sent to model (matches unified_training image_msg_index=1):
        [sys] + [vis_tokens(32)] + [user_context] + [asst_header]

    user_context = [PATIENT_SYMPTOMS] (from slots) + [DENSENET_STATISTICS]

    PRIMARY PATH (USE_MOCK_VISION=False):
        projected_embedding tensor from vision_service is used directly.

    FALLBACK PATH (USE_MOCK_VISION=True but USE_MOCK_LLM=False):
        Re-runs DenseNet + projection from state["image_path"].
    """
    tokenizer, model, device = _load_llm()
    densenet = state.get("densenet_output") or {}

    # NEW dataset d — single user message with symptoms + densenet stats
    user_context = _build_dataset_d_user_context(state)
    # history is a list with one user message — generate_with_image places it
    # after visual tokens: [sys] + [vis_tokens] + [user_context] + [asst_header]
    diagnosis_history = [{"role": "user", "content": user_context}]

    projected_embedding = densenet.get("projected_embedding")

    if isinstance(projected_embedding, torch.Tensor):
        # PRIMARY PATH: pre-computed visual tokens from vision_service
        raw = _generate_with_adapter(
            adapter_name="image",
            fn=generate_with_image,
            model=model, tokenizer=tokenizer,
            history=diagnosis_history,
            vis_tokens=projected_embedding,
            max_new_tokens=_MAX_NEW_TOKENS,
            device=device,
            system_prompt=_SYSTEM_PROMPT_DIAGNOSIS,
        )
        return _truncate_after_diagnosis_close(raw)

    # --------------------------------------------------------------------
    # FALLBACK PATH: re-run visual pipeline from image_path.
    # Activated when projected_embedding is a string (mock vision) or None.
    # --------------------------------------------------------------------
    image_path: str | None = state.get("image_path")
    if not image_path:
        raise ValueError(
            "[llm_service] _real_generate_diagnosis_response: no projected_embedding "
            "tensor and no image_path in state.  Cannot generate a visual diagnosis.\n"
            "Either set USE_MOCK_VISION=False so vision_service produces a real tensor, "
            "or ensure image_path is set before joint_diagnosis runs."
        )

    # Lazy-import vision models — reuses the module-level cache in vision_service
    # so we do NOT load a second copy of DenseNet + projection into memory.
    from langgraph_orchestrator.services.vision_service import (  # noqa: PLC0415
        _load_vision_models,
        _IMAGE_TRANSFORM,
    )
    from PIL import Image  # type: ignore  # noqa: PLC0415
    import torch.nn.functional as F  # noqa: PLC0415

    feat_extractor, projection_mlp, vis_device = _load_vision_models()
    proj_dtype = next(projection_mlp.parameters()).dtype

    image      = Image.open(image_path).convert("RGB")
    img_tensor = _IMAGE_TRANSFORM(image).unsqueeze(0).to(vis_device)

    with torch.no_grad():
        pooled_features, _ = feat_extractor(img_tensor)          # (1, 1024)
        vis_tokens = projection_mlp(
            pooled_features.to(proj_dtype)
        )                                                         # (1, 32, llm_dim)

    raw = _generate_with_adapter(  # separate prompts — MultiLoRA dispatch
        adapter_name="image",
        fn=generate_with_image,
        model=model, tokenizer=tokenizer,
        history=diagnosis_history,
        vis_tokens=vis_tokens,
        max_new_tokens=_MAX_NEW_TOKENS,
        device=device,
        system_prompt=_SYSTEM_PROMPT_DIAGNOSIS,
    )
    return _truncate_after_diagnosis_close(raw)


def _real_generate_followup_response(state: dict[str, Any]) -> str:
    """REAL: call the fine-tuned LLM to answer a follow-up question.

    STATUS: FULLY WIRED.

    RAG injection:
        If state["rag_context"] is set, it is prepended to the last user
        message as a [CONTEXT]...[/CONTEXT] block.  This matches the format
        described in _SYSTEM_PROMPT ("When a [CONTEXT] block is provided,
        answer using only that context.") and the model's training data.

        The injection modifies a local copy of the history — state is not
        mutated.

    No RAG:
        If rag_context is None, the full conversation history is used as-is
        and the model answers from its own knowledge.
    """
    tokenizer, model, device = _load_llm()
    history     = list(state.get("messages", []))
    rag_context = state.get("rag_context")

    if rag_context and history:
        last_msg = history[-1]
        if last_msg["role"] == "user":
            # Inject context directly into the last user message so the model
            # sees [CONTEXT]...[/CONTEXT] immediately before the question.
            # This matches the SYSTEM_PROMPT instruction about [CONTEXT] blocks.
            history = history[:-1] + [
                {
                    "role":    "user",
                    "content": (
                        f"[CONTEXT]\n{rag_context}\n[/CONTEXT]\n\n"
                        f"{last_msg['content']}"
                    ),
                }
            ]
        # If the last message is not from the user (unusual state), fall
        # through without injection — safer than crashing.

    return _generate_with_adapter(  # separate prompts — MultiLoRA dispatch
        adapter_name="rag", fn=generate_text,
        model=model, tokenizer=tokenizer, history=history,
        max_new_tokens=_MAX_NEW_TOKENS, device=device,
        system_prompt=_SYSTEM_PROMPT_FOLLOWUP,
        extract_first_turn=False,  # follow-up answers may be longer and less prone to token loops, so keep the full response instead of truncating to the first turn.
    )
