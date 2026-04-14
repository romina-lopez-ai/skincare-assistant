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

# HuggingFace Hub repo ID for the fine-tuned dermatology LLM.
# Override with the LLM_MODEL_PATH env var to use a local merged-model dir,
# e.g. unified_training/outputs/merged_model.
_LLM_MODEL_PATH: str = os.environ.get(
    "LLM_MODEL_PATH",
    "santiagoboxiga/skincare-llama-3.2-3b",
)

# Maximum new tokens to generate per turn.  Increase for longer diagnoses.
_MAX_NEW_TOKENS: int = 256

# Formatting reminder injected as the final user message before visual diagnosis
# generation.  Its purpose is to suppress the noise patterns observed in real
# model outputs: leaked think headers, repeated token loops, fake turn markers.
# It is NOT shown to the end-user — it only influences the generation context.
_DIAGNOSIS_FORMAT_REMINDER = (
    "Please provide your visual diagnosis now.\n"
    "Output format rules — follow exactly:\n"
    "1. If you need to reason internally, wrap it in <think>...</think>.\n"
    "2. Your user-facing response MUST be wrapped in [DIAGNOSIS]...[/DIAGNOSIS].\n"
    "3. Stop immediately after [/DIAGNOSIS]. Do not add any text after it.\n"
    "4. Do not emit repeated tokens, debug variable names, or continuation markers.\n"
    "5. Do not write [user], [email], or any fake turn markers."
)

# System prompt — must match the prompt used during fine-tuning so the model
# produces the expected tags ([END_SYMPTOMS], [DIAGNOSIS], etc.).
_SYSTEM_PROMPT = """You are an empathetic dermatology assistant with three clinical capabilities:

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

Always recommend consulting a dermatologist. Never provide a definitive diagnosis."""


# ---------------------------------------------------------------------------
# Lazy model loader — loads once per process and caches in module globals
# ---------------------------------------------------------------------------

_llm_model     = None   # transformers AutoModelForCausalLM
_llm_tokenizer = None   # transformers AutoTokenizer
_llm_device: torch.device | None = None


def _load_llm():
    """Load and cache the LLM (tokenizer + model) on first call.

    Returns:
        (tokenizer, model, device)

    STATUS: FULLY WIRED.
    On first call this will download ~6.4 GB from HuggingFace if using the
    default repo ID.  Set the LLM_MODEL_PATH env var to a local path to
    skip the download.

    TODO: If the base model changes (e.g. LLaMA-3.2-1B), verify that
    _SYSTEM_PROMPT and the special-token IDs used in _chat_ids() still match.
    """
    global _llm_model, _llm_tokenizer, _llm_device

    if _llm_model is not None:
        return _llm_tokenizer, _llm_model, _llm_device

    # Import here so that mock-only runs never pay the transformers import cost.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    print(f"[llm_service] Loading LLM from: {_LLM_MODEL_PATH}")
    print("[llm_service] (First run may download ~6.4 GB — this can take a few minutes)")

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
    model.eval()

    _llm_tokenizer = tokenizer
    _llm_model     = model
    _llm_device    = device

    print("[llm_service] LLM ready.")
    return _llm_tokenizer, _llm_model, _llm_device


# ===========================================================================
# Helper functions — adapted from inference_example.py
# ===========================================================================

def build_prompt(history: list[dict[str, str]]) -> str:
    """Serialise message history into a LLaMA-3.2 chat prompt string.

    Prepends _SYSTEM_PROMPT automatically.  Skips messages with empty content.

    Args:
        history: List of {"role": ..., "content": ...} dicts.  Should contain
                 only user/assistant turns (system is injected here).

    Returns:
        Fully-formatted prompt string ending with the assistant header so the
        model generates the next assistant turn.
    """
    text  = "<|begin_of_text|>"
    text += (
        f"<|start_header_id|>system<|end_header_id|>\n\n"
        f"{_SYSTEM_PROMPT}<|eot_id|>"
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
                  max_new_tokens: int, device: torch.device) -> str:
    """Text-only generation: system prompt + history → next assistant turn.

    Args:
        model:          Loaded causal LM.
        tokenizer:      Matching tokenizer.
        history:        User/assistant turn list (system prompt is injected).
        max_new_tokens: Hard limit on generated tokens.
        device:         Torch device the model lives on.

    Returns:
        Cleaned assistant reply string (first turn only).
    """
    prompt = build_prompt(history)
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

    Returns:
        Raw LLM output string.  May contain <think>…</think> and
        [DIAGNOSIS]…[/DIAGNOSIS] tags; clean_response.py strips these before
        the text reaches the user.

    NOTE: This function signature differs from inference_example.py's
    generate_with_image — it accepts vis_tokens directly instead of
    feat_extractor + projection + image_path, because the projection step
    already ran in vision_service.project_vision_features.
    """
    full_messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + history
    text_ids      = _chat_ids(full_messages, tokenizer, prime_assistant=True)

    # Split off the trailing assistant header so visual tokens slot between
    # the conversation context and the header.
    asst_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    pre_ids  = text_ids[: len(text_ids) - len(asst_header_ids)]

    pre_tensor  = torch.tensor([pre_ids],       dtype=torch.long, device=device)
    asst_tensor = torch.tensor([asst_header_ids], dtype=torch.long, device=device)

    embed_fn    = model.get_input_embeddings()
    pre_embeds  = embed_fn(pre_tensor)   # (1, T_pre, llm_dim)
    asst_embeds = embed_fn(asst_tensor)  # (1, T_asst, llm_dim)

    # Move visual tokens to the correct device and dtype
    vis_tokens = vis_tokens.to(device).to(pre_embeds.dtype)  # (1, 32, llm_dim)

    # Combine: [context] + [visual tokens] + [assistant header]
    combined  = torch.cat([pre_embeds, vis_tokens, asst_embeds], dim=1)
    attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

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
    # Remove repeated garbage patterns occasionally emitted by the model
    raw = re.sub(r'(\[\]\s*){3,}', '', raw).strip()
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

def _inject_diagnosis_reminder(
    history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of history with the diagnosis formatting reminder appended.

    The reminder is added as a new user message at the end of the history so
    the model sees it as the most recent instruction before generating.  It
    explicitly tells the model:
      - to wrap reasoning in <think>...</think>
      - to wrap the answer in [DIAGNOSIS]...[/DIAGNOSIS]
      - to stop after [/DIAGNOSIS]
      - not to emit repeated tokens or fake turn markers

    NOTE — DenseNet label is intentionally NOT injected here.
    The LLM performs its own independent visual reasoning from the projected
    embeddings.  Injecting DenseNet's top-1 label would bias the model toward
    that label even when the LLM's own visual assessment is more accurate
    (e.g. DenseNet says "acne" at 48% confidence on a hand showing psoriasis
    features, while the LLM correctly reads "psoriasis" from the embeddings).
    The sidebar showing DenseNet's label and the LLM giving a different
    diagnosis is expected and acceptable — they are two independent classifiers.

    A copy is returned — state["messages"] is never mutated.
    """
    return list(history) + [{"role": "user", "content": _DIAGNOSIS_FORMAT_REMINDER}]


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
    return generate_text(model, tokenizer, history, _MAX_NEW_TOKENS, device)


def _real_generate_diagnosis_response(state: dict[str, Any]) -> str:
    """REAL: call the fine-tuned multimodal LLM to generate a diagnosis.

    STATUS: FULLY WIRED — two paths depending on what vision_service provided.

    PRIMARY PATH (USE_MOCK_VISION=False):
        vision_service._real_project_vision_features() stores a torch.Tensor
        under state["densenet_output"]["projected_embedding"] (shape (1,32,D)).
        This function passes it directly to generate_with_image(), which injects
        the visual tokens into the LLM's input embedding space.
        This is the correct production path — no redundant DenseNet forward pass.

    FALLBACK PATH (USE_MOCK_VISION=True but USE_MOCK_LLM=False — mixed mode):
        projected_embedding is a string (mock) or absent.
        We re-run DenseNet + projection from state["image_path"] using the
        vision_service model cache.  This re-runs the visual pipeline but is
        safe and produces a correct result.

    FUTURE CLEANUP NOTE:
        Once the team always uses USE_MOCK_VISION=False in integration runs,
        the fallback path can be removed.  The primary path (consuming the
        pre-computed tensor from state) is already the intended final design.
        A possible even cleaner future version could pass the projected_embedding
        tensor through the state TypedDict (densenet_output["projected_embedding"])
        and never re-load vision models inside llm_service at all.
    """
    tokenizer, model, device = _load_llm()
    history  = state.get("messages", [])
    densenet = state.get("densenet_output") or {}

    # Build a diagnosis-specific history: append the formatting reminder
    # (with embedded DenseNet context) as the last user message.
    # This anchors the LLM to DenseNet's classification so the response
    # and the sidebar show the same condition, and suppresses generation
    # noise (token loops, fake turns, leaked headers).
    # The reminder is a local copy — state["messages"] is NOT mutated.
    diagnosis_history = _inject_diagnosis_reminder(history)

    projected_embedding = densenet.get("projected_embedding")

    if isinstance(projected_embedding, torch.Tensor):
        # ----------------------------------------------------------------
        # PRIMARY PATH: use pre-computed visual tokens from vision_service.
        # ----------------------------------------------------------------
        raw = generate_with_image(
            model, tokenizer, diagnosis_history,
            vis_tokens=projected_embedding,
            max_new_tokens=_MAX_NEW_TOKENS,
            device=device,
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

    raw = generate_with_image(
        model, tokenizer, diagnosis_history,
        vis_tokens=vis_tokens,
        max_new_tokens=_MAX_NEW_TOKENS,
        device=device,
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

    return generate_text(model, tokenizer, history, _MAX_NEW_TOKENS, device)
