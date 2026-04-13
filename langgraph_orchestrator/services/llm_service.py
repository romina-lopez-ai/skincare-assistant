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
2. Fill in each _real_* function below with your model calls.
3. The public functions and all node files stay unchanged.

The _mock_* functions are preserved and continue to work exactly as before.
"""

from __future__ import annotations

from typing import Any

from langgraph_orchestrator.config import USE_MOCK_LLM
from langgraph_orchestrator.services.symptom_service import REQUIRED_SLOTS


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
    slots = state.get("symptom_slots", {})
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
# Real implementations — replace these bodies with actual model calls
# ===========================================================================

def _real_generate_symptom_response(state: dict[str, Any]) -> str:
    """REAL: call the fine-tuned LLM to generate the next symptom question.

    Implementation guide:
    1. Serialise state["symptom_slots"] and state["messages"] into a prompt.
    2. Call the model (e.g. pipeline("text-generation", ...)(prompt)).
    3. Return the raw model output string.
       - Include [END_SYMPTOMS] in the output when all slots are satisfied.
    """
    raise NotImplementedError(
        "_real_generate_symptom_response: replace this with your LLM call. "
        "Set USE_MOCK_LLM = True in config.py to use the mock instead."
    )


def _real_generate_diagnosis_response(state: dict[str, Any]) -> str:
    """REAL: call the fine-tuned multimodal LLM to generate a diagnosis.

    Implementation guide:
    1. Serialise state["symptom_slots"] into a structured prompt section.
    2. Attach state["densenet_output"]["projected_embedding"] as visual context.
    3. Pass state["uncertainty_flag"] so the model can adjust its tone.
    4. Call the model and return the raw output including <think> and
       [DIAGNOSIS] tags.  clean_response.py handles stripping before display.
    """
    raise NotImplementedError(
        "_real_generate_diagnosis_response: replace this with your LLM call. "
        "Set USE_MOCK_LLM = True in config.py to use the mock instead."
    )


def _real_generate_followup_response(state: dict[str, Any]) -> str:
    """REAL: call the fine-tuned LLM to answer a follow-up question.

    Implementation guide:
    - If state["rag_context"] is set, inject it into the prompt and instruct
      the model to answer using only that context.
    - If state["rag_context"] is None, answer from the model's own knowledge
      using the full conversation history in state["messages"].
    """
    raise NotImplementedError(
        "_real_generate_followup_response: replace this with your LLM call. "
        "Set USE_MOCK_LLM = True in config.py to use the mock instead."
    )
