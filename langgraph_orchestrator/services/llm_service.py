"""llm_service.py — LLM backend abstraction for the dermatology chatbot.

All three functions are **mocks** that reproduce the placeholder text that
previously lived inside the node files.

Teammate integration guide
--------------------------
Replace each function body with a real call to your fine-tuned LLM + LoRA
adapters.  The function signatures and return types must not change so the
node files keep working without modification.

Expected real implementation pattern (per function):
    1. Build a prompt from the state fields shown in each docstring.
    2. Call the model (e.g. model.generate(...) or pipeline(...)).
    3. Return the raw model output as a string.
       - generate_symptom_response  → plain question text
       - generate_diagnosis_response → raw text that may contain
         <think>...</think> and [DIAGNOSIS]...[/DIAGNOSIS] tags
         (clean_response.py strips these before showing to the user)
       - generate_followup_response → plain answer text
"""

from __future__ import annotations

from typing import Any

from langgraph_orchestrator.services.symptom_service import REQUIRED_SLOTS


def generate_symptom_response(state: dict[str, Any]) -> str:
    """Generate the next symptom-collection question.

    MOCK: asks about the first unfilled slot.

    Real implementation should:
    - Format the current symptom_slots and conversation history into a prompt.
    - Call the fine-tuned LLM to decide what to ask next.
    - Detect [END_SYMPTOMS] in the model output to signal completion.

    Args:
        state: Current ChatState dict.

    Returns:
        Assistant reply string.  When all slots are filled the real model
        should include [END_SYMPTOMS] in its output.
    """
    # IMPORTANT: this function must only look at symptom_slots.
    # It must never branch on image_uploaded, image_acknowledged, or phase.
    # Image acknowledgement is the sole responsibility of await_or_store_image.
    slots = state.get("symptom_slots", {})
    missing = [k for k in REQUIRED_SLOTS if not slots.get(k)]

    if missing:
        return f"Could you tell me more about the {missing[0]} of your skin concern?"
    return "Thank you. I have all the information I need. [END_SYMPTOMS]"


def generate_diagnosis_response(state: dict[str, Any]) -> str:
    """Generate a multimodal diagnosis using symptoms + image features.

    MOCK: returns a hardcoded <think> + [DIAGNOSIS] block.

    Real implementation should:
    - Serialise state["symptom_slots"] into a structured prompt section.
    - Attach state["densenet_output"]["projected_embedding"] as visual context.
    - Call the fine-tuned multimodal LLM.
    - Return the raw output including <think>...</think> and
      [DIAGNOSIS]...[/DIAGNOSIS] tags.
      (clean_response.py handles stripping before display.)

    Args:
        state: Current ChatState dict (densenet_output must be populated).

    Returns:
        Raw LLM output string with chain-of-thought and diagnosis tags.
    """
    densenet = state.get("densenet_output") or {}
    label      = densenet.get("label", "unknown")
    confidence = densenet.get("confidence", 0.0)

    return (
        f"<think>Combining symptom profile with DenseNet result "
        f"({label}, conf={confidence:.2f}). "
        "Patient reports dry, itchy patches. Consistent with eczema.</think>\n"
        "[DIAGNOSIS]Based on your symptoms and the image, this appears to be "
        "eczema (atopic dermatitis). The affected area shows characteristic dry, "
        "scaly patches. I recommend consulting a dermatologist for a confirmed "
        "diagnosis and personalised treatment plan.[/DIAGNOSIS]"
    )


def generate_followup_response(state: dict[str, Any]) -> str:
    """Generate a follow-up answer, optionally grounded in RAG context.

    MOCK: returns a hardcoded placeholder answer.

    Real implementation should:
    - If state["rag_context"] is set, inject it into the prompt and instruct
      the LLM to answer using only that context.
    - If state["rag_context"] is None, answer from the LLM's own knowledge
      using the conversation history.

    Args:
        state: Current ChatState dict.  May contain rag_context.

    Returns:
        Plain assistant reply string (no special tags required).
    """
    return (
        "Based on the available medical information: "
        "Eczema is typically managed with regular moisturising using emollients and, "
        "during flares, topical corticosteroids. Identifying and avoiding personal "
        "triggers (e.g. soaps, fabrics, allergens) is also recommended."
    )
