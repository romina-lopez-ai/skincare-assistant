from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def joint_diagnosis(state: ChatState) -> ChatState:
    """Generate a multimodal diagnosis using both symptom history and image features.

    Responsibilities:
    - Build a combined prompt that includes:
        * the collected symptom slots,
        * the projected image embedding / DenseNet result,
        * the full conversation history for context.
    - Call the fine-tuned multimodal LLM.
    - Expect the model output to contain:
        <think>...</think>          — internal chain-of-thought (stripped later)
        [DIAGNOSIS]...[/DIAGNOSIS]  — the diagnosis to show the user
    - Store the raw LLM output in messages (clean_response strips <think> later).
    - Set diagnosis_done = True and write a short summary to diagnosis_summary.
    - Set uncertainty_flag = True if the model expresses low confidence or if
      DenseNet confidence was already flagged as uncertain.

    Key rules:
    - LLM is ALWAYS the final response generator; DenseNet result is an input.
    - This node is only reached when BOTH symptoms and image are ready.

    Real implementation will:
    - Serialise symptom_slots into a structured prompt section.
    - Attach the projected embedding as a special token / prefix.
    - Parse [DIAGNOSIS]...[/DIAGNOSIS] from the raw response.

    Returns:
        Updated state with the raw assistant message appended, diagnosis_done
        set to True, and diagnosis_summary populated.
    """
    slots = state["symptom_slots"]
    densenet = state.get("densenet_output") or {}

    # --- Placeholder LLM call ---
    mock_raw = (
        "<think>Combining symptom profile with DenseNet result "
        f"({densenet.get('label', 'unknown')}, "
        f"conf={densenet.get('confidence', 0.0):.2f}). "
        "Patient reports dry, itchy patches. Consistent with eczema.</think>\n"
        "[DIAGNOSIS]Based on your symptoms and the image, this appears to be "
        "eczema (atopic dermatitis). The affected area shows characteristic dry, "
        "scaly patches. I recommend consulting a dermatologist for a confirmed "
        "diagnosis and personalised treatment plan.[/DIAGNOSIS]"
    )

    summary = f"Preliminary diagnosis: {densenet.get('label', 'unknown')} " \
              f"(confidence {densenet.get('confidence', 0.0):.0%})"

    messages = state["messages"] + [{"role": "assistant", "content": mock_raw}]

    print(f"[joint_diagnosis] {summary}")

    return {
        **state,
        "messages": messages,
        "diagnosis_done": True,
        "diagnosis_summary": summary,
        "phase": "follow_up",
    }
