from __future__ import annotations

from langgraph_orchestrator.services.llm_service import generate_diagnosis_response
from langgraph_orchestrator.state import ChatState


def joint_diagnosis(state: ChatState) -> ChatState:
    """Generate a multimodal diagnosis using symptoms + projected image features.

    Orchestration responsibilities (kept here):
    - Set diagnosis_done = True.
    - Build diagnosis_summary from densenet_output metadata.
    - Advance phase to "follow_up".
    - Append raw model output to messages.

    Model behavior (delegated to llm_service):
    - Generating the diagnosis text is handled by generate_diagnosis_response().
      The raw output may contain <think> and [DIAGNOSIS] tags;
      clean_response.py strips them before display.
    """
    raw_reply = generate_diagnosis_response(state)

    densenet  = state.get("densenet_output") or {}
    label     = densenet.get("label", "unknown")
    confidence = densenet.get("confidence", 0.0)
    summary   = f"Preliminary diagnosis: {label} (confidence {confidence:.0%})"

    messages = state["messages"] + [{"role": "assistant", "content": raw_reply}]

    print(f"[joint_diagnosis] {summary}")

    return {
        **state,
        "messages":          messages,
        "diagnosis_done":    True,
        "diagnosis_summary": summary,
        "phase":             "follow_up",
    }
