from __future__ import annotations

from langgraph_orchestrator.services.llm_service import generate_diagnosis_response
from langgraph_orchestrator.state import ChatState
from langgraph_orchestrator.utils.response_parser import (
    extract_first_diagnosis_block,
    is_valid_diagnosis_response,
    parse_model_response,
)


def joint_diagnosis(state: ChatState) -> ChatState:
    """Generate a multimodal diagnosis using symptoms + projected image features.

    Orchestration responsibilities (kept here):
    - Validate the model output before committing the diagnosis milestone.
    - Set diagnosis_done = True ONLY when a valid diagnosis is confirmed.
    - Build diagnosis_summary from densenet_output metadata.
    - Advance phase to "follow_up" on success.
    - On malformed output: set uncertainty/fallback flags and leave
      diagnosis_done = False so the graph can recover without getting stuck
      in an incomplete follow_up state.

    Milestone protection rule
    -------------------------
    diagnosis_done = True is irreversible within a session (only reset()
    clears it).  We therefore validate content before setting it:
    - The model output must contain a parseable [DIAGNOSIS] block, OR
      the cleaned full text must be at least 30 meaningful characters.
    - Empty, tag-only, or extremely short outputs are rejected.

    Model behavior (delegated to llm_service):
    - generate_diagnosis_response() produces the raw output.
    - response_parser handles cleaning — clean_response.py strips
      <think> and tag wrappers before the message is shown to the user.
    """
    raw_reply = generate_diagnosis_response(state)

    # ------------------------------------------------------------------
    # Validate the diagnosis before locking in the milestone.
    # extract_first_diagnosis_block returns the inner text of the first
    # [DIAGNOSIS]…[/DIAGNOSIS] block, or None if no block is present.
    # is_valid_diagnosis_response checks that the content is substantive.
    # ------------------------------------------------------------------
    parsed_content = extract_first_diagnosis_block(raw_reply)

    # Fall back to the full cleaned text if no block tag was found
    if parsed_content is None:
        parsed_content = parse_model_response(raw_reply)

    diagnosis_valid = is_valid_diagnosis_response(parsed_content)

    messages = state["messages"] + [{"role": "assistant", "content": raw_reply}]

    if not diagnosis_valid:
        # Malformed or empty output — do NOT lock in diagnosis_done.
        # Set flags so the graph recovers via the fallback path.
        print(
            "[joint_diagnosis] WARNING: output failed validation "
            f"(parsed content: {repr(parsed_content[:60]) if parsed_content else 'None'}). "
            "Setting uncertainty + fallback flags."
        )
        return {
            **state,
            "messages":          messages,
            "diagnosis_done":    False,   # milestone NOT committed
            "uncertainty_flag":  True,
            "fallback_triggered": True,
            # phase stays at "joint_diagnosis" so the next router call
            # re-evaluates rather than sliding to "follow_up" incorrectly
        }

    # ------------------------------------------------------------------
    # Valid diagnosis — commit the milestone.
    # After this point, routing.py guarantees no backward transition.
    # ------------------------------------------------------------------
    densenet    = state.get("densenet_output") or {}
    label       = densenet.get("label", "unknown")
    confidence  = densenet.get("confidence", 0.0)
    is_cautious = bool(state.get("uncertainty_flag"))

    # Build a summary that reflects whether the diagnosis was made confidently
    # or with hedged language due to medium image confidence.
    if is_cautious:
        summary = (
            f"Preliminary indication: {label} (confidence {confidence:.0%}, "
            "low-confidence — treat as rough indication only)"
        )
    else:
        summary = f"Preliminary diagnosis: {label} (confidence {confidence:.0%})"

    print(
        f"[joint_diagnosis] Milestone committed — {summary} "
        f"({'cautious' if is_cautious else 'confident'})"
    )

    return {
        **state,
        "messages":           messages,
        "diagnosis_done":     True,          # irreversible until session reset
        "diagnosis_summary":  summary,
        "phase":              "follow_up",
        # Preserve uncertainty_flag for cautious diagnoses so downstream nodes
        # (e.g. llm_followup) can qualify their answers accordingly.
        # fallback_triggered is always cleared — we committed a diagnosis, so
        # there is nothing left to fall back from.
        "uncertainty_flag":   is_cautious,   # True → cautious; False → confident
        "fallback_triggered": False,
    }
