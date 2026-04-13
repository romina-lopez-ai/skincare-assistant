from __future__ import annotations

from langgraph_orchestrator.state import ChatState
from langgraph_orchestrator.utils.response_parser import parse_model_response


def clean_response(state: ChatState) -> ChatState:
    """Strip internal model tags from the latest assistant message.

    Orchestration responsibilities (kept here):
    - Locate the last assistant message in state["messages"].
    - Replace its raw content with the cleaned, user-facing version.
    - Leave all earlier messages untouched (preserves the reasoning trace).

    Parsing behavior (delegated to response_parser):
    - Removes <think>…</think> blocks.
    - Extracts the first [DIAGNOSIS]…[/DIAGNOSIS] block; discards duplicates.
    - Strips [END_SYMPTOMS], [NOT_AVAILABLE], and similar control tags.
    - Removes trailing bracket/paren artifacts.
    - Collapses excessive blank lines.
    """
    messages = state["messages"]

    # Find the last assistant message
    last_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_idx = i
            break

    if last_idx is None:
        return state  # Nothing to clean

    raw = messages[last_idx]["content"]
    cleaned = parse_model_response(raw)

    print(f"[clean_response] {len(raw)} chars → {len(cleaned)} chars")

    updated_messages = list(messages)
    updated_messages[last_idx] = {**messages[last_idx], "content": cleaned}

    return {**state, "messages": updated_messages}
