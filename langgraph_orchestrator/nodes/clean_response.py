from __future__ import annotations

import re

from langgraph_orchestrator.state import ChatState

# Matches the <think>...</think> block produced by chain-of-thought models.
_THINK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

# Matches [DIAGNOSIS]...[/DIAGNOSIS] wrapper — strips the tags, keeps content.
_DIAGNOSIS_PATTERN = re.compile(r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", re.DOTALL)


def clean_response(state: ChatState) -> ChatState:
    """Strip internal reasoning tags from the latest assistant message.

    Responsibilities:
    - Find the most recent assistant message in state["messages"].
    - Remove any <think>...</think> block (internal chain-of-thought).
    - Unwrap [DIAGNOSIS]...[/DIAGNOSIS] tags, keeping only the inner text.
    - Replace the raw message with the cleaned version so the user never sees
      the model's internal reasoning.

    Key rule:
        Only the last assistant message is cleaned; earlier messages are left
        unchanged so the full reasoning trace is preserved for debugging.

    Real implementation will:
    - Apply the same regex pipeline; no further changes expected unless
      additional internal tags are introduced (e.g. [PLAN], [UNCERTAINTY]).

    Returns:
        Updated state with the last assistant message replaced by its
        cleaned, user-facing version.
    """
    messages = state["messages"]

    # Find the last assistant message index
    last_assistant_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            last_assistant_idx = i
            break

    if last_assistant_idx is None:
        # Nothing to clean
        return state

    raw_content = messages[last_assistant_idx]["content"]

    # 1. Remove <think>...</think>
    cleaned = _THINK_PATTERN.sub("", raw_content)

    # 2. Unwrap [DIAGNOSIS]...[/DIAGNOSIS]
    match = _DIAGNOSIS_PATTERN.search(cleaned)
    if match:
        cleaned = match.group(1)

    cleaned = cleaned.strip()

    updated_messages = list(messages)
    updated_messages[last_assistant_idx] = {
        **messages[last_assistant_idx],
        "content": cleaned,
    }

    print(f"[clean_response] Cleaned message (length {len(raw_content)} → {len(cleaned)})")

    return {**state, "messages": updated_messages}
