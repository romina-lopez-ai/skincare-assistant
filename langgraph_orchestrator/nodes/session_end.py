from __future__ import annotations

from langgraph_orchestrator.state import ChatState, create_initial_state


def session_end(state: ChatState) -> ChatState:
    """Cleanly terminate the current session and reset state.

    Responsibilities:
    - Optionally persist the session (messages, diagnosis summary) to a log or
      database before resetting.
    - Generate a polite closing message and append it to messages.
    - Reset the state to a clean initial state, preserving only the session_id
      so the same user can start a fresh conversation without re-authentication.

    Triggered by:
    - The user sending an [END_SESSION] signal.
    - An inactivity timeout (handled externally; this node resets the state).
    - An unrecoverable fallback error.

    Real implementation will:
    - Flush conversation history to persistent storage (database / file).
    - Optionally send a summary email or notification.
    - Invalidate any cached embeddings associated with this session.

    Returns:
        A freshly initialised ChatState for the same session_id, with a
        closing assistant message as the first entry in messages.
    """
    closing_message = {
        "role": "assistant",
        "content": (
            "[PLACEHOLDER] Thank you for using the dermatology assistant. "
            "Your session has ended. Feel free to start a new conversation "
            "if you have further questions. Take care!"
        ),
    }

    print(f"[session_end] Closing session={state['session_id']}  "
          f"turns={len(state['messages'])}  "
          f"diagnosis_done={state['diagnosis_done']}")

    # Reset to a clean state, keeping the same session_id
    fresh_state = create_initial_state(state["session_id"])
    fresh_state = {**fresh_state, "messages": [closing_message], "phase": "end_session"}

    return fresh_state
