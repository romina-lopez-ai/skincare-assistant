"""service.py — LangGraph session management and workflow integration layer.

Responsibilities
----------------
- Maintain an in-memory session store (session_id → ChatState).
- Compile the LangGraph graph once (lazy singleton).
- Expose simple functions that the FastAPI endpoints call.
- Infer which nodes ran each turn for the debug panel.

Integration points
------------------
When real models are ready, only the individual node files under
langgraph_orchestrator/nodes/ need updating.  This service layer and the
FastAPI layer do not need to change.
"""

from __future__ import annotations

import uuid
from typing import Any

from langgraph_orchestrator.build_graph import build_graph
from langgraph_orchestrator.state import ChatState, create_initial_state

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------

_sessions: dict[str, dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Lazy graph singleton — compiled once on first use
# ---------------------------------------------------------------------------

_graph = None


def _get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def start_session(session_id: str | None = None) -> dict[str, Any]:
    """Create a new session and return its initial state."""
    sid = session_id or str(uuid.uuid4())
    state = dict(create_initial_state(sid))
    _sessions[sid] = state
    return state


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return the current state for session_id, or None if not found."""
    return _sessions.get(session_id)


def reset_session(session_id: str) -> dict[str, Any]:
    """Reset a session to its initial state, keeping the same session_id."""
    state = dict(create_initial_state(session_id))
    _sessions[session_id] = state
    return state


# ---------------------------------------------------------------------------
# Turn execution
# ---------------------------------------------------------------------------

def run_turn(session_id: str, user_message: str) -> dict[str, Any]:
    """Append a user message and run one full LangGraph turn.

    Args:
        session_id:   Active session identifier.
        user_message: The user's raw text input.

    Returns:
        A dict with keys: response, phase, symptoms_complete, image_uploaded,
        diagnosis_done, rag_context_used, nodes_visited.
    """
    state = _sessions.get(session_id)
    if state is None:
        state = start_session(session_id)

    # Snapshot state before the turn for path inference
    before = dict(state)

    # Append user message
    state = {
        **state,
        "messages": state["messages"] + [{"role": "user", "content": user_message}],
    }

    graph = _get_graph()
    final_state: dict[str, Any] = dict(state)

    try:
        result = graph.invoke(state)
        final_state = dict(result)
    except Exception as exc:
        # Fallback: keep state changes from before the error and inject a
        # placeholder response so the UI doesn't break.
        error_msg = (
            f"[WORKFLOW ERROR — placeholder response]\n"
            f"The graph raised an exception: {exc}\n"
            "Check the server console for the full traceback."
        )
        final_state["messages"] = final_state["messages"] + [
            {"role": "assistant", "content": error_msg}
        ]
        print(f"[service.run_turn] Graph error: {exc}")

    _sessions[session_id] = final_state

    # Extract the last assistant message
    last_response = ""
    for msg in reversed(final_state.get("messages", [])):
        if msg.get("role") == "assistant":
            last_response = msg["content"]
            break

    nodes_visited = _infer_path(before, final_state)

    return {
        "response":         last_response,
        "phase":            final_state.get("phase", "unknown"),
        "symptoms_complete": final_state.get("symptoms_complete", False),
        "image_uploaded":   final_state.get("image_uploaded", False),
        "diagnosis_done":   final_state.get("diagnosis_done", False),
        "rag_context_used": final_state.get("rag_context") is not None,
        "nodes_visited":    nodes_visited,
    }


# ---------------------------------------------------------------------------
# Image upload
# ---------------------------------------------------------------------------

def store_image(session_id: str, file_path: str) -> dict[str, Any]:
    """Store an uploaded image path in session state silently.

    This function does NOT generate any assistant message.
    The single acknowledgement is sent by the await_or_store_image node on the
    next user message turn — having two sources of acknowledgement caused the
    duplicate-message bug.

    The vision pipeline fires automatically once symptoms_complete and
    image_uploaded are both True (handled by input_router → run_vision).

    Args:
        session_id: Active session identifier.
        file_path:  Absolute or relative path to the saved image file.

    Returns:
        response is always "" — the frontend must not add an assistant bubble.
    """
    state = _sessions.get(session_id)
    if state is None:
        state = start_session(session_id)

    # Update image flags only — do NOT touch messages
    state = {
        **state,
        "image_uploaded":    True,
        "image_path":        file_path,
        "image_acknowledged": False,   # reset so await_or_store_image sends exactly one ack
    }
    _sessions[session_id] = state

    return {
        "response":          "",       # empty — frontend shows a toast, not a bubble
        "phase":             state.get("phase", "unknown"),
        "symptoms_complete": state.get("symptoms_complete", False),
        "image_uploaded":    True,
        "diagnosis_done":    state.get("diagnosis_done", False),
        "rag_context_used":  False,
        "nodes_visited":     ["[image_stored]"],
    }


# ---------------------------------------------------------------------------
# State summary (for GET /session/{id}/state)
# ---------------------------------------------------------------------------

def get_state_summary(session_id: str) -> dict[str, Any] | None:
    """Return a structured state snapshot for the debug panel."""
    state = _sessions.get(session_id)
    if state is None:
        return None

    return {
        "session_id":       state.get("session_id", session_id),
        "phase":            state.get("phase", "unknown"),
        "symptom_slots":    state.get("symptom_slots", {}),
        "symptom_turns":    state.get("symptom_turns", 0),
        "symptoms_complete": state.get("symptoms_complete", False),
        "image_uploaded":   state.get("image_uploaded", False),
        "image_path":       state.get("image_path"),
        "image_processed":  state.get("image_processed", False),
        "densenet_output":  state.get("densenet_output"),
        "diagnosis_done":   state.get("diagnosis_done", False),
        "diagnosis_summary": state.get("diagnosis_summary"),
        "rag_context":      state.get("rag_context"),
        "fallback_triggered": state.get("fallback_triggered", False),
        "uncertainty_flag": state.get("uncertainty_flag", False),
        "message_count":    len(state.get("messages", [])),
        "messages":         state.get("messages", []),
    }


# ---------------------------------------------------------------------------
# Path inference helper
# ---------------------------------------------------------------------------

def _infer_path(before: dict[str, Any], after: dict[str, Any]) -> list[str]:
    """Infer the sequence of nodes visited by comparing before/after state.

    This is a heuristic for the debug panel.  It is not guaranteed to be
    exact when multiple paths share the same state changes, but it is correct
    for all standard flows defined in the plan.
    """
    path: list[str] = ["session_entry", "input_router"]

    diagnosis_just_done = (
        after.get("diagnosis_done") and not before.get("diagnosis_done")
    )
    vision_just_ran = (
        after.get("image_processed") and not before.get("image_processed")
    )
    was_follow_up = before.get("diagnosis_done", False)
    symptoms_just_done = (
        after.get("symptoms_complete") and not before.get("symptoms_complete")
    )

    if vision_just_ran:
        path += ["run_vision", "project_image_features"]

    if diagnosis_just_done:
        path += ["joint_diagnosis", "clean_response"]

    elif was_follow_up:
        path.append("followup_router")
        # Detect RAG path: rag_context was temporarily populated then cleared,
        # OR the assistant response contains the RAG placeholder marker.
        last_assistant = ""
        for msg in reversed(after.get("messages", [])):
            if msg.get("role") == "assistant":
                last_assistant = msg.get("content", "")
                break
        if "RAG answer" in last_assistant or "rag_answer" in last_assistant.lower():
            path += ["retrieve_context", "rag_answer"]
        else:
            path.append("llm_followup")

    elif (
        # await_or_store_image ran for an early upload:
        # image_acknowledged flipped False → True this turn
        after.get("image_uploaded")
        and after.get("image_acknowledged")
        and not before.get("image_acknowledged", False)
        and not after.get("symptoms_complete")
    ):
        path.append("await_or_store_image")

    elif after.get("symptoms_complete") and not after.get("image_uploaded"):
        # await_or_store_image ran to prompt the user for an image
        path.append("await_or_store_image")

    elif symptoms_just_done or not after.get("symptoms_complete"):
        path.append("collect_symptoms")

    if after.get("phase") == "end_session":
        path.append("session_end")

    return path
