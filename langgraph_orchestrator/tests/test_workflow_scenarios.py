"""test_workflow_scenarios.py — Scenario-based workflow tests for the LangGraph pipeline.

What these tests cover
----------------------
Each test validates the orchestration logic (routing, state transitions,
flag management) using the mock service backends.  They do NOT check exact
LLM wording — they check that the right nodes ran and the right state
fields changed.

Running the tests
-----------------
From the skincare_model/ directory:

    pytest langgraph_orchestrator/tests/ -v

All tests use USE_MOCK_LLM / USE_MOCK_VISION / USE_MOCK_RAG = True (the
defaults in config.py) so no real models are required.

Test index
----------
1.  test_symptom_collection_fills_slots_and_completes
2.  test_early_image_acknowledged_exactly_once
3.  test_completion_by_slot_threshold_not_end_signal
4.  test_max_turn_fallback_forces_completion
5.  test_full_diagnosis_path_commits_milestone
6.  test_low_confidence_vision_triggers_reupload_not_symptom_loop
7.  test_diagnosis_milestone_not_reversed_by_stale_tags
8a. test_parse_strip_think_blocks
8b. test_parse_extract_first_diagnosis_block
8c. test_parse_remove_noise_tags
8d. test_parse_strip_trailing_artifacts
8e. test_parse_full_pipeline
9a. test_followup_knowledge_question_routes_to_rag
9b. test_followup_conversational_question_routes_to_llm
10. test_reset_session_returns_clean_state
"""

from __future__ import annotations

import pytest

from langgraph_orchestrator.tests.helpers import (
    ALL_SLOTS_MESSAGE,
    FOUR_SLOTS_MESSAGE,
    all_assistant_messages,
    last_assistant_message,
    make_state,
    run_turn,
)


# ===========================================================================
# Scenario 1 — Normal symptom collection flow
# ===========================================================================

def test_symptom_collection_fills_slots_and_completes(graph):
    """Symptom slots fill across turns; symptoms_complete becomes True;
    phase advances to await_image once complete."""

    state = make_state()

    # Turn 1 — no slot keywords, session initialises
    state = run_turn(graph, state, "Hi, I think I have a skin problem.")
    assert state["phase"] == "symptom_collection"
    assert state["symptoms_complete"] is False
    assert last_assistant_message(state) != ""  # model asked a question

    # Turn 2 — message fills all 5 slots at once
    state = run_turn(graph, state, ALL_SLOTS_MESSAGE)

    assert state["symptoms_complete"] is True
    assert state["symptom_slots"]["location"]   is not None
    assert state["symptom_slots"]["duration"]   is not None
    assert state["symptom_slots"]["appearance"] is not None
    assert state["symptom_slots"]["severity"]   is not None
    assert state["symptom_slots"]["triggers"]   is not None
    # collect_symptoms sets phase="await_image" on completion
    assert state["phase"] == "await_image"


# ===========================================================================
# Scenario 2 — Early image upload
# ===========================================================================

def test_early_image_acknowledged_exactly_once(graph):
    """When an image is uploaded before symptoms are complete:
    - The workflow sends exactly one acknowledgement message.
    - image_acknowledged is set to True.
    - Subsequent turns continue symptom collection (no second ack).
    """

    # Simulate: image arrived before symptoms are done
    state = make_state(image_uploaded=True, image_path="/fake/early.jpg")
    # Default: image_acknowledged=False, symptoms_complete=False

    # Turn 1 — image acknowledgement should fire (Case B in await_or_store_image)
    state = run_turn(graph, state, "Here is a photo of my rash.")
    assert state["image_acknowledged"] is True
    assert state["symptoms_complete"] is False

    msgs_t1 = all_assistant_messages(state)
    assert len(msgs_t1) == 1
    ack = msgs_t1[0].lower()
    # Acknowledgement message should mention image/photo
    assert "image" in ack or "photo" in ack or "saved" in ack

    # Turn 2 — symptom collection continues; NO second acknowledgement
    state = run_turn(graph, state, "I have some pain.")
    msgs_t2 = all_assistant_messages(state)
    # Exactly one new message added (the symptom question), not another ack
    assert len(msgs_t2) == 2
    # The second message must differ from the first (it's a symptom question)
    assert msgs_t2[1] != msgs_t2[0]
    # image_acknowledged must stay True (not re-fired)
    assert state["image_acknowledged"] is True


# ===========================================================================
# Scenario 3 — Slot-threshold completion (no [END_SYMPTOMS] needed)
# ===========================================================================

def test_completion_by_slot_threshold_not_end_signal(graph):
    """When 4 slots are filled, symptoms_complete becomes True even if the
    mock LLM does not emit [END_SYMPTOMS].

    The mock only emits [END_SYMPTOMS] when ALL 5 slots are filled.
    With exactly 4 filled (triggers missing), it asks about triggers — no
    signal — but is_symptoms_complete() fires on slot count alone.
    """

    pre_filled = {
        "location":   "arm",
        "duration":   "two weeks",
        "appearance": "red rash",
        "severity":   "moderate",
        "triggers":   None,   # intentionally left empty
    }
    state = make_state(symptom_slots=pre_filled)

    # Completely neutral message — no slot keywords → mock asks about triggers,
    # no [END_SYMPTOMS] emitted.  4 slots already filled >= MIN_SLOTS_REQUIRED.
    state = run_turn(graph, state, "Okay.")

    assert state["symptoms_complete"] is True
    # Confirm the response does NOT contain [END_SYMPTOMS]
    # (proves completion was driven by slot count, not the model signal)
    assert "[END_SYMPTOMS]" not in last_assistant_message(state)


# ===========================================================================
# Scenario 4 — Max-turn fallback
# ===========================================================================

def test_max_turn_fallback_forces_completion(graph):
    """When symptom_turns reaches MAX_SYMPTOM_TURNS the workflow forces
    symptoms_complete = True regardless of how many slots are filled,
    preventing an infinite symptom-collection loop.
    """

    from langgraph_orchestrator.services.symptom_service import MAX_SYMPTOM_TURNS

    # One turn away from the cap, only one slot filled
    state = make_state(
        symptom_turns=MAX_SYMPTOM_TURNS - 1,
        symptom_slots={
            "location":   "back",
            "duration":   None,
            "appearance": None,
            "severity":   None,
            "triggers":   None,
        },
    )

    # This turn pushes symptom_turns to MAX_SYMPTOM_TURNS
    state = run_turn(graph, state, "I really don't know what else to tell you.")

    assert state["symptoms_complete"] is True
    assert state["symptom_turns"] == MAX_SYMPTOM_TURNS


# ===========================================================================
# Scenario 5 — Full diagnosis path
# ===========================================================================

def test_full_diagnosis_path_commits_milestone(graph):
    """With symptoms complete and image uploaded, the graph runs the full
    vision + diagnosis pipeline in a single turn:
      input_router → run_vision → project_image_features
                  → joint_diagnosis → clean_response

    After the turn: diagnosis_done=True, phase="follow_up", tags stripped.
    """

    state = make_state(
        symptoms_complete=True,
        image_uploaded=True,
        image_path="/fake/image.jpg",
        image_processed=False,
        phase="await_image",
    )

    state = run_turn(graph, state, "Please analyse my image.")

    # Milestone committed
    assert state["diagnosis_done"] is True
    assert state["phase"] == "follow_up"
    assert state["image_processed"] is True
    assert state["diagnosis_summary"] is not None

    # Mock returns confidence=0.82 → Tier 1 (confident)
    assert state["uncertainty_flag"] is False
    assert state["fallback_triggered"] is False

    response = last_assistant_message(state)
    assert response != ""
    # clean_response must have stripped internal tags
    assert "<think>" not in response
    assert "[DIAGNOSIS]" not in response


# ===========================================================================
# Scenario 6 — Low-confidence image fallback
# ===========================================================================

def test_low_confidence_vision_triggers_reupload_not_symptom_loop(graph, monkeypatch):
    """When DenseNet confidence is critically low (< 0.40):
    - Diagnosis is NOT committed.
    - The workflow sends a targeted re-upload prompt (not a symptom question).
    - image_uploaded and image_processed are reset for clean re-upload.
    - fallback_triggered is cleared (no routing lock).
    - Already-collected symptom slots are fully preserved.
    """

    import langgraph_orchestrator.services.vision_service as vs

    monkeypatch.setattr(
        vs,
        "_mock_run_vision_inference",
        lambda _path: {"label": "unknown", "confidence": 0.25, "morphology": "unclear"},
    )

    pre_filled = {
        "location":   "arm",
        "duration":   "two weeks",
        "appearance": "red rash",
        "severity":   "moderate",
        "triggers":   "soap",
    }
    state = make_state(
        symptoms_complete=True,
        image_uploaded=True,
        image_path="/fake/image.jpg",
        image_processed=False,
        symptom_slots=pre_filled,
        phase="await_image",
    )

    state = run_turn(graph, state, "Please analyse my image.")

    # Diagnosis must NOT be committed
    assert state["diagnosis_done"] is False

    # Image flags reset — user must re-upload; flags are clean for next attempt
    assert state["image_uploaded"] is False
    assert state["image_processed"] is False
    assert state["fallback_triggered"] is False

    # Symptom slots fully preserved — no re-collection needed
    assert state["symptom_slots"]["location"]   == "arm"
    assert state["symptom_slots"]["duration"]   is not None
    assert state["symptom_slots"]["appearance"] is not None
    assert state["symptom_slots"]["severity"]   is not None
    assert state["symptom_slots"]["triggers"]   is not None

    # Response must be a re-upload prompt, NOT a symptom question
    response = last_assistant_message(state).lower()
    assert "clearer" in response or "upload" in response
    assert "Could you tell me more about" not in last_assistant_message(state)


# ===========================================================================
# Scenario 7 — Diagnosis milestone protection
# ===========================================================================

def test_diagnosis_milestone_not_reversed_by_stale_tags(graph):
    """Once diagnosis_done=True, the workflow must not route backward into
    symptom collection regardless of what tags appear in the user message.

    Stray [END_SYMPTOMS] and [NOT_AVAILABLE] in a follow-up message must
    not cause regression.
    """

    state = make_state(
        diagnosis_done=True,
        phase="follow_up",
        diagnosis_summary="Preliminary diagnosis: eczema (82%)",
        symptoms_complete=True,
        image_uploaded=True,
        image_processed=True,
    )

    # Noisy message with stale internal tags
    state = run_turn(
        graph, state,
        "Thanks for the diagnosis. [END_SYMPTOMS] [NOT_AVAILABLE] What now?",
    )

    # Milestone preserved
    assert state["diagnosis_done"] is True
    assert state["phase"] == "follow_up"

    # Response must be a follow-up answer, not a symptom question
    response = last_assistant_message(state)
    assert "Could you tell me more about" not in response


# ===========================================================================
# Scenario 8 — Response parsing / cleanup (unit tests, no graph needed)
# ===========================================================================

def test_parse_strip_think_blocks():
    """<think>…</think> blocks are removed; surrounding text is kept."""
    from langgraph_orchestrator.utils.response_parser import strip_think_blocks

    assert strip_think_blocks("<think>reasoning</think>Clean answer.") == "Clean answer."
    assert strip_think_blocks("<think>a</think>Mid<think>b</think>End") == "MidEnd"
    # Multi-line block
    assert strip_think_blocks("<think>\nline1\nline2\n</think>Result") == "Result"
    # No block — unchanged
    assert strip_think_blocks("No tags here.") == "No tags here."


def test_parse_extract_first_diagnosis_block():
    """Only the first [DIAGNOSIS]…[/DIAGNOSIS] block is returned; the rest discarded."""
    from langgraph_orchestrator.utils.response_parser import extract_first_diagnosis_block

    assert extract_first_diagnosis_block("[DIAGNOSIS]You have eczema.[/DIAGNOSIS]") \
        == "You have eczema."
    # Only first block kept
    result = extract_first_diagnosis_block(
        "[DIAGNOSIS]First.[/DIAGNOSIS] extra [DIAGNOSIS]Second.[/DIAGNOSIS]"
    )
    assert result == "First."
    assert "Second" not in result
    # Returns None when no block present
    assert extract_first_diagnosis_block("No tags here.") is None


def test_parse_remove_noise_tags():
    """Known internal control tags are stripped from the output."""
    from langgraph_orchestrator.utils.response_parser import remove_known_noise_tags

    assert remove_known_noise_tags("Answer here. [END_SYMPTOMS]").strip() \
        == "Answer here."
    assert remove_known_noise_tags("[NOT_AVAILABLE]").strip() == ""
    # [CONTEXT] and [/CONTEXT] are tag markers only — the content between them
    # is NOT erased, only the surrounding tag tokens are stripped.
    result = remove_known_noise_tags("text [CONTEXT]ctx[/CONTEXT] end")
    assert "ctx" in result            # content preserved
    assert "[CONTEXT]" not in result  # opening tag removed
    assert "[/CONTEXT]" not in result # closing tag removed


def test_parse_strip_trailing_artifacts():
    """Stray closing brackets or parens at the end of text are removed."""
    from langgraph_orchestrator.utils.response_parser import strip_trailing_artifacts

    assert strip_trailing_artifacts("Consult a doctor.)") == "Consult a doctor."
    assert strip_trailing_artifacts("See a specialist.]") == "See a specialist."
    assert strip_trailing_artifacts("Normal text.") == "Normal text."


def test_parse_full_pipeline():
    """The full normalisation pipeline produces clean, user-facing text.

    Checks all steps together:
    - <think> removed
    - First [DIAGNOSIS] block extracted
    - Duplicate [DIAGNOSIS] block discarded
    - [END_SYMPTOMS] stripped
    - Final text trimmed
    """
    from langgraph_orchestrator.utils.response_parser import parse_model_response

    raw = (
        "<think>internal reasoning</think>\n"
        "[DIAGNOSIS]Clean diagnosis text.[/DIAGNOSIS]\n"
        "[DIAGNOSIS]Duplicate — must be discarded.[/DIAGNOSIS]\n"
        "[END_SYMPTOMS]"
    )
    result = parse_model_response(raw)

    assert result == "Clean diagnosis text."
    assert "<think>" not in result
    assert "[DIAGNOSIS]" not in result
    assert "[END_SYMPTOMS]" not in result
    assert "Duplicate" not in result


# ===========================================================================
# Scenario 9 — Follow-up routing
# ===========================================================================

def test_followup_knowledge_question_routes_to_rag(graph):
    """A question containing a RAG keyword (treatment, causes, etc.) routes
    through retrieve_context → rag_answer, producing a knowledge-grounded reply.

    The mock RAG answer contains "available medical information" — a reliable
    signal that the RAG path (not the plain LLM path) was taken.
    """

    state = make_state(
        diagnosis_done=True,
        phase="follow_up",
        diagnosis_summary="Preliminary diagnosis: eczema (82%)",
        symptoms_complete=True,
        image_processed=True,
    )

    state = run_turn(graph, state, "What is the treatment for this condition?")

    response = last_assistant_message(state).lower()
    # Mock RAG path returns "Based on the available medical information…"
    assert "available medical information" in response or "emollient" in response
    # rag_context is cleared after use (rag_answer node does this)
    assert state.get("rag_context") is None


def test_followup_conversational_question_routes_to_llm(graph):
    """A conversational question with no RAG keywords routes to llm_followup,
    producing the LLM placeholder response.

    The inline _llm_followup node (build_graph.py) returns text containing
    "PLACEHOLDER" — a reliable signal that the plain LLM path was taken.
    """

    state = make_state(
        diagnosis_done=True,
        phase="follow_up",
        diagnosis_summary="Preliminary diagnosis: eczema (82%)",
        symptoms_complete=True,
        image_processed=True,
    )

    # No RAG keywords — should go to plain LLM path
    state = run_turn(graph, state, "Thank you, that is very helpful.")

    response = last_assistant_message(state)
    assert "PLACEHOLDER" in response


# ===========================================================================
# Scenario 10 — Session reset
# ===========================================================================

def test_reset_session_returns_clean_state():
    """reset_session() returns a fully clean state.

    Verified:
    - All diagnosis / symptom / image flags cleared.
    - phase returns to "start".
    - messages list is empty.
    - session_id is preserved (by design — same ID, new state).
    """

    from langgraph_orchestrator.web.api.service import (
        get_session,
        reset_session,
        start_session,
    )

    sid = "test-reset-scenario-10"
    start_session(sid)

    # Reset and inspect the returned state directly
    state = reset_session(sid)

    assert state["diagnosis_done"] is False
    assert state["symptoms_complete"] is False
    assert state["image_uploaded"] is False
    assert state["image_processed"] is False
    assert state["phase"] == "start"
    assert state["messages"] == []
    assert state["session_id"] == sid       # ID preserved after reset
    assert state["symptom_slots"] == {      # all slots cleared
        "location": None,
        "duration": None,
        "appearance": None,
        "severity": None,
        "triggers": None,
    }

    # get_session must also return the clean state
    retrieved = get_session(sid)
    assert retrieved is not None
    assert retrieved["phase"] == "start"
    assert retrieved["diagnosis_done"] is False
