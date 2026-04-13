"""routing.py — Pure routing functions for the LangGraph dermatology workflow.

Each function in this module is a *conditional edge resolver*: it receives the
current ChatState and returns a string that LangGraph uses to pick the next
node.  None of these functions mutate state.

Separation of concerns
-----------------------
- Node functions  (nodes/)  → mutate state, produce output.
- Routing functions (here)  → read state, return a destination string.

Keeping routing logic here (rather than inside each node) means:
1. The full transition table is visible in one file.
2. Node files stay focused on their single responsibility.
3. Unit-testing routing decisions requires no mocks — just build a state dict.
"""

from __future__ import annotations

from langgraph_orchestrator.state import ChatState

# ---------------------------------------------------------------------------
# Route constants
# Every destination string used in the graph is defined once here so that
# the graph builder and these functions always stay in sync.
# ---------------------------------------------------------------------------

# Top-level input routing
ROUTE_COLLECT_SYMPTOMS = "collect_symptoms"
ROUTE_AWAIT_OR_STORE_IMAGE = "await_or_store_image"
ROUTE_RUN_VISION = "run_vision"
ROUTE_FOLLOWUP_ROUTER = "followup_router"
ROUTE_SESSION_END = "session_end"

# Post-vision routing
ROUTE_JOINT_DIAGNOSIS = "joint_diagnosis"

# Follow-up routing
ROUTE_RETRIEVE_CONTEXT = "retrieve_context"
ROUTE_LLM_FOLLOWUP = "llm_followup"


# ---------------------------------------------------------------------------
# 1. after_session_entry
#    Called on the edge leaving session_entry.
#    Decides whether this is the very first turn (go to input_router) or
#    whether the session is already over.
# ---------------------------------------------------------------------------

def after_session_entry(state: ChatState) -> str:
    """Route after the session entry node initialises the turn.

    Logic:
    - phase == "end_session"  → session_end   (rare: re-entry into a closed session)
    - anything else           → input_router  (normal path)
    """
    if state["phase"] == "end_session":
        return ROUTE_SESSION_END
    return "input_router"


# ---------------------------------------------------------------------------
# 2. route_from_input_router
#    The main routing hub.  Called on the conditional edge leaving input_router.
#    Encodes the full priority order for deciding what to do this turn.
# ---------------------------------------------------------------------------

def route_from_input_router(state: ChatState) -> str:
    """Decide the high-level processing path for this turn.

    Priority order (highest → lowest):
    1. Session end requested                 → session_end
    2. Diagnosis already done                → followup_router
    3. Symptoms + image both ready           → run_vision   (or joint_diagnosis
                                               if image was already processed)
    4. Symptoms complete, no image yet       → await_or_store_image
    5. Image uploaded early (during intake)  → await_or_store_image
       (image stored; symptom collection continues on next turn)
    6. Default                               → collect_symptoms

    Milestone protection (Phase 4):
    - Two independent guards prevent backward regression after diagnosis:
      1. PRIMARY:   diagnosis_done == True  → always followup_router
      2. SECONDARY: phase == "follow_up"    → always followup_router

    Uncertainty / low-confidence fallback (Phase 5):
    - Tier 3 (conf < 0.40): route_after_vision → await_or_store_image.
      That node sends a targeted "re-upload" message and resets
      image_processed=False + image_uploaded=False, so the NEXT image the
      user provides re-enters via branch 3 above (symptoms_complete +
      image_uploaded + not image_processed → run_vision).  No special
      override is needed here; the standard priority ladder handles it.
    - Malformed diagnosis output: joint_diagnosis leaves image_processed=True
      and diagnosis_done=False, so on re-entry branch 3 resolves to
      joint_diagnosis (image_processed=True → ROUTE_JOINT_DIAGNOSIS) for
      a clean retry.
    """
    # --- Session end ---
    if state["phase"] == "end_session":
        return ROUTE_SESSION_END

    # --- MILESTONE GUARD (primary): diagnosis flag is set ---
    # diagnosis_done = True is irreversible within a session.
    # Once set, ALL future turns route to follow-up — no exceptions.
    if state["diagnosis_done"]:
        return ROUTE_FOLLOWUP_ROUTER

    # --- MILESTONE GUARD (secondary): phase is already follow_up ---
    # Defensive check: if the phase advanced to follow_up but diagnosis_done
    # was somehow not committed (e.g. malformed output on first attempt),
    # still honour the forward progress rather than sliding backward.
    if state["phase"] == "follow_up":
        return ROUTE_FOLLOWUP_ROUTER

    # --- Both symptoms and image are ready ---
    if state["symptoms_complete"] and state["image_uploaded"]:
        if state["image_processed"]:
            # Vision already ran (e.g. re-entry); go straight to diagnosis
            return ROUTE_JOINT_DIAGNOSIS
        return ROUTE_RUN_VISION

    # --- Symptoms done but still waiting for image ---
    if state["symptoms_complete"] and not state["image_uploaded"]:
        return ROUTE_AWAIT_OR_STORE_IMAGE

    # --- Early image upload during symptom collection ---
    # Only route here on the FIRST turn after upload (image_acknowledged=False).
    # Once await_or_store_image sends the acknowledgement it sets
    # image_acknowledged=True, and subsequent turns fall through to
    # ROUTE_COLLECT_SYMPTOMS so symptom collection continues normally.
    if (
        state["image_uploaded"]
        and not state.get("image_acknowledged", False)
        and not state["symptoms_complete"]
    ):
        return ROUTE_AWAIT_OR_STORE_IMAGE

    # --- Default: keep collecting symptoms ---
    return ROUTE_COLLECT_SYMPTOMS


# ---------------------------------------------------------------------------
# 3. route_after_vision
#    Called on the edge leaving project_image_features.
#    At this point DenseNet has run and features have been projected.
#    The only valid destination is joint_diagnosis.
# ---------------------------------------------------------------------------

def route_after_vision(state: ChatState) -> str:
    """Route after image features have been projected.

    Logic:
    - If image was processed successfully → joint_diagnosis
    - If confidence was critically low (both flags set) → await_or_store_image
      The await node sends a focused "please re-upload" message, resets
      image_processed and image_uploaded, then ends the turn.  The user's
      next image upload re-enters the normal run_vision path automatically.
    """
    if state["uncertainty_flag"] and state["fallback_triggered"]:
        return ROUTE_AWAIT_OR_STORE_IMAGE
    return ROUTE_JOINT_DIAGNOSIS


# ---------------------------------------------------------------------------
# 4. route_from_followup_router
#    Called on the conditional edge leaving followup_router.
#    Decides between a RAG-grounded answer and a plain LLM answer.
# ---------------------------------------------------------------------------

def route_from_followup_router(state: ChatState) -> str:
    """Route a post-diagnosis follow-up question.

    Reads the "_route" key that followup_router wrote into state.

    Possible destinations:
    - retrieve_context  — knowledge question: treatment, causes, prevention …
    - llm_followup      — conversational / clarification question
    - session_end       — user signalled end of session
    """
    # End-session signal takes priority
    if state["phase"] == "end_session":
        return ROUTE_SESSION_END

    # followup_router node writes its decision to state["_route"]
    return state.get("_route", ROUTE_LLM_FOLLOWUP)


# ---------------------------------------------------------------------------
# 5. route_after_diagnosis
#    Called on the edge leaving clean_response (after joint_diagnosis).
#    After the first diagnosis the conversation always moves to follow_up.
# ---------------------------------------------------------------------------

def route_after_diagnosis(state: ChatState) -> str:
    """Route after a diagnosis has been generated and cleaned.

    Logic:
    - diagnosis_done = True  → followup_router (user may ask follow-ups)
    - diagnosis_done = False → collect_symptoms (diagnosis failed; collect more)
    """
    if state["diagnosis_done"]:
        return ROUTE_FOLLOWUP_ROUTER
    return ROUTE_COLLECT_SYMPTOMS


# ---------------------------------------------------------------------------
# 6. Convenience: phase_to_route
#    Maps the `phase` field directly to a node name.
#    Useful as a simple fallback resolver when the phase is the only signal.
# ---------------------------------------------------------------------------

_PHASE_MAP: dict[str, str] = {
    "start":              "session_entry",
    "symptom_collection": ROUTE_COLLECT_SYMPTOMS,
    "await_image":        ROUTE_AWAIT_OR_STORE_IMAGE,
    "joint_diagnosis":    ROUTE_JOINT_DIAGNOSIS,
    "follow_up":          ROUTE_FOLLOWUP_ROUTER,
    "end_session":        ROUTE_SESSION_END,
}


def phase_to_route(state: ChatState) -> str:
    """Map the current phase directly to its default node.

    Intended as a last-resort resolver, not the primary routing mechanism.
    Falls back to ROUTE_COLLECT_SYMPTOMS for any unrecognised phase so the
    graph never gets stuck.
    """
    return _PHASE_MAP.get(state["phase"], ROUTE_COLLECT_SYMPTOMS)
