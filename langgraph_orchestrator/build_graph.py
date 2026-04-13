"""build_graph.py — Assembles and compiles the LangGraph workflow.

Graph topology
--------------
                    START
                      │
               session_entry
              ╱             ╲
       (normal)           (end_session)
            │                  │
       input_router        session_end ── END
      ╱    │    │    ╲
     ↓     ↓    ↓     ↓
collect  await  run   followup
symptoms store  vision router
  │      image   │   ╱       ╲
  END      │   project     (RAG)       (LLM)
           END  features  retrieve   llm_followup
                  │       context      │
                  ↓          │        END
            joint_diag    rag_answer
                  │          │
            clean_response  END
                  │
                 END

Key rules reflected in the edges
----------------------------------
1. Diagnosis requires BOTH symptoms and image — vision pipeline only triggers
   once symptoms_complete AND image_uploaded.
2. Early image upload (during symptom collection) is handled by
   await_or_store_image and loops back to END so symptom collection
   continues on the next user turn.
3. The LLM always generates the final message — DenseNet output flows into
   project_image_features then into joint_diagnosis as context, never
   directly to the user.
4. RAG is conditional — only reached for knowledge/treatment follow-up
   questions, not for conversational replies.
5. Each path that produces an assistant message ends at END so the graph
   pauses and waits for the next user input before re-entering.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from langgraph_orchestrator.nodes import (
    await_or_store_image,
    clean_response,
    collect_symptoms,
    followup_router,
    input_router,
    joint_diagnosis,
    project_image_features,
    rag_answer,
    retrieve_context,
    run_vision,
    session_end,
    session_entry,
)
from langgraph_orchestrator.routing import (
    ROUTE_AWAIT_OR_STORE_IMAGE,
    ROUTE_COLLECT_SYMPTOMS,
    ROUTE_FOLLOWUP_ROUTER,
    ROUTE_JOINT_DIAGNOSIS,
    ROUTE_LLM_FOLLOWUP,
    ROUTE_RETRIEVE_CONTEXT,
    ROUTE_RUN_VISION,
    ROUTE_SESSION_END,
    after_session_entry,
    route_after_vision,
    route_from_followup_router,
    route_from_input_router,
)
from langgraph_orchestrator.state import ChatState


# ---------------------------------------------------------------------------
# Inline placeholder node
# ---------------------------------------------------------------------------

def _llm_followup(state: ChatState) -> ChatState:
    """Placeholder: plain LLM response for conversational follow-up questions.

    This handles the 'normal' follow-up path (non-knowledge questions) after
    diagnosis, e.g. "What does eczema look like?" or "Is this serious?".

    Not a separate file because it is a thin wrapper around a single LLM call
    with no special pre- or post-processing.  The real implementation will call
    the fine-tuned LLM with the full conversation history.
    """
    reply = (
        "[PLACEHOLDER — LLM followup] That is a great question. "
        "Based on your diagnosis and our conversation so far, here is what I can share."
    )
    print("[llm_followup] Generating conversational follow-up response.")
    return {
        **state,
        "messages": state["messages"] + [{"role": "assistant", "content": reply}],
    }


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the dermatology chatbot LangGraph.

    Returns:
        A compiled LangGraph ``CompiledGraph`` ready to be invoked with a
        ``ChatState`` dict via ``graph.invoke(state)`` or streamed with
        ``graph.stream(state)``.

    Usage::

        from langgraph_orchestrator.build_graph import build_graph
        from langgraph_orchestrator.state import create_initial_state
        import uuid

        graph = build_graph()
        state = create_initial_state(str(uuid.uuid4()))
        state["messages"].append({"role": "user", "content": "Hi, I have a rash."})
        result = graph.invoke(state)
    """
    graph = StateGraph(ChatState)

    # ------------------------------------------------------------------
    # Register nodes
    # ------------------------------------------------------------------
    graph.add_node("session_entry",          session_entry)
    graph.add_node("input_router",           input_router)
    graph.add_node("collect_symptoms",       collect_symptoms)
    graph.add_node("await_or_store_image",   await_or_store_image)
    graph.add_node("run_vision",             run_vision)
    graph.add_node("project_image_features", project_image_features)
    graph.add_node("joint_diagnosis",        joint_diagnosis)
    graph.add_node("clean_response",         clean_response)
    graph.add_node("followup_router",        followup_router)
    graph.add_node("retrieve_context",       retrieve_context)
    graph.add_node("rag_answer",             rag_answer)
    graph.add_node("llm_followup",           _llm_followup)
    graph.add_node("session_end",            session_end)

    # ------------------------------------------------------------------
    # Entry edge
    # ------------------------------------------------------------------
    graph.add_edge(START, "session_entry")

    # ------------------------------------------------------------------
    # session_entry → input_router  (normal)
    #              → session_end    (if phase is already "end_session")
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "session_entry",
        after_session_entry,
        {
            "input_router":   "input_router",
            ROUTE_SESSION_END: "session_end",
        },
    )

    # ------------------------------------------------------------------
    # input_router — main routing hub
    #
    # Priority (encoded in route_from_input_router):
    #   fallback active          → collect_symptoms
    #   end_session              → session_end
    #   diagnosis_done           → followup_router
    #   symptoms + image (unprocessed) → run_vision
    #   symptoms + image (processed)   → joint_diagnosis  (re-entry)
    #   symptoms, no image       → await_or_store_image
    #   early image, no symptoms → await_or_store_image   (store + continue)
    #   default                  → collect_symptoms
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "input_router",
        route_from_input_router,
        {
            ROUTE_COLLECT_SYMPTOMS:     "collect_symptoms",
            ROUTE_AWAIT_OR_STORE_IMAGE: "await_or_store_image",
            ROUTE_RUN_VISION:           "run_vision",
            ROUTE_JOINT_DIAGNOSIS:      "joint_diagnosis",
            ROUTE_FOLLOWUP_ROUTER:      "followup_router",
            ROUTE_SESSION_END:          "session_end",
        },
    )

    # ------------------------------------------------------------------
    # collect_symptoms
    # Produces an assistant message then waits for the next user turn.
    # ------------------------------------------------------------------
    graph.add_edge("collect_symptoms", END)

    # ------------------------------------------------------------------
    # await_or_store_image
    # Either prompts the user to upload an image, or acknowledges an early
    # upload and continues symptom collection on the next user turn.
    # Both sub-cases produce an assistant message → END.
    # ------------------------------------------------------------------
    graph.add_edge("await_or_store_image", END)

    # ------------------------------------------------------------------
    # Vision pipeline: run_vision → project_image_features → …
    #
    # run_vision runs DenseNet and sets image_processed = True.
    # project_image_features converts DenseNet output into LLM embeddings.
    # After projection the graph routes to joint_diagnosis (normal) or
    # back to collect_symptoms (high uncertainty + fallback triggered).
    # ------------------------------------------------------------------
    graph.add_edge("run_vision", "project_image_features")

    graph.add_conditional_edges(
        "project_image_features",
        route_after_vision,
        {
            ROUTE_JOINT_DIAGNOSIS:      "joint_diagnosis",
            # Tier 3 (conf < 0.40): send a focused re-upload prompt and reset
            # image flags so the next upload triggers run_vision cleanly.
            ROUTE_AWAIT_OR_STORE_IMAGE: "await_or_store_image",
        },
    )

    # ------------------------------------------------------------------
    # Diagnosis pipeline: joint_diagnosis → clean_response → END
    #
    # joint_diagnosis produces raw LLM output containing <think> blocks
    # and [DIAGNOSIS]…[/DIAGNOSIS] tags.
    # clean_response strips the internal reasoning so only the final
    # diagnosis text is shown to the user, then we END to wait for
    # the user's follow-up.
    # ------------------------------------------------------------------
    graph.add_edge("joint_diagnosis",  "clean_response")
    graph.add_edge("clean_response",   END)

    # ------------------------------------------------------------------
    # Follow-up routing: followup_router decides RAG vs plain LLM
    #
    # route_from_followup_router reads the "_route" key that the
    # followup_router node wrote into state, then maps it to a node.
    # ------------------------------------------------------------------
    graph.add_conditional_edges(
        "followup_router",
        route_from_followup_router,
        {
            ROUTE_RETRIEVE_CONTEXT: "retrieve_context",
            ROUTE_LLM_FOLLOWUP:     "llm_followup",
            ROUTE_SESSION_END:      "session_end",
        },
    )

    # ------------------------------------------------------------------
    # RAG path: retrieve_context → rag_answer → END
    # ------------------------------------------------------------------
    graph.add_edge("retrieve_context", "rag_answer")
    graph.add_edge("rag_answer",        END)

    # ------------------------------------------------------------------
    # Plain LLM follow-up path: llm_followup → END
    # ------------------------------------------------------------------
    graph.add_edge("llm_followup", END)

    # ------------------------------------------------------------------
    # Session end → END
    # ------------------------------------------------------------------
    graph.add_edge("session_end", END)

    # ------------------------------------------------------------------
    # Compile
    # ------------------------------------------------------------------
    return graph.compile()
