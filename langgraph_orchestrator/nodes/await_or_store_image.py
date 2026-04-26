from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def await_or_store_image(state: ChatState) -> ChatState:
    """Request an image from the user, or silently store one that arrived early.

    Three sub-cases handled by this node:

    Case A — symptoms complete, no image yet:
        Ask the user to upload a photo of the affected skin area.

    Case B — image uploaded during symptom collection:
        Acknowledge the image, store the path in state, and instruct the user
        that symptom questions will continue.  Diagnosis will only happen once
        both symptoms and image are present.

    Case C — vision ran but DenseNet confidence was critically low (< 0.40):
        Send a focused re-upload prompt.  Reset image_processed, image_uploaded,
        and fallback_triggered so that the next image the user provides enters
        the standard run_vision path without any special-case routing.
        Symptom slots are never touched — all collected information is preserved.

    Key rule:
        This node must never set image_processed = True; that is run_vision's job.

    Returns:
        Updated state with an assistant message appended and, for Case C,
        image flags reset to their pre-upload values.
    """
    messages = state["messages"]
    updates: dict = {}

    if state.get("image_processed") and state.get("fallback_triggered"):
        # Case C: vision ran but confidence was critically low.
        # Reset image flags so the next upload triggers run_vision cleanly.
        # Symptom slots and all other collected state are preserved.
        reply = (
            "I wasn't able to get a confident reading from the image you provided — "
            "the image quality or lighting may be making it difficult to analyse. "
            "Could you upload a clearer, well-lit photo of the affected area? "
            "Your symptom information has been kept and will be used in the analysis."
        )
        updates["image_processed"]   = False  # allow run_vision to fire again
        updates["image_uploaded"]    = False  # user must re-upload
        updates["fallback_triggered"] = False  # clear flag; standard routing resumes
        updates["phase"]             = "await_image"
        print("[await_or_store_image] Vision fallback — prompting for clearer image; flags reset.")

    elif state["image_uploaded"] and not state["symptoms_complete"]:
        # Case B: early image, not yet acknowledged — send acknowledgement once
        # and set image_acknowledged = True so the router never returns here again.
        reply = (
            "Thanks for the image — I've saved it. "
            "Let's continue with a few more questions before I analyse it."
        )
        updates["image_acknowledged"] = True
        updates["phase"] = "symptom_collection"
        print("[await_or_store_image] Early image acknowledged; returning to symptom collection.")

    else:
        # Case A: symptoms done, waiting for image.
        reply = (
            "Great, I have all your symptom details. "
            "Please upload a clear photo of the affected area so I can take a look."
        )
        updates["phase"] = "await_image"
        print("[await_or_store_image] Prompting user to upload image.")

    updates["messages"] = messages + [{"role": "assistant", "content": reply}]
    return {**state, **updates}
