from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def await_or_store_image(state: ChatState) -> ChatState:
    """Request an image from the user, or silently store one that arrived early.

    Two sub-cases handled by this node:

    Case A — symptoms complete, no image yet:
        Ask the user to upload a photo of the affected skin area.

    Case B — image uploaded during symptom collection:
        Acknowledge the image, store the path in state, and instruct the user
        that symptom questions will continue.  Diagnosis will only happen once
        both symptoms and image are present.

    Key rule:
        An image alone is never enough to trigger diagnosis.  This node must
        never set image_processed = True; that is the job of run_vision.

    Real implementation will:
    - Inspect the latest message for an image attachment to distinguish the two
      cases above.
    - Validate the uploaded file (format, size, readability).
    - Persist the file path / object key in state["image_path"].

    Returns:
        Updated state with image_uploaded / image_path set (Case B) or an
        assistant prompt asking for the image appended to messages (Case A).
    """
    messages = state["messages"]
    updates: dict = {}

    if state["image_uploaded"] and not state["symptoms_complete"]:
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
        # Case A: symptoms done, waiting for image
        reply = (
            "[PLACEHOLDER] Great, I have all your symptom details. "
            "Please upload a clear photo of the affected area so I can take a look."
        )
        updates["phase"] = "await_image"
        print("[await_or_store_image] Prompting user to upload image.")

    updates["messages"] = messages + [{"role": "assistant", "content": reply}]
    return {**state, **updates}
