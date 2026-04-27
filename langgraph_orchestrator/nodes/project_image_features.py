from __future__ import annotations

from langgraph_orchestrator.services.vision_service import project_vision_features
from langgraph_orchestrator.state import ChatState


def project_image_features(state: ChatState) -> ChatState:
    """Convert DenseNet output into LLM-compatible embeddings.

    Orchestration responsibilities (kept here):
    - Read densenet_output from state.
    - Write the updated dict (with projected_embedding) back to state.

    Model behavior (delegated to vision_service):
    - The projection itself is handled by project_vision_features().
    """
    densenet_out = state.get("densenet_output") or {}
    updated      = project_vision_features(densenet_out)

    print(f"[project_image_features] {updated.get('projected_embedding', '')}")

    return {**state, "densenet_output": updated}
