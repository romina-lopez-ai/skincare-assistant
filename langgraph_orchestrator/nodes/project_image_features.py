from __future__ import annotations

from langgraph_orchestrator.state import ChatState


def project_image_features(state: ChatState) -> ChatState:
    """Convert DenseNet output into LLM-compatible embeddings via the projection layer.

    Responsibilities:
    - Read state["densenet_output"] (label, confidence, morphology).
    - Pass the raw image features / DenseNet logits through the trained
      projection layer (a small MLP or linear map).
    - Store the resulting embedding vector in state["densenet_output"] under
      the key "projected_embedding" so the joint_diagnosis node can attach it
      to the LLM input.

    When this node is active:
        image_uploaded = True AND symptoms_complete = True AND image_processed = True

    Key rule:
        This node is only reached when an image is present.  It must never be
        called for text-only sessions.

    Real implementation will:
    - Load the projection layer weights.
    - Accept a feature tensor from DenseNet's penultimate layer (not just the
      predicted label) for richer visual information.
    - Return a fixed-size embedding (e.g. 4096-dim) compatible with the LLM's
      input space.

    Returns:
        Updated state where state["densenet_output"]["projected_embedding"]
        holds the projected feature vector (placeholder: a descriptive string).
    """
    densenet_out = state.get("densenet_output") or {}

    # --- Placeholder projection ---
    # Real code: embedding = projection_layer(densenet_feature_tensor)
    projected = (
        f"[PROJECTED] condition={densenet_out.get('label', 'unknown')}  "
        f"confidence={densenet_out.get('confidence', 0.0):.2f}  "
        f"morphology='{densenet_out.get('morphology', '')}'"
    )

    updated_densenet = {**densenet_out, "projected_embedding": projected}

    print(f"[project_image_features] {projected}")

    return {**state, "densenet_output": updated_densenet}
