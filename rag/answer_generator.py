"""answer_generator.py : generate a grounded answer.

Receives context evidences from retrieve_context_node and produces
a response using the fine-tuned Ollama model.

The model was trained on Dataset C to handle three cases:
  1. 1 evidence chunk  → cites the source, answers from it
  2. 2 evidence chunks → compares priorities, synthesises answer
  3. 0 evidence chunks → honestly states no relevant info was found

Temperature is set low (0.0) to maximise faithfulness to context.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.prompt_factory import build_clinical_prompt
from rag.model_loader import get_llm

logger = logging.getLogger(__name__)

def answer_generator(state: dict) -> dict:
    """
    LangGraph node: generate a grounded dermatology answer.

    Input state keys:
        question  (str):  the user's question.
        evidences (list): evidence dicts from retrieve_context_node.
                          May be empty — the model handles this gracefully.

    Output state keys:
        answer          (str):  the model's response.
        answer_metadata (dict): diagnostic info for logging/evaluation.
    """
    query     = state.get("question", "").strip()
    evidences = state.get("evidences", [])

    if not query:
        logger.warning("rag_answer_node: empty question in state")
        return {
            "answer": "I didn't receive a question. Please try again.",
            "answer_metadata": {"has_context": False, "num_evidences": 0, "error": "empty_question"},
        }

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = build_clinical_prompt(query, evidences)
    logger.debug(f"Prompt built ({len(prompt)} chars) | evidences={len(evidences)}")

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        llm        = get_llm()
        raw_output = llm.invoke(prompt)
    except Exception as e:
        logger.error(f"Ollama generation failed: {e}")
        return {
            "answer": "I encountered an error generating a response. Please try again.",
            "answer_metadata": {
                "has_context":   len(evidences) > 0,
                "num_evidences": len(evidences),
                "error":         str(e),
            },
        }

    # ── Parse output ──────────────────────────────────────────────────────────
    # The prompt ends with '<think>' so the model output begins mid-think.
    # We prepend '<think>' to reconstruct the full response with reasoning.
    answer = raw_output

    # Log source titles for traceability
    sources = [
        ev.get("meta", {}).get("title") or ev.get("meta", {}).get("source", "unknown")
        for ev in evidences
    ]

    logger.info(
        f"Answer generated | query='{query[:50]}' | "
        f"evidences={len(evidences)} | sources={sources} | "
        f"output_len={len(answer)} chars"
    )

    return {
        "answer": answer,
        "answer_metadata": {
            "has_context":   len(evidences) > 0,
            "num_evidences": len(evidences),
            "sources":       sources,
        },
    }