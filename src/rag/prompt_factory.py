"""prompt_factory.py — Build prompts with [CONTEXT] and PRIORITY tags.

The format produced here matches exactly the format used in Dataset C training:
  - [CONTEXT] block with PRIORITY labels and source attribution
  - [QUESTION] block
  - [ANSWER] opening with <think> tag

Three cases handled, mirroring Dataset C edge cases:
  1. evidences has 1 item  → PRIORITY 1 only  (one-chunk case)
  2. evidences has 2 items → PRIORITY 1 + 2   (two-chunk case)
  3. evidences is empty    → empty [CONTEXT]  (no-context case — model says "I couldn't find...")
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.config import SYSTEM_PROMPT


def _format_source_header(evidence: dict, priority: int) -> str:
    """Build the '## PRIORITY N: Source Name (Year) (Relevance: X%)' header line."""
    meta = evidence.get("meta", {})

    source = (
        meta.get("title")    or
        meta.get("source")   or
        meta.get("filename") or
        f"Source {priority}"
    )
    # Truncate very long titles so the header stays readable
    if len(source) > 80:
        source = source[:77] + "..."

    year     = meta.get("year", "")
    year_str = f" ({year})" if year else ""

    score     = evidence.get("score", 0)
    score_str = f" (Relevance: {int(score * 100)}%)" if score else ""

    return f"## PRIORITY {priority}: {source}{year_str}{score_str}"


def build_clinical_prompt(query: str, evidences: list) -> str:
    """
    Build the full inference prompt for the fine-tuned LLM.

    Args:
        query:     The user's dermatology question (already processed by the
                   LangGraph orchestrator — this is the final query sent to RAG).
        evidences: List of evidence dicts from retrieve_context_node.
                   Each dict has keys: 'text', 'score', 'meta'.
                   Pass [] to trigger the no-context response path.

    Returns:
        Formatted prompt string ending with '<think>' so the model
        immediately enters its reasoning chain.
    """
    # ── Build [CONTEXT] block ─────────────────────────────────────────────────
    if not evidences:
        # No-context case: empty block signals the model to admit it has no info.
        # This matches the edge_no_context examples in Dataset C.
        context_block = "[CONTEXT]\n[/CONTEXT]"
    else:
        lines = ["[CONTEXT]\n"]
        for i, ev in enumerate(evidences[:2], start=1):   # hard cap at 2 chunks
            header = _format_source_header(ev, priority=i)
            source = (
                ev.get("meta", {}).get("title")   or
                ev.get("meta", {}).get("source")  or
                ev.get("meta", {}).get("filename") or
                f"Source {i}"
            )
            lines.append("")
            lines.append(header)
            lines.append(f"Source: {source}")
            lines.append(ev.get("text", "").strip())
            lines.append("")
        lines.append("[/CONTEXT]")
        context_block = "\n".join(lines)

    # ── Assemble full prompt ──────────────────────────────────────────────────
    # The prompt ends with '<think>' so the model's first token is the
    # opening of its reasoning chain — exactly as trained in Dataset C.
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"{context_block}\n\n"
        f"[QUESTION]\n{query}\n\n"
        f"[ANSWER]\n"
    )