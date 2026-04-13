"""symptom_service.py — Deterministic symptom slot extraction and completion logic.

This module owns all of the rule-based symptom-tracking logic so that
collect_symptoms.py stays thin and the completion rules are testable
independently of the graph.

Teammate integration guide
--------------------------
- `extract_slots_from_message` uses keyword heuristics now.
  Replace it with a real slot-filling call (e.g. an NER model, a structured
  LLM output, or a regex parser) when your model is ready.
  The function signature and the returned dict structure must not change.

- `is_symptoms_complete` is purely rule-based and does NOT need to change
  when the real model is integrated.  It is the single source of truth for
  completion decisions.

Completion rules (in priority order)
--------------------------------------
1. Model emitted [END_SYMPTOMS]              → complete  (trusted signal)
2. MIN_SLOTS_REQUIRED (4) or more filled    → complete  (structural guarantee)
3. MAX_SYMPTOM_TURNS reached                → complete  (safety cap)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SLOTS: tuple[str, ...] = (
    "location", "duration", "appearance", "severity", "triggers"
)

MIN_SLOTS_REQUIRED = 4   # complete when at least this many slots are filled
MAX_SYMPTOM_TURNS  = 10  # hard cap — prevents infinite loops


# ---------------------------------------------------------------------------
# Keyword map for heuristic slot detection
#
# Each entry maps a slot name to a list of signal words.  If any signal word
# appears in the user message the slot is considered mentioned and is filled
# with a short extracted snippet.
#
# Teammate note: replace _find_slot_evidence() and this map with your real
# slot-filling model; keep the dict structure so the extraction loop below
# continues to work unchanged.
# ---------------------------------------------------------------------------

_SLOT_KEYWORDS: dict[str, list[str]] = {
    "location": [
        "face", "cheek", "forehead", "chin", "nose",
        "neck", "scalp", "head", "ear",
        "arm", "elbow", "wrist", "hand", "finger",
        "leg", "knee", "shin", "ankle", "foot", "toe",
        "back", "chest", "stomach", "abdomen",
        "shoulder", "groin", "thigh", "buttock",
        "body", "skin",
    ],
    "duration": [
        "day", "days", "week", "weeks", "month", "months",
        "year", "years", "since", " ago", "started", "began",
        "for about", "long", "yesterday", "recently",
        "last night", "this morning", "a while",
    ],
    "appearance": [
        "red", "pink", "brown", "dark", "pale", "white", "yellow",
        "itchy", "itching", "itch",
        "dry", "flaky", "scaly", "crusty", "rough",
        "swollen", "raised", "flat", "bumpy",
        "blister", "rash", "patch", "spot", "mark", "lesion", "sore",
        "oozing", "weeping", "bleeding", "cracking",
    ],
    "severity": [
        "mild", "slight", "minor", "not that bad",
        "moderate", "medium", "manageable",
        "severe", "bad", "very bad", "terrible",
        "unbearable", "extreme", "awful",
        "worse", "worsening", "getting worse",
        "/10", "out of 10",
        "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    ],
    "triggers": [
        "stress", "anxious", "anxiety", "nervous",
        "soap", "detergent", "washing",
        "food", "eating", "dairy", "gluten", "nut",
        "sweat", "sweating", "exercise", "heat",
        "sun", "sunlight", "cold", "wind", "weather",
        "fabric", "wool", "synthetic", "clothes",
        "when i", "after i", "after eating", "when stressed",
        "nothing", "don't know", "no idea", "not sure",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_slot_evidence(message_lower: str, keywords: list[str]) -> str | None:
    """Return a short text snippet from message_lower if any keyword matches.

    Tries longer keywords first to prefer more specific matches.
    Returns a window of up to ~35 characters around the matched term,
    which gives enough context for the debug panel without storing the
    whole message.

    Returns None when no keyword is found.
    """
    for kw in sorted(keywords, key=len, reverse=True):
        idx = message_lower.find(kw)
        if idx != -1:
            start   = max(0, idx - 6)
            end     = min(len(message_lower), idx + len(kw) + 20)
            snippet = message_lower[start:end].strip(" .,;!?")
            return snippet
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_slots_from_message(
    message: str,
    current_slots: dict[str, str | None],
) -> dict[str, str | None]:
    """Update symptom slots based on evidence found in a user message.

    Only fills slots that are currently None — already-filled slots are
    never overwritten, preserving earlier information.

    MOCK / heuristic: uses keyword matching via _SLOT_KEYWORDS.

    Real implementation should:
    - Replace or extend this with a structured output from the LLM
      (e.g. JSON slot values parsed from the model response) or an NER model.

    Args:
        message:       The raw user message text.
        current_slots: The existing symptom_slots dict from ChatState.

    Returns:
        An updated copy of current_slots with any newly detected values filled.
    """
    msg_lower = message.lower()
    updated   = dict(current_slots)

    for slot, keywords in _SLOT_KEYWORDS.items():
        if updated.get(slot) is not None:
            continue  # already filled — don't overwrite
        evidence = _find_slot_evidence(msg_lower, keywords)
        if evidence:
            updated[slot] = evidence

    return updated


def is_symptoms_complete(
    slots: dict[str, str | None],
    turns: int,
    end_signal: bool,
) -> bool:
    """Apply the deterministic completion rules.

    Rules (checked in order, first True wins):
    1. end_signal is True        → model explicitly signalled done
    2. filled_count >= MIN_SLOTS_REQUIRED (4) → enough structured data
    3. turns >= MAX_SYMPTOM_TURNS            → hard turn cap

    Args:
        slots:      Current symptom_slots dict.
        turns:      symptom_turns value AFTER incrementing for this turn.
        end_signal: True if [END_SYMPTOMS] appeared in the model reply.

    Returns:
        True if the symptom collection phase should end.
    """
    filled_count = sum(1 for v in slots.values() if v is not None)

    if end_signal:
        return True
    if filled_count >= MIN_SLOTS_REQUIRED:
        return True
    if turns >= MAX_SYMPTOM_TURNS:
        return True
    return False


def filled_slot_count(slots: dict[str, str | None]) -> int:
    """Return the number of non-None slots — useful for logging and tests."""
    return sum(1 for v in slots.values() if v is not None)
