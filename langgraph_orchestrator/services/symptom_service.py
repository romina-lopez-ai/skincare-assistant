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

import re

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SLOTS: tuple[str, ...] = (
    "location", "duration", "appearance", "severity", "triggers"
)

MIN_SLOTS_REQUIRED = 4   # complete when at least this many slots are filled
MAX_SYMPTOM_TURNS  = 10  # hard cap — prevents infinite loops
MIN_SYMPTOM_TURNS  = 3   # minimum assistant turns before allowing completion


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
        "neck", "scalp", "head", "ear", "behind the ear",
        "arm", "forearm", "upper arm", "inner arm", "elbow", "wrist", "hand", "finger", "between fingers",
        "leg", "lower leg", "knee", "shin", "ankle", "foot", "toe",
        "back", "chest", "stomach", "abdomen",
        "shoulder", "groin", "thigh", "inner thigh", "buttock",
        "lip", "mouth", "eye", "around the eye",
        # "body", "skin",  # Summary: removed — match in almost every message, too noisy
    ],
    "duration": [
        "hour", "hours",
        "day", "days", "week", "weeks", "month", "months",
        "year", "years", "since", " ago", "started", "began",
        "for about", "yesterday", "recently",
        "last night", "this morning", "a while",
        "chronic", "on and off", "comes and goes", "flare",
        # "long",  # Summary: removed — ambiguous ("long scratch" fills duration incorrectly)
    ],
    "appearance": [
        "red", "pink", "brown", "dark", "pale", "white", "yellow", "purple", "silvery", "orange",
        "itchy", "itching", "itch",
        "dry", "flaky", "scaly", "crusty", "rough", "thick", "smooth",
        "swollen", "raised", "flat", "bumpy",
        "blister", "rash", "patch", "spot", "mark", "lesion", "sore",
        "pimple", "acne", "hive", "welt",
        "ring", "circular", "ring-shaped", "coin-shaped",
        "fluid-filled", "clear", "fluid",
        "oozing", "weeping", "bleeding", "cracking",
    ],
    "severity": [
        "mild", "slight", "minor", "not that bad",
        "moderate", "medium", "manageable", "tolerable",
        "severe", "bad", "very bad", "terrible",
        "unbearable", "extreme", "awful", "intense",
        "worse", "worsening", "getting worse",
        "painful", "pain", "hurts", "burning", "stinging", "tender", "uncomfortable",
        "/10", "out of 10",
        # "1"-"10" removed — Summary: standalone digits trigger on durations/dates
        # e.g. "2 weeks" fills severity with "2", "3 days ago" fills with "3"
    ],
    "triggers": [
        "stress", "anxious", "anxiety", "nervous",
        "soap", "detergent", "washing",
        "food", "eating", "dairy", "gluten", "nut",
        "sweat", "sweating", "exercise", "heat",
        "sun", "sunlight", "cold", "wind", "weather", "seasonal", "humidity", "dry air",
        "fabric", "wool", "synthetic", "clothes",
        "alcohol", "perfume", "makeup", "cosmetics", "shampoo", "lotion",
        "dust", "pollen", "allergy", "allergic",
        "hormones", "period", "pregnancy",
        "when i", "after i", "after eating", "when stressed",
        "nothing", "don't know", "no idea", "not sure",
    ],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_slot_evidence(message_lower: str, keywords: list[str]) -> str | None:
    """Return a phrase from message_lower that contains a keyword match.

    Summary: rewritten to fix three issues with the original 6+20 char window:
      1. Words cut mid-phrase — now snaps to word boundaries on both sides.
      2. Wrong slot — longer keywords tried first (unchanged), but now the
         returned phrase is wide enough to show real context.
      3. Duplicate-looking values — phrase ends at the nearest clause separator
         (, . ! ? newline) so each slot captures its own clause.

    Tries longer keywords first to prefer more specific matches.
    Returns None when no keyword is found.
    """
    for kw in sorted(keywords, key=len, reverse=True):
        kw_stripped = kw.strip()

        # Summary: use \b word-boundary matching when keyword starts and ends with
        # a word character — prevents "arm" matching "warm/farm", "ear" matching
        # "early/hear". Keywords like "/10" or " ago" skip boundaries (non-word edges).
        if re.match(r"\w", kw_stripped) and re.search(r"\w$", kw_stripped):
            pattern = r"\b" + re.escape(kw_stripped) + r"\b"
        else:
            pattern = re.escape(kw_stripped)

        m = re.search(pattern, message_lower)
        if m is None:
            continue

        idx = m.start()

        # Summary: snap left to the nearest word boundary (don't cut mid-word)
        start = idx
        while start > 0 and message_lower[start - 1] not in " .,;!?\n\t":
            start -= 1

        # Summary: extend right to the next clause boundary or 60 chars max
        tail = message_lower[start:]
        clause_end = len(tail)
        for sep_char in (",", ".", "!", "?", "\n"):
            pos = tail.find(sep_char)
            if 0 < pos < clause_end:
                clause_end = pos

        snippet = tail[:min(clause_end, 60)].strip(" .,;!?")
        if snippet:
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

    # Special case for severity: capture plain number answers to the "1-10" question.
    # Can't keep "1"-"10" in _SLOT_KEYWORDS because they false-trigger on durations
    # ("2 weeks" → severity "2"). Negative lookahead excludes numbers followed by
    # time words so "7" matches but "2 weeks" and "3 days ago" do not.
    if updated.get("severity") is None:
        _sev_m = re.search(
            r"\b(10|[1-9])\b(?!\s*(?:days?|weeks?|months?|years?|hours?))",
            msg_lower,
        )
        if _sev_m:
            _start = _sev_m.start()
            updated["severity"] = msg_lower[_start : _start + 12].strip(" .,;!?")

    return updated


def is_symptoms_complete(
    slots: dict[str, str | None],
    turns: int,
    end_signal: bool,
) -> bool:
    """Apply the deterministic completion rules.

    Rules (checked in order, first True wins):
    1. end_signal is True AND turns >= MIN_SYMPTOM_TURNS
       → model explicitly signalled done (but not prematurely)
    2. filled_count >= MIN_SLOTS_REQUIRED AND turns >= MIN_SYMPTOM_TURNS
       → enough structured data AND enough conversation
    3. turns >= MAX_SYMPTOM_TURNS → hard turn cap (always fires)

    Args:
        slots:      Current symptom_slots dict.
        turns:      symptom_turns value AFTER incrementing for this turn.
        end_signal: True if [END_SYMPTOMS] appeared in the model reply.

    Returns:
        True if the symptom collection phase should end.
    """
    filled_count = sum(1 for v in slots.values() if v is not None)

    if end_signal and turns >= MIN_SYMPTOM_TURNS:
        return True
    if filled_count >= MIN_SLOTS_REQUIRED and turns >= MIN_SYMPTOM_TURNS:
        return True
    if turns >= MAX_SYMPTOM_TURNS:
        return True
    return False


def filled_slot_count(slots: dict[str, str | None]) -> int:
    """Return the number of non-None slots — useful for logging and tests."""
    return sum(1 for v in slots.values() if v is not None)
