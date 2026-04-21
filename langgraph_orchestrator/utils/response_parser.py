"""response_parser.py — Reusable response cleaning helpers.

Purpose
-------
Real model outputs are noisy.  This module normalises raw LLM output into
clean user-facing text before it is shown in the chat interface.

Each helper does exactly one thing and can be used independently or composed
via the top-level `parse_model_response()` function.

Teammate extension guide
------------------------
To handle a new tag or artifact:
1. Add a targeted helper function (follow the pattern of the existing ones).
2. Call it from `normalize_response_text()` in the right position.
3. Add a test case in tests/.

Avoid chaining all logic into a single mega-regex — isolated helpers are far
easier to debug and extend.

Handled patterns
----------------
| Pattern                              | Action                                |
|--------------------------------------|---------------------------------------|
| <think>…</think>                     | Removed entirely                      |
| [DIAGNOSIS]…[/DIAGNOSIS]             | First block extracted, rest discarded |
| [END_SYMPTOMS]                       | Stripped (never user-facing)          |
| [NOT_AVAILABLE]                      | Stripped (model uncertainty marker)   |
| Duplicate [DIAGNOSIS] blocks         | Only first kept                       |
| Stray bracket noise  )) ]] etc       | Stripped from end of text             |
| Excessive blank lines                | Collapsed to one                      |
| Leaked think header (no tags)        | Stripped from start                   |
| Fake turn markers [user]/[email]     | Everything from marker onwards cut    |
| Repeated junk tokens (dense.dense…)  | Collapsed/removed                     |
| Unicode box/block garbage  □□□       | Stripped                              |
| Malformed trailing token streams     | Stripped                              |
"""

from __future__ import annotations

import re
from collections import Counter

# ---------------------------------------------------------------------------
# Compiled patterns — defined once, reused on every call
# ---------------------------------------------------------------------------

# <think>…</think> — DOTALL so the block can span multiple lines
_RE_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)

# First [DIAGNOSIS]…[/DIAGNOSIS] block (non-greedy, DOTALL)
_RE_DIAGNOSIS_FIRST = re.compile(
    r"\[DIAGNOSIS\](.*?)\[/DIAGNOSIS\]", re.DOTALL | re.IGNORECASE
)

# Any remaining [DIAGNOSIS]…[/DIAGNOSIS] blocks after the first was extracted
_RE_DIAGNOSIS_ANY = re.compile(
    r"\[/?DIAGNOSIS\]", re.IGNORECASE
)

# Known internal control tags that must never reach the user
_NOISE_TAGS: tuple[str, ...] = (
    "[END_SYMPTOMS]",
    "[/END_SYMPTOMS]",
    "[NOT_AVAILABLE]",
    "[UNAVAILABLE]",
    "[UNKNOWN]",
    "[CONTEXT]",
    "[/CONTEXT]",
)

# Stray trailing characters: closing brackets/parens with optional whitespace
_RE_TRAILING_NOISE = re.compile(r"[\)\]\}]+\s*$")

# Three or more consecutive blank lines → collapse to one
_RE_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

# ---------------------------------------------------------------------------
# Real-model noise patterns (new)
# ---------------------------------------------------------------------------

# Leaked think header WITHOUT proper tags — the model sometimes emits lines like:
#   "think Think Dense (Dense)\nDense: 0.901\nThink: 0.099\n"
# at the very start of the output instead of wrapping in <think>...</think>.
# Matches: optional leading whitespace, then "think" word at start of line,
# followed by any number of "Key: value" debug lines.
_RE_LEAKED_THINK_HEADER = re.compile(
    r"^[\s]*think\b[^\n]*\n(?:[A-Za-z][A-Za-z0-9 _]*:\s*[^\n]+\n)*\s*",
    re.IGNORECASE,
)

# Fake conversation-continuation markers the model sometimes hallucinates.
# Everything from "[user]", "[email]", "[inst]", or "[/inst]" onwards is junk.
_RE_FAKE_CONTINUATION = re.compile(
    r"\[(user|email|assistant|inst|\/inst|human)\].*$",
    re.DOTALL | re.IGNORECASE,
)

# Same-word repeated 5+ times joined by dots or spaces:
#   "dense.dense.dense.dense.dense" or "think think think think think"
# These are hallucinated token loops, never valid medical text.
_RE_REPEATED_WORD = re.compile(
    r"\b(\w{2,})[. ]+(?:\1[. ]+){4,}\1?\b",
    re.IGNORECASE,
)

# Unicode block/box/replacement characters (□ U+25A1, ▪ U+25AA, ▸ U+25B8,
# U+FFFD replacement char, and the solid block range U+2580-U+259F).
# These never appear in legitimate clinical text.
_RE_UNICODE_JUNK = re.compile(
    r"[\u2580-\u259f\u25a0-\u25ff\ufffd\u2400-\u243f□▪▸]+"
)

# Long runs of punctuation-separated tokens that are clearly a token stream
# rather than English prose: 10+ word.word.word or Primary:Psoriasis.DenseNet
# sequences.  Separators include dots, commas, and colons; tokens may contain
# hyphens (e.g. "Silvery-White.Scales").
# Uses [ \t]* (not \s*) so the match never spans across newlines — prevents
# eating the last word of a real sentence that happens to end with a period.
_RE_TOKEN_STREAM = re.compile(
    r"(?:[\w][\w-]*[.,:][ \t]*){10,}",
    re.IGNORECASE,
)

# URLs — never valid in a clinical diagnosis response.
# Used by is_valid_diagnosis_response to reject hallucinated URL loops.
_RE_URL = re.compile(r"https?://\S+", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Individual helpers — each does exactly one thing
# ---------------------------------------------------------------------------

def strip_think_blocks(text: str) -> str:
    """Remove all <think>…</think> blocks from text.

    These are internal chain-of-thought sections generated by the model.
    They should never be shown to the user.

    >>> strip_think_blocks("<think>reasoning</think>Answer here.")
    'Answer here.'
    """
    return _RE_THINK.sub("", text)


def extract_first_diagnosis_block(text: str) -> str | None:
    """Return the content of the first [DIAGNOSIS]…[/DIAGNOSIS] block.

    Returns None if no well-formed block is found.  Only the FIRST block is
    returned; duplicate or repeated blocks are discarded.

    >>> extract_first_diagnosis_block("[DIAGNOSIS]You have eczema.[/DIAGNOSIS]")
    'You have eczema.'
    >>> extract_first_diagnosis_block("No tags here") is None
    True
    """
    match = _RE_DIAGNOSIS_FIRST.search(text)
    if match:
        return match.group(1).strip()
    return None


def remove_known_noise_tags(text: str) -> str:
    """Strip known internal control tags that should never appear in output.

    Handles: [END_SYMPTOMS], [NOT_AVAILABLE], [CONTEXT]/[/CONTEXT], etc.
    Tag matching is case-insensitive.

    >>> remove_known_noise_tags("Here is the answer. [END_SYMPTOMS]")
    'Here is the answer.'
    """
    result = text
    for tag in _NOISE_TAGS:
        # Case-insensitive removal without regex overhead for simple literals
        result = re.sub(re.escape(tag), "", result, flags=re.IGNORECASE)
    return result


def strip_residual_diagnosis_tags(text: str) -> str:
    """Remove any leftover [DIAGNOSIS] / [/DIAGNOSIS] tag fragments.

    Called after the first block has already been extracted, so that any
    duplicate block markers (e.g. a second [DIAGNOSIS]…[/DIAGNOSIS] from a
    malformed repeated output) don't leak into the final text.

    >>> strip_residual_diagnosis_tags("text [DIAGNOSIS] extra [/DIAGNOSIS]")
    'text  extra '
    """
    return _RE_DIAGNOSIS_ANY.sub("", text)


def strip_trailing_artifacts(text: str) -> str:
    """Remove stray closing-bracket or paren sequences from the end of text.

    Catches patterns like "...eczema.)" or "...treatment.]" that sometimes
    appear when the model closes a bracket it never opened.

    >>> strip_trailing_artifacts("Consult a dermatologist.)")
    'Consult a dermatologist.'
    """
    return _RE_TRAILING_NOISE.sub("", text)


def collapse_blank_lines(text: str) -> str:
    """Collapse three or more consecutive blank lines down to a single one."""
    return _RE_EXCESS_BLANK_LINES.sub("\n\n", text)


# ---------------------------------------------------------------------------
# Real-model noise helpers (new)
# ---------------------------------------------------------------------------

def strip_leaked_think_header(text: str) -> str:
    """Strip a leaked think-style header that appears WITHOUT proper tags.

    The model sometimes emits debug-like lines at the very start of its output
    instead of properly wrapping them in <think>...</think>, e.g.:

        think Think Dense (Dense)
        Dense: 0.901
        Think: 0.099

        Visual analysis suggests...

    This helper removes that header block when it appears at the start of the
    text (after any leading whitespace).  It is applied BEFORE <think> tag
    stripping so the two steps don't interfere.

    >>> strip_leaked_think_header("think Think Dense\\nDense: 0.9\\n\\nDiagnosis here.")
    'Diagnosis here.'
    """
    return _RE_LEAKED_THINK_HEADER.sub("", text)


def strip_fake_continuation(text: str) -> str:
    """Remove everything from a hallucinated [user]/[email]/[inst] marker onwards.

    The model occasionally appends a fake next conversation turn starting with
    markers like "[user]thank you[email]dense.think.dense..." after the real
    response ends.  Everything from the first such marker to end-of-string is
    junk.

    >>> strip_fake_continuation("Good diagnosis.\\n\\n[user]thank you")
    'Good diagnosis.'
    """
    return _RE_FAKE_CONTINUATION.sub("", text)


def strip_repeated_tokens(text: str) -> str:
    """Remove same-word-repeated-many-times token loops.

    Targets patterns like "dense.dense.dense.dense.dense" or
    "think think think think think" which are hallucinated token loops that
    the real model can emit when generation goes off the rails.

    >>> strip_repeated_tokens("See a doctor. dense.dense.dense.dense.dense end")
    'See a doctor.  end'
    """
    return _RE_REPEATED_WORD.sub("", text)


def strip_unicode_junk(text: str) -> str:
    """Remove Unicode block/box/garbage characters (□ ▪ ▸ etc.).

    These codepoints never appear in legitimate clinical text and indicate
    the model emitted raw binary noise or token-embedding artefacts.

    >>> strip_unicode_junk("Diagnosis.\\u25a0\\u25a0\\u25a0")
    'Diagnosis.'
    """
    return _RE_UNICODE_JUNK.sub("", text)


def strip_token_streams(text: str) -> str:
    """Remove long runs of dot/comma-separated word tokens.

    Catches trailing streams like "dense,think,dense,dense,dense,dense,dense..."
    that are clearly internal token sequences rather than English prose.
    Only fires on sequences of 10+ tokens to avoid stripping lists of medical
    terms.
    """
    return _RE_TOKEN_STREAM.sub("", text)


def strip_repeated_lines(text: str) -> str:
    """Remove any non-trivial line that appears 3 or more times.

    The real model sometimes enters a full-sentence generation loop, emitting
    the same sentence repeatedly, e.g.:

        DenseNet model visualized: https://www.tensorflow.org/model_search/denseface
        DenseNet model visualized: https://www.tensorflow.org/model_search/denseface
        ...  (×12)

    This is different from the same-WORD loop caught by strip_repeated_tokens.
    Here the repeated unit is an entire line.

    Algorithm:
    1. Count occurrences of each stripped non-empty line.
    2. Any line that appears ≥ 3 times is a loop artefact — remove ALL copies.
       The threshold is 3 (not 2) to preserve legitimate repeated short lines
       like "---" separators or "Always consult a dermatologist." if it appears
       twice.

    >>> strip_repeated_lines("A\\nB\\nB\\nB\\nC")
    'A\\nC'
    """
    lines = text.split("\n")
    counts: Counter[str] = Counter(
        line.strip() for line in lines if line.strip()
    )
    junk: set[str] = {line for line, n in counts.items() if n >= 3}
    if not junk:
        return text
    return "\n".join(line for line in lines if line.strip() not in junk)


def strip_trailing_fragment(text: str) -> str:
    """Remove short orphaned text after the last sentence-ending punctuation.

    After token-stream stripping, a few junk words (e.g. "Itchy Dry The") can
    be left dangling at the end.  If the last sentence-ending punctuation (. ! ?)
    is followed by fewer than 40 characters that don't themselves end a sentence,
    those trailing characters are noise and get trimmed.

    >>> strip_trailing_fragment("Good diagnosis. Itchy Dry The")
    'Good diagnosis.'
    >>> strip_trailing_fragment("Good diagnosis. This is a real sentence.")
    'Good diagnosis. This is a real sentence.'
    """
    last_punct = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_punct == -1:
        return text
    after = text[last_punct + 1:]
    if 0 < len(after.strip()) < 40:
        return text[: last_punct + 1].rstrip()
    return text


def _junk_ratio(text: str) -> float:
    """Return the fraction of characters that are repeated-word junk tokens.

    Used by is_valid_diagnosis_response to reject outputs dominated by garbage.
    """
    if not text:
        return 1.0
    junk_chars = sum(len(m.group()) for m in _RE_REPEATED_WORD.finditer(text))
    junk_chars += sum(len(m.group()) for m in _RE_UNICODE_JUNK.finditer(text))
    return junk_chars / len(text)


# ---------------------------------------------------------------------------
# Composed normaliser — the main entry point
# ---------------------------------------------------------------------------

def normalize_response_text(text: str) -> str:
    """Apply the full normalisation pipeline to a raw model output string.

    Pipeline (in order):
    1.  Strip Unicode junk characters (□ ▪ etc.) — must run first so later
        patterns operate on clean ASCII/UTF-8 text.
    2.  Strip leaked think headers (debug lines without <think> tags).
    3.  Strip fake continuation markers ([user]/[email]/…) and everything after.
    4.  Strip repeated token loops (dense.dense.dense…).
    5.  Strip long token-stream runs.
    6.  Strip proper <think>…</think> blocks.
    7.  Try to extract the first [DIAGNOSIS] block.
        - If found: use its content going forward.
        - If not: use the full text (already cleaned above).
    8.  Remove residual [DIAGNOSIS]/[/DIAGNOSIS] tag fragments.
    9.  Remove known noise tags ([END_SYMPTOMS], [NOT_AVAILABLE], etc.).
    10. Strip trailing bracket/paren artifacts.
    11. Collapse excessive blank lines.
    12. Final strip of leading/trailing whitespace.

    Args:
        text: Raw string from the model.

    Returns:
        Clean, user-facing string.
    """
    # Step 1 — unicode junk (run before any regex that matches word chars)
    text = strip_unicode_junk(text)

    # Step 2 — leaked think-style header without tags
    text = strip_leaked_think_header(text)

    # Step 3 — fake turn continuation (everything after [user]/[email]/etc.)
    text = strip_fake_continuation(text)

    # Step 4 — repeated token loops
    text = strip_repeated_tokens(text)

    # Step 5a — long token-stream runs
    text = strip_token_streams(text)

    # Step 5b — repeated-sentence loops (e.g. "DenseNet model visualized: URL" ×12)
    text = strip_repeated_lines(text)

    # Step 6 — proper <think>…</think> chain-of-thought blocks
    text = strip_think_blocks(text)

    # Step 7 — extract diagnosis block if present, else keep full cleaned text
    diagnosis_content = extract_first_diagnosis_block(text)
    if diagnosis_content is not None:
        text = diagnosis_content

    # Step 8 — drop any leftover [DIAGNOSIS] tag fragments
    text = strip_residual_diagnosis_tags(text)

    # Step 9 — drop known internal control tags
    text = remove_known_noise_tags(text)

    # Step 10 — trailing artifact cleanup
    text = strip_trailing_artifacts(text)

    # Step 10b — trim orphaned trailing fragments left after token-stream removal.
    # If the text ends with a short fragment (< 40 chars) after the last sentence-
    # ending punctuation, it's almost certainly junk residue, not real content.
    text = strip_trailing_fragment(text)

    # Step 11 — collapse blank lines
    text = collapse_blank_lines(text)

    # Step 12 — final whitespace trim
    return text.strip()


# ---------------------------------------------------------------------------
# Convenience alias — the name nodes should import
# ---------------------------------------------------------------------------

def parse_model_response(raw: str) -> str:
    """Alias for normalize_response_text.

    Node files import this name so the call sites read naturally:

        from langgraph_orchestrator.utils.response_parser import parse_model_response
        clean = parse_model_response(raw_llm_output)
    """
    return normalize_response_text(raw)


# ---------------------------------------------------------------------------
# Diagnosis validation — used by joint_diagnosis.py before locking the flag
# ---------------------------------------------------------------------------

def is_valid_diagnosis_response(text: str) -> bool:
    """Return True if the diagnosis content is non-empty and meaningful.

    Used by joint_diagnosis.py to decide whether to commit the
    diagnosis_done = True milestone.

    Validation rules (all must pass):
    1. Non-empty after whitespace stripping.
    2. At least 40 characters of actual content after stripping noise tags
       (raised from 20 — real model can produce short-but-legit noise strings
       that a 20-char threshold would accept).
    3. Junk ratio < 0.30 — output must not be dominated by repeated tokens,
       unicode garbage, or token-stream artefacts.
    4. Contains at least one sentence-ending punctuation mark (. ! ?) — a
       response made entirely of fragments or partial tokens is rejected.
    5. Does not still contain a fake-turn marker ([user]/[email]) — indicates
       cleanup failed to strip a hallucinated continuation.

    Note: this function expects PARSED content — the inner text of the
    [DIAGNOSIS] block already extracted (or the full cleaned text if no block
    was present).  Run normalize_response_text() before calling this.

    Args:
        text: Parsed diagnosis content string.

    Returns:
        True if the diagnosis is substantive and should lock in diagnosis_done.
    """
    if not text or not text.strip():
        return False

    content = remove_known_noise_tags(text).strip()

    # Rule 2 — minimum length
    if len(content) < 40:
        return False

    # Rule 3 — junk ratio (word loops, unicode garbage)
    if _junk_ratio(content) >= 0.30:
        return False

    # Rule 4 — must contain at least one sentence-end punctuation
    if not re.search(r"[.!?]", content):
        return False

    # Rule 5 — no residual fake-turn markers
    if re.search(r"\[(user|email|inst)\]", content, re.IGNORECASE):
        return False

    # Rule 6 — no URLs (clinical text never contains http links; their presence
    # means the model hallucinated a "model visualized: URL" loop or similar)
    if _RE_URL.search(content):
        return False

    # Rule 7 — no repeated-line loop survived cleanup (safety net: if any non-
    # trivial line still appears 3+ times the output is still a loop artefact)
    non_empty_lines = [l.strip() for l in content.split("\n") if len(l.strip()) > 10]
    if non_empty_lines:
        most_common_count = Counter(non_empty_lines).most_common(1)[0][1]
        if most_common_count >= 3:
            return False

    return True
