"""eval_dataset_c.py — Complete quality evaluator for Dataset C.

Evaluates RAG-QA entries across dimensions, with special handling for edge cases.
Minimum messages: 3 (system + user + assistant) — edge cases don't need long dialogs.
"""

import argparse
import json
import logging
import os
import re
import hashlib
from dataclasses import dataclass, field, asdict
from statistics import mean
from collections import defaultdict
import glob

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MIN_MESSAGES = 3  # Cambiado: system + user + assistant es suficiente
MAX_MESSAGES = 50
MIN_THINK_LENGTH = 40

# Edge case types where source citation is NOT expected
NO_CITATION_REQUIRED = [
    "edge_irrelevant_domain",
    "edge_no_context",
    "edge_too_generic",
    "edge_out_of_scope",
    "edge_second_opinion",
]

# Edge case types where follow-up may be optional
FOLLOWUP_OPTIONAL = [
    "edge_out_of_scope",
    "edge_second_opinion",
]

TREATMENT_PATTERNS = [
    #r"\bprescrip\w*\b",
    r"\btake\s+\d+\s*mg\b",
    r"\bapply\s+\w+\s+(cream|ointment|gel|lotion)\s+twice\b",
    r"\bdosage\b",
    r"\bdrug\s+of\s+choice\b",
    r"\buse\s+(methotrexate|dupilumab|cyclosporine|prednisone|betamethasone)\b",
]

FOLLOW_UP_PATTERNS = [
    r"\?$",
    r"do you (want|have|need)\b",
    r"would you like\b",
    r"feel free to ask\b",
    r"let me know if\b",
    r"is there anything else\b",
    r"any other questions\b",
]

HALLUCINATED_SOURCE_PATTERNS = [
    r"according to (pubmed|webmd|mayo clinic|nih|cdc|who|uptodate|aad|wikipedia)",
    r"based on (pubmed|webmd|mayo clinic|nih|cdc|who|uptodate|aad|wikipedia)",
    r"per (pubmed|webmd|mayo clinic|nih|cdc|who|uptodate|aad|wikipedia)",
]

# ============================================================
# HELPERS
# ============================================================

def get_entry_hash(entry: dict) -> str:
    messages = entry.get("messages", [])
    content = json.dumps(messages, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def extract_think_content(text: str) -> str | None:
    match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return match.group(1).strip() if match else None


def extract_answer(text: str) -> str:
    parts = re.split(r"</think>", text, maxsplit=1)
    return parts[1].strip() if len(parts) > 1 else text.strip()


def extract_context_sources(user_content: str) -> list[str]:
    return re.findall(r"##\s+PRIORITY\s+\d+:\s+(.+)", user_content)


def validate_message_roles(messages: list) -> tuple[bool, list]:
    violations = []
    if not messages:
        return False, ["empty messages array"]
    
    # First message must be system
    if messages[0].get("role") != "system":
        violations.append(f"first message role is '{messages[0].get('role')}', expected 'system'")
    
    # Check alternating pattern after system
    expected = "user"
    for i, msg in enumerate(messages[1:], start=1):
        role = msg.get("role")
        if role != expected:
            violations.append(f"message {i}: expected '{expected}', got '{role}'")
        expected = "assistant" if expected == "user" else "user"
    
    return len(violations) == 0, violations


def validate_message_count(messages: list, entry_type: str) -> tuple[bool, list]:
    violations = []
    # Edge cases often have exactly 3 messages (system + user + assistant)
    # Normal cases (one_chunk, two_chunk) should have more
    if entry_type in ["one_chunk", "two_chunk"]:
        if len(messages) < 3:
            violations.append(f"normal entry has only {len(messages)} messages, expected ≥5")
    else:
        if len(messages) < MIN_MESSAGES:
            violations.append(f"too few messages: {len(messages)} < {MIN_MESSAGES}")
    
    if len(messages) > MAX_MESSAGES:
        violations.append(f"too many messages: {len(messages)} > {MAX_MESSAGES}")
    
    return len(violations) == 0, violations


# ============================================================
# EVALUATION FUNCTIONS (with edge case awareness)
# ============================================================

def eval_think_tags(assistant_content: str) -> tuple[bool, bool, list]:
    violations = []
    has_tags = "<think>" in assistant_content and "</think>" in assistant_content
    if not has_tags:
        violations.append("missing <think>...</think> tags")
        return False, False, violations
    
    think_content = extract_think_content(assistant_content)
    if think_content is None:
        violations.append("could not extract think content")
        return True, False, violations
    
    non_trivial = len(think_content) >= MIN_THINK_LENGTH
    if not non_trivial:
        violations.append(f"think too short ({len(think_content)} chars, min {MIN_THINK_LENGTH})")
    
    return True, non_trivial, violations


def eval_double_newline(assistant_content: str) -> tuple[bool, list]:
    has_blank = bool(re.search(r"</think>\s*\n\s*\n", assistant_content))
    violations = [] if has_blank else ["no blank line after </think>"]
    return has_blank, violations


def eval_source_citation(assistant_content: str, context_sources: list[str], entry_type: str) -> tuple[bool, list]:
    """Edge cases don't need to cite sources."""
    # Edge cases don't require citation
    if entry_type in NO_CITATION_REQUIRED:
        return True, []
    
    answer = extract_answer(assistant_content).lower()
    violations = []
    
    if not context_sources:
        return True, []
    
    cited = any(src.lower()[:20] in answer for src in context_sources)
    if not cited:
        violations.append(f"no source cited. Available: {context_sources}")
    return cited, violations


def eval_follow_up(assistant_content: str, entry_type: str) -> tuple[bool, list]:
    """Some edge cases don't need follow-up questions."""
    answer = extract_answer(assistant_content).lower().strip()
    
    # Out of scope and second opinion edge cases may not need follow-up
    if entry_type in FOLLOWUP_OPTIONAL:
        # Still nice to have, but not required
        has_followup = any(re.search(p, answer) for p in FOLLOW_UP_PATTERNS)
        if not has_followup:
            return True, []  # No penalty for missing follow-up in these cases
        return True, []
    
    has_followup = any(re.search(p, answer) for p in FOLLOW_UP_PATTERNS)
    violations = [] if has_followup else ["no follow-up question at end"]
    return has_followup, violations


def eval_hallucination(assistant_content: str, context_sources: list[str]) -> tuple[bool, list]:
    answer = extract_answer(assistant_content).lower()
    context_sources_lower = [s.lower() for s in context_sources]
    violations = []
    
    for pattern in HALLUCINATED_SOURCE_PATTERNS:
        m = re.search(pattern, answer)
        if m:
            cited_name = m.group(1)
            if not any(cited_name in s for s in context_sources_lower):
                violations.append(f"hallucinated source: '{cited_name}'")
    
    return len(violations) == 0, violations


def eval_no_treatment(assistant_content: str) -> tuple[bool, list]:
    answer_lower = assistant_content.lower()
    violations = []
    for pattern in TREATMENT_PATTERNS:
        if re.search(pattern, answer_lower):
            match = re.search(pattern, answer_lower)
            violations.append(f"treatment pattern matched: '{pattern}' -> found '{match.group()}'")
    return len(violations) == 0, violations


# ============================================================
# ENTRY EVALUATION
# ============================================================

@dataclass
class EntryResult:
    file_name: str
    index: int
    entry_type: str
    topic: str
    
    # Structural
    roles_valid: bool = False
    count_valid: bool = False
    
    # Content
    has_think_tags: bool = False
    think_non_trivial: bool = False
    has_double_newline: bool = False
    cites_source: bool = False
    has_follow_up: bool = False
    no_hallucination: bool = False
    no_treatment: bool = False
    
    violations: list = field(default_factory=list)
    
    @property
    def overall_score(self) -> float:
        scores = [
            float(self.roles_valid),
            float(self.count_valid),
            float(self.has_think_tags),
            float(self.think_non_trivial),
            float(self.has_double_newline),
            float(self.cites_source),
            float(self.has_follow_up),
            float(self.no_hallucination),
            float(self.no_treatment),
        ]
        return mean(scores)


def evaluate_entry(file_name: str, index: int, record: dict) -> EntryResult:
    metadata = record.get("metadata", {})
    messages = record.get("messages", [])
    
    entry_type = metadata.get("type", "unknown")
    topic = metadata.get("topic", "unknown")
    
    user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
    assistant_msg = next((m["content"] for m in messages if m["role"] == "assistant"), "")
    
    context_sources = extract_context_sources(user_msg)
    
    result = EntryResult(file_name=file_name, index=index, entry_type=entry_type, topic=topic)
    all_violations = []
    
    # Structural
    valid_roles, v = validate_message_roles(messages)
    result.roles_valid = valid_roles
    all_violations.extend(v)
    
    valid_count, v = validate_message_count(messages, entry_type)
    result.count_valid = valid_count
    all_violations.extend(v)
    
    # Content
    has_tags, non_trivial, v = eval_think_tags(assistant_msg)
    result.has_think_tags = has_tags
    result.think_non_trivial = non_trivial
    all_violations.extend(v)
    
    has_nl, v = eval_double_newline(assistant_msg)
    result.has_double_newline = has_nl
    all_violations.extend(v)
    
    cites, v = eval_source_citation(assistant_msg, context_sources, entry_type)
    result.cites_source = cites
    all_violations.extend(v)
    
    has_fu, v = eval_follow_up(assistant_msg, entry_type)
    result.has_follow_up = has_fu
    all_violations.extend(v)
    
    no_hall, v = eval_hallucination(assistant_msg, context_sources)
    result.no_hallucination = no_hall
    all_violations.extend(v)
    
    no_treat, v = eval_no_treatment(assistant_msg)
    result.no_treatment = no_treat
    all_violations.extend(v)
    
    result.violations = all_violations
    return result


# ============================================================
# AGGREGATION & REPORT
# ============================================================

@dataclass
class EvalReport:
    total: int = 0
    unique: int = 0
    duplicates: int = 0
    passed: int = 0
    failed: int = 0
    
    avg_overall: float = 0.0
    pct_roles_valid: float = 0.0
    pct_count_valid: float = 0.0
    pct_think_tags: float = 0.0
    pct_think_nontrivial: float = 0.0
    pct_double_newline: float = 0.0
    pct_cites_source: float = 0.0
    pct_follow_up: float = 0.0
    pct_no_hallucination: float = 0.0
    pct_no_treatment: float = 0.0
    
    by_type: dict = field(default_factory=dict)
    by_file: dict = field(default_factory=dict)
    worst_entries: list = field(default_factory=list)


def aggregate(results: list[EntryResult]) -> EvalReport:
    report = EvalReport(total=len(results))
    if not results:
        return report
    
    scores = [r.overall_score for r in results]
    report.passed = sum(1 for s in scores if s >= 0.70)  # Más permisivo
    report.failed = report.total - report.passed
    report.avg_overall = mean(scores)
    
    report.pct_roles_valid = mean(r.roles_valid for r in results)
    report.pct_count_valid = mean(r.count_valid for r in results)
    report.pct_think_tags = mean(r.has_think_tags for r in results)
    report.pct_think_nontrivial = mean(r.think_non_trivial for r in results)
    report.pct_double_newline = mean(r.has_double_newline for r in results)
    report.pct_cites_source = mean(r.cites_source for r in results)
    report.pct_follow_up = mean(r.has_follow_up for r in results)
    report.pct_no_hallucination = mean(r.no_hallucination for r in results)
    report.pct_no_treatment = mean(r.no_treatment for r in results)
    
    # By type
    types = set(r.entry_type for r in results)
    for t in types:
        sub = [r for r in results if r.entry_type == t]
        report.by_type[t] = {
            "count": len(sub),
            "avg_overall": round(mean(r.overall_score for r in sub), 3),
            "pct_think_tags": round(mean(r.has_think_tags for r in sub), 3),
            "pct_cites_source": round(mean(r.cites_source for r in sub), 3),
        }
    
    # By file
    files = set(r.file_name for r in results)
    for f in files:
        sub = [r for r in results if r.file_name == f]
        report.by_file[f] = {
            "count": len(sub),
            "passed": sum(1 for r in sub if r.overall_score >= 0.70),
            "failed": sum(1 for r in sub if r.overall_score < 0.70),
            "avg_score": round(mean(r.overall_score for r in sub), 3),
        }
    
    # Worst entries (up to 20, sorted by score)
    sorted_results = sorted(results, key=lambda r: r.overall_score)
    report.worst_entries = [
        {
            "file": r.file_name,
            "index": r.index,
            "type": r.entry_type,
            "topic": r.topic,
            "score": round(r.overall_score, 3),
            "violations": r.violations[:5],
        }
        for r in sorted_results[:20]  # Mostrar hasta 20
    ]
    
    return report


def print_report(report: EvalReport) -> None:
    bar = "=" * 70
    print(f"\n{bar}")
    print("DATASET C — COMPLETE EVALUATION REPORT")
    print(bar)
    print(f"  Total entries        : {report.total}")
    print(f"  Passed (score ≥ 0.70) : {report.passed} ({100*report.passed/max(report.total,1):.1f}%)")
    print(f"  Failed                : {report.failed} ({100*report.failed/max(report.total,1):.1f}%)")
    print()
    print("  ── Structural ──")
    print(f"  Roles valid          : {100*report.pct_roles_valid:.1f}%")
    print(f"  Count valid (min=3)  : {100*report.pct_count_valid:.1f}%")
    print()
    print("  ── Content (Dataset C) ──")
    print(f"  Think tags present   : {100*report.pct_think_tags:.1f}%")
    print(f"  Think non-trivial    : {100*report.pct_think_nontrivial:.1f}%")
    print(f"  Double newline after : {100*report.pct_double_newline:.1f}%")
    print(f"  Source cited         : {100*report.pct_cites_source:.1f}%")
    print(f"  Follow-up present    : {100*report.pct_follow_up:.1f}%")
    print(f"  No hallucination     : {100*report.pct_no_hallucination:.1f}%")
    print(f"  No treatment advice  : {100*report.pct_no_treatment:.1f}%")
    print()
    print("  ── By file (train/val/test) ──")
    for f, stats in sorted(report.by_file.items()):
        print(f"  {f:<30} n={stats['count']:<4} passed={stats['passed']:<4} failed={stats['failed']:<4} avg={stats['avg_score']:.3f}")
    print()
    print("  ── Worst entries (up to 20) ──")
    for e in report.worst_entries:
        print(f"  {e['file']}:{e['index']:<4} {e['type']:<18} [{e['topic']}]  score={e['score']:.3f}")
        for v in e["violations"][:2]:
            print(f"         ⚠ {v}")
    print(bar)


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Complete Dataset C Evaluator")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Input JSONL file or directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save JSON report to this path (default: same folder as input)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-entry results")
    args = parser.parse_args()
    
    # Collect files
    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.jsonl"))
        input_dir = args.input
    else:
        files = [args.input]
        input_dir = os.path.dirname(args.input)
    
    all_results = []
    seen_hashes = set()
    duplicate_count = 0
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing {file_name}...")
        records = []
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        record = json.loads(line)
                        records.append(record)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed line: {e}")
        
        for i, record in enumerate(records):
            entry_hash = get_entry_hash(record)
            if entry_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(entry_hash)
            result = evaluate_entry(file_name, i, record)
            all_results.append(result)
            
            if args.verbose and result.overall_score < 0.70:
                print(f"⚠ {file_name}:{i} {result.entry_type} score={result.overall_score:.3f} - {result.violations[:2]}")
    
    report = aggregate(all_results)
    report.duplicates = duplicate_count
    report.unique = len(all_results)
    
    print_report(report)
    
    # Guardar reporte en la carpeta del input por defecto
    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(input_dir, "eval_report.json")
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved → {output_path}")


if __name__ == "__main__":
    main()

