"""eval_dataset_b.py — Automatic quality evaluator for Dataset B.

Evaluates symptom-collection dialogues across 7 dimensions:
  1. Structural integrity   — roles alternate, starts with user, ends with assistant
  2. Message count          — within reasonable range (3-50 messages)
  3. Symptom closure        — [END_SYMPTOMS] present in last assistant message
  4. Clinical coverage      — how many of the 12 domains are addressed
  5. Single-question rule   — assistant never asks >1 question per turn
  6. Identity compliance    — assistant never diagnoses, recommends, or goes off-role
  7. Message length variety — mix of short and long messages

Supports both .jsonl (line by line) and .json (full array) formats.

Usage:
    python eval_dataset_b.py --input datasets/dataset_b/train.jsonl
    python eval_dataset_b.py --input prepare_datasets/temporal_datasets/dataset_b/
    python eval_dataset_b.py --input prepare_datasets/temporal_datasets/dataset_b/ --verbose
"""

import argparse
import json
import logging
import os
import re
import hashlib
import glob
from dataclasses import dataclass, field, asdict
from statistics import mean, stdev
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# CONSTANTS
# ============================================================

MIN_MESSAGES = 3
MAX_MESSAGES = 50

CLINICAL_DOMAINS = {
    "onset_duration":     ["when", "how long", "started", "began", "ago", "duration", "onset", "first notice"],
    "morphology":         ["shape", "size", "color", "colour", "texture", "border", "scale", "scaly", "raised", "flat", "bump", "blister", "patch", "plaque"],
    "distribution":       ["spread", "where", "location", "area", "both", "one side", "symmetric", "localized", "body part"],
    "sensations":         ["itch", "itching", "pain", "burn", "burning", "tingle", "tingling", "sting", "hurt", "discomfort", "sensation"],
    "temporal_patterns":  ["constant", "intermittent", "comes and goes", "worse at", "time of day", "night", "morning", "seasonal", "flare"],
    "aggravating":        ["worse", "aggravate", "trigger", "heat", "sweat", "sun", "stress", "food", "friction", "soap", "detergent"],
    "alleviating":        ["better", "help", "relief", "improve", "reduce", "cream", "cool", "cold", "alleviate"],
    "prior_treatments":   ["tried", "treatment", "cream", "medication", "medicine", "over the counter", "otc", "home remedy", "applied", "used"],
    "triggers_exposures": ["new product", "travel", "contact", "allerg", "exposure", "changed", "diet", "cosmetic", "laundry"],
    "systemic_symptoms":  ["fever", "fatigue", "tired", "joint", "lymph", "swollen", "weight", "nausea", "systemic"],
    "personal_history":   ["history", "allerg", "atop", "asthma", "skin condition", "before", "previously", "chronic"],
    "family_history":     ["family", "parent", "mother", "father", "sibling", "brother", "sister", "relative", "heredit", "genetic"],
}

FORBIDDEN_PATTERNS = [
    r"\bdiagnos\w*\b",
    r"\bprescri\w*\b",
    r"\btake\s+\d+\s*mg\b",
    r"\brecommend\s+(using|taking|applying)\s+\w+\s+(cream|pill|medication|drug)\b",
    r"\byou\s+have\s+(psoriasis|eczema|acne|rosacea|dermatitis)\b",
    r"\bcall\s+(the\s+)?(clinic|doctor|hospital|us)\b",
    r"\bphone\b",
    r"\bappointment\b",
    r"\bemergency\s+room\b",
    r"\b911\b",
    r"\bER\b",
]

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_entry_hash(record: dict) -> str:
    messages = record.get("messages", [])
    content = json.dumps(messages, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()


def strip_system_message(messages: list) -> list:
    return [m for m in messages if m["role"] != "system"]


def load_records_from_file(file_path: str) -> list:
    """Load records from .jsonl (line by line) or .json (full array)."""
    records = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        
        if file_path.endswith(".jsonl"):
            for line in content.splitlines():
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"JSONL parse error in {file_path}: {e}")
        else:
            try:
                data = json.loads(content)
                if isinstance(data, list):
                    records = data
                else:
                    records = [data]
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error in {file_path}: {e}")
    
    return records


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def check_structure(messages: list) -> tuple[bool, list]:
    violations = []
    msgs = strip_system_message(messages)

    if not msgs:
        return False, ["empty dialogue"]
    if msgs[0]["role"] != "user":
        violations.append("does not start with user")
    for i in range(len(msgs) - 1):
        if msgs[i]["role"] == msgs[i + 1]["role"]:
            violations.append(f"consecutive same role at turn {i}")
    if msgs[-1]["role"] != "assistant":
        violations.append("does not end with assistant")
    return len(violations) == 0, violations


def check_message_count(messages: list) -> tuple[bool, list]:
    violations = []
    msgs = strip_system_message(messages)
    
    if len(msgs) < MIN_MESSAGES:
        violations.append(f"too few messages: {len(msgs)} < {MIN_MESSAGES}")
    if len(msgs) > MAX_MESSAGES:
        violations.append(f"too many messages: {len(msgs)} > {MAX_MESSAGES}")
    
    return len(violations) == 0, violations


def check_end_symptoms(messages: list) -> bool:
    msgs = strip_system_message(messages)
    assistant_msgs = [m for m in msgs if m["role"] == "assistant"]
    if not assistant_msgs:
        return False
    return "[END_SYMPTOMS]" in assistant_msgs[-1]["content"]


def check_clinical_coverage(messages: list) -> float:
    full_text = " ".join(m["content"] for m in messages).lower()
    covered = sum(
        1 for keywords in CLINICAL_DOMAINS.values()
        if any(kw in full_text for kw in keywords)
    )
    return covered / len(CLINICAL_DOMAINS)


def check_single_question_rule(messages: list) -> tuple[float, list]:
    msgs = strip_system_message(messages)
    assistant_turns = [(i, m) for i, m in enumerate(msgs) if m["role"] == "assistant"]
    
    if not assistant_turns:
        return 1.0, []

    violations = []
    for i, msg in assistant_turns:
        q_count = msg["content"].count("?")
        if q_count > 1:
            violations.append(i)

    rate = 1.0 - len(violations) / len(assistant_turns)
    return rate, violations


def check_identity_compliance(messages: list) -> tuple[bool, list]:
    violations = []
    msgs = strip_system_message(messages)
    
    for i, msg in enumerate(msgs):
        if msg["role"] != "assistant":
            continue
        content_lower = msg["content"].lower()
        for pattern in FORBIDDEN_PATTERNS:
            if re.search(pattern, content_lower):
                violations.append(f"turn {i}: matched '{pattern}'")
    return len(violations) == 0, violations


def check_length_variety(messages: list) -> float:
    msgs = strip_system_message(messages)
    lengths = [len(m["content"]) for m in msgs]
    
    if len(lengths) < 2:
        return 0.0
    m = mean(lengths)
    if m == 0:
        return 0.0
    cv = stdev(lengths) / m
    return min(cv / 0.5, 1.0)


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass
class DialogueResult:
    file_name: str
    index: int
    edge_type: str | None
    dialogue_type: str
    num_turns: int

    structural_ok: bool = False
    count_ok: bool = False
    has_end_symptoms: bool = False
    clinical_coverage: float = 0.0
    single_q_rate: float = 0.0
    identity_ok: bool = False
    length_variety: float = 0.0

    violations: list = field(default_factory=list)

    @property
    def overall_score(self) -> float:
        scores = [
            float(self.structural_ok),
            float(self.count_ok),
            float(self.has_end_symptoms),
            self.clinical_coverage,
            self.single_q_rate,
            float(self.identity_ok),
            min(self.length_variety, 1.0),
        ]
        return mean(scores)


def evaluate_dialogue(file_name: str, index: int, record: dict) -> DialogueResult:
    messages = record.get("messages", [])
    metadata = record.get("metadata", {})
    edge_type = metadata.get("edge_type")
    dtype = metadata.get("type", "unknown")
    num_turns = len([m for m in messages if m["role"] != "system"])

    result = DialogueResult(
        file_name=file_name,
        index=index,
        edge_type=edge_type,
        dialogue_type=dtype,
        num_turns=num_turns,
    )

    all_violations = []

    struct_ok, struct_v = check_structure(messages)
    result.structural_ok = struct_ok
    all_violations.extend(struct_v)

    count_ok, count_v = check_message_count(messages)
    result.count_ok = count_ok
    all_violations.extend(count_v)

    result.has_end_symptoms = check_end_symptoms(messages)
    if not result.has_end_symptoms:
        all_violations.append("missing [END_SYMPTOMS]")

    result.clinical_coverage = check_clinical_coverage(messages)

    sq_rate, sq_v = check_single_question_rule(messages)
    result.single_q_rate = sq_rate
    for v in sq_v:
        all_violations.append(f"double-question at turn {v}")

    id_ok, id_v = check_identity_compliance(messages)
    result.identity_ok = id_ok
    all_violations.extend(id_v)

    result.length_variety = check_length_variety(messages)
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
    avg_clinical_coverage: float = 0.0
    avg_single_q_rate: float = 0.0
    avg_length_variety: float = 0.0

    pct_structural_ok: float = 0.0
    pct_count_ok: float = 0.0
    pct_end_symptoms: float = 0.0
    pct_identity_ok: float = 0.0

    by_type: dict = field(default_factory=dict)
    by_file: dict = field(default_factory=dict)
    worst_dialogues: list = field(default_factory=list)


def aggregate(results: list[DialogueResult]) -> EvalReport:
    report = EvalReport(total=len(results))
    if not results:
        return report

    scores = [r.overall_score for r in results]
    report.passed = sum(1 for s in scores if s >= 0.70)
    report.failed = report.total - report.passed

    report.avg_overall = mean(scores)
    report.avg_clinical_coverage = mean(r.clinical_coverage for r in results)
    report.avg_single_q_rate = mean(r.single_q_rate for r in results)
    report.avg_length_variety = mean(r.length_variety for r in results)
    report.pct_structural_ok = mean(r.structural_ok for r in results)
    report.pct_count_ok = mean(r.count_ok for r in results)
    report.pct_end_symptoms = mean(r.has_end_symptoms for r in results)
    report.pct_identity_ok = mean(r.identity_ok for r in results)

    types = set(r.dialogue_type for r in results)
    for t in types:
        sub = [r for r in results if r.dialogue_type == t]
        report.by_type[t] = {
            "count": len(sub),
            "avg_overall": round(mean(r.overall_score for r in sub), 3),
            "avg_coverage": round(mean(r.clinical_coverage for r in sub), 3),
            "pct_end_symptoms": round(mean(r.has_end_symptoms for r in sub), 3),
            "pct_identity_ok": round(mean(r.identity_ok for r in sub), 3),
        }

    files = set(r.file_name for r in results)
    for f in files:
        sub = [r for r in results if r.file_name == f]
        report.by_file[f] = {
            "count": len(sub),
            "passed": sum(1 for r in sub if r.overall_score >= 0.70),
            "failed": sum(1 for r in sub if r.overall_score < 0.70),
            "avg_score": round(mean(r.overall_score for r in sub), 3),
        }

    sorted_results = sorted(results, key=lambda r: r.overall_score)
    report.worst_dialogues = [
        {
            "file": r.file_name,
            "index": r.index,
            "type": r.dialogue_type,
            "edge_type": r.edge_type,
            "score": round(r.overall_score, 3),
            "violations": r.violations[:5],
        }
        for r in sorted_results[:10]
    ]

    return report


def print_report(report: EvalReport) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print("DATASET B — EVALUATION REPORT")
    print(bar)
    print(f"  Total dialogues      : {report.total}")
    print(f"  Unique               : {report.unique}")
    print(f"  Duplicates           : {report.duplicates}")
    print(f"  Passed (score≥0.70)  : {report.passed}  ({100*report.passed/max(report.total,1):.1f}%)")
    print(f"  Failed               : {report.failed}  ({100*report.failed/max(report.total,1):.1f}%)")
    print()
    print("  ── Dimension averages ──")
    print(f"  Overall score        : {report.avg_overall:.3f}")
    print(f"  Clinical coverage    : {report.avg_clinical_coverage:.3f}")
    print(f"  Single-question rate : {report.avg_single_q_rate:.3f}")
    print(f"  Length variety       : {report.avg_length_variety:.3f}")
    print(f"  Structural OK        : {100*report.pct_structural_ok:.1f}%")
    print(f"  Message count OK     : {100*report.pct_count_ok:.1f}%")
    print(f"  END_SYMPTOMS present : {100*report.pct_end_symptoms:.1f}%")
    print(f"  Identity compliance  : {100*report.pct_identity_ok:.1f}%")
    print()
    print("  ── By file ──")
    for f, stats in sorted(report.by_file.items()):
        print(f"  {f:<30} n={stats['count']:<4} passed={stats['passed']:<4} failed={stats['failed']:<4} avg={stats['avg_score']:.3f}")
    print()
    print("  ── By dialogue type ──")
    for t, stats in sorted(report.by_type.items()):
        print(f"  {t:<20} n={stats['count']:<4} score={stats['avg_overall']:.3f} coverage={stats['avg_coverage']:.3f}")
    print()
    print("  ── Worst 10 dialogues ──")
    for d in report.worst_dialogues:
        et = f" [{d['edge_type']}]" if d['edge_type'] else ""
        print(f"  {d['file']}:{d['index']:<4} {d['type']}{et:<30} score={d['score']:.3f}")
        for v in d['violations'][:2]:
            print(f"         ⚠ {v}")
    print(bar)


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Dataset B")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to JSONL/JSON file or directory")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Save JSON report to this path")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print per-dialogue results")
    args = parser.parse_args()

    if os.path.isdir(args.input):
        files = glob.glob(os.path.join(args.input, "*.json")) + \
                glob.glob(os.path.join(args.input, "*.jsonl"))
        input_dir = args.input
    else:
        files = [args.input]
        input_dir = os.path.dirname(args.input)

    if not files:
        logger.error(f"No JSON/JSONL files found in {args.input}")
        return

    all_results = []
    seen_hashes = set()
    duplicate_count = 0

    for file_path in files:
        file_name = os.path.basename(file_path)
        logger.info(f"Processing {file_name}...")
        
        records = load_records_from_file(file_path)

        for i, record in enumerate(records):
            if "messages" not in record:
                logger.warning(f"Skipping record {i} in {file_name}: no 'messages' field")
                continue
            
            entry_hash = get_entry_hash(record)
            if entry_hash in seen_hashes:
                duplicate_count += 1
                continue
            seen_hashes.add(entry_hash)
            
            result = evaluate_dialogue(file_name, i, record)
            all_results.append(result)
            
            if args.verbose and result.overall_score < 0.70:
                et = f" [{result.edge_type}]" if result.edge_type else ""
                print(f"⚠ {file_name}:{i} {result.dialogue_type}{et} score={result.overall_score:.3f}")

    report = aggregate(all_results)
    report.unique = len(all_results)
    report.duplicates = duplicate_count
    report.total = len(all_results) + duplicate_count

    print_report(report)

    if args.output:
        output_path = args.output
    else:
        output_path = os.path.join(input_dir, "eval_report_b.json")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved → {output_path}")


if __name__ == "__main__":
    main()