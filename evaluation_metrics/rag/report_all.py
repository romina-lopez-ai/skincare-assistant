"""report_all.py — Print a consolidated summary from the two evaluation result files.

Reads:
  - results/rag_eval_results.json       (hit rate — from evaluate_hit_rate.py)
  - results/ragas_detailed_results.json (RAGAS   — from evaluate_ragas.py)

Does NOT re-run any evaluation. Pure read + print.

Run from repo root:
    python evaluation_metrics/rag/report_all.py
"""

import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import (
    RESULTS_DIR, RESULTS_PATH, RAGAS_THRESHOLDS,
    HR_THRESHOLD, FP_RATE_THRESHOLD,
)

DETAILED_RESULTS_PATH = RESULTS_DIR / "ragas_detailed_results.json"

CONFIGS = ["finetuned_rag", "base_rag", "finetuned_no_rag", "base_no_rag"]
CONFIG_LABELS = {
    "finetuned_rag":    "Fine-tuned + RAG   [our system]",
    "base_rag":         "Base LLaMA + RAG",
    "finetuned_no_rag": "Fine-tuned, no RAG [parametric]",
    "base_no_rag":      "Base LLaMA, no RAG [pure baseline]",
}


def sep(title: str = "", width: int = 64):
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{'─'*pad} {title} {'─'*pad}")
    else:
        print("─" * width)


def pf(val, threshold, higher_is_better=True):
    if val is None:
        return "N/A"
    ok = val >= threshold if higher_is_better else val <= threshold
    return "PASS" if ok else "FAIL"


def print_hit_rate(hr: dict):
    sep("1. Retrieval Quality — Hit Rate")
    overall = hr.get("overall", {})

    print(f"\n  {'k':<4}  {'Overall':>8}  {'Acne':>8}  {'Eczema':>8}  {'Psoriasis':>8}")
    print("  " + "─" * 46)
    per_cond = hr.get("per_condition", {})
    for k in [1, 2]:
        overall_val = overall.get(f"hr_at_{k}", 0.0)
        flag = f"  [{pf(overall_val, HR_THRESHOLD)}]" if k == 2 else ""
        acne  = per_cond.get("acne",      {}).get(f"hr_at_{k}", 0.0)
        ecz   = per_cond.get("eczema",    {}).get(f"hr_at_{k}", 0.0)
        psor  = per_cond.get("psoriasis", {}).get(f"hr_at_{k}", 0.0)
        print(f"  HR@{k}  {overall_val:>8.4f}  {acne:>8.4f}  {ecz:>8.4f}  {psor:>8.4f}{flag}")

    fp = hr.get("fp_rate", None)
    fp_flag = f"  [{pf(fp, FP_RATE_THRESHOLD, higher_is_better=False)}]" if fp is not None else ""
    print(f"\n  FP rate  {fp:.4f}{fp_flag}  (out-of-index queries that retrieved something)")
    print(f"\n  n_in_index={hr.get('n_in_index')}  n_negatives={hr.get('n_negatives')}")


def print_ragas(rg: dict):
    sep("2. Generation Quality — RAGAS")
    meta    = rg.get("metadata", {})
    summary = rg.get("summary", {})

    print(f"\n  Judge    : {meta.get('judge_model', 'N/A')} via {meta.get('judge_provider', 'N/A')}")
    print(f"  Samples  : {meta.get('n_samples')}  |  Filter: {meta.get('condition_filter', 'all')}")
    print(f"  Run date : {meta.get('run_date', 'N/A')}")

    metrics = list(RAGAS_THRESHOLDS.keys())

    # Header
    col_w = 14
    header = f"  {'Metric':<22}" + "".join(f"  {CONFIG_LABELS[c][:col_w]:>{col_w}}" for c in CONFIGS)
    print(f"\n{header}")
    print("  " + "─" * (22 + (col_w + 2) * len(CONFIGS)))

    for m in metrics:
        threshold = RAGAS_THRESHOLDS[m]
        row = f"  {m:<22}"
        for cfg in CONFIGS:
            agg  = summary.get(cfg, {}).get(m, {})
            mean = agg.get("mean", 0.0)
            std  = agg.get("std",  0.0)
            cell = f"{mean:.3f}±{std:.3f}"
            row += f"  {cell:>{col_w}}"
        # pass/fail on our system only
        our_mean = summary.get("finetuned_rag", {}).get(m, {}).get("mean", 0.0)
        row += f"  [{pf(our_mean, threshold)}] (thr≥{threshold})"
        print(row)

    # Deltas
    print(f"\n  {'Delta (adapter = fine-tuned+RAG − base+RAG)'}")
    print(f"  {'Delta (RAG     = fine-tuned+RAG − fine-tuned+no-RAG)'}")
    print(f"  {'Delta (total   = fine-tuned+RAG − base+no-RAG)'}")
    print(f"\n  {'Metric':<22}  {'Δ adapter':>10}  {'Δ RAG':>10}  {'Δ total':>10}")
    print("  " + "─" * 58)
    for m in metrics:
        ft_rag  = summary.get("finetuned_rag",    {}).get(m, {}).get("mean", 0.0)
        base_rag = summary.get("base_rag",         {}).get(m, {}).get("mean", 0.0)
        ft_nr   = summary.get("finetuned_no_rag",  {}).get(m, {}).get("mean", 0.0)
        base_nr = summary.get("base_no_rag",       {}).get(m, {}).get("mean", 0.0)
        print(f"  {m:<22}  {ft_rag-base_rag:>+10.4f}  {ft_rag-ft_nr:>+10.4f}  {ft_rag-base_nr:>+10.4f}")


def print_per_condition(rg: dict):
    sep("3. Per-condition RAGAS — fine-tuned + RAG")
    per_c   = rg.get("per_condition", {})
    metrics = list(RAGAS_THRESHOLDS.keys())

    short = ["Faith", "AnswRel", "CtxPrec", "CtxRec"]
    print(f"\n  {'Condition':<12}" + "".join(f"  {s:>9}" for s in short) + "   n")
    print("  " + "─" * 56)
    for cond in ["acne", "eczema", "psoriasis"]:
        vals = per_c.get(cond, {}).get("finetuned_rag") or per_c.get(cond, {}).get("ft_rag", {})
        row  = f"  {cond:<12}" + "".join(
            f"  {vals.get(m, {}).get('mean', 0.0):>9.4f}" for m in metrics
        )
        row += f"   {per_c.get(cond, {}).get('n', 0)}"
        print(row)


def print_qualitative(rg: dict, n: int = 3):
    sep("4. Qualitative samples")
    samples = rg.get("samples", [])
    printed = 0
    for s in samples:
        if printed >= n:
            break
        no_ctx = s.get("no_context", False)
        ft_ans = s.get("answers", {}).get("finetuned_rag", "")
        if not ft_ans:
            continue
        print(f"\n  [{s['sample_number']:02d}] {s['condition'].upper()} | {s['query']}")
        print(f"  No context: {no_ctx}")
        answers = s.get("answers", {})
        for cfg in CONFIGS:
            label = CONFIG_LABELS[cfg]
            ans   = answers.get(cfg, "")
            print(f"  {label[:28]}: {ans[:120].strip()}...")
        ragas = s.get("ragas", {})
        ft_scores = ragas.get("finetuned_rag", {})
        scores_str = "  ".join(
            f"{m[:5]}={ft_scores.get(m, 0):.2f}" for m in RAGAS_THRESHOLDS
        )
        print(f"  Scores [fine-tuned+RAG]: {scores_str}")
        printed += 1


def print_threshold_summary(hr: dict, rg: dict):
    sep("5. Pass / Fail — acceptance thresholds")
    summary = rg.get("summary", {})
    ft_rag  = summary.get("finetuned_rag", {})
    fp      = hr.get("fp_rate", None)

    rows = [
        ("HR@2",              hr.get("overall", {}).get("hr_at_2", 0.0), HR_THRESHOLD,        True),
        ("FP rate",           fp if fp is not None else 0.0,             FP_RATE_THRESHOLD,   False),
        ("Faithfulness",      ft_rag.get("faithfulness",      {}).get("mean", 0.0), RAGAS_THRESHOLDS["faithfulness"],      True),
        ("Answer Relevancy",  ft_rag.get("answer_relevancy",  {}).get("mean", 0.0), RAGAS_THRESHOLDS["answer_relevancy"],  True),
        ("Context Precision", ft_rag.get("context_precision", {}).get("mean", 0.0), RAGAS_THRESHOLDS["context_precision"], True),
        ("Context Recall",    ft_rag.get("context_recall",    {}).get("mean", 0.0), RAGAS_THRESHOLDS["context_recall"],    True),
    ]
    print(f"\n  {'Metric':<22}  {'Value':>8}  {'Threshold':>10}  {'Status'}")
    print("  " + "─" * 52)
    for name, val, thr, hi in rows:
        thr_str = f"{'≥' if hi else '≤'} {thr}"
        flag    = pf(val, thr, higher_is_better=hi)
        mark    = "✓" if flag == "PASS" else "✗"
        print(f"  {name:<22}  {val:>8.4f}  {thr_str:>10}  {mark} {flag}")


class _Tee:
    """Writes to both terminal and a file at the same time."""
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            s.flush()


def _run_report(hr_data: dict, rg_data: dict):
    sep("RAG EVALUATION REPORT", width=64)

    if hr_data:
        print_hit_rate(hr_data)
    if rg_data:
        print_ragas(rg_data)
        print_per_condition(rg_data)
        print_qualitative(rg_data, n=3)
    if hr_data and rg_data:
        print_threshold_summary(hr_data, rg_data)

    sep()
    if hr_data:
        print(f"  Hit rate  → {RESULTS_PATH}")
    if rg_data:
        print(f"  RAGAS     → {DETAILED_RESULTS_PATH}")
    sep()


def main():
    hr_data = {}
    rg_data = {}

    if RESULTS_PATH.exists():
        hr_data = json.loads(RESULTS_PATH.read_text(encoding="utf-8-sig")).get("hit_rate", {})
    else:
        print(f"[WARNING] Hit rate results not found at {RESULTS_PATH}")
        print("  Run: python evaluation_metrics/rag/evaluate_hit_rate.py")

    if DETAILED_RESULTS_PATH.exists():
        rg_data = json.loads(DETAILED_RESULTS_PATH.read_text(encoding="utf-8-sig"))
    else:
        print(f"[WARNING] RAGAS detailed results not found at {DETAILED_RESULTS_PATH}")
        print("  Run: python evaluation_metrics/rag/evaluate_ragas.py")

    if not hr_data and not rg_data:
        print("\nNo results available. Run the evaluations first.")
        return

    report_path = RESULTS_DIR / "report_summary.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as f:
        with redirect_stdout(_Tee(sys.stdout, f)):
            _run_report(hr_data, rg_data)

    print(f"\n  Report saved → {report_path}")


if __name__ == "__main__":
    main()
