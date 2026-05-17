"""generate_viz_data.py — Extract a clean summary JSON for visualizations.

Reads:
  results/rag_eval_results.json       (hit rate)
  results/ragas_detailed_results.json (RAGAS)

Writes:
  results/viz_summary.json            (clean, ready for visualize.py)

Does NOT re-run any evaluation. Run whenever either source JSON changes.

Run from repo root:
    python evaluation_metrics/rag/generate_viz_data.py
"""

import json
import sys
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import RESULTS_DIR, RESULTS_PATH

RAGAS_PATH   = RESULTS_DIR / "ragas_detailed_results.json"
VIZ_OUT_PATH = RESULTS_DIR / "viz_summary.json"

METRICS    = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
CONDITIONS = ["acne", "eczema", "psoriasis"]

CONFIG_LABELS = {
    "finetuned_rag":    "FT + RAG (ours)",
    "base_rag":         "Base + RAG",
    "finetuned_no_rag": "FT, no RAG",
    "base_no_rag":      "Base, no RAG",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    missing = [p for p in (RAGAS_PATH, RESULTS_PATH) if not p.exists()]
    if missing:
        for p in missing:
            print(f"[ERROR] Not found: {p}")
        sys.exit(1)

    rg = load_json(RAGAS_PATH)
    hr = load_json(RESULTS_PATH)["hit_rate"]

    # ── configs: mean ± std for each metric ───────────────────────────────────
    configs = {}
    for cfg, label in CONFIG_LABELS.items():
        entry = {"label": label}
        for m in METRICS:
            agg = rg["summary"].get(cfg, {}).get(m, {})
            entry[m] = {"mean": round(agg.get("mean", 0.0), 4),
                        "std":  round(agg.get("std",  0.0), 4)}
        configs[cfg] = entry

    # ── deltas (additive decomposition) ───────────────────────────────────────
    def mean(cfg, m):
        return configs[cfg][m]["mean"]

    deltas = {}
    for name, (a, b) in {
        "adapter": ("finetuned_rag", "base_rag"),
        "rag":     ("finetuned_rag", "finetuned_no_rag"),
        "total":   ("finetuned_rag", "base_no_rag"),
    }.items():
        deltas[name] = {m: round(mean(a, m) - mean(b, m), 4) for m in METRICS}

    # ── per-condition (FT+RAG scores + HR@2) ──────────────────────────────────
    per_condition = {}
    for cond in CONDITIONS:
        pc = rg.get("per_condition", {}).get(cond, {})
        # per_condition keys in the source JSON use "ft_rag" (not "finetuned_rag")
        ft = pc.get("ft_rag") or pc.get("finetuned_rag", {})
        row = {m: round(ft.get(m, {}).get("mean", 0.0), 4) for m in METRICS}
        row["hr_at_2"] = hr.get("per_condition", {}).get(cond, {}).get("hr_at_2", 0.0)
        per_condition[cond] = row

    out = {
        "generated":         str(date.today()),
        "n_ragas_samples":   rg["metadata"]["n_samples"],
        "n_hitrate_samples": hr["n_in_index"],
        "judge_model":       rg["metadata"]["judge_model"],
        "metrics":           METRICS,
        "configs":           configs,
        "deltas":            deltas,
        "per_condition":     per_condition,
        "hr_overall": {
            "hr_at_2":     hr["overall"]["hr_at_2"],
            "fp_rate":     hr["fp_rate"],
            "n_in_index":  hr["n_in_index"],
            "n_negatives": hr["n_negatives"],
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_OUT_PATH.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"viz_summary.json written → {VIZ_OUT_PATH}")


if __name__ == "__main__":
    main()
