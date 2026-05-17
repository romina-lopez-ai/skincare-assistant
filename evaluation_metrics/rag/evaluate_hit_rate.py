"""evaluate_hit_rate.py -- Compute HR@k for the RAG retrieval pipeline.

Loads eval_dataset.json, runs the retriever for each query, and checks
whether the expected source appears among the top-k returned chunks.
Also measures the false positive rate on out-of-index queries.

Run from repo root:
    python evaluation_metrics/rag/evaluate_hit_rate.py

Output: evaluation_metrics/rag/results/rag_eval_results.json  (hit_rate section)
"""

import json
import re
import sys
import time
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import (
    EVAL_DATASET_PATH, RESULTS_DIR, RESULTS_PATH,
    CONDITIONS, HR_PRIMARY_K, HR_THRESHOLD, FP_RATE_THRESHOLD,
)

K_VALUES = [1, 2]

# Mirrors retrieve_context._GENERIC_REFS — vague pronouns that point to the diagnosed condition.
_GENERIC_REFS = re.compile(
    r"\b(it|this|that|the condition|my condition|this condition|the disease)\b",
    re.IGNORECASE,
)


def _reformulate(query: str, condition: str) -> str:
    """Mirror of retrieve_context._reformulate_query() without ChatState dependency.

    Replaces vague pronouns with the condition name, or prefixes short queries,
    exactly as the production pipeline does before calling the retriever.
    """
    if not condition:
        return query
    if _GENERIC_REFS.search(query):
        return _GENERIC_REFS.sub(condition, query)
    if len(query.split()) < 5:
        return f"{condition} {query}"
    return query


def load_eval_dataset() -> list[dict]:
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Eval dataset not found at {EVAL_DATASET_PATH}. "
            "Run generate_eval_dataset.py first."
        )
    return json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8-sig"))


def source_matches(retrieved_sources: list[str], expected_source: str) -> bool:
    """Fuzzy match: expected_source appears as substring of any retrieved source."""
    expected_lower = expected_source.lower()
    for src in retrieved_sources:
        if expected_lower in src.lower() or src.lower() in expected_lower:
            return True
    return False


def retrieve_sources(query: str, condition: str = "", top_k: int = 5) -> list[str]:
    """Run the full retrieval pipeline and return source names for the top chunks.

    Applies the same query reformulation as retrieve_context node so that
    vague queries like 'Is it contagious?' become 'Is psoriasis contagious?'
    before hitting the reranker — matching real production behaviour.
    """
    from rag.retriever import retriever as rag_retriever

    reformulated = _reformulate(query, condition)
    if reformulated != query:
        print(f"           Reformulated: '{query}' → '{reformulated}'")

    result    = rag_retriever({"question": reformulated})
    evidences = result.get("evidences", [])
    sources   = []
    for ev in evidences[:top_k]:
        meta   = ev.get("meta", {})
        source = meta.get("title") or meta.get("source", "")
        if source:
            sources.append(source)
    return sources


def evaluate_hit_rate(samples: list[dict]) -> dict:
    in_index  = [s for s in samples if s["type"] == "in_index"]
    negatives = [s for s in samples if s["type"] == "out_of_index"]

    print("\n" + "="*60)
    print(f"Hit Rate@k Evaluation  ({len(in_index)} in-index | {len(negatives)} out-of-index)")
    print("="*60)

    per_sample_results  = []
    hits                = defaultdict(int)
    per_condition_hits  = defaultdict(lambda: defaultdict(int))
    per_condition_total = defaultdict(int)

    for i, sample in enumerate(in_index, start=1):
        query    = sample["query"]
        expected = sample["expected_source"]
        cond     = sample["condition"]
        print(f"[{i:02d}/{len(in_index)}] {cond:<12} | {query[:58]}")

        t0      = time.time()
        sources = retrieve_sources(query, condition=cond, top_k=max(K_VALUES))
        elapsed = time.time() - t0

        hit_flags = {}
        for k in K_VALUES:
            hit = source_matches(sources[:k], expected)
            hits[k] += int(hit)
            per_condition_hits[cond][k] += int(hit)
            hit_flags[f"hit_at_{k}"] = hit

        per_condition_total[cond] += 1
        per_sample_results.append({
            **sample,
            "retrieved_sources": sources,
            "latency_s":        round(elapsed, 3),
            **hit_flags,
        })
        status = "HIT" if hit_flags.get(f"hit_at_{HR_PRIMARY_K}") else "MISS"
        print(f"           HR@{HR_PRIMARY_K}={status}  sources={[s[:40] for s in sources[:2]]}")

    n_in       = len(in_index)
    hr_overall = {
        f"hr_at_{k}": round(hits[k] / n_in, 4) if n_in else 0.0
        for k in K_VALUES
    }

    # per-condition
    per_condition = {}
    for cond in CONDITIONS:
        n = per_condition_total[cond]
        per_condition[cond] = {
            f"hr_at_{k}": round(per_condition_hits[cond][k] / n, 4) if n else 0.0
            for k in K_VALUES
        }
        per_condition[cond]["n_samples"] = n

    # out-of-index false positive rate
    fp_count    = 0
    neg_results = []
    print(f"\n-- Out-of-index queries ({len(negatives)}) --")
    for i, sample in enumerate(negatives, start=1):
        sources  = retrieve_sources(sample["query"], top_k=2)
        has_hit  = len(sources) > 0
        fp_count += int(has_hit)
        neg_results.append({**sample, "retrieved_sources": sources, "false_positive": has_hit})
        status = "FP" if has_hit else "OK"
        print(f"  [{i:02d}] {status} | {sample['query'][:58]}")

    fp_rate = round(fp_count / len(negatives), 4) if negatives else 0.0

    # summary
    print("\n-- Summary " + "-"*49)
    for k in K_VALUES:
        threshold_flag = f"  [threshold>={HR_THRESHOLD}]" if k == HR_PRIMARY_K else ""
        pass_fail = ""
        if k == HR_PRIMARY_K:
            pass_fail = "  [PASS]" if hr_overall[f"hr_at_{k}"] >= HR_THRESHOLD else "  [FAIL]"
        print(f"  HR@{k} = {hr_overall[f'hr_at_{k}']:.4f}{pass_fail}{threshold_flag}")
    fp_pass = fp_rate <= FP_RATE_THRESHOLD
    print(f"  FP rate = {fp_rate:.4f}  ({'PASS' if fp_pass else 'FAIL'}) [threshold<={FP_RATE_THRESHOLD}]")
    print("-"*60)

    return {
        "overall":        hr_overall,
        "per_condition":  per_condition,
        "fp_rate":        fp_rate,
        "n_in_index":     n_in,
        "n_negatives":    len(negatives),
        "per_sample":     per_sample_results + neg_results,
    }


def save_results(hit_rate_results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    existing = {}
    if RESULTS_PATH.exists():
        existing = json.loads(RESULTS_PATH.read_text(encoding="utf-8-sig"))

    existing["hit_rate"] = {
        "overall":        hit_rate_results["overall"],
        "per_condition":  hit_rate_results["per_condition"],
        "fp_rate":        hit_rate_results["fp_rate"],
        "n_in_index":     hit_rate_results["n_in_index"],
        "n_negatives":    hit_rate_results["n_negatives"],
    }

    if "samples" not in existing:
        existing["samples"] = hit_rate_results["per_sample"]
    else:
        sample_map = {s["id"]: s for s in hit_rate_results["per_sample"]}
        for s in existing["samples"]:
            if s["id"] in sample_map:
                s.update({
                    k: v for k, v in sample_map[s["id"]].items()
                    if k not in ("query", "condition", "type", "ground_truth_answer")
                })

    RESULTS_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {RESULTS_PATH}")


def main():
    samples = load_eval_dataset()
    results = evaluate_hit_rate(samples)
    save_results(results)


if __name__ == "__main__":
    main()
