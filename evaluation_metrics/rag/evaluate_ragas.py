"""evaluate_ragas.py - RAGAS quality evaluation: 4-way model comparison.

Configurations evaluated per in-index query:
  1. Fine-tuned V4 + RAG          [our production system]
  2. Base LLaMA 3.2-3B + RAG      [measures adapter contribution]
  3. Fine-tuned V4, NO RAG        [parametric knowledge of fine-tuned model]
  4. Base LLaMA 3.2-3B, NO RAG   [pure parametric baseline — no adapter, no context]

Each configuration is scored per sample by a Groq/OpenAI LLM judge on:
  faithfulness, answer_relevancy, context_precision, context_recall

Output: evaluation_metrics/rag/results/ragas_detailed_results.json
  Contains: full per-sample detail (query, chunks+scores, context sent to LLM,
  all 4 answers, RAGAS scores, raw Groq judge prompts+responses).

Run from repo root:
    python evaluation_metrics/rag/evaluate_ragas.py
    CONDITION=acne python evaluation_metrics/rag/evaluate_ragas.py

Env vars:
    JUDGE_PROVIDER=groq     (default) or openai
    GROQ_API_KEY=gsk_...
    OPENAI_API_KEY=sk-...
    SKIP_LLM=1              skip HF model loading (uses ground_truth as placeholder)
    CONDITION=acne          evaluate only one condition (saves cost during testing)
"""

import json
import math
import os
import sys
import time
import statistics
from datetime import datetime
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import (
    EVAL_DATASET_PATH, RESULTS_DIR, RESULTS_PATH,
    CONDITIONS, RAGAS_THRESHOLDS,
    JUDGE_PROVIDER, JUDGE_MODEL_GROQ, JUDGE_MODEL_OPENAI,
    GROQ_API_KEY, OPENAI_API_KEY,
    EMBEDDING_MODEL, SKIP_LLM, MAX_NEW_TOKENS,
)

DETAILED_RESULTS_PATH = RESULTS_DIR / "ragas_detailed_results.json"

# ---------------------------------------------------------------------------
# Query reformulation — mirrors retrieve_context node behaviour
# ---------------------------------------------------------------------------
import re as _re

_GENERIC_REFS = _re.compile(
    r"\b(it|this|that|the condition|my condition|this condition|the disease)\b",
    _re.IGNORECASE,
)


def _reformulate(query: str, condition: str) -> str:
    if not condition:
        return query
    if _GENERIC_REFS.search(query):
        return _GENERIC_REFS.sub(condition, query)
    if len(query.split()) < 5:
        return f"{condition} {query}"
    return query


# ---------------------------------------------------------------------------
# Context formatting
# ---------------------------------------------------------------------------

def format_context_string(evidences: list) -> str:
    blocks = []
    for i, ev in enumerate(evidences, start=1):
        text   = ev.get("text", "").strip()
        meta   = ev.get("meta", {})
        source = meta.get("title") or meta.get("source", "Unknown source")
        score  = ev.get("score", 0.0)
        if text:
            blocks.append(
                f"## PRIORITY {i}: {source}\n{text}"
            )
    if not blocks:
        return ""
    return "[CONTEXT]\n\n" + "\n\n".join(blocks) + "\n[/CONTEXT]"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, condition: str = "") -> tuple:
    from rag.retriever import retriever as rag_retriever
    reformulated = _reformulate(query, condition)
    if reformulated != query:
        print(f"           Reformulated: '{query}' → '{reformulated}'")
    result    = rag_retriever({"question": reformulated})
    evidences = result.get("evidences", [])
    contexts  = [ev.get("text", "").strip() for ev in evidences if ev.get("text")]
    return evidences, contexts


# ---------------------------------------------------------------------------
# LLM inference — 4 configurations
# ---------------------------------------------------------------------------

def _strip_think(text: str) -> str:
    think_end = text.find("</think>")
    return text[think_end + len("</think>"):].strip() if think_end != -1 else text


def generate_finetuned_rag_answer(query: str, context_str: str) -> str:
    """Config 1: Fine-tuned V4 + RAG — calls the production pipeline."""
    from langgraph_orchestrator.services.llm_service import generate_followup_response
    state = {
        "messages":        [{"role": "user", "content": query}],
        "rag_context":     context_str if context_str else None,
        "densenet_output": None,
    }
    return generate_followup_response(state)


def generate_base_rag_answer(query: str, context_str: str) -> str:
    """Config 2: Base LLaMA (no adapter) + RAG context."""
    from langgraph_orchestrator.services.llm_service import (
        _load_llm, _generate_with_adapter, generate_text, _SYSTEM_PROMPT_FOLLOWUP_RAG,
    )
    tokenizer, model, device = _load_llm()
    content = f"{context_str}\n\n[QUESTION]\n{query}" if context_str else query
    history = [{"role": "user", "content": content}]
    raw = _generate_with_adapter(
        adapter_name="__base__", fn=generate_text,
        model=model, tokenizer=tokenizer, history=history,
        max_new_tokens=MAX_NEW_TOKENS, device=device,
        system_prompt=_SYSTEM_PROMPT_FOLLOWUP_RAG,
        extract_first_turn=False,
    )
    return _strip_think(raw)


def generate_finetuned_no_rag_answer(query: str) -> str:
    """Config 3: Fine-tuned V4 adapter, NO context — parametric knowledge only."""
    from langgraph_orchestrator.services.llm_service import (
        _load_llm, _generate_with_adapter, generate_text, _SYSTEM_PROMPT_FOLLOWUP_RAG,
    )
    tokenizer, model, device = _load_llm()
    history = [{"role": "user", "content": query}]
    raw = _generate_with_adapter(
        adapter_name="rag", fn=generate_text,
        model=model, tokenizer=tokenizer, history=history,
        max_new_tokens=MAX_NEW_TOKENS, device=device,
        system_prompt=_SYSTEM_PROMPT_FOLLOWUP_RAG,
        extract_first_turn=False,
    )
    return _strip_think(raw)


def generate_base_no_rag_answer(query: str) -> str:
    """Config 4: Base LLaMA (no adapter), NO context — pure parametric baseline."""
    from langgraph_orchestrator.services.llm_service import (
        _load_llm, _generate_with_adapter, generate_text, _SYSTEM_PROMPT_FOLLOWUP_RAG,
    )
    tokenizer, model, device = _load_llm()
    history = [{"role": "user", "content": query}]
    raw = _generate_with_adapter(
        adapter_name="__base__", fn=generate_text,
        model=model, tokenizer=tokenizer, history=history,
        max_new_tokens=MAX_NEW_TOKENS, device=device,
        system_prompt=_SYSTEM_PROMPT_FOLLOWUP_RAG,
        extract_first_turn=False,
    )
    return _strip_think(raw)


# ---------------------------------------------------------------------------
# RAGAS setup
# ---------------------------------------------------------------------------

def build_ragas_llm():
    if JUDGE_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Export it or set JUDGE_PROVIDER=openai.")
        from langchain_groq import ChatGroq
        from ragas.llms import LangchainLLMWrapper
        print(f"[ragas] Judge LLM: Groq / {JUDGE_MODEL_GROQ}")
        return LangchainLLMWrapper(ChatGroq(
            model=JUDGE_MODEL_GROQ,
            groq_api_key=GROQ_API_KEY,
            temperature=0,
        ))
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set.")
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper
    print(f"[ragas] Judge LLM: OpenAI / {JUDGE_MODEL_OPENAI}")
    return LangchainLLMWrapper(ChatOpenAI(
        model=JUDGE_MODEL_OPENAI,
        openai_api_key=OPENAI_API_KEY,
        temperature=0,
    ))


def build_ragas_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper
    print(f"[ragas] Embeddings: {EMBEDDING_MODEL}")
    return LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL))


# ---------------------------------------------------------------------------
# Judge capture callback — records every raw Groq prompt+response
# ---------------------------------------------------------------------------

from langchain_core.callbacks.base import BaseCallbackHandler as _BaseCallbackHandler

class _JudgeCapture(_BaseCallbackHandler):
    """Captures raw LLM judge prompts and responses for every RAGAS evaluation step."""

    def __init__(self):
        self.calls: list   = []
        self._current: dict = {}

    def on_llm_start(self, serialized, prompts, **kwargs):
        self._current = {"prompt": prompts[0] if prompts else ""}

    def on_llm_end(self, response, **kwargs):
        if response.generations:
            self._current["response"] = response.generations[0][0].text
        self.calls.append(dict(self._current))
        self._current = {}

    def reset(self):
        self.calls = []


# ---------------------------------------------------------------------------
# RAGAS scoring — one sample at a time for per-sample judge capture
# ---------------------------------------------------------------------------

def _safe_float(v) -> float:
    try:
        f = float(v)
        return 0.0 if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return 0.0


def run_ragas(
    questions, answers, contexts_list, ground_truths,
    ragas_llm, ragas_embs,
    label: str = "",
) -> tuple:
    """Run RAGAS one sample at a time.

    Returns:
        scores_list      — list of {metric: float} per sample
        judge_calls_list — list of [{prompt, response}, ...] per sample (raw Groq calls)
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    for metric in [faithfulness, answer_relevancy, context_precision, context_recall]:
        metric.llm = ragas_llm
    answer_relevancy.embeddings = ragas_embs

    capture = _JudgeCapture()
    underlying = getattr(ragas_llm, "langchain_llm", None)
    if underlying is not None:
        underlying.callbacks = [capture]

    scores_list:      list = []
    judge_calls_list: list = []
    n = len(questions)

    for idx, (q, a, ctx, gt) in enumerate(zip(questions, answers, contexts_list, ground_truths), start=1):
        print(f"  [{idx:02d}/{n}] scoring{(' [' + label + ']') if label else ''} | {q[:55]}")
        capture.reset()
        dataset = Dataset.from_dict({
            "question":     [q],
            "answer":       [a],
            "contexts":     [ctx],
            "ground_truth": [gt],
        })
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        row = result.to_pandas().iloc[0]
        scores_list.append({
            "faithfulness":      _safe_float(row.get("faithfulness")),
            "answer_relevancy":  _safe_float(row.get("answer_relevancy")),
            "context_precision": _safe_float(row.get("context_precision")),
            "context_recall":    _safe_float(row.get("context_recall")),
        })
        judge_calls_list.append(list(capture.calls))

    return scores_list, judge_calls_list


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_scores(scores_list: list) -> dict:
    if not scores_list:
        return {}
    metrics = list(scores_list[0].keys())
    result  = {}
    for m in metrics:
        valid = [s[m] for s in scores_list if s.get(m) is not None and not math.isnan(s[m])]
        result[m] = {
            "mean": round(statistics.mean(valid), 4) if valid else 0.0,
            "std":  round(statistics.stdev(valid), 4) if len(valid) > 1 else 0.0,
        }
    return result


def threshold_report(agg: dict, label: str):
    print(f"\n  [{label}]")
    for metric, threshold in RAGAS_THRESHOLDS.items():
        mean = agg.get(metric, {}).get("mean", 0.0)
        std  = agg.get(metric, {}).get("std",  0.0)
        flag = "PASS" if mean >= threshold else "FAIL"
        print(f"    {metric:<22} {mean:.4f} +/- {std:.4f}  (threshold >= {threshold})  {flag}")


def delta_report(agg_a: dict, agg_b: dict, label: str):
    print(f"\n  [{label}]")
    for m in RAGAS_THRESHOLDS:
        delta = agg_a.get(m, {}).get("mean", 0) - agg_b.get(m, {}).get("mean", 0)
        print(f"    {m:<22} {delta:+.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(f"Eval dataset not found at {EVAL_DATASET_PATH}.")

    all_samples = json.loads(EVAL_DATASET_PATH.read_text(encoding="utf-8-sig"))
    in_index    = [s for s in all_samples if s["type"] == "in_index"]

    only_condition = os.getenv("CONDITION", "").strip().lower()
    if only_condition:
        in_index = [s for s in in_index if s["condition"] == only_condition]

    print(f"\n{'='*60}")
    print(f"RAGAS Evaluation  ({len(in_index)} in-index samples)")
    if only_condition:
        print(f"  Condition filter: {only_condition}")
    print(f"{'='*60}")

    ragas_llm  = build_ragas_llm()
    ragas_embs = build_ragas_embeddings()

    # ── Step 1: Retrieve contexts ─────────────────────────────────────────────
    print("\n-- Step 1: Retrieving contexts --")
    retrieval_data = []
    for i, sample in enumerate(in_index, start=1):
        print(f"  [{i:02d}/{len(in_index)}] {sample['query'][:70]}")
        evidences, contexts = retrieve(sample["query"], condition=sample.get("condition", ""))
        retrieval_data.append({
            "sample":      sample,
            "evidences":   evidences,
            "contexts":    contexts,
            "context_str": format_context_string(evidences),
        })

    # ── Step 2: LLM inference — 4 configurations ─────────────────────────────
    ft_rag_answers      = []
    base_rag_answers    = []
    ft_no_rag_answers   = []
    base_no_rag_answers = []

    if SKIP_LLM:
        print("\n-- Step 2: LLM inference SKIPPED (SKIP_LLM=1) — using ground_truth --")
        for rd in retrieval_data:
            gt = rd["sample"].get("ground_truth_answer", "")
            ft_rag_answers.append(gt)
            base_rag_answers.append(gt)
            ft_no_rag_answers.append(gt)
            base_no_rag_answers.append(gt)
    else:
        def _run_inference(label, answers_list, fn):
            print(f"\n-- Step 2: [{label}] --")
            for i, rd in enumerate(retrieval_data, start=1):
                q = rd["sample"]["query"]
                print(f"  [{i:02d}/{len(retrieval_data)}] {q[:60]}")
                t0  = time.time()
                ans = fn(rd)
                print(f"           ({time.time()-t0:.1f}s) → {ans[:80]}...")
                answers_list.append(ans)

        _run_inference(
            "Fine-tuned V4 + RAG", ft_rag_answers,
            lambda rd: generate_finetuned_rag_answer(rd["sample"]["query"], rd["context_str"]),
        )
        _run_inference(
            "Base LLaMA 3.2-3B + RAG", base_rag_answers,
            lambda rd: generate_base_rag_answer(rd["sample"]["query"], rd["context_str"]),
        )
        _run_inference(
            "Fine-tuned V4, NO RAG", ft_no_rag_answers,
            lambda rd: generate_finetuned_no_rag_answer(rd["sample"]["query"]),
        )
        _run_inference(
            "Base LLaMA 3.2-3B, NO RAG", base_no_rag_answers,
            lambda rd: generate_base_no_rag_answer(rd["sample"]["query"]),
        )

    # ── Step 3: RAGAS scoring ─────────────────────────────────────────────────
    questions           = [rd["sample"]["query"] for rd in retrieval_data]
    contexts_list       = [rd["contexts"] for rd in retrieval_data]
    contexts_list_empty = [["No context provided."] for _ in retrieval_data]
    ground_truths       = [rd["sample"].get("ground_truth_answer", "") for rd in retrieval_data]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _checkpoint = RESULTS_DIR / "_ragas_checkpoint.json"

    if _checkpoint.exists():
        print(f"\n-- Step 3: Loading RAGAS scores from checkpoint (skipping Groq calls) --")
        _cp = json.loads(_checkpoint.read_text(encoding="utf-8-sig"))
        ft_rag_scores      = _cp["ft_rag"]
        base_rag_scores    = _cp["base_rag"]
        ft_no_rag_scores   = _cp["ft_no_rag"]
        base_no_rag_scores = _cp["base_no_rag"]
        ft_rag_judge       = _cp.get("ft_rag_judge",      [[] for _ in ft_rag_scores])
        base_rag_judge     = _cp.get("base_rag_judge",    [[] for _ in base_rag_scores])
        ft_no_rag_judge    = _cp.get("ft_no_rag_judge",   [[] for _ in ft_no_rag_scores])
        base_no_rag_judge  = _cp.get("base_no_rag_judge", [[] for _ in base_no_rag_scores])
        print(f"  Loaded {len(ft_rag_scores)} samples from checkpoint.")
    else:
        print("\n-- Step 3: RAGAS scoring --")
        ft_rag_scores,      ft_rag_judge      = run_ragas(questions, ft_rag_answers,      contexts_list,       ground_truths, ragas_llm, ragas_embs, "Fine-tuned+RAG")
        base_rag_scores,    base_rag_judge    = run_ragas(questions, base_rag_answers,    contexts_list,       ground_truths, ragas_llm, ragas_embs, "Base+RAG")
        ft_no_rag_scores,   ft_no_rag_judge   = run_ragas(questions, ft_no_rag_answers,   contexts_list_empty, ground_truths, ragas_llm, ragas_embs, "Fine-tuned+no-RAG")
        base_no_rag_scores, base_no_rag_judge = run_ragas(questions, base_no_rag_answers, contexts_list_empty, ground_truths, ragas_llm, ragas_embs, "Base+no-RAG")

        _checkpoint.write_text(
            json.dumps({
                "ft_rag":           ft_rag_scores,
                "base_rag":         base_rag_scores,
                "ft_no_rag":        ft_no_rag_scores,
                "base_no_rag":      base_no_rag_scores,
                "ft_rag_judge":     ft_rag_judge,
                "base_rag_judge":   base_rag_judge,
                "ft_no_rag_judge":  ft_no_rag_judge,
                "base_no_rag_judge": base_no_rag_judge,
            }, indent=2),
            encoding="utf-8-sig",
        )
        print(f"  [checkpoint saved → {_checkpoint}]")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    ft_rag_agg      = aggregate_scores(ft_rag_scores)
    base_rag_agg    = aggregate_scores(base_rag_scores)
    ft_no_rag_agg   = aggregate_scores(ft_no_rag_scores)
    base_no_rag_agg = aggregate_scores(base_no_rag_scores)

    pc: dict = {
        "ft_rag":      defaultdict(list),
        "base_rag":    defaultdict(list),
        "ft_no_rag":   defaultdict(list),
        "base_no_rag": defaultdict(list),
    }
    for rd, s1, s2, s3, s4 in zip(
        retrieval_data, ft_rag_scores, base_rag_scores, ft_no_rag_scores, base_no_rag_scores
    ):
        cond = rd["sample"]["condition"]
        pc["ft_rag"][cond].append(s1)
        pc["base_rag"][cond].append(s2)
        pc["ft_no_rag"][cond].append(s3)
        pc["base_no_rag"][cond].append(s4)

    per_condition_summary = {}
    for cond in CONDITIONS:
        entry = {key: aggregate_scores(pc[key][cond]) for key in pc}
        entry["n"] = len(pc["ft_rag"][cond])
        per_condition_summary[cond] = entry

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("RAGAS Summary")
    print(f"{'='*60}")
    threshold_report(ft_rag_agg,      "Fine-tuned V4 + RAG          [our system]")
    threshold_report(base_rag_agg,    "Base LLaMA 3.2-3B + RAG")
    threshold_report(ft_no_rag_agg,   "Fine-tuned V4, NO RAG        [parametric fine-tuned]")
    threshold_report(base_no_rag_agg, "Base LLaMA 3.2-3B, NO RAG   [pure baseline]")

    delta_report(ft_rag_agg, base_rag_agg,    "Delta: fine-tuned+RAG vs base+RAG       (adapter contribution)")
    delta_report(ft_rag_agg, ft_no_rag_agg,   "Delta: fine-tuned+RAG vs fine-tuned+no-RAG (RAG contribution)")
    delta_report(ft_rag_agg, base_no_rag_agg, "Delta: fine-tuned+RAG vs base+no-RAG   (total improvement)")

    # ── Build detailed per-sample output ──────────────────────────────────────
    from rag.config import SYSTEM_PROMPT

    samples_out = []
    for i, (rd, s1, s2, s3, s4, a1, a2, a3, a4, j1, j2, j3, j4) in enumerate(zip(
        retrieval_data,
        ft_rag_scores, base_rag_scores, ft_no_rag_scores, base_no_rag_scores,
        ft_rag_answers, base_rag_answers, ft_no_rag_answers, base_no_rag_answers,
        ft_rag_judge, base_rag_judge, ft_no_rag_judge, base_no_rag_judge,
    ), start=1):
        s         = rd["sample"]
        evidences = rd["evidences"]
        samples_out.append({
            "sample_number":   i,
            "id":              s["id"],
            "condition":       s["condition"],
            "query":           s["query"],
            "ground_truth_answer":     s.get("ground_truth_answer"),
            "expected_source":         s.get("expected_source"),
            "no_context":              len(evidences) == 0,
            "retrieved_chunks": [
                {
                    "rank":   j + 1,
                    "source": (ev.get("meta") or {}).get("title") or (ev.get("meta") or {}).get("source", "unknown"),
                    "score":  round(ev.get("score", 0.0), 4),
                    "text":   ev.get("text", "").strip(),
                }
                for j, ev in enumerate(evidences)
            ],
            "context_str_sent_to_llm": rd["context_str"],
            "answers": {
                "finetuned_rag":    a1,
                "base_rag":         a2,
                "finetuned_no_rag": a3,
                "base_no_rag":      a4,
            },
            "ragas": {
                "finetuned_rag":    dict(**s1, judge_calls=j1),
                "base_rag":         dict(**s2, judge_calls=j2),
                "finetuned_no_rag": dict(**s3, judge_calls=j3),
                "base_no_rag":      dict(**s4, judge_calls=j4),
            },
        })

    # ── Save detailed results ─────────────────────────────────────────────────
    output = {
        "metadata": {
            "run_date":         datetime.now().isoformat(timespec="seconds"),
            "judge_model":      JUDGE_MODEL_GROQ if JUDGE_PROVIDER == "groq" else JUDGE_MODEL_OPENAI,
            "judge_provider":   JUDGE_PROVIDER,
            "skip_llm":         SKIP_LLM,
            "condition_filter": only_condition or "all",
            "n_samples":        len(samples_out),
            "system_prompt":    SYSTEM_PROMPT,
            "thresholds":       RAGAS_THRESHOLDS,
        },
        "summary": {
            "finetuned_rag":    ft_rag_agg,
            "base_rag":         base_rag_agg,
            "finetuned_no_rag": ft_no_rag_agg,
            "base_no_rag":      base_no_rag_agg,
        },
        "per_condition": per_condition_summary,
        "samples":       samples_out,
    }

    DETAILED_RESULTS_PATH.write_text(
        json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8-sig"
    )
    print(f"\nDetailed results saved to {DETAILED_RESULTS_PATH}")


if __name__ == "__main__":
    main()
