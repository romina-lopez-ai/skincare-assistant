"""generate_eval_dataset.py — Build the RAG evaluation dataset from ChromaDB chunks.

Reads chunks directly from the ChromaDB collection, samples N per condition,
calls an LLM (Groq/OpenAI) to generate one QA pair per chunk, adds hard-coded
out-of-index (negative) queries, and saves everything to data/eval_dataset.json.

Run from repo root (ONE TIME — output is committed to the repo):
    python evaluation_metrics/rag/generate_eval_dataset.py

Env vars:
    JUDGE_PROVIDER=groq   (default) or openai
    GROQ_API_KEY=gsk_...
    OPENAI_API_KEY=sk-...
"""

import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import (
    CHROMA_DB_PATH, COLLECTION_NAME, CONDITIONS,
    SAMPLES_PER_CONDITION, NEGATIVE_QUERIES,
    JUDGE_PROVIDER, JUDGE_MODEL_GROQ, JUDGE_MODEL_OPENAI,
    GROQ_API_KEY, OPENAI_API_KEY,
    DATA_DIR, EVAL_DATASET_PATH,
)

GENERATION_PROMPT = """You are a medical question generator for a dermatology AI evaluation.

Given the following medical text about {condition}, generate ONE specific clinical question.

Text:
{chunk_text}

Source: {source}

Rules:
- The question must be DIRECTLY and SPECIFICALLY answered by the text above
- It should be something a patient with {condition} would realistically ask
- Do not ask about topics not covered in the text

Respond ONLY in this exact format (no extra text):
QUESTION: [your question here]
ANSWER: [1-3 sentence reference answer based only on the text above]"""


def detect_condition(text: str, metadata: dict) -> str | None:
    combined = (text + " " + metadata.get("title", "") + " " + metadata.get("source", "")).lower()
    if "acne" in combined:
        return "acne"
    if "eczema" in combined or "atopic dermatitis" in combined:
        return "eczema"
    if "psoriasis" in combined:
        return "psoriasis"
    return None


def build_judge_llm():
    if JUDGE_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Export it or set JUDGE_PROVIDER=openai.")
        from langchain_groq import ChatGroq
        print(f"[generator] Using Groq judge: {JUDGE_MODEL_GROQ}")
        return ChatGroq(model=JUDGE_MODEL_GROQ, groq_api_key=GROQ_API_KEY, temperature=0.3)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set. Export it or set JUDGE_PROVIDER=groq.")
    from langchain_openai import ChatOpenAI
    print(f"[generator] Using OpenAI judge: {JUDGE_MODEL_OPENAI}")
    return ChatOpenAI(model=JUDGE_MODEL_OPENAI, openai_api_key=OPENAI_API_KEY, temperature=0.3)


def generate_qa_pair(chunk_text: str, source: str, condition: str, llm) -> dict | None:
    prompt = GENERATION_PROMPT.format(
        condition=condition,
        chunk_text=chunk_text[:1200],
        source=source,
    )
    try:
        response = llm.invoke(prompt)
        content  = response.content if hasattr(response, "content") else str(response)
        question, answer = None, None
        for line in content.strip().splitlines():
            if line.startswith("QUESTION:"):
                question = line.replace("QUESTION:", "").strip()
            elif line.startswith("ANSWER:"):
                answer = line.replace("ANSWER:", "").strip()
        if question and answer:
            return {"query": question, "ground_truth_answer": answer}
    except Exception as e:
        print(f"  [WARNING] LLM call failed: {e}")
    return None


def load_chromadb_chunks() -> dict[str, list[tuple[str, dict]]]:
    """Return {condition: [(text, metadata), ...]} from ChromaDB."""
    import chromadb

    if not CHROMA_DB_PATH.exists():
        raise FileNotFoundError(
            f"ChromaDB not found at {CHROMA_DB_PATH}. "
            "Run rag_index_pubmed.py or rag_index_local.py first."
        )

    client    = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    coll      = client.get_or_create_collection(COLLECTION_NAME)
    total     = coll.count()
    print(f"[generator] ChromaDB has {total} chunks in '{COLLECTION_NAME}'")

    raw       = coll.get()
    docs      = raw.get("documents", [])
    metadatas = raw.get("metadatas", []) or [{}] * len(docs)

    by_condition: dict[str, list] = {c: [] for c in CONDITIONS}
    for text, meta in zip(docs, metadatas):
        if not text or not text.strip():
            continue
        cond = detect_condition(text, meta or {})
        if cond:
            by_condition[cond].append((text, meta or {}))

    for cond in CONDITIONS:
        print(f"  {cond:<12}: {len(by_condition[cond])} chunks available")
    return by_condition


def main():
    random.seed(42)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("RAG Eval Dataset Generator")
    print("="*60)

    llm            = build_judge_llm()
    by_condition   = load_chromadb_chunks()
    samples: list  = []
    counters       = {c: 0 for c in CONDITIONS}

    for cond in CONDITIONS:
        available = by_condition[cond]
        if not available:
            print(f"[WARNING] No chunks found for '{cond}' — skipping.")
            continue

        # Deduplicate by source to maximise diversity
        seen_sources: set = set()
        diverse: list     = []
        for text, meta in available:
            src = meta.get("title") or meta.get("source", "")
            if src not in seen_sources:
                seen_sources.add(src)
                diverse.append((text, meta))

        pool     = diverse if len(diverse) >= SAMPLES_PER_CONDITION else available
        selected = random.sample(pool, min(SAMPLES_PER_CONDITION, len(pool)))

        print(f"\n[{cond.upper()}] Generating {len(selected)} QA pairs...")
        for i, (text, meta) in enumerate(selected, start=1):
            source = meta.get("title") or meta.get("source", "Unknown source")
            print(f"  [{i}/{len(selected)}] Source: {source[:60]}")
            qa = generate_qa_pair(text, source, cond, llm)
            if qa is None:
                print(f"  [SKIP] Could not generate QA pair for this chunk.")
                continue

            counters[cond] += 1
            samples.append({
                "id":                  f"{cond}_{counters[cond]:03d}",
                "condition":           cond,
                "type":                "in_index",
                "query":               qa["query"],
                "ground_truth_answer": qa["ground_truth_answer"],
                "expected_source":     source,
                "chunk_preview":       text[:150].strip(),
            })

    # Negative (out-of-index) examples
    print(f"\n[NEGATIVE] Adding {len(NEGATIVE_QUERIES)} out-of-index queries...")
    for i, query in enumerate(NEGATIVE_QUERIES, start=1):
        samples.append({
            "id":                  f"neg_{i:03d}",
            "condition":           None,
            "type":                "out_of_index",
            "query":               query,
            "ground_truth_answer": None,
            "expected_source":     None,
            "chunk_preview":       None,
        })

    # Save
    EVAL_DATASET_PATH.write_text(json.dumps(samples, indent=2, ensure_ascii=False))

    in_index_count = sum(1 for s in samples if s["type"] == "in_index")
    neg_count      = sum(1 for s in samples if s["type"] == "out_of_index")
    print(f"\n{'='*60}")
    print(f"Saved {len(samples)} samples to {EVAL_DATASET_PATH}")
    print(f"  in_index    : {in_index_count}")
    print(f"  out_of_index: {neg_count}")
    for cond in CONDITIONS:
        count = sum(1 for s in samples if s["condition"] == cond)
        print(f"    {cond:<12}: {count}")
    print(f"{'='*60}")
    print("Next step: run evaluate_hit_rate.py")


if __name__ == "__main__":
    main()
