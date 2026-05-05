"""inspect_chromadb.py — Explore indexed chunks before generating the eval dataset.

Run from repo root:
    python evaluation_metrics/rag/inspect_chromadb.py

Shows: total chunk count, per-condition breakdown, sample chunks.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evaluation_metrics.rag.config import (
    CHROMA_DB_PATH, PARENT_STORE_PATH, COLLECTION_NAME, CONDITIONS
)


def detect_condition(text: str, metadata: dict) -> str | None:
    combined = (text + " " + metadata.get("title", "") + " " + metadata.get("source", "")).lower()
    if "acne" in combined:
        return "acne"
    if "eczema" in combined or "atopic dermatitis" in combined:
        return "eczema"
    if "psoriasis" in combined:
        return "psoriasis"
    return None


def main():
    import chromadb
    from rag.model_loader import get_embedding_model
    from rag.vector_store import EmbeddingFunctionWrapper

    print(f"\n{'='*60}")
    print("ChromaDB Inspection")
    print(f"{'='*60}")
    print(f"Path       : {CHROMA_DB_PATH}")
    print(f"Collection : {COLLECTION_NAME}")

    if not CHROMA_DB_PATH.exists():
        print("\n[ERROR] ChromaDB path not found. Run the indexing scripts first.")
        return

    client     = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    total      = collection.count()
    print(f"Total chunks indexed: {total}\n")

    if total == 0:
        print("[WARNING] Collection is empty. Run rag_index_pubmed.py or rag_index_local.py first.")
        return

    # Pull all documents
    raw       = collection.get()
    docs      = raw.get("documents", [])
    metadatas = raw.get("metadatas", []) or [{}] * len(docs)

    # Group by condition
    by_condition: dict[str, list] = {c: [] for c in CONDITIONS}
    untagged: list = []

    for text, meta in zip(docs, metadatas):
        cond = detect_condition(text or "", meta or {})
        if cond:
            by_condition[cond].append((text, meta))
        else:
            untagged.append((text, meta))

    print("── Condition breakdown ──────────────────────────────────")
    for cond in CONDITIONS:
        print(f"  {cond:<12}: {len(by_condition[cond]):>4} chunks")
    print(f"  {'untagged':<12}: {len(untagged):>4} chunks")
    print(f"  {'TOTAL':<12}: {total:>4} chunks")

    # Unique sources with chunk counts
    sources: dict[str, int] = {}
    for meta in metadatas:
        src = (meta or {}).get("title") or (meta or {}).get("source", "unknown")
        sources[src] = sources.get(src, 0) + 1

    print("\n── Indexed sources ──────────────────────────────────────")
    for src, count in sorted(sources.items(), key=lambda x: -x[1]):
        print(f"  [{count:>3} chunks]  {src}")

    # Sample 2 chunks per condition
    print("\n── Sample chunks ────────────────────────────────────────")
    for cond in CONDITIONS:
        chunks = by_condition[cond]
        if not chunks:
            print(f"\n[{cond.upper()}] No chunks found.")
            continue
        print(f"\n[{cond.upper()}] — {len(chunks)} chunks total")
        for text, meta in chunks[:2]:
            source = meta.get("title") or meta.get("source", "Unknown")
            print(f"  Source : {source}")
            print(f"  Preview: {(text or '')[:180].strip()}...")
            print()

    if untagged:
        print(f"\n── Untagged sample (first 2) ────────────────────────────")
        for text, meta in untagged[:2]:
            source = meta.get("title") or meta.get("source", "Unknown")
            print(f"  Source : {source}")
            print(f"  Preview: {(text or '')[:180].strip()}...")
            print()

    print("="*60)
    print("Next step: run generate_eval_dataset.py")


if __name__ == "__main__":
    main()
