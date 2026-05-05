"""migrate_medlineplus_titles.py — One-time migration: prefix MedlinePlus chunk titles.

Updates existing ChromaDB chunks that came from MedlinePlus so their title
changes from e.g. "Psoriasis" to "MedlinePlus – Psoriasis".

Run once from repo root:
    python rag/scripts/migrate_medlineplus_titles.py

Safe to re-run: already-prefixed titles are skipped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import chromadb
from rag.config import LOCAL_CHROMA_DB_PATH, COLLECTION_NAME

PREFIX = "MedlinePlus – "


def main():
    client     = chromadb.PersistentClient(path=str(LOCAL_CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)
    total      = collection.count()

    print(f"\nChromaDB: {total} chunks in '{COLLECTION_NAME}'")

    if total == 0:
        print("Collection is empty — nothing to migrate.")
        return

    raw   = collection.get(include=["metadatas", "documents"])
    ids   = raw["ids"]
    metas = raw["metadatas"]

    update_ids:   list[str]  = []
    update_metas: list[dict] = []

    for chunk_id, meta in zip(ids, metas):
        if not meta:
            continue
        if meta.get("source") != "medlineplus":
            continue

        title = meta.get("title", "")
        if title.startswith(PREFIX):
            continue  # already migrated

        new_meta = {**meta, "title": f"{PREFIX}{title}"}
        update_ids.append(chunk_id)
        update_metas.append(new_meta)

    if not update_ids:
        print("Nothing to update — all MedlinePlus titles already prefixed.")
        return

    print(f"Updating {len(update_ids)} chunks...")
    collection.update(ids=update_ids, metadatas=update_metas)

    # Verify
    print("\nUpdated titles:")
    seen: set[str] = set()
    for meta in update_metas:
        t = meta.get("title", "")
        if t not in seen:
            seen.add(t)
            print(f"  {t}")

    print(f"\nDone. {len(update_ids)} chunks updated in ChromaDB.")


if __name__ == "__main__":
    main()
