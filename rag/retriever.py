"""retriever.py : retrieve grounded clinical context.

Full pipeline per query:
  1. Query expansion     — add clinical keywords
  2. Dense retrieval     — ChromaDB semantic search 
  3. BM25 retrieval      — lexical search over same corpus 
  4. RRF fusion          — Reciprocal Rank Fusion merges + deduplicates both lists
  5. FlashRank reranking — cross-encoder refines top-x RRF candidates
  6. Score filter        — drop chunks below RELEVANCE_THRESHOLD (default x)
  7. Return top-2        — matches 1-chunk / 2-chunk format from Dataset C

BM25 index is built once at module load from the same documents in ChromaDB,
so both retrievers always operate over the same corpus. If the collection is
empty, BM25 is skipped and only dense retrieval is used.
"""

import logging
from langchain_community.retrievers import BM25Retriever

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.vector_store import RAGEngine
from rag.query_processor import DocumentProcessor
from rag.config import DEFAULT_TOP_K, DEFAULT_TOP_K_R,FINAL_TOP_K, RELEVANCE_THRESHOLD

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETONS
# Initialised once when the module is imported, reused across all queries.
# ─────────────────────────────────────────────────────────────────────────────
engine = RAGEngine()
processor = DocumentProcessor()

# Dense retriever
_dense_retriever = None


def _get_dense_retriever():
    """Get or create the dense retriever (Singleton)."""
    global _dense_retriever
    if _dense_retriever is None:
        _dense_retriever = engine.get_vectorstore().as_retriever(
            search_type="similarity",
            search_kwargs={"k": DEFAULT_TOP_K},
        )
        logger.info(f"Dense retriever created (k={DEFAULT_TOP_K})")
    return _dense_retriever


def _build_bm25_retriever(k: int = DEFAULT_TOP_K) -> BM25Retriever | None:
    """
    Build a BM25 retriever from documents currently in ChromaDB.

    Called once at startup. Returns None if the collection is empty
    (user hasn't run rag_index_pubmed.py or rag_index_local.py yet).

    The BM25 index mirrors the ChromaDB corpus so both retrievers
    always operate over the same set of documents.
    """
    try:
        count = engine.get_collection_count()
        if count == 0:
            logger.warning(
                "ChromaDB is empty — BM25 not built. "
                "Run rag_index_pubmed.py or rag_index_local.py first."
            )
            return None

        # Pull all stored documents from ChromaDB to build the BM25 index.
        # One-time cost at startup; ~1-2s for 1000 docs, negligible afterwards.
        raw = engine.get_vectorstore().get()
        if not raw or not raw.get("documents"):
            logger.warning("No documents found in vectorstore for BM25 index.")
            return None

        from langchain_core.documents import Document
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(raw["documents"], raw["metadatas"])
            if text and text.strip()
        ]

        if not docs:
            return None

        retriever = BM25Retriever.from_documents(docs)
        retriever.k = k
        logger.info(f"BM25 index built: {len(docs)} documents (k={k})")
        return retriever

    except Exception as e:
        logger.error(f"Failed to build BM25 retriever: {e}")
        return None


# Build BM25 index once at import time
_bm25: BM25Retriever | None = _build_bm25_retriever(k=DEFAULT_TOP_K)


# ─────────────────────────────────────────────────────────────────────────────
# NODE
# ─────────────────────────────────────────────────────────────────────────────

def retriever(state: dict) -> dict:
    """
    LangGraph node: retrieve grounded clinical evidence for a user query.

    Input state:
        question (str): user's dermatology question.

    Output state:
        evidences (list[dict]): up to FINAL_TOP_K evidence chunks, each:
            text  (str)   — chunk content
            score (float) — FlashRank relevance score
            meta  (dict)  — source metadata (title, pmid, year, source, ...)

        Returns [] if no chunk passes RELEVANCE_THRESHOLD.
        rag_answer_node handles the empty case with the no-context response
        trained in Dataset C (edge_no_context examples).
    """
    query: str = state.get("question", "").strip()
    if not query:
        logger.warning("retrieve_context_node: empty question")
        return {"evidences": [], "question": ""}
  
    # ── 1. Query expansion ────────────────────────────────────────────────────
    expanded = processor.expand_query(query)
 
    # ── 2. Dense retrieval (usando Singleton) ─────────────────────────────────
    try:
        dense_retriever = _get_dense_retriever()
        dense_docs = dense_retriever.invoke(expanded)
        logger.info(f"Dense retrieval: {len(dense_docs)} candidates")
    except Exception as e:
        logger.error(f"Dense retrieval failed: {e}")
        dense_docs = []

    # ── 3. BM25 retrieval ─────────────────────────────────────────────────────
    bm25_docs = []
    if _bm25 is not None:
        try:
            bm25_docs = _bm25.invoke(expanded)
            bm25_docs = bm25_docs[:DEFAULT_TOP_K]
            logger.info(f"BM25 retrieval: {len(bm25_docs)} candidates")
        except Exception as e:
            logger.error(f"BM25 retrieval failed: {e}")
    else:
        logger.info("BM25 not available — dense retrieval only")
    
    # ── 4. RRF fusion ─────────────────────────────────────────────────────────
    fused = processor.rrf_fuse(bm25_docs, dense_docs)
    if not fused:
        logger.info("No candidates after RRF — returning empty evidences")
        return {"evidences": [], "question": query}
    
    # ── 5. FlashRank reranking ────────────────────────────────────────────────
    reranked = processor.rerank(query, fused, top_k=DEFAULT_TOP_K_R)  # rerank top-10 from RRF to limit latency
    
   # ── 6. Score filter ───────────────────────────────────────────────────────
    filtered = processor.filter_by_score(reranked, threshold=RELEVANCE_THRESHOLD)
  
    # ── 7. Return top-N ───────────────────────────────────────────────────────
    evidences = filtered[:FINAL_TOP_K]
    print(f" [7] Final Evidences: {len(evidences)}")
    if evidences:
        logger.info(
            f"Returning {len(evidences)} evidence chunk(s) | "
            f"scores: {[e['score'] for e in evidences]}"
        )
    else:
        logger.info(
            "No evidence passed threshold "
            f"(threshold={RELEVANCE_THRESHOLD}) — empty context."
        )

    return {"evidences": evidences, "question": query}