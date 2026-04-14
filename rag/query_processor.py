
"""query_processor.py — Query expansion, RRF fusion, and FlashRank reranking.

Pipeline:
  1. expand_query()    — append clinical keywords for better recall
  2. rrf_fuse()        — Reciprocal Rank Fusion merges BM25 + dense ranked lists
  3. rerank()          — FlashRank cross-encoder refines top-K candidates
  4. filter_by_score() — drop anything below RELEVANCE_THRESHOLD

Why RRF instead of positional scoring:
  Positional scoring (score = 1 - rank/N) gives rank-1 of BM25 a score of 1.0
  regardless of whether it is actually relevant. A general-medicine introduction
  that mentions "dermatology" ranks high in BM25 but is not useful for a
  specific eczema question.

  RRF (Reciprocal Rank Fusion) solves this:
    score(doc) = Σ 1/(k + rank_in_list)  for each list the doc appears in
  A document that appears at rank-1 in ONLY one list gets a lower combined
  score than a document that appears at rank-3 in BOTH lists.
  This correctly rewards documents that are consistently relevant across
  both retrieval methods.

Why FlashRank is kept (not removed):
  FlashRank ms-marco-MiniLM-L-6-v2 is a cross-encoder with ~22M parameters.
  Benchmarked at ~80-150ms for 10 passages on CPU — acceptable for a
  dermatology chatbot where the user expects a short thinking delay.
  It is only called on the top-K candidates after RRF (typically 10),
  not on the full corpus, so the cost is bounded and predictable.
  It provides the final semantic quality gate that BM25+dense alone miss.
"""

import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import RERANKER_MODEL, RELEVANCE_THRESHOLD, EXPANSION_SUFFIX
from rag.model_loader import get_ranker

logger = logging.getLogger(__name__)

# RRF constant — standard value from the original paper (Cormack et al. 2009)
RRF_K = 60

class DocumentProcessor:
    """
    Handles query expansion, RRF fusion, FlashRank reranking, and score filtering.
    All operations are CPU-safe and mobile-compatible.
    """

    def __init__(self):
        self._ranker = get_ranker()
        self._has_reranker = self._ranker is not None
        
        if self._has_reranker:
            logger.info(f"FlashRank reranker loaded (Singleton)")
        else:
            logger.warning(
                "FlashRank not available — reranking disabled. "
                "Install with: pip install flashrank"
            )

    # ── 1. Query expansion ────────────────────────────────────────────────────

    def expand_query(self, query: str) -> str:
        """
        Append clinical keywords to improve recall of medical documents.

        Helps when patients use informal language:
          "bumpy red skin" → retrieves "erythematous papular lesions"
        """
        return f"{query} {EXPANSION_SUFFIX}"

    # ── 2. RRF fusion ─────────────────────────────────────────────────────────

    def rrf_fuse(self, *ranked_lists, k: int = RRF_K) -> list:
        """
        Reciprocal Rank Fusion across multiple ranked document lists.

        Args:
            *ranked_lists: Each argument is a list of LangChain Documents,
                           ordered from most to least relevant (rank 0 = best).
            k:             RRF constant. Higher k reduces the impact of
                           top-ranked documents. Standard value is 60.

        Returns:
            List of dicts: {'doc': Document, 'rrf_score': float}
            Sorted by rrf_score descending (best first).

        Example:
            BM25 returns:  [Introduction(rank=0), EczemaDoc(rank=1)]
            Dense returns: [EczemaDoc(rank=0), AtopicDoc(rank=1)]

            RRF scores:
              EczemaDoc    = 1/(60+1) + 1/(60+0) = 0.0164 + 0.0166 = 0.0330  ← winner
              Introduction = 1/(60+0)             = 0.0166
              AtopicDoc    = 1/(60+1)             = 0.0164
        """
        scores: dict[str, dict] = {}   # key: content fingerprint

        for ranked_list in ranked_lists:
            for rank, doc in enumerate(ranked_list):
                text        = doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
                fingerprint = text[:120].strip()

                if fingerprint not in scores:
                    scores[fingerprint] = {"doc": doc, "rrf_score": 0.0}

                scores[fingerprint]["rrf_score"] += 1.0 / (k + rank + 1)

        fused = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
        logger.info(
            f"RRF fusion: {sum(len(lst) for lst in ranked_lists)} total candidates "
            f"→ {len(fused)} unique (deduped)"
        )
        return fused

    # ── 3. FlashRank reranking ────────────────────────────────────────────────

    def rerank(self, query: str, fused_candidates: list, top_k: int = 10) -> list:
        """
        Rerank the top-K RRF candidates using FlashRank cross-encoder.

        Only runs on the top candidates (default 10) to keep latency low.
        Each passage is scored by comparing the query and document together
        (cross-encoder), which is more accurate than embedding similarity alone.

        Args:
            query:            The original (unexpanded) user question.
            fused_candidates: Output of rrf_fuse() — list of dicts with 'doc' key.
            top_k:            How many candidates to pass to the reranker.

        Returns:
            List of evidence dicts: {'text', 'score', 'meta'}
            Sorted by FlashRank score descending.
            If FlashRank is unavailable, returns RRF-ranked candidates directly.
        """
        # Take only top-K from RRF to bound reranker latency
        candidates_for_rerank = fused_candidates[:top_k]

        if not candidates_for_rerank:
            return []

        # Fallback if FlashRank not installed
        if not self._has_reranker:
            logger.warning("FlashRank unavailable — returning RRF-ranked candidates")
            return [
                {
                    "text":  c["doc"].page_content if hasattr(c["doc"], "page_content") else c["doc"].get("text", ""),
                    "score": round(c["rrf_score"], 4),
                    "meta":  c["doc"].metadata if hasattr(c["doc"], "metadata") else c["doc"].get("meta", {}),
                }
                for c in candidates_for_rerank
            ]

        from flashrank import RerankRequest

        passages = []
        for i, c in enumerate(candidates_for_rerank):
            doc  = c["doc"]
            text = doc.page_content if hasattr(doc, "page_content") else doc.get("text", "")
            meta = doc.metadata    if hasattr(doc, "metadata")    else doc.get("meta", {})
            passages.append({"id": i, "text": text, "meta": meta})

        results = self._ranker.rerank(RerankRequest(query=query, passages=passages))

        # Map back to evidence format
        reranked = []
        for r in results:
            reranked.append({
                "text":  r["text"],
                "score": round(float(r["score"]), 4),
                "meta":  r.get("meta", {}),
            })

        logger.info(
            f"FlashRank reranked {len(passages)} candidates | "
            f"top score: {reranked[0]['score'] if reranked else 'N/A'}"
        )
        return reranked

    # ── 4. Score filter ───────────────────────────────────────────────────────

    def filter_by_score(self, candidates: list,
                        threshold: float = RELEVANCE_THRESHOLD) -> list:
        """
        Drop evidence chunks below the relevance threshold.

        FlashRank scores are not normalised (range varies by model).
        For ms-marco-MiniLM-L-6-v2, scores typically range from -10 to +10.
        The default RELEVANCE_THRESHOLD of 0.4 is calibrated for this model.

        If scores are all below threshold (very different query/corpus domain),
        the model will receive an empty context and correctly respond:
        "I couldn't find relevant information in my database."
        """
        passed = [c for c in candidates if c.get("score", 0) >= threshold]

        logger.info(
            f"Score filter (threshold={threshold}): "
            f"{len(candidates)} candidates → {len(passed)} passed"
        )

        if not passed and candidates:
            best = max(candidates, key=lambda c: c.get("score", 0))
            logger.info(
                f"All candidates below threshold. Best score was "
                f"{best.get('score', 0):.4f} — returning empty context."
            )

        return passed