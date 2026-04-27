"""scripts/rag_index_pubmed.py — Index PubMed abstracts with keyword filtering.

Fetches abstracts from PubMed via Biopython Entrez, filters by medical
keyword density to avoid indexing off-topic articles, and stores them
in the RAGEngine (ChromaDB + parent document store).

Usage:
    python rag_index_pubmed.py                        # batch: all topics from config
    python rag_index_pubmed.py --topic "psoriasis"    # single topic
    python rag_index_pubmed.py --topic "acne" --limit 50
"""

import argparse
import logging
import re
from typing import Dict, Optional
from Bio import Entrez
from langchain_core.documents import Document
from sympy import limit
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.config import (
    ENTREZ_EMAIL,
    PUBMED_TOPICS_TO_INDEX,
    MIN_KEYWORD_DENSITY,
    MEDICAL_KEYWORDS,
    RETMAX_PER_TOPIC,
)

from src.rag.vector_store import RAGEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Set email once at module load 
Entrez.email = ENTREZ_EMAIL

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def compute_keyword_density(text: str) -> float:
    text = text.lower()
    words = text.split()
    if not words:
        return 0.0
    
    count = 0
    for keyword in MEDICAL_KEYWORDS:
        count += text.count(keyword.lower())
    
    return count / len(words)

def extract_year(article: Dict) -> Optional[str]:
    """
    Extract publication year from a PubMed article dict.
    Returns year string or None if not found.
    """
    try:
        pub_date = (
            article["MedlineCitation"]["Article"]
            ["Journal"]["JournalIssue"]["PubDate"]
        )
        year = pub_date.get("Year")
        if year:
            return str(year)
        # Fallback: MedlineDate like "2024 Jan-Feb"
        medline = pub_date.get("MedlineDate", "")
        m = re.search(r"(19|20)\d{2}", medline)
        return m.group(0) if m else None
    except (KeyError, AttributeError):
        return None


def article_to_document(article: Dict, topic: str) -> Optional[Document]:
    """
    Convert a PubMed article to a LangChain Document.
    Returns None if the article lacks an abstract or has low keyword density.
    """
    try:
        citation     = article["MedlineCitation"]
        title        = str(citation["Article"]["ArticleTitle"])
        abstract     = " ".join(citation["Article"]["Abstract"]["AbstractText"])
        pmid         = str(citation["PMID"])
        year         = extract_year(article)
        density      = compute_keyword_density(abstract)

        if density < MIN_KEYWORD_DENSITY:
            return None   # skip low-relevance abstracts

        metadata = {
            "source":          "pubmed",
            "quality":         "peer_reviewed",   # used by score filter
            "title":           title,
            "pmid":            pmid,
            "topic":           topic,
            "keyword_density": round(density, 4),
        }
        if year:
            metadata["year"] = year

        return Document(page_content=abstract, metadata=metadata)

    except (KeyError, TypeError) as e:
        logger.debug(f"Skipping article — missing field: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# INDEXING
# ─────────────────────────────────────────────────────────────────────────────

def index_topic(topic: str, limit: int = RETMAX_PER_TOPIC) -> int:
    """
    Fetch and index PubMed abstracts for one topic.

    Returns:
        Number of abstracts successfully indexed.
    """
    logger.info(f"Searching PubMed: '{topic}' (limit={limit})")

    # Search
    try:
        #handle = Entrez.esearch(db="pubmed", term=topic, retmax=limit)
        term_with_filter = f"{topic} AND (Review[pt] OR Practice Guideline[pt] OR Systematic Review[pt])"
        handle = Entrez.esearch(db="pubmed", term=term_with_filter, retmax=limit)
        ids    = Entrez.read(handle)["IdList"]
        handle.close()
    except Exception as e:
        logger.error(f"PubMed search failed for '{topic}': {e}")
        return 0

    if not ids:
        logger.warning(f"No PubMed results for '{topic}'")
        return 0

    # Fetch
    try:
        handle  = Entrez.efetch(db="pubmed", id=",".join(ids), retmode="xml")
        records = Entrez.read(handle)
        handle.close()
    except Exception as e:
        logger.error(f"PubMed fetch failed for '{topic}': {e}")
        return 0

    # Convert and filter
    docs = []
    for article in records.get("PubmedArticle", []):
        doc = article_to_document(article, topic)
        if doc:
            docs.append(doc)

    skipped = len(records.get("PubmedArticle", [])) - len(docs)
    logger.info(f"  {len(docs)} high-quality abstracts | {skipped} skipped (low density)")

    if not docs:
        return 0

    # Index
    
    engine = RAGEngine()
    engine.get_retriever().add_documents(docs)

    # Log a sample for verification
    sample = docs[0].metadata
    logger.info(
        f"  Sample: [{sample.get('year', '?')}] {sample.get('title', '')[:60]}... "
        f"(density={sample.get('keyword_density', 0):.2f})"
    )
    return len(docs)


def index_batch() -> Dict[str, int]:
    """Run batch indexing over all topics defined in config."""
    results     = {}
    total       = 0

    print("=" * 60)
    print("PubMed Batch Indexing")
    print("=" * 60)

    for topic, limit in PUBMED_TOPICS_TO_INDEX:
        count         = index_topic(topic, limit)
        results[topic] = count
        total         += count

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for topic, count in results.items():
        print(f"  {topic:<40} {count} articles")
    print(f"  {'TOTAL':<40} {total} articles")
    print("=" * 60)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Index PubMed abstracts into the dermatology RAG knowledge base"
    )
    ap.add_argument("--topic", type=str, default=None,
                    help="Single PubMed topic to index")
    ap.add_argument("--limit", type=int, default=RETMAX_PER_TOPIC,
                    help=f"Max articles per topic (default: {RETMAX_PER_TOPIC})")
    args = ap.parse_args()

    if args.topic:
        n = index_topic(args.topic, args.limit)
        print(f"\n✓ Indexed {n} articles for '{args.topic}'")
    else:
        index_batch()