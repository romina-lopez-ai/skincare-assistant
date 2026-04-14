import sys
import argparse
from pathlib import Path

# Configurar paths
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import RAGEngine
from src.rag.query_processor import DocumentProcessor

def debug_full_pipeline(question: str):
    engine = RAGEngine()
    processor = DocumentProcessor()
    
    print("\n" + "="*80)
    print(f"🔬 RAG PIPELINE DEEP DEBUG: '{question}'")
    print("="*80)

    # --- ETAPA 1: QUERY EXPANSION ---
    expanded = processor.expand_query(question)
    print(f"\n[1] QUERY EXPANSION")
    print(f"    Original: {question}")
    print(f"    Expanded: {expanded}")

    # --- ETAPA 2: RETRIEVAL INICIAL (DENSE) ---
    # Obtenemos los 20 mejores de Chroma para tener una base amplia
    print(f"\n[2] DENSE RETRIEVAL (ChromaDB - Top 20)")
    raw_results = engine.vectorstore.similarity_search_with_relevance_scores(expanded, k=20)
    
    dense_docs = []
    print(f"    {'#':<3} | {'Score':<7} | {'Title':<20} | {'Content Snippet'}")
    for i, (doc, score) in enumerate(raw_results, 1):
        dense_docs.append(doc)
        title = doc.metadata.get('title', 'N/A')[:20]
        snippet = doc.page_content.replace('\n', ' ')[:50]
        print(f"    {i:<3} | {score:>6.3f} | {title:<20} | {snippet}...")

    # --- ETAPA 3: RRF FUSION ---
    # En este test usamos solo la lista de Dense, pero RRF la preparará para el Reranker
    fused_results = processor.rrf_fuse(dense_docs)
    print(f"\n[3] RRF FUSION (Normalization & Dedup)")
    print(f"    Total fused unique docs: {len(fused_results)}")

    # --- ETAPA 4: FLASHRANK RERANKING ---
    print(f"\n[4] FLASHRANK RERANKING (Semantic Quality Gate)")
    reranked_results = processor.rerank(question, fused_results, top_k=10)
    
    print(f"    {'#':<3} | {'New Score':<9} | {'Source':<15} | {'Title'}")
    print(f"    " + "-"*60)
    for i, res in enumerate(reranked_results, 1):
        # FlashRank suele dar scores como 0.99 para muy relevantes o negativos para irrelevantes
        score = res['score']
        title = res['meta'].get('title', 'N/A')[:30]
        print(f"    {i:<3} | {score:>9.4f} | {res['meta'].get('source', '???'):<15} | {title}")

    # --- ETAPA 5: SCORE FILTERING ---
    final_evidence = processor.filter_by_score(reranked_results)
    print(f"\n[5] FINAL FILTERING (Threshold Check)")
    print(f"    Final candidates passed to LLM: {len(final_evidence)}")
    
    if final_evidence:
        print("\nTOP EVIDENCE FOR LLM:")
        print(f"--- START CONTEXT ---")
        print(final_evidence[0]['text'][:500] + "...")
        print(f"--- END CONTEXT ---")
    else:
        print("❌ NO EVIDENCE PASSED THE THRESHOLD.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    args = parser.parse_args()
    debug_full_pipeline(args.query)