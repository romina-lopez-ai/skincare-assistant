import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import RAGEngine
# Asumiendo que tienes un nodo o función de rerank, la importamos. 
# Si no, aquí verás dónde insertarla.
# from src.nodes.rerank_node import rank_documents 

def debug_pipeline(question: str):
    engine = RAGEngine()
    retriever = engine.get_retriever()

    print("\n" + "="*80)
    print(f"🔍 DEBUGGING RAG PIPELINE FOR: '{question}'")
    print("="*80)

    # --- ETAPA 1: RETRIEVAL INICIAL (TOP 20) ---
    print(f"\n[STEP 1] Initial Retrieval (ChromaDB + Parent Store)")
    # Accedemos directamente al vectorstore para ver los scores crudos
    # Usamos n_results=20 como pediste
    results = engine.vectorstore.similarity_search_with_relevance_scores(question, k=20)
    
    print(f"{'#':<3} | {'Score':<7} | {'Source':<15} | {'Snippet'}")
    print("-" * 80)
    
    retrieved_docs = []
    for i, (doc, score) in enumerate(results, 1):
        source = doc.metadata.get('source', 'unknown')
        title = doc.metadata.get('title', 'N/A')[:15]
        content = doc.page_content.replace('\n', ' ')[:60]
        print(f"{i:<3} | {score:>6.3f} | {title:<15} | {content}...")
        retrieved_docs.append(doc)

    # --- ETAPA 2: RERANKING (Si tienes implementado uno) ---
    print(f"\n[STEP 2] Reranking Process")
    # Aquí simulamos o llamamos a tu función de rerank
    # Si usas FlashRank, Cohere o Cross-Encoders:
    print("... Applying Reranker logic (Ordering by contextual relevance) ...")
    
    # Ejemplo de cómo se vería si pasas de 20 a los mejores 5:
    # ranked_docs = rank_documents(question, retrieved_docs) 
    ranked_docs = retrieved_docs[:5] # Por ahora solo tomamos los top 5 para el ejemplo
    
    for i, doc in enumerate(ranked_docs, 1):
        print(f"🔥 Rank {i}: {doc.metadata.get('title', 'N/A')[:40]}")

    # --- ETAPA 3: CONTEXTO FINAL PARA EL PROMPT ---
    print(f"\n[STEP 3] Final Context Construction")
    full_context = "\n---\n".join([d.page_content for d in ranked_docs])
    print(f"Total tokens/chars in context: {len(full_context)} chars")
    print(f"Final Context Preview:\n{full_context[:300]}...")

    # --- ETAPA 4: LLM GENERATION ---
    print(f"\n[STEP 4] LLM Response Generation")
    # Aquí llamarías a tu rag_answer_node(state)
    print("Waiting for Llama-3B...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True)
    args = parser.parse_args()
    debug_pipeline(args.query)