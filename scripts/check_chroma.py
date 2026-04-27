"""scripts/check_chroma.py — Check what's in your ChromaDB.

Usage:
    python check_chroma.py              # Show summary
    python check_chroma.py --verbose    # Show sample documents
    python check_chroma.py --query "psoriasis"  # Search for specific topic
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.vector_store import RAGEngine
from src.rag.config import COLLECTION_NAME


def check_collection(verbose: bool = False, query: str = None):
    """Check contents of ChromaDB collection."""
    engine = RAGEngine()
    
    # Get collection directly
    collection = engine.collection
    
    # Get count
    count = collection.count()
    print(f"\n ChromaDB Collection: {COLLECTION_NAME}")
    print(f"   Total documents: {count}")
    
    if count == 0:
        print("\n Collection is empty! Run index_local.py or index_pubmed.py first.")
        return
    
    # Get sample
    if verbose:
        print("\n Sample documents (first 3):")
        results = collection.peek(limit=3)
        for i, (doc_id, doc, meta) in enumerate(zip(
            results['ids'], results['documents'], results['metadatas']
        )):
            print(f"\n   [{i+1}] ID: {doc_id}")
            print(f"       Source: {meta.get('source', 'unknown')}")
            print(f"       Title: {meta.get('title', 'N/A')[:60]}...")
            print(f"       Year: {meta.get('year', 'N/A')}")
            print(f"       Content preview: {doc[:1500]}...")
    
    # Search for specific query
    if query:
        print(f"\n Searching for: '{query}'")
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if results['documents'] and results['documents'][0]:
            print(f"\n   Found {len(results['documents'][0])} results:")
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            )):
                print(f"\n   [{i+1}] Score: {1 - dist:.3f}")
                print(f"       Source: {meta.get('source', 'unknown')}")
                print(f"       Title: {meta.get('title', 'N/A')[:50]}...")
                print(f"       Content: {doc[:120]}...")
        else:
            print("   No results found.")


def main():
    parser = argparse.ArgumentParser(description="Check ChromaDB contents")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Show sample documents")
    parser.add_argument("--query", "-q", type=str, 
                       help="Search for specific query")
    args = parser.parse_args()
    
    check_collection(verbose=args.verbose, query=args.query)


if __name__ == "__main__":
    main()