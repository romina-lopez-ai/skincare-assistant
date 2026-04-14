"""scripts/test_rag_nodes.py — Test RAG nodes without API.

Usage:
    python scripts/test_rag_nodes.py
    python scripts/test_rag_nodes.py --question "What causes psoriasis?"
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.model_loader import load_all_models, get_models_status
from src.nodes.retrieve_context_node import retrieve_context_node
from src.nodes.rag_answer_node import rag_answer_node
from src.rag.prompt_factory import build_clinical_prompt

def test_rag_pipeline(question: str, verbose: bool = True):
    """
    Test the complete RAG pipeline: retrieve → answer.

    Args:
        question: User query
        verbose: Print detailed output

    Returns:
        dict with 'evidences' and 'answer'
    """
    print("\n" + "=" * 60)
    print(f" TESTING RAG PIPELINE")
    print("=" * 60)
    print(f" Question: {question}\n")

    # Step 1: Retrieve context
    print(" Step 1: Retrieving context...")
    state = {"question": question}
    state = retrieve_context_node(state)

    evidences = state.get("evidences", [])
    print(f"   Retrieved {len(evidences)} evidence chunks")

    if verbose and evidences:
        print("\n   --- Evidences ---")
        for i, ev in enumerate(evidences, 1):
            score = ev.get("score", 0)
            title = ev["meta"].get("title", "Unknown")
            print(f"\n   [{i}] Score: {score:.3f} | Source: {title[:60]}")
            print(f"       Text: {ev['text'][:200]}...")

    # Step 2: Generate answer
    print("\n Step 2: Generating answer...")

    if verbose:
        print("\n" + "=" * 60)
        print(" PROMPT SENT TO MODEL")
        print("=" * 60)
        
        query = state.get("question", "")
        evidences = state.get("evidences", [])
        test_prompt = build_clinical_prompt(query, evidences)
        
        print(test_prompt)
        print("=" * 60)
        print("\n")
    # ─────────────────────────────────────────────────────────────
    
    state = rag_answer_node(state)

    answer = state.get("answer", "")
    answer_metadata = state.get("answer_metadata", {})

    print(f"\n Step 3: Answer generated")
    print(f"   Has context: {answer_metadata.get('has_context', False)}")
    print(f"   Num evidences used: {answer_metadata.get('num_evidences', 0)}")

    print("\n" + "=" * 60)
    print(" FINAL ANSWER")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    return {"evidences": evidences, "answer": answer, "metadata": answer_metadata}


def test_batch_queries():
    """Run multiple test queries."""
    test_queries = [
        "What are the symptoms of psoriasis?",
        "How is eczema treated?",
        "What causes acne?",
        "Is there a cure for vitiligo?",
        "What does melanoma look like?",
    ]

    results = []
    for query in test_queries:
        print("\n" + " " * 30)
        result = test_rag_pipeline(query, verbose=False)
        results.append({
            "query": query,
            "has_context": result["metadata"].get("has_context", False),
            "answer_preview": result["answer"][:150] + "..."
        })

    print("\n" + "=" * 60)
    print(" BATCH TEST SUMMARY")
    print("=" * 60)
    for r in results:
        status = "ok" if r["has_context"] else "warning"
        print(f"{status} {r['query'][:50]:<50} | Context: {r['has_context']}")
        print(f"   Answer: {r['answer_preview']}\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test RAG nodes")
    parser.add_argument("--question", "-q", type=str, default=None,
                        help="Single question to test")
    parser.add_argument("--batch", action="store_true",
                        help="Run batch of test queries")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed evidence output")

    args = parser.parse_args()

    if args.batch:
        test_batch_queries()
    elif args.question:
        test_rag_pipeline(args.question, verbose=args.verbose)
    else:
        # Default: interactive mode
        print("RAG Node Tester - Interactive Mode")
        print("Type 'quit' to exit\n")
        while True:
            question = input(" Question: ").strip()
            if question.lower() in ["quit", "exit", "q"]:
                break
            if question:
                test_rag_pipeline(question, verbose=args.verbose)
            print()