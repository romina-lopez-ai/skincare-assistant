"""scripts/rag_index_medlineplus.py — Robust Multi-Result API Indexing."""

import requests
import xml.etree.ElementTree as ET
import re
import time
import sys
from pathlib import Path
from datetime import datetime
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rag.vector_store import RAGEngine
from rag.config import MEDLINEPLUS_URL, MEDLINEPLUS_BASE_URL, MEDLINEPLUS_DB, MEDLINEPLUS_TERMS, MEDLINEPLUS_MAX_RESULTS

max_results =  MEDLINEPLUS_MAX_RESULTS

def fetch_medlineplus_api(term: str, max_results:int) -> list:
    """Queries the API and returns a LIST of Documents."""
    base_url = MEDLINEPLUS_BASE_URL
    params = {
        "db": MEDLINEPLUS_DB,
        "term": term,
        "retMax": max_results
    }
    
    docs_list = []
    try:
        response = requests.get(base_url, params=params, timeout=15)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        
        for doc_node in root.findall(".//document"):
            data = {}
         
            for content in doc_node.findall("content"):
                data[content.get("name")] = content.text

            raw_title = data.get("title", term.title())
            clean_title = re.sub(r'<[^>]*>', '', raw_title)

            raw_summary = data.get("FullSummary", "")
            if not raw_summary:
                continue

            clean_summary = re.sub(r'<[^>]*>', '', raw_summary)
            title = data.get("title", term.title())

            term_lower = term.lower()
            title_lower = clean_title.lower()
            
            is_relevant = (term_lower in title_lower) or (title_lower in term_lower)
            
            if not is_relevant:
                print(f"     Skipping: {clean_title}") 
                continue
            
            clean_summary = re.sub(r'<[^>]*>', '', raw_summary)
            url = doc_node.get("url", MEDLINEPLUS_URL)

            docs_list.append(Document(
                page_content=f"# {clean_title}\n\n{clean_summary}",
                metadata={
                    "source": "medlineplus",
                    "title": f"MedlinePlus – {clean_title}",
                    "url": url,
                    "review_date": datetime.now().strftime("%m/%d/%Y"),
                    "term": term
                }
            ))
        return docs_list
        
    except Exception as e:
        print(f"    API Error for '{term}': {e}")
        return []

def index_medical_knowledge():
    engine = RAGEngine()
    total_indexed = 0
    
    print("=" * 60)
    print(" MULTI-RESULT MEDICAL INDEXER")
    print("=" * 60)
    
    for term in MEDLINEPLUS_TERMS:
        print(f" Searching for multiple resources on: {term}...")
        
        found_docs = fetch_medlineplus_api(term, max_results)
        
        if found_docs:
            engine.get_retriever().add_documents(found_docs)
            total_indexed += len(found_docs)
            print(f"    Success: Indexed {len(found_docs)} documents for '{term}'")
        else:
            print(f"    No results for: {term}")
        
        time.sleep(1)

    print("\n" + "=" * 60)
    print(f" DONE: {total_indexed} documents added to your Vector Store")
    print("=" * 60)

if __name__ == "__main__":
    index_medical_knowledge()