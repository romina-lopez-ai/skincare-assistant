"""scripts/medlineplus_auto_index.py — Index MedlinePlus content reliably."""

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
import sys
from pathlib import Path
import time
import re

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.rag.vector_store import RAGEngine

# ============================================================
# CONFIGURACIÓN
# ============================================================

# URLs reales de MedlinePlus (funcionan)
'''
MEDLINEPLUS_URLS = {
    "psoriasis": "https://medlineplus.gov/psoriasis.html",
    "psoriasis_ency": "https://medlineplus.gov/ency/article/000434.htm",
    "acne": "https://medlineplus.gov/acne.html",
    "acne_ency": "https://medlineplus.gov/ency/article/000873.htm",
    "eczema": "https://medlineplus.gov/eczema.html",
    "eczema_ency": "https://medlineplus.gov/ency/article/000853.htm",
    "atopic_dermatitis": "https://medlineplus.gov/ency/article/000853.htm",
    "skin_cancer": "https://medlineplus.gov/skincancer.html",
    "rosacea": "https://medlineplus.gov/rosacea.html",
    "vitiligo": "https://medlineplus.gov/vitiligo.html",
}
'''
MEDLINEPLUS_URLS = {
    "psoriasis": "https://medlineplus.gov/psoriasis.html",}

def fetch_medlineplus_page(url: str, title_override: str = None) -> Document:
    """
    Extrae el contenido de una página de MedlinePlus.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Extraer título
        if title_override:
            title = title_override
        else:
            title_tag = soup.find('h1')
            title = title_tag.text.strip() if title_tag else url.split('/')[-1].replace('.html', '')
        
        # Extraer contenido principal
        content_parts = [f"# {title}"]
        
        # Buscar contenido en diferentes selectores
        main_content = soup.find('div', {'id': 'main-content'}) or soup.find('main') or soup.find('article')
        
        if main_content:
            for elem in main_content.find_all(['p', 'li', 'h2', 'h3']):
                text = elem.get_text().strip()
                if text and len(text) > 20:
                    if elem.name in ['h2', 'h3']:
                        content_parts.append(f"\n## {text}")
                    else:
                        content_parts.append(text)
        else:
            # Fallback: buscar todo el cuerpo
            body = soup.find('body')
            if body:
                for elem in body.find_all(['p']):
                    text = elem.get_text().strip()
                    if text and len(text) > 30:
                        content_parts.append(text)
        
        # Limpiar contenido
        full_content = "\n".join(content_parts)
        # Eliminar caracteres no deseados
        full_content = re.sub(r'\s+', ' ', full_content)
        full_content = re.sub(r'\[\d+\]', '', full_content)
        
        if len(full_content) < 200:
            return None
        
        return Document(
            page_content=full_content,
            metadata={
                "source": "medlineplus",
                "title": title,
                "url": url,
                "type": "patient_education"
            }
        )
        
    except Exception as e:
        print(f"   Error fetching {url}: {e}")
        return None


def index_medlineplus():
    """Indexa todo el contenido de MedlinePlus."""
    engine = RAGEngine()
    total_indexed = 0
    
    print("=" * 60)
    print("📚 Indexing MedlinePlus Content")
    print("=" * 60)
    
    for name, url in MEDLINEPLUS_URLS.items():
        print(f"\n🔍 Processing: {name}")
        doc = fetch_medlineplus_page(url, name.replace('_', ' ').title())
        
        if doc:
            engine.get_retriever().add_documents([doc])
            total_indexed += 1
            print(f"   ✅ Indexed: {doc.metadata['title']} ({len(doc.page_content)} chars)")
        else:
            print(f"   ⚠️ Failed to extract content")
        
        time.sleep(0.5)  # Ser respetuoso con el servidor
    
    print("\n" + "=" * 60)
    print(f"📊 TOTAL: {total_indexed} documents indexed from MedlinePlus")
    print("=" * 60)


def search_medlineplus_by_query(query: str, max_results: int = 3) -> list:
    """
    Busca en MedlinePlus usando búsqueda web y devuelve documentos.
    """
    search_url = "https://medlineplus.gov/search"
    params = {"query": query, "language": "en"}
    
    try:
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        results = []
        
        for result in soup.find_all('div', class_='search-result-item')[:max_results]:
            link = result.find('a')
            if link and link.get('href'):
                title = link.get_text().strip()
                url = link.get('href')
                if not url.startswith('http'):
                    url = f"https://medlineplus.gov{url}"
                
                snippet = result.find('p', class_='snippet')
                snippet_text = snippet.get_text().strip() if snippet else ""
                
                if title and snippet_text:
                    results.append({
                        "title": title,
                        "content": snippet_text,
                        "url": url
                    })
        
        return results
        
    except Exception as e:
        print(f"   Search error: {e}")
        return []


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--search", type=str, help="Search MedlinePlus for a topic")
    parser.add_argument("--index", action="store_true", help="Index predefined MedlinePlus content")
    args = parser.parse_args()
    
    if args.search:
        results = search_medlineplus_by_query(args.search)
        print(f"\n🔍 Search results for '{args.search}':")
        for r in results:
            print(f"   - {r['title']}\n     {r['url']}\n     {r['content'][:200]}...\n")
    else:
        index_medlineplus()