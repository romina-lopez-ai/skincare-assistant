"""scripts/rag_index_local.py — Index local documents (PDF, CSV, TXT, DOCX) with metadata.

Uses simple document loaders to avoid complex dependencies from unstructured.
For medical documents, basic text extraction is usually sufficient and more reliable.

Convention: Include year in filename like 'document_2024.pdf' to auto-extract year.
If no year found, 'year' field is omitted from metadata.

Usage:
    python rag_index_local.py --file document.pdf
    python index_local.py --folder ./data/documents/
"""

import argparse
import re
from typing import List, Dict, Any

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_classic.schema import Document

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rag.vector_store import RAGEngine

# ============================================================
# CONFIGURATION
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".csv": "csv",
    ".txt": "text",
    ".doc": "document",
    ".docx": "document",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_year_from_filename(filename: str) -> str:
    """
    Extract year from filename like 'document_2024.pdf' or 'guidelines-2023.pdf'.
    Returns year as string, or None if not found.
    """
    # Match 4-digit year (1900-2099)
    match = re.search(r'(19|20)\d{2}', filename)
    return match.group(0) if match else None


def get_loader(file_path: str):
    """
    Return the appropriate loader based on file extension.
    
    Uses simple loaders to avoid complex dependencies from unstructured.
    For medical documents, basic text extraction is usually sufficient.
    """
    ext = Path(file_path).suffix.lower()
    
    if ext == ".pdf":
        # Simple PDF loader - no need for complex unstructured processing
        return PyPDFLoader(file_path)
    elif ext == ".csv":
        import csv
        csv.field_size_limit(10**6)
        return CSVLoader(
            file_path,
            encoding="utf-8",
            autodetect_encoding=False 
        )
    elif ext == ".txt":
        return TextLoader(file_path)
    elif ext == ".md":
        # Markdown is just text, use simple text loader
        return TextLoader(file_path)
    elif ext in [".doc", ".docx"]:
        # Simple docx loader without unstructured dependencies
        return Docx2txtLoader(file_path)
    else:
        raise ValueError(f"Unsupported format: {ext}. Supported: {list(SUPPORTED_EXTENSIONS.keys())}")


def add_metadata_to_documents(documents: List[Document], file_path: str, file_type: str) -> List[Document]:
    """
    Add consistent metadata to all documents for source attribution.
    Extracts year from filename if present.
    """
    file_path_obj = Path(file_path)
    file_name = file_path_obj.name
    year = extract_year_from_filename(file_name)
    
    for i, doc in enumerate(documents):
        # Preserve existing metadata
        existing_meta = doc.metadata.copy() if doc.metadata else {}
        
        # Build base metadata
        metadata: Dict[str, Any] = {
            "source": file_type,
            "filename": file_name,
            "filepath": str(file_path),
            "page": existing_meta.get("page", i + 1),
            "title": existing_meta.get("title", file_name),
        }
        
        # Add year only if found
        if year:
            metadata["year"] = year
        
        # Merge with existing metadata
        metadata.update(existing_meta)
        doc.metadata = metadata
    
    return documents


def index_file(file_path: str):
    """Index a single file with metadata."""
    file_path_obj = Path(file_path)
    
    if not file_path_obj.exists():
        print(f"[ERROR] File not found: {file_path}")
        return
    
    ext = file_path_obj.suffix.lower()
    file_type = SUPPORTED_EXTENSIONS.get(ext, "unknown")
    
    print(f" Indexing: {file_path_obj.name}")
    
    # Load documents
    loader = get_loader(str(file_path))
    documents = loader.load()
    
    # Add metadata
    documents = add_metadata_to_documents(documents, str(file_path), file_type)
    
    # Index
    engine = RAGEngine()
    retriever = engine.get_retriever()
    retriever.add_documents(documents)

    # Show metadata sample
    if documents:
        sample_meta = documents[0].metadata
        year_info = f" | year: {sample_meta.get('year', 'N/A')}" if sample_meta.get('year') else ""
        print(f"      Indexed {len(documents)} pages")
        print(f"      Source: {sample_meta.get('source')}{year_info}")
        print(f"      File: {sample_meta.get('filename')}")


def index_folder(folder_path: str):
    """Index all supported files in a folder."""
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"[ERROR] Path not found: {folder_path}")
        return

    # Find all supported files
    files = []
    for ext in SUPPORTED_EXTENSIONS.keys():
        files.extend(folder.rglob(f"*{ext}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    
    files = list(set(files))  # Remove duplicates
    
    if not files:
        print(f"[WARNING] No supported files found in {folder_path}")
        print(f"   Supported formats: {list(SUPPORTED_EXTENSIONS.keys())}")
        return
    
    print(f" Found {len(files)} files to index in {folder_path}")
    print("-" * 50)
    
    for file_path in files:
        try:
            index_file(str(file_path))
        except Exception as e:
            print(f"    Failed: {e}")
    
    print("-" * 50)
    print(f" Indexing complete. Checked {len(files)} files.")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index local documents (PDF, CSV, TXT, DOCX) with metadata",
        epilog="Tip: Include year in filename like 'document_2024.pdf' for automatic year extraction."
    )
    parser.add_argument("--file", type=str, help="Single file to index")
    parser.add_argument("--folder", type=str, help="Folder with documents to index")
    args = parser.parse_args()

    if args.file:
        index_file(args.file)
    elif args.folder:
        index_folder(args.folder)
    else:
        print("Usage: python index_local.py --file document.pdf")
        print("       python index_local.py --folder ./data/documents/")
        print("\nSupported formats: PDF, CSV, TXT, DOC, DOCX")
        print("Year extraction: Include 4-digit year in filename (e.g., 'guidelines_2024.pdf')")