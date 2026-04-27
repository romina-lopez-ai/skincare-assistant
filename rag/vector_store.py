"""vector_store.py — RAGEngine with local and cloud support."""

import logging
from annotated_types import doc
from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter, json

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import (
    STORAGE_TYPE,
    LOCAL_CHROMA_DB_PATH, LOCAL_PARENT_STORE_PATH, 
    CHROMA_TENANT, CHROMA_DATABASE, CHROMA_API_KEY,
    CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP,
    PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP,
    SEMANTIC_SEPARATOR,
    COLLECTION_NAME
)

from rag.model_loader import get_embedding_model

logger = logging.getLogger(__name__)


class EmbeddingFunctionWrapper:
    """
    Wrapper que adapta SentenceTransformer al formato que Chroma espera.
    Chroma requiere un objeto con el método embed_query().
    """
    def __init__(self, model):
        self.model = model
    
    def embed_query(self, text: str):
        """Embed a single query (used by Chroma)."""
        return self.model.encode(text).tolist()
    
    def embed_documents(self, texts: list):
        """Embed multiple documents (used by Chroma)."""
        return self.model.encode(texts).tolist()


def _get_chroma_client():
    """Get Chroma client based on STORAGE_TYPE."""
    import chromadb
    
    if STORAGE_TYPE == "cloud":
        logger.info(f"Connecting to Chroma Cloud: tenant={CHROMA_TENANT}, db={CHROMA_DATABASE}")
        return chromadb.CloudClient(
            tenant=CHROMA_TENANT,
            database=CHROMA_DATABASE,
            api_key=CHROMA_API_KEY
        )
    else:
        logger.info(f"Using local ChromaDB at {LOCAL_CHROMA_DB_PATH}")
        return chromadb.PersistentClient(path=str(LOCAL_CHROMA_DB_PATH))


def _get_embedding_function():
    """
    Get embedding function for Chroma using the Singleton model.
    Returns a wrapper that has embed_query() and embed_documents() methods.
    """
    model = get_embedding_model()
    return EmbeddingFunctionWrapper(model)

class RAGEngine:
    """
    Core RAG Engine: Manages VectorStore and Document Storage.
    Supports both local (PersistentClient) and cloud (CloudClient).
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        try:
            # Embeddings
            embedding_function = _get_embedding_function()
            semantic_separators = SEMANTIC_SEPARATOR
            # Child splitter
            self.child_splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHILD_CHUNK_SIZE,
                chunk_overlap=CHILD_CHUNK_OVERLAP,
                separators=semantic_separators,
                keep_separator=True,  # Keep separators to preserve semantic boundaries
                strip_whitespace=True,  # Remove leading/trailing whitespace from chunks
            )
            # Provides richer context than child chunks alone.
            self.parent_splitter = RecursiveCharacterTextSplitter(
                chunk_size=PARENT_CHUNK_SIZE,
                chunk_overlap=PARENT_CHUNK_OVERLAP,
                separators=semantic_separators,
                keep_separator=True,
                strip_whitespace=True,
            )

            # Chroma client (local or cloud)
            self.chroma_client = _get_chroma_client()
            self.collection = self.chroma_client.get_or_create_collection(COLLECTION_NAME)
            
            # LangChain Chroma wrapper
            self.vectorstore = Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=embedding_function,
                client=self.chroma_client,  
            )

            # Local document store (for parent documents)
            LOCAL_PARENT_STORE_PATH.mkdir(parents=True, exist_ok=True)
            byte_store = LocalFileStore(str(LOCAL_PARENT_STORE_PATH))
            self.docstore = create_kv_docstore(byte_store)

            # Parent Document Retriever
            self.retriever = ParentDocumentRetriever(
                vectorstore=self.vectorstore,
                docstore=self.docstore,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )

            self._initialized = True
        except Exception:
            self._initialized = False
            raise

        logger.info(
            f"RAGEngine ready | storage={STORAGE_TYPE} | "
            f"child_chunk={CHILD_CHUNK_SIZE} | parent_chunk={PARENT_CHUNK_SIZE}"
        )
    
    def get_retriever(self):
        return self.retriever

    def get_vectorstore(self):
        return self.vectorstore
    
    def get_collection_count(self) -> int:
        """Return number of child chunks currently indexed."""
        return self.collection.count()