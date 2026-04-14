"""model_loader.py — Centralized Singleton model loading.

All models are loaded once and shared across the entire application.
This is the single source of truth for model loading.

Models managed:
    - Embedding model (for query expansion)
    - Reranker model (FlashRank cross-encoder)
    - Ollama model (fine-tuned LLM) — loaded via langchain_ollama

Usage:
    from rag.model_loader import (
        get_embedding_model,
        get_ranker,
        get_llm,
        load_all_models,
        unload_models
    )
    
    embedding_model = get_embedding_model()
    ranker = get_ranker()
    llm = get_llm()
"""

import logging
from sentence_transformers import SentenceTransformer
from flashrank import Ranker
from langchain_ollama import OllamaLLM 

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.rag.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    OLLAMA_MODEL,
    OLLAMA_TEMPERATURE,
    OLLAMA_MAX_TOKENS
)

logger = logging.getLogger(__name__)

# ============================================================
# SINGLETON INSTANCES (private)
# ============================================================
_embedding_model = None
_ranker = None
_llm = None


# ============================================================
# EMBEDDING MODEL
# ============================================================
def _load_embedding_model():
    """Load embedding model (internal)."""
    logger.info(f" Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(f" Embedding model loaded")
    return model


def get_embedding_model():
    """Get embedding model (Singleton)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _load_embedding_model()
    return _embedding_model


# ============================================================
# RERANKER MODEL (FlashRank)
# ============================================================
def _load_ranker():
    """Load reranker model (internal)."""
    logger.info(f" Loading reranker model: {RERANKER_MODEL}")
    ranker = Ranker(model_name=RERANKER_MODEL)
    logger.info(f" Reranker model loaded")
    return ranker


def get_ranker():
    """Get reranker model (Singleton)."""
    global _ranker
    if _ranker is None:
        _ranker = _load_ranker()
    return _ranker


# ============================================================
# OLLAMA LLM MODEL
# ============================================================
def _load_llm():
    """Load Ollama LLM model (internal)."""
    logger.info(f" Loading Ollama model: {OLLAMA_MODEL}")
    logger.info(f"   Temperature: {OLLAMA_TEMPERATURE}, Max tokens: {OLLAMA_MAX_TOKENS}")
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=OLLAMA_TEMPERATURE,
        num_predict=OLLAMA_MAX_TOKENS,
        timeout=120,
    )
    logger.info(f" Ollama model loaded")
    return llm


def get_llm():
    """Get Ollama LLM model (Singleton)."""
    global _llm
    if _llm is None:
        _llm = _load_llm()
    return _llm


# ============================================================
# UTILITY FUNCTIONS
# ============================================================
def load_all_models():
    """Preload all models (call at application startup)."""
    logger.info(" Preloading all models...")
    get_embedding_model()
    get_ranker()
    get_llm()
    logger.info(" All models preloaded")


def unload_models():
    """Unload all models (useful for testing or memory cleanup)."""
    global _embedding_model, _ranker, _llm
    _embedding_model = None
    _ranker = None
    _llm = None
    logger.info(" Models unloaded")


def get_models_status() -> dict:
    """Get status of all models (for debugging)."""
    return {
        "embedding_model": {
            "loaded": _embedding_model is not None,
            "name": EMBEDDING_MODEL,
        },
        "ranker": {
            "loaded": _ranker is not None,
            "name": RERANKER_MODEL,
        },
        "llm": {
            "loaded": _llm is not None,
            "name": OLLAMA_MODEL,
            "temperature": OLLAMA_TEMPERATURE,
            "max_tokens": OLLAMA_MAX_TOKENS,
        },
    }