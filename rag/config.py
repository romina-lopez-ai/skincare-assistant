"""config.py — Configuration for the RAG module.

All values can be overridden via environment variables.
Supports local ChromaDB (default) and Chroma Cloud.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# STORAGE TYPE
# ─────────────────────────────────────────────────────────────────────────────
# "local"  → PersistentClient (on-disk, default)
# "cloud"  → ChromaDB Cloud (requires CHROMA_* credentials)
STORAGE_TYPE = os.getenv("RAG_STORAGE_TYPE", "local")

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL STORAGE PATHS
# ─────────────────────────────────────────────────────────────────────────────
LOCAL_CHROMA_DB_PATH    = Path(os.getenv("RAG_CHROMA_PATH",  "./data/chroma_db"))
LOCAL_PARENT_STORE_PATH = Path(os.getenv("RAG_STORE_PATH",   "./data/parent_docs_store"))
#LOCAL_PARENT_DB_PATH = Path(os.getenv("RAG_PARENT_DB_PATH", "./data/parent_documents.db"))
# ─────────────────────────────────────────────────────────────────────────────
# CLOUD STORAGE (Chroma Cloud)
# ─────────────────────────────────────────────────────────────────────────────
CHROMA_TENANT   = os.getenv("CHROMA_TENANT",   "")
CHROMA_DATABASE = os.getenv("CHROMA_DATABASE", "")
CHROMA_API_KEY  = os.getenv("CHROMA_API_KEY",  "")

# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING MODEL
# Lightweight model that runs on CPU — no GPU needed.
# ─────────────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# RERANKER MODEL
# FlashRank cross-encoder — 22M params, ~80-150ms on CPU for 10 passages.
# Scores range roughly -10 to +10; RELEVANCE_THRESHOLD filters the output.
# ─────────────────────────────────────────────────────────────────────────────
RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "ms-marco-MiniLM-L-12-v2")

# ─────────────────────────────────────────────────────────────────────────────
# LLM MODEL
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_MODEL       = os.getenv("OLLAMA_MODEL","llama-phase3")
HF_MODEL           = os.getenv("HF_MODEL","RominaLopez/skincare-llama-3.2-3b-4bit-quantized")
TEMPERATURE        = float(os.getenv("OLLAMA_TEMPERATURE", "0.1"))
MAX_TOKENS         = int(os.getenv("OLLAMA_MAX_TOKENS",    "512"))
LLM_BACKEND        = os.getenv("LLM_BACKEND", "ollama")  # "ollama" or "hf"
#LLM_BACKEND        = os.getenv("LLM_BACKEND", "hf") 

# ─────────────────────────────────────────────────────────────────────────────
# RAG HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
# RELEVANCE_THRESHOLD: minimum FlashRank score (0–1) to pass a chunk to the LLM.
# 0.7 is deliberately conservative for medical context — prefer no answer over a wrong one.
RELEVANCE_THRESHOLD = float(os.getenv("RAG_RELEVANCE_THRESHOLD", "0.7"))

# Number of candidates retrieved from each retriever before score filtering.
DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "10"))
DEFAULT_TOP_K_R = int(os.getenv("RAG_DEFAULT_TOP_K_R", "10"))  # top-K passed to FlashRank for reranking
# Maximum chunks passed to the LLM after filtering (matches Dataset C: 1 or 2 chunks).
FINAL_TOP_K = int(os.getenv("RAG_FINAL_TOP_K", "2"))

# Child chunk: what gets embedded and searched.
# Smaller = more precise retrieval, less context per chunk.
CHILD_CHUNK_SIZE    = int(os.getenv("RAG_CHILD_CHUNK_SIZE", "300"))
CHILD_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP",    "100"))

# Parent chunk: what gets returned to the LLM as context.
# Larger = more context, but must fit within OLLAMA_MAX_TOKENS budget.
PARENT_CHUNK_SIZE    = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "1200"))
PARENT_CHUNK_OVERLAP = int(os.getenv("RAG_PARENT_CHUNK_OVERLAP", "100"))

SEMANTIC_SEPARATOR = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]
# ─────────────────────────────────────────────────────────────────────────────
# QUERY EXPANSION
# Appended to the user query before retrieval to improve recall of medical docs.
# ─────────────────────────────────────────────────────────────────────────────
EXPANSION_SUFFIX = os.getenv(
    "RAG_EXPANSION_SUFFIX",
    "clinical symptoms diagnosis treatment dermatology skin medical"
)

# ─────────────────────────────────────────────────────────────────────────────
# CHROMA COLLECTION
# ─────────────────────────────────────────────────────────────────────────────
COLLECTION_NAME = os.getenv("RAG_COLLECTION_NAME", "skin_health")

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPT
# Must match the SYSTEM_UNIFIED used during fine-tuning (system_prompt.py).
# Imported from there to keep a single source of truth.
# ─────────────────────────────────────────────────────────────────────────────
#try:
#    from prepare_datasets.system_prompt import SYSTEM_UNIFIED as SYSTEM_PROMPT
#except ImportError:
    # Fallback if system_prompt.py is not on the path yet
SYSTEM_PROMPT = (
"You are an empathetic dermatology assistant with three clinical capabilities.\n"
"In this answer, only the third capability FOLLOW-UP QA is relevant.\n"
"When a [CONTEXT] block is provided, your goal is to extract the answer directly from it.\n"
"RULES:\n"
"1. USE WHAT IS AVAILABLE: If the context contains a definition, description, or specific facts, use them to answer. \n"
"2. BE CONCISE: Summarize the key points if the context is extensive.\n"
"3. MINIMUM VIABLE ANSWER: Even if the context is brief (e.g., just a single sentence definition), use that sentence to answer the question. \n"
"4. DO NOT COMPLAIN about insufficient context unless the context is literally empty or completely unrelated to the question.\n"
"5. NO SPECULATION: Do not add external medical knowledge.\n"
"6. NO PRESCRIPTIONS: Never suggest specific dosages or medications.\n"
"7. NO META-COMMENTARY: Do not explain your reasoning. Do not say -The context provided discusses...-. Just give the answer directly.\n"
"8. SOURCE ATTRIBUTION: Mention the source of the facts you report.\n"
"9. MULTI-BLOCK SCANNING: Analyze each [CONTEXT] block independently. If Block 1 is irrelevant but Block 2 contains the answer, use Block 2 and ignore the irrelevance of Block 1.\n"
"10. POSITIVE EXTRACTION: Do not mention what the context lacks or does not say. Instead, focus exclusively on reporting the facts that ARE present.\n"
)
# ─────────────────────────────────────────────────────────────────────────────
# PUBMED INDEXING
# ─────────────────────────────────────────────────────────────────────────────
# Required by NCBI Entrez — must be a valid email address.
ENTREZ_EMAIL = os.getenv("ENTREZ_EMAIL", "r.lopez1@postgrad.curtin.edu.au")

RETMAX_PER_TOPIC   = int(os.getenv("RETMAX_PER_TOPIC",    "30"))
MIN_KEYWORD_DENSITY = float(os.getenv("MIN_KEYWORD_DENSITY", "0.01"))

# Medical keywords used to filter low-relevance abstracts before indexing.
MEDICAL_KEYWORDS = {
    # General dermatology terms
    "skin", "rash", "lesion", "dermatitis",
    "cause", "origin", "lifestyle", "prevention", "routine", 
    "home", "avoid", "daily", "understanding", "explain", "meaning",
    
    # Specific conditions and symptoms
    "psoriasis", "plaque", "scale", "scaly", "erythema",
    "acne", "comedone", "papule", "pustule", "cyst", "inflammatory",
    "eczema", "atopic", "atopic dermatitis", "itch", "pruritus",
    
    # General medical terms
    "topical", "corticosteroid", "emollient", "moisturizer",
    "treatment", "therapy", "management",
    
    # Clinical terms
    "diagnosis", "clinical", "patient", "guideline",
}
PUBMED_TOPICS_TO_INDEX = [
    # Psoriasis
    ("psoriasis AND \"Patient Education Handout\"[Publication Type]", 30),
    ("psoriasis AND patient education[Title/Abstract]", 25),
    ("psoriasis AND self care[MeSH]", 20),
    ("psoriasis AND quality of life[MeSH]", 20),
    
    # Acne
    ("acne vulgaris AND \"Patient Education Handout\"[Publication Type]", 30),
    ("acne AND patient education[Title/Abstract]", 25),
    ("acne AND self care[MeSH]", 20),
    
    # Eczema
    ("atopic dermatitis AND \"Patient Education Handout\"[Publication Type]", 30),
    ("eczema AND patient education[Title/Abstract]", 25),
    ("atopic dermatitis AND self care[MeSH]", 20),
]
'''
PUBMED_TOPICS_TO_INDEX = [
    # Psoriasis — reviews y guidelines
    ("psoriasis Review[pt]", 20),
    ("psoriasis pathophysiology Review[pt]", 15),
    ("psoriasis clinical features Practice Guideline[pt]", 15),
    
    # Acne — reviews y guidelines  
    ("acne vulgaris Review[pt]", 20),
    ("acne vulgaris pathogenesis Review[pt]", 15),
    ("acne treatment Practice Guideline[pt]", 15),
    
    # Eczema — reviews y guidelines
    ("atopic dermatitis Review[pt]", 20),
    ("eczema clinical features Review[pt]", 15),
    ("atopic dermatitis Practice Guideline[pt]", 15),
]
'''
# ─────────────────────────────────────────────────────────────────────────────
# MEDLINEPLUS INDEXING
# ─────────────────────────────────────────────────────────────────────────────

MEDLINEPLUS_TERMS = {
    "psoriasis", "acne","eczema", "skin conditions",
    "itching", "rashes", "skin lesion", "dermatitis", 
    "skin aging", "skin plaque", "skin scales",
    "comedones", "papule", "pustule", "phototherapy"
    }
MEDLINEPLUS_URL = "https://medlineplus.gov"
MEDLINEPLUS_BASE_URL = "https://wsearch.nlm.nih.gov/ws/query"
MEDLINEPLUS_DB = "healthTopics"
MEDLINEPLUS_MAX_RESULTS = 3 
