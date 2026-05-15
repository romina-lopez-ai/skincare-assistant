"""config.py — Configuration for the RAG evaluation module."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Repo root: evaluation_metrics/rag/ → two levels up
REPO_ROOT = Path(__file__).parent.parent.parent

# ── ChromaDB (mirrors rag/config.py env vars) ─────────────────────────────────
CHROMA_DB_PATH   = REPO_ROOT / "data" / "chroma_db"
PARENT_STORE_PATH = REPO_ROOT / "data" / "parent_docs_store"
COLLECTION_NAME  = os.getenv("RAG_COLLECTION_NAME", "skin_health")

# ── Output paths ──────────────────────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
EVAL_DATASET_PATH = DATA_DIR    / "all_eval_dataset.json"
RESULTS_PATH      = RESULTS_DIR / "rag_eval_results.json"

# ── Eval dataset composition ──────────────────────────────────────────────────
CONDITIONS           = ["acne", "eczema", "psoriasis"]
SAMPLES_PER_CONDITION = 10
NEGATIVE_SAMPLES     = 10

# Hard-coded out-of-index queries (clearly non-dermatological)
NEGATIVE_QUERIES = [
    "What are the stages of colorectal cancer?",
    "How is type 2 diabetes managed with insulin?",
    "What is the standard treatment for hypertension?",
    "How do ACE inhibitors work for heart failure?",
    "What are the symptoms of appendicitis?",
    "How is hypothyroidism diagnosed with blood tests?",
    "What are the dietary recommendations for chronic kidney disease?",
    "How is rheumatoid arthritis different from osteoarthritis?",
    "What vaccines are recommended for pneumonia prevention?",
    "How is COPD managed in elderly patients?",
]

# ── LLM judge (question generation + RAGAS) ───────────────────────────────────
# Options: "groq" (free, open source) or "openai" (paid fallback)
JUDGE_PROVIDER      = os.getenv("JUDGE_PROVIDER", "openai")
JUDGE_MODEL_GROQ    = os.getenv("JUDGE_MODEL_GROQ",   "llama-3.3-70b-versatile")
JUDGE_MODEL_OPENAI  = os.getenv("JUDGE_MODEL_OPENAI", "gpt-4o-mini")

GROQ_API_KEY   = os.getenv("GROQ_API_KEY",   "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Embeddings for RAGAS answer_relevancy metric (uses the same model already in the project)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Models for fine-tuned vs base comparison ──────────────────────────────────
BASE_MODEL_ID = "RominaLopez/llama-3.2-3b-4bit-quantized"

# Set SKIP_LLM=1 to skip HF model loading (use when GPU/model is unavailable)
SKIP_LLM = os.getenv("SKIP_LLM", "0") == "1"
MAX_NEW_TOKENS = 512

# ── Evaluation thresholds ─────────────────────────────────────────────────────
HR_PRIMARY_K      = 2     # matches RAG pipeline FINAL_TOP_K
HR_THRESHOLD      = 0.40  # minimum acceptable HR@2
FP_RATE_THRESHOLD = 0.1  # max acceptable false positive rate on out-of-index queries

RAGAS_THRESHOLDS = {
    "faithfulness":       0.80,
    "answer_relevancy":   0.75,
    "context_precision":  0.70,
    "context_recall":     0.65,
}
