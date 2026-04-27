"""model_loader.py — Centralized Singleton model loading."""

import logging
from pathlib import Path
import sys

import torch
from sentence_transformers import SentenceTransformer
from flashrank import Ranker
from langchain_ollama import OllamaLLM
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import (
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    LLM_BACKEND,
    OLLAMA_MODEL,
    HF_MODEL,
    TEMPERATURE,
    MAX_TOKENS,
)

logger = logging.getLogger(__name__)

_embedding_model = None
_ranker = None
_llm = None


def _load_embedding_model():
    logger.info(f" Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    logger.info(" Embedding model loaded")
    return model


def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = _load_embedding_model()
    return _embedding_model


def _load_ranker():
    logger.info(f" Loading reranker model: {RERANKER_MODEL}")
    ranker = Ranker(model_name=RERANKER_MODEL)
    logger.info(" Reranker model loaded")
    return ranker


def get_ranker():
    global _ranker
    if _ranker is None:
        _ranker = _load_ranker()
    return _ranker


class HFLLMWrapper:
    def __init__(self, tokenizer, model, device, max_tokens):
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.max_tokens = max_tokens

    def invoke(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        attention_mask = inputs.attention_mask.to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        generated = outputs[0, input_ids.shape[-1] :].tolist()
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def _load_ollama_llm():
    logger.info(f" Loading Ollama model: {OLLAMA_MODEL}")
    logger.info(f"   Temperature: {TEMPERATURE}, Max tokens: {MAX_TOKENS}")
    llm = OllamaLLM(
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        num_predict=MAX_TOKENS,
        timeout=120,
    )
    logger.info(" Ollama model loaded")
    return llm


def _load_hf_llm():
    logger.info(f" Loading HuggingFace model: {HF_MODEL}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL,
        torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()
    logger.info(" HuggingFace model loaded")
    return HFLLMWrapper(tokenizer, model, device, MAX_TOKENS)


def _load_llm():
    if LLM_BACKEND == "hf":
        return _load_hf_llm()
    if LLM_BACKEND == "ollama":
        return _load_ollama_llm()
    raise ValueError(f"Unknown LLM_BACKEND: {LLM_BACKEND!r}. Use 'hf' or 'ollama'.")


def get_llm():
    global _llm
    if _llm is None:
        _llm = _load_llm()
    return _llm


def load_all_models():
    logger.info(" Preloading all models...")
    get_embedding_model()
    get_ranker()
    get_llm()
    logger.info(" All models preloaded")


def unload_models():
    global _embedding_model, _ranker, _llm
    _embedding_model = None
    _ranker = None
    _llm = None
    logger.info(" Models unloaded")


def get_models_status() -> dict:
    return {
        "embedding_model": {"loaded": _embedding_model is not None, "name": EMBEDDING_MODEL},
        "ranker": {"loaded": _ranker is not None, "name": RERANKER_MODEL},
        "llm": {"loaded": _llm is not None, "name": HF_MODEL if LLM_BACKEND == "hf" else OLLAMA_MODEL},
    }