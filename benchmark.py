#!/usr/bin/env python3
"""
Benchmark: Skincare Model vs LLaVA-1.5-7B vs LLaMA-3.2-3B-Instruct (base)
Evaluates visual diagnosis on Dataset D validation split.

All models receive the same skin images and patient symptoms.
Models are loaded one at a time to fit within 16 GB VRAM (T4).

Usage:
    python benchmark.py
    python benchmark.py --max-samples 20
    python benchmark.py --models skincare llava-1.5
    python benchmark.py --models llama-3b-base
"""

import argparse
import gc
import json
import logging
import os
import re
import sys
import time
from collections import defaultdict

import numpy as np
from dotenv import load_dotenv
load_dotenv()

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATASET_D_VAL = os.path.join(PROJECT_ROOT, "datasets", "dataset_d", "validation.jsonl")
IMAGES_DIR = os.path.join(PROJECT_ROOT, "datasets", "images_dataset")
DENSENET_CKPT = os.path.join(PROJECT_ROOT, "densenet_model", "outputs", "model.pth")
PROJECTION_CKPT = os.path.join(PROJECT_ROOT, "projection.pth")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "benchmark_outputs")

BASE_MODEL = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "model", "adapters")
LLAVA_15_REPO = "llava-hf/llava-1.5-7b-hf"
ALL_MODELS = ["skincare", "llava-1.5", "llama-3b-base"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def free_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def bnb_4bit():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_jsonl(path, max_samples=None):
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def resolve_image_path(item):
    raw = item["image_path"].replace("images/", "", 1)
    return os.path.join(IMAGES_DIR, raw)


def extract_patient_symptoms(messages):
    for msg in messages:
        if msg["role"] == "user" and "[PATIENT_SYMPTOMS]" in msg.get("content", ""):
            match = re.search(
                r"\[PATIENT_SYMPTOMS\](.*?)\[/PATIENT_SYMPTOMS\]",
                msg["content"], re.DOTALL,
            )
            if match:
                return match.group(1).strip()
    return ""


def extract_densenet_stats(messages):
    for msg in messages:
        if msg["role"] == "user" and "[DENSENET_STATISTICS]" in msg.get("content", ""):
            match = re.search(
                r"\[DENSENET_STATISTICS\](.*?)\[/DENSENET_STATISTICS\]",
                msg["content"], re.DOTALL,
            )
            if match:
                return match.group(1).strip()
    return ""


def ground_truth_condition(item):
    path = item["image_path"]
    for cond in ("acne", "eczema", "psoriasis"):
        if cond in path:
            return cond
    return item.get("DenseNet_output", {}).get("primary", "unknown")


# ── Metrics ───────────────────────────────────────────────────────────────────

CONDITION_LABELS = ["acne", "eczema", "psoriasis"]

NEGATION_PATTERNS = [
    r"(?:not|n't|no)\s+(?:\w+\s+){0,3}%s",
    r"(?:unlikely|improbable|doubtful)\s+(?:to be\s+)?(?:\w+\s+){0,2}%s",
    r"(?:rule[sd]?\s+out|ruled\s+out|ruling\s+out)\s+(?:\w+\s+){0,2}%s",
    r"(?:excluded?|eliminat(?:ed?|ing))\s+%s",
    r"%s\s+(?:is|was|seems?|appears?)\s+(?:unlikely|improbable|ruled out)",
    r"%s\s+(?:can|could|should)\s+be\s+(?:ruled out|excluded|eliminated)",
]


def _is_negated(condition, text):
    text_lower = text.lower()
    for pattern in NEGATION_PATTERNS:
        if re.search(pattern % re.escape(condition), text_lower):
            return True
    return False


def extract_diagnosis(response):
    response_lower = response.lower()
    affirmed = []
    for cond in CONDITION_LABELS:
        if cond in response_lower and not _is_negated(cond, response):
            affirmed.append(cond)

    if len(affirmed) == 1:
        return affirmed[0]

    if len(affirmed) > 1:
        primary_patterns = [
            r"(?:diagnosis|assessment|impression|consistent with|suggestive of|"
            r"appears? to be|likely|most likely|indicates?)\s*(?::?\s*)(?:\w+\s+){0,3}(%s)",
            r"(%s)\s+(?:is|appears?|seems?)\s+(?:the\s+)?(?:most\s+)?(?:likely|probable|primary)",
        ]
        cond_pattern = "|".join(re.escape(c) for c in affirmed)
        for pat in primary_patterns:
            m = re.search(pat % cond_pattern, response_lower)
            if m:
                matched = m.group(1)
                for c in affirmed:
                    if c in matched:
                        return c
        first_pos = {c: response_lower.index(c) for c in affirmed}
        return min(first_pos, key=first_pos.get)

    return "unknown"


def extract_reference_answer(messages):
    for msg in reversed(messages):
        if msg["role"] == "assistant" and msg["content"].strip():
            return msg["content"].strip()
    return ""


def compute_metrics(response, gt_condition, patient_symptoms):
    response_lower = response.lower()
    predicted = extract_diagnosis(response)
    has_disclaimer = bool(
        re.search(r"dermatologist|medical professional|healthcare", response_lower)
    )

    return {
        "gt_condition": gt_condition,
        "predicted_condition": predicted,
        "condition_correct": predicted == gt_condition,
        "has_disclaimer": has_disclaimer,
        "response_length": len(response.split()),
    }


def compute_text_similarity(predictions, references):
    from bert_score import score as bert_score
    from rouge_score import rouge_scorer

    logger.info("Computing BERTScore over %d samples ...", len(predictions))
    P, R, F1 = bert_score(
        predictions, references, lang="en", verbose=False, device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bert_scores = {"precision": P.tolist(), "recall": R.tolist(), "f1": F1.tolist()}

    logger.info("Computing ROUGE-L ...")
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l_scores = []
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref, pred)
        rouge_l_scores.append(s["rougeL"].fmeasure)

    return bert_scores, rouge_l_scores


def bootstrap_ci(values, n_bootstrap=1000, ci=95):
    values = np.array(values, dtype=float)
    boot_means = np.array([
        np.mean(np.random.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return round(float(np.mean(values)), 3), round(float(lower), 3), round(float(upper), 3)


def aggregate_metrics(results, bert_scores=None, rouge_l_scores=None):
    n = len(results)
    if n == 0:
        return {}

    y_true = [r["gt_condition"] for r in results]
    y_pred = [r["predicted_condition"] for r in results]

    acc = round(accuracy_score(y_true, y_pred), 3)
    macro_f1 = round(f1_score(y_true, y_pred, labels=CONDITION_LABELS, average="macro", zero_division=0), 3)
    macro_prec = round(precision_score(y_true, y_pred, labels=CONDITION_LABELS, average="macro", zero_division=0), 3)
    macro_rec = round(recall_score(y_true, y_pred, labels=CONDITION_LABELS, average="macro", zero_division=0), 3)

    correct = [float(r["condition_correct"]) for r in results]
    acc_mean, acc_lo, acc_hi = bootstrap_ci(correct)

    per_condition = {}
    by_cond = defaultdict(list)
    for r in results:
        by_cond[r["gt_condition"]].append(float(r["condition_correct"]))
    for cond in sorted(by_cond):
        mean, lo, hi = bootstrap_ci(by_cond[cond])
        per_condition[cond] = {"accuracy": mean, "ci_95": [lo, hi], "n": len(by_cond[cond])}

    cm = confusion_matrix(y_true, y_pred, labels=CONDITION_LABELS)
    cm_dict = {
        "labels": CONDITION_LABELS,
        "matrix": cm.tolist(),
    }

    report = classification_report(
        y_true, y_pred, labels=CONDITION_LABELS, output_dict=True, zero_division=0,
    )
    per_class_report = {
        cond: {
            "precision": round(report[cond]["precision"], 3),
            "recall": round(report[cond]["recall"], 3),
            "f1": round(report[cond]["f1-score"], 3),
            "support": report[cond]["support"],
        }
        for cond in CONDITION_LABELS if cond in report
    }

    summary = {
        "n": n,
        "accuracy": acc,
        "accuracy_ci_95": [acc_lo, acc_hi],
        "macro_f1": macro_f1,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "per_class": per_class_report,
        "per_condition_accuracy": per_condition,
        "confusion_matrix": cm_dict,
        "disclaimer_rate": round(sum(r["has_disclaimer"] for r in results) / n, 3),
        "avg_response_length": round(sum(r["response_length"] for r in results) / n, 1),
    }

    if bert_scores is not None:
        bs_mean, bs_lo, bs_hi = bootstrap_ci(bert_scores["f1"])
        summary["bert_score_f1"] = bs_mean
        summary["bert_score_f1_ci_95"] = [bs_lo, bs_hi]

    if rouge_l_scores is not None:
        rl_mean, rl_lo, rl_hi = bootstrap_ci(rouge_l_scores)
        summary["rouge_l"] = rl_mean
        summary["rouge_l_ci_95"] = [rl_lo, rl_hi]

    return summary


# ── Skincare Model ────────────────────────────────────────────────────────────

def load_skincare(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    import torch.nn.functional as F

    hf_token = os.environ.get("HF_TOKEN", None)

    logger.info("Loading base model: %s", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_4bit(),
        device_map="auto",
        token=hf_token,
    )

    image_adapter = os.path.join(ADAPTER_DIR, "image")
    symptoms_adapter = os.path.join(ADAPTER_DIR, "symptoms")

    logger.info("Loading image LoRA adapter from %s", image_adapter)
    model = PeftModel.from_pretrained(base, image_adapter, adapter_name="image")

    logger.info("Loading symptoms LoRA adapter from %s", symptoms_adapter)
    model.load_adapter(symptoms_adapter, adapter_name="symptoms")

    model.set_adapter("image")
    model.eval()

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "visual_alignment"))
    from projection import ProjectionMLP

    llm_dim = model.config.hidden_size
    projection = ProjectionMLP(densenet_dim=1024, llm_dim=llm_dim,
                               num_tokens=32, hidden_dim=2048)
    projection.load_state_dict(torch.load(PROJECTION_CKPT, map_location="cpu"))
    projection.eval().to(device).to(torch.float16)

    sys.path.insert(0, os.path.join(PROJECT_ROOT, "densenet_model", "src"))
    from model import build_model
    densenet_full = build_model(pretrained=False, fine_tune=False, num_classes=3)
    densenet_full.load_state_dict(torch.load(DENSENET_CKPT, map_location="cpu"))
    densenet_features = densenet_full.features
    densenet_pool = torch.nn.AdaptiveAvgPool2d((1, 1))

    class FeatExtractor(torch.nn.Module):
        def __init__(self, features, pool):
            super().__init__()
            self.features = features
            self.pool = pool
        def forward(self, x):
            feat = F.relu(self.features(x))
            return self.pool(feat).view(feat.size(0), -1)

    feat_extractor = FeatExtractor(densenet_features, densenet_pool).eval().to(device)
    for p in feat_extractor.parameters():
        p.requires_grad = False

    from visual_alignment.dataset import IMAGE_TRANSFORM

    logger.info("Skincare model ready — active adapter: image (%.1f GB VRAM used)",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)

    return {
        "model": model, "tokenizer": tokenizer, "projection": projection,
        "feat_extractor": feat_extractor, "image_transform": IMAGE_TRANSFORM,
    }


def _chat_ids(messages, tokenizer, prime_assistant=True):
    bos = tokenizer.bos_token_id
    ids = [bos] if bos is not None else []
    start_h = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    end_h = tokenizer.encode("<|end_header_id|>", add_special_tokens=False)
    eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
    nl2 = tokenizer.encode("\n\n", add_special_tokens=False)

    for msg in messages:
        if msg["content"] == "":
            continue
        role_ids = tokenizer.encode(msg["role"], add_special_tokens=False)
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        ids += start_h + role_ids + end_h + nl2 + content_ids + eot

    if prime_assistant:
        asst_ids = tokenizer.encode("assistant", add_special_tokens=False)
        ids += start_h + asst_ids + end_h + nl2

    return ids


@torch.no_grad()
def run_skincare(bundle, sample, device, max_new_tokens):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    projection = bundle["projection"]
    feat_extractor = bundle["feat_extractor"]
    transform = bundle["image_transform"]

    messages = sample["messages"]
    image_msg_index = sample["image_msg_index"]
    image_path = resolve_image_path(sample)

    pre_msgs = messages[:image_msg_index]
    text_ids = _chat_ids(pre_msgs, tokenizer, prime_assistant=True)
    asst_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    pre_ids = text_ids[:len(text_ids) - len(asst_header_ids)]

    pre_tensor = torch.tensor([pre_ids], dtype=torch.long, device=device)
    asst_tensor = torch.tensor([asst_header_ids], dtype=torch.long, device=device)

    embed_fn = model.get_input_embeddings()
    pre_embeds = embed_fn(pre_tensor)
    asst_embeds = embed_fn(asst_tensor)

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    vis_feat = feat_extractor(img_tensor)
    vis_tokens = projection(vis_feat.to(torch.float16))
    vis_tokens = vis_tokens.to(pre_embeds.dtype)

    combined = torch.cat([pre_embeds, vis_tokens, asst_embeds], dim=1)
    attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)

    out = model.generate(
        inputs_embeds=combined,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ── LLaVA-1.5 ────────────────────────────────────────────────────────────────

LLAVA_SYSTEM_PROMPT = (
    "You are a dermatology assistant. Analyze the provided skin image along "
    "with the patient's reported symptoms. Reason step by step using "
    "<think>...</think> covering: visual morphology, symptom correlation, "
    "and differential diagnosis. Then respond inside [DIAGNOSIS]...[/DIAGNOSIS] "
    "as a doctor speaking to the patient: plain language, 3-4 short paragraphs, "
    "explain what you see and why. Always recommend consulting a dermatologist. "
    "Never provide a definitive diagnosis."
)


def load_llava15(device):
    from transformers import LlavaForConditionalGeneration, AutoProcessor

    logger.info("Loading LLaVA-1.5-7B from HF: %s", LLAVA_15_REPO)
    hf_token = os.environ.get("HF_TOKEN", None)

    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_15_REPO,
        quantization_config=bnb_4bit(),
        device_map="auto",
        token=hf_token,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(LLAVA_15_REPO, token=hf_token)

    logger.info("LLaVA-1.5 ready (%.1f GB VRAM used)",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
    return {"model": model, "processor": processor}


def _build_llava_prompt(patient_symptoms):
    user_text = (
        f"{LLAVA_SYSTEM_PROMPT}\n\n"
        f"Patient reported symptoms:\n{patient_symptoms}\n\n"
        "Please analyze the skin image and provide your assessment."
    )
    return f"USER: <image>\n{user_text}\nASSISTANT:"


@torch.no_grad()
def run_llava15(bundle, sample, device, max_new_tokens):
    model = bundle["model"]
    processor = bundle["processor"]

    image_path = resolve_image_path(sample)
    image = Image.open(image_path).convert("RGB")
    patient_symptoms = extract_patient_symptoms(sample["messages"])

    prompt = _build_llava_prompt(patient_symptoms)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device)

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    generated = out[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ── LLaMA 3.2 3B Instruct (base, no adapters) ───────────────────────────────

LLAMA_BASE_SYSTEM_PROMPT = (
    "You are a dermatology assistant. Based on the patient's reported symptoms, "
    "reason step by step using <think>...</think> covering: symptom analysis "
    "and differential diagnosis. Then respond inside [DIAGNOSIS]...[/DIAGNOSIS] "
    "as a doctor speaking to the patient: plain language, 3-4 short paragraphs, "
    "explain your reasoning and assessment. Always recommend consulting a dermatologist. "
    "Never provide a definitive diagnosis."
)


def load_llama_base(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_token = os.environ.get("HF_TOKEN", None)
    logger.info("Loading base LLaMA 3.2 3B Instruct: %s", BASE_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_4bit(),
        device_map="auto",
        token=hf_token,
    )
    model.eval()

    logger.info("LLaMA 3.2 3B Instruct (base) ready (%.1f GB VRAM used)",
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0)
    return {"model": model, "tokenizer": tokenizer}


@torch.no_grad()
def run_llama_base(bundle, sample, device, max_new_tokens):
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]

    patient_symptoms = extract_patient_symptoms(sample["messages"])

    messages = [
        {"role": "system", "content": LLAMA_BASE_SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Patient reported symptoms:\n{patient_symptoms}\n\n"
            "Based on these symptoms, what skin condition could the patient have? "
            "Please provide your assessment."
        )},
    ]

    input_ids = _chat_ids(messages, tokenizer, prime_assistant=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    out = model.generate(
        input_ids=input_tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated = out[0][input_tensor.shape[-1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ── Runner ────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "skincare": {
        "load": load_skincare,
        "run": run_skincare,
        "name": "Skincare (3B + MultiLoRA)",
        "params": "~3B",
    },
    "llava-1.5": {
        "load": load_llava15,
        "run": run_llava15,
        "name": "LLaVA-1.5-7B",
        "params": "~7B",
    },
    "llama-3b-base": {
        "load": load_llama_base,
        "run": run_llama_base,
        "name": "LLaMA-3.2-3B-Instruct (base)",
        "params": "~3B",
    },
}


def run_benchmark(model_key, samples, device, max_new_tokens):
    reg = MODEL_REGISTRY[model_key]
    logger.info("=" * 60)
    logger.info("BENCHMARKING: %s (%s params)", reg["name"], reg["params"])
    logger.info("=" * 60)

    start = time.time()
    bundle = reg["load"](device)
    load_time = time.time() - start
    logger.info("Model loaded in %.1f s", load_time)

    if bundle is None:
        logger.warning("Skipping %s — model failed to load", model_key)
        return None, None

    results = []
    responses = []
    pred_texts = []
    ref_texts = []

    pbar = tqdm(samples, desc=model_key, unit="sample")
    for sample in pbar:
        patient_symptoms = extract_patient_symptoms(sample["messages"])
        gt_cond = ground_truth_condition(sample)

        try:
            response = reg["run"](bundle, sample, device, max_new_tokens)
        except Exception as e:
            logger.warning("Error on sample %s: %s", sample.get("image_path", "?"), e)
            response = f"(error: {e})"

        metrics = compute_metrics(response, gt_cond, patient_symptoms)
        results.append(metrics)
        responses.append({
            "image_path": sample["image_path"],
            "response": response,
            **metrics,
        })

        pred_texts.append(response)
        ref_texts.append(extract_reference_answer(sample["messages"]))

        running_acc = sum(r["condition_correct"] for r in results) / len(results)
        pbar.set_postfix(acc=f"{running_acc:.0%}")

    del bundle
    free_gpu()

    bert_scores, rouge_l_scores = None, None
    if ref_texts and any(ref_texts):
        bert_scores, rouge_l_scores = compute_text_similarity(pred_texts, ref_texts)
        for i, resp in enumerate(responses):
            resp["bert_score_f1"] = round(bert_scores["f1"][i], 3)
            resp["rouge_l"] = round(rouge_l_scores[i], 3)

    total_time = time.time() - start
    summary = aggregate_metrics(results, bert_scores, rouge_l_scores)
    summary["load_time_s"] = round(load_time, 1)
    summary["total_time_s"] = round(total_time, 1)
    summary["avg_time_per_sample_s"] = round((total_time - load_time) / max(len(results), 1), 2)

    return summary, responses


# ── Reporting ─────────────────────────────────────────────────────────────────

def _fmt_ci(mean, ci):
    if ci:
        return f"{mean:.1%} [{ci[0]:.1%}, {ci[1]:.1%}]"
    return f"{mean:.1%}"


def print_comparison(summaries):
    models = list(summaries.keys())
    col_w = 30
    label_w = 24

    print("\n" + "=" * (label_w + 2 + col_w * len(models)))
    print("BENCHMARK RESULTS — Visual Diagnosis (Dataset D)")
    print("=" * (label_w + 2 + col_w * len(models)))

    def _model_name(key):
        return MODEL_REGISTRY[key]["name"] if key in MODEL_REGISTRY else key

    header = f"  {'Metric':<{label_w}}"
    for m in models:
        header += f"  {_model_name(m):>{col_w - 2}}"
    print(header)
    print("  " + "-" * (label_w + col_w * len(models)))

    print("\n  Classification Metrics:")
    for key, label in [
        ("accuracy", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("macro_precision", "Macro Precision"),
        ("macro_recall", "Macro Recall"),
    ]:
        row = f"    {label:<{label_w - 2}}"
        for m in models:
            val = summaries[m].get(key, 0)
            ci = summaries[m].get(f"{key}_ci_95")
            row += f"  {_fmt_ci(val, ci):>{col_w - 2}}"
        print(row)

    print("\n  Text Similarity:")
    for key, label in [
        ("bert_score_f1", "BERTScore F1"),
        ("rouge_l", "ROUGE-L"),
    ]:
        row = f"    {label:<{label_w - 2}}"
        for m in models:
            val = summaries[m].get(key)
            if val is None:
                row += f"  {'n/a':>{col_w - 2}}"
            else:
                ci = summaries[m].get(f"{key}_ci_95")
                row += f"  {_fmt_ci(val, ci):>{col_w - 2}}"
        print(row)

    print("\n  Other:")
    for key, label, is_pct in [
        ("disclaimer_rate", "Disclaimer Present", True),
        ("avg_response_length", "Avg Response Length", False),
        ("avg_time_per_sample_s", "Avg Time/Sample (s)", False),
    ]:
        row = f"    {label:<{label_w - 2}}"
        for m in models:
            val = summaries[m].get(key, 0)
            if is_pct:
                row += f"  {val:>{col_w - 2}.1%}"
            else:
                row += f"  {val:>{col_w - 2}}"
        print(row)

    print("\n  Per-class Precision / Recall / F1:")
    for cond in CONDITION_LABELS:
        row = f"    {cond:<{label_w - 2}}"
        for m in models:
            pc = summaries[m].get("per_class", {}).get(cond, {})
            p = pc.get("precision", 0)
            r = pc.get("recall", 0)
            f = pc.get("f1", 0)
            cell = f"P={p:.2f} R={r:.2f} F1={f:.2f}"
            row += f"  {cell:>{col_w - 2}}"
        print(row)

    print("\n  Per-condition Accuracy (with 95% CI):")
    for cond in CONDITION_LABELS:
        row = f"    {cond:<{label_w - 2}}"
        for m in models:
            pc = summaries[m].get("per_condition_accuracy", {}).get(cond, {})
            acc = pc.get("accuracy", 0)
            ci = pc.get("ci_95")
            row += f"  {_fmt_ci(acc, ci):>{col_w - 2}}"
        print(row)

    print("\n  Confusion Matrix (rows=true, cols=predicted):")
    for m in models:
        cm = summaries[m].get("confusion_matrix", {})
        labels = cm.get("labels", CONDITION_LABELS)
        matrix = cm.get("matrix", [])
        print(f"\n    {_model_name(m)}:")
        print(f"    {'':>12}  " + "  ".join(f"{l:>10}" for l in labels))
        for i, lbl in enumerate(labels):
            if i < len(matrix):
                print(f"    {lbl:>12}  " + "  ".join(f"{v:>10}" for v in matrix[i]))

    print("\n" + "=" * (label_w + 2 + col_w * len(models)))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark skincare model vs LLaVA models")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit number of validation samples (default: all 137)")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Max generation tokens per sample")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        choices=ALL_MODELS, help="Which models to benchmark")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    if torch.cuda.is_available():
        logger.info("GPU: %s (%.1f GB)", torch.cuda.get_device_name(),
                     torch.cuda.get_device_properties(0).total_memory / 1e9)

    logger.info("Loading Dataset D validation ...")
    samples = load_jsonl(DATASET_D_VAL, args.max_samples)
    logger.info("Loaded %d samples", len(samples))

    for s in samples:
        img = resolve_image_path(s)
        if not os.path.exists(img):
            logger.error("Image not found: %s", img)
            sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_summaries = {}
    all_responses = {}

    for model_key in args.models:
        summary, responses = run_benchmark(
            model_key, samples, device, args.max_new_tokens,
        )
        if summary is not None:
            all_summaries[model_key] = summary
            all_responses[model_key] = responses

    summary_path = os.path.join(OUTPUT_DIR, "benchmark_summary.json")
    detail_path = os.path.join(OUTPUT_DIR, "benchmark_responses.json")

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            existing_summaries = json.load(f)
        existing_summaries.pop("llava-med", None)
        existing_summaries.update(all_summaries)
        all_summaries = existing_summaries

    if os.path.exists(detail_path):
        with open(detail_path) as f:
            existing_responses = json.load(f)
        existing_responses.pop("llava-med", None)
        existing_responses.update(all_responses)
        all_responses = existing_responses

    if all_summaries:
        print_comparison(all_summaries)

    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    with open(detail_path, "w") as f:
        json.dump(all_responses, f, indent=2)

    logger.info("Summary  → %s", summary_path)
    logger.info("Responses → %s", detail_path)


if __name__ == "__main__":
    main()
