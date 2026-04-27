"""Evaluation — Phase 4 fine-tuned vs zero-shot Llama on Dataset F.

Evaluates *behavioral* compliance for the image + symptoms → diagnosis task.

Dataset F metrics (fine-tuned uses real visual tokens; zero-shot uses text-serialized DenseNet):
  - think_block_rate      : <think>...</think> present
  - diagnosis_tag_rate    : [DIAGNOSIS] present
  - condition_match_rate  : primary condition from DenseNet appears in response
  - disclaimer_rate       : response mentions "dermatologist"
  - symptoms_referenced_rate : <think> block references >= 1 symptom term from patient

Usage (from project root):
    python unified_training/evaluate.py
    python unified_training/evaluate.py --max-samples 5
    python unified_training/evaluate.py --skip-zeroshot

Outputs:
    unified_training/outputs/eval_comparison.json  — per-sample detail
    unified_training/outputs/eval_summary.json     — aggregate behavioral table
"""

import argparse
import json
import logging
import os
import re
import sys

from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_this_dir, "..")
for _p in (_this_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visual_alignment.projection import ProjectionMLP
from visual_alignment.dataset import IMAGE_TRANSFORM
from unified_training.config import Phase4Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DenseNet extractor (identical to train.py)
# ---------------------------------------------------------------------------

class DenseNetFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        sys.path.insert(0, os.path.join(_project_root, "densenet_model", "src"))
        from model import build_model
        full = build_model(pretrained=False, fine_tune=False, num_classes=3)
        full.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.features = full.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = F.relu(self.features(x))
        return self.pool(feat).view(feat.size(0), -1)


# ---------------------------------------------------------------------------
# Behavioral checkers
# ---------------------------------------------------------------------------

def has_diagnosis_tag(text: str) -> bool:
    return "[DIAGNOSIS]" in text

def has_think_block(text: str) -> bool:
    return bool(re.search(r"<think>.*?</think>", text, re.DOTALL))

def extract_think_block(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1) if m else ""

def has_disclaimer(text: str) -> bool:
    return bool(re.search(r"dermatologist", text, re.IGNORECASE))

def condition_in_response(response: str, primary: str) -> bool:
    return primary.lower() in response.lower()

def symptoms_referenced(think_text: str, symptom_words: set[str]) -> bool:
    if not symptom_words:
        return False
    think_words = set(re.findall(r"[a-zA-Z]{3,}", think_text.lower()))
    return bool(think_words & symptom_words)


# ---------------------------------------------------------------------------
# Chat prompt builder (text tokens only)
# ---------------------------------------------------------------------------

def _chat_ids(messages: list, tokenizer, prime_assistant: bool = True) -> list[int]:
    """Build LLaMA-3.2 chat format token IDs for a message list.

    If prime_assistant=True, appends the assistant header to trigger generation.
    Skips empty-content messages (image placeholders).
    """
    bos     = tokenizer.bos_token_id
    ids     = [bos] if bos is not None else []
    start_h = tokenizer.encode("<|start_header_id|>", add_special_tokens=False)
    end_h   = tokenizer.encode("<|end_header_id|>",   add_special_tokens=False)
    eot     = tokenizer.encode("<|eot_id|>",          add_special_tokens=False)
    nl2     = tokenizer.encode("\n\n",                add_special_tokens=False)

    for msg in messages:
        if msg["content"] == "":
            continue
        role_ids    = tokenizer.encode(msg["role"],    add_special_tokens=False)
        content_ids = tokenizer.encode(msg["content"], add_special_tokens=False)
        ids += start_h + role_ids + end_h + nl2 + content_ids + eot

    if prime_assistant:
        asst_ids = tokenizer.encode("assistant", add_special_tokens=False)
        ids += start_h + asst_ids + end_h + nl2

    return ids


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _bnb():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def _load_base(config: Phase4Config, device: torch.device):
    kwargs = dict(token=config.hf_token or None)
    if device.type == "cuda":
        return AutoModelForCausalLM.from_pretrained(
            config.slm_model_name, quantization_config=_bnb(),
            device_map="auto", **kwargs
        )
    return AutoModelForCausalLM.from_pretrained(
        config.slm_model_name, **kwargs
    ).to(device)

def _load_tok(config: Phase4Config):
    tok = AutoTokenizer.from_pretrained(
        config.slm_model_name, token=config.hf_token or None
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

def load_finetuned(config: Phase4Config, device: torch.device):
    logger.info("Loading fine-tuned model …")
    base  = _load_base(config, device)
    model = PeftModel.from_pretrained(
        base, os.path.join(config.output_dir, "lora_adapters")
    )
    model.eval()
    tok = _load_tok(config)

    llm_dim = model.config.hidden_size
    projection = ProjectionMLP(
        densenet_dim=config.densenet_dim,
        llm_dim=llm_dim,
        num_tokens=config.num_visual_tokens,
        hidden_dim=config.projection_hidden_dim,
    )
    projection.load_state_dict(
        torch.load(os.path.join(config.output_dir, "projection.pth"), map_location="cpu")
    )
    projection.eval().to(device).to(torch.bfloat16)

    feat_extractor = DenseNetFeatureExtractor(config.densenet_checkpoint).eval().to(device)

    logger.info("Fine-tuned model ready.")
    return model, tok, projection, feat_extractor

def load_zeroshot(config: Phase4Config, device: torch.device):
    logger.info("Loading zero-shot (base) model …")
    model = _load_base(config, device)
    model.eval()
    tok = _load_tok(config)
    logger.info("Zero-shot model ready.")
    return model, tok


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_from_ids(model, tokenizer, input_ids: torch.Tensor,
                      max_new_tokens: int, device: torch.device) -> str:
    input_ids = input_ids.to(device)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    new_ids = out[0, input_ids.shape[-1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_from_embeds(model, tokenizer, inputs_embeds: torch.Tensor,
                         max_new_tokens: int) -> str:
    attn_mask = torch.ones(inputs_embeds.shape[:2], dtype=torch.long,
                           device=inputs_embeds.device)
    out = model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Visual token injection for fine-tuned path
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_visual_embeds(
    model, tokenizer, messages: list, image_msg_index: int,
    image_path: str, feat_extractor, projection,
    device: torch.device, num_visual_tokens: int,
) -> torch.Tensor:
    """Build combined text+visual embeddings for one sample.

    Layout: [pre-image text tokens] + [N visual tokens] + [assistant header tokens]
    """
    pre_image_msgs = messages[:image_msg_index]
    text_ids = _chat_ids(pre_image_msgs, tokenizer, prime_assistant=True)
    asst_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    pre_ids = text_ids[: len(text_ids) - len(asst_header_ids)]

    pre_tensor  = torch.tensor([pre_ids],        dtype=torch.long, device=device)
    asst_tensor = torch.tensor([asst_header_ids], dtype=torch.long, device=device)

    embed_fn    = model.get_input_embeddings()
    pre_embeds  = embed_fn(pre_tensor)
    asst_embeds = embed_fn(asst_tensor)

    image = Image.open(image_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    vis_feat   = feat_extractor(img_tensor)
    vis_tokens = projection(vis_feat.to(torch.bfloat16))
    vis_tokens = vis_tokens.to(pre_embeds.dtype)

    combined = torch.cat([pre_embeds, vis_tokens, asst_embeds], dim=1)
    return combined


# ---------------------------------------------------------------------------
# Zero-shot text serialization (no visual tokens)
# ---------------------------------------------------------------------------

def serialize_densenet(densenet_output: dict) -> str:
    primary   = densenet_output.get("primary", "unknown")
    conf      = densenet_output.get("confidence", 0.0)
    secondary = densenet_output.get("secondary", "")
    conf2     = densenet_output.get("confidence_2", 0.0)

    lines = [
        "[VISUAL_INPUT]",
        f"DenseNet prediction: {primary} ({conf:.0%} confidence)",
    ]
    if secondary:
        lines.append(f"Secondary: {secondary} ({conf2:.0%})")
    lines.append("[/VISUAL_INPUT]")
    return "\n".join(lines)


def build_zeroshot_messages(messages: list, image_msg_index: int,
                            densenet_output: dict) -> list:
    """Replace the image placeholder with a text-serialized DenseNet message."""
    result = list(messages[:image_msg_index])
    result.append({"role": "user", "content": serialize_densenet(densenet_output)})
    return result


# ---------------------------------------------------------------------------
# Symptom words extraction
# ---------------------------------------------------------------------------

def extract_symptom_words(messages: list, image_msg_index: int) -> set[str]:
    """Content words from user turns before the image placeholder."""
    stop = {"the", "a", "an", "is", "are", "was", "I", "my", "it", "yes",
            "no", "and", "or", "but", "for", "with", "on", "at", "by",
            "have", "has", "been", "some", "they", "that", "this", "its"}
    words: set[str] = set()
    for msg in messages[:image_msg_index]:
        if msg["role"] == "user" and msg["content"]:
            for w in re.findall(r"[a-zA-Z]{3,}", msg["content"].lower()):
                if w not in stop:
                    words.add(w)
    return words


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------

def evaluate_sample_finetuned(
    model, tokenizer, messages: list, image_msg_index: int,
    image_path: str, densenet_output: dict,
    feat_extractor, projection,
    max_new_tokens: int, device: torch.device,
) -> dict:
    embeds        = build_visual_embeds(
        model, tokenizer, messages, image_msg_index,
        image_path, feat_extractor, projection, device,
        projection.num_tokens,
    )
    response      = generate_from_embeds(model, tokenizer, embeds, max_new_tokens)
    primary       = densenet_output.get("primary", "")
    think_text    = extract_think_block(response)
    symptom_words = extract_symptom_words(messages, image_msg_index)

    return {
        "has_think_block":        has_think_block(response),
        "has_diagnosis_tag":      has_diagnosis_tag(response),
        "condition_match":        condition_in_response(response, primary),
        "has_disclaimer":         has_disclaimer(response),
        "symptoms_referenced":    symptoms_referenced(think_text, symptom_words),
        "response":               response,
        "primary_condition":      primary,
    }


def evaluate_sample_zeroshot(
    model, tokenizer, messages: list, image_msg_index: int,
    densenet_output: dict, max_new_tokens: int, device: torch.device,
) -> dict:
    input_msgs    = build_zeroshot_messages(messages, image_msg_index, densenet_output)
    ids           = torch.tensor([_chat_ids(input_msgs, tokenizer)], dtype=torch.long)
    response      = generate_from_ids(model, tokenizer, ids, max_new_tokens, device)
    primary       = densenet_output.get("primary", "")
    think_text    = extract_think_block(response)
    symptom_words = extract_symptom_words(messages, image_msg_index)

    return {
        "has_think_block":        has_think_block(response),
        "has_diagnosis_tag":      has_diagnosis_tag(response),
        "condition_match":        condition_in_response(response, primary),
        "has_disclaimer":         has_disclaimer(response),
        "symptoms_referenced":    symptoms_referenced(think_text, symptom_words),
        "response":               response,
        "primary_condition":      primary,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "n": n,
        "think_block_rate":          round(sum(r["has_think_block"]      for r in results) / n, 3),
        "diagnosis_tag_rate":        round(sum(r["has_diagnosis_tag"]    for r in results) / n, 3),
        "condition_match_rate":      round(sum(r["condition_match"]      for r in results) / n, 3),
        "disclaimer_rate":           round(sum(r["has_disclaimer"]       for r in results) / n, 3),
        "symptoms_referenced_rate":  round(sum(r["symptoms_referenced"] for r in results) / n, 3),
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str, max_samples: int | None = None) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break
    return samples


def resolve_image_path(item: dict, images_root: str) -> str:
    raw = item["image_path"].replace("images/", "", 1)
    return os.path.join(images_root, raw)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Behavioral evaluation: fine-tuned vs zero-shot")
    parser.add_argument("--max-samples",    type=int, default=None,
                        help="Cap samples (default: all)")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-zeroshot",  action="store_true")
    args = parser.parse_args()

    config = Phase4Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    val_path = os.path.join(_project_root, config.dataset_val)
    samples = load_jsonl(val_path, args.max_samples)
    logger.info(f"Dataset F: {len(samples)} validation samples")

    images_root = os.path.join(_project_root, config.images_dir)

    all_results: dict = {}
    summary: dict     = {}

    # ---- Fine-tuned ----
    ft_model, ft_tok, ft_proj, ft_feat = load_finetuned(config, device)
    ft_results = []

    pbar = tqdm(enumerate(samples), total=len(samples),
                desc="finetuned", unit="sample")
    for i, item in pbar:
        r = evaluate_sample_finetuned(
            ft_model, ft_tok, item["messages"],
            item["image_msg_index"],
            resolve_image_path(item, images_root),
            item["DenseNet_output"],
            ft_feat, ft_proj,
            args.max_new_tokens, device,
        )
        r["sample_idx"] = i
        ft_results.append(r)
        pbar.set_postfix(sample=i + 1)

    all_results["finetuned"] = ft_results
    summary["finetuned"] = aggregate(ft_results)
    logger.info(f"  [finetuned] {summary['finetuned']}")

    del ft_model, ft_tok, ft_proj, ft_feat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Zero-shot ----
    if not args.skip_zeroshot:
        zs_model, zs_tok = load_zeroshot(config, device)
        zs_results = []

        pbar = tqdm(enumerate(samples), total=len(samples),
                    desc="zeroshot", unit="sample")
        for i, item in pbar:
            r = evaluate_sample_zeroshot(
                zs_model, zs_tok, item["messages"],
                item["image_msg_index"],
                item["DenseNet_output"],
                args.max_new_tokens, device,
            )
            r["sample_idx"] = i
            zs_results.append(r)
            pbar.set_postfix(sample=i + 1)

        all_results["zeroshot"] = zs_results
        summary["zeroshot"] = aggregate(zs_results)
        logger.info(f"  [zeroshot] {summary['zeroshot']}")

        del zs_model, zs_tok
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Print summary ----
    logger.info("\n" + "=" * 72)
    logger.info("BEHAVIORAL EVALUATION SUMMARY — Dataset F")
    logger.info("=" * 72)

    def delta(a, b): s = a - b; return f"{'+'if s>=0 else ''}{s:.1%}"

    header = f"  {'Model':<14} {'ThinkBlock':>11} {'Diagnosis':>10} {'CondMatch':>10} {'Disclaimer':>11} {'SymptomsRef':>12}"
    logger.info(f"\n  [zero-shot uses text-serialized DenseNet]")
    logger.info(header)
    logger.info("  " + "-" * 66)

    for tag in ("finetuned", "zeroshot"):
        if tag in summary:
            m = summary[tag]
            logger.info(
                f"  {tag:<14} "
                f"{m['think_block_rate']:>11.1%} "
                f"{m['diagnosis_tag_rate']:>10.1%} "
                f"{m['condition_match_rate']:>10.1%} "
                f"{m['disclaimer_rate']:>11.1%} "
                f"{m['symptoms_referenced_rate']:>12.1%}"
            )

    if "finetuned" in summary and "zeroshot" in summary:
        ft, zs = summary["finetuned"], summary["zeroshot"]
        logger.info("  " + "-" * 66)
        logger.info(
            f"  {'delta (ft-zs)':<14} "
            f"{delta(ft['think_block_rate'], zs['think_block_rate']):>11} "
            f"{delta(ft['diagnosis_tag_rate'], zs['diagnosis_tag_rate']):>10} "
            f"{delta(ft['condition_match_rate'], zs['condition_match_rate']):>10} "
            f"{delta(ft['disclaimer_rate'], zs['disclaimer_rate']):>11} "
            f"{delta(ft['symptoms_referenced_rate'], zs['symptoms_referenced_rate']):>12}"
        )

    logger.info("\n" + "=" * 72)

    # ---- Save ----
    os.makedirs(config.output_dir, exist_ok=True)
    detail_path  = os.path.join(config.output_dir, "eval_comparison.json")
    summary_path = os.path.join(config.output_dir, "eval_summary.json")

    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=2)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Per-sample detail → {detail_path}")
    logger.info(f"Aggregate summary → {summary_path}")


if __name__ == "__main__":
    main()
