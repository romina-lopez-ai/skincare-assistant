"""Evaluation orchestrator — Phase 4 fine-tuned vs zero-shot Llama.

Evaluates *behavioral* compliance rather than lexical similarity.
The model is not trained to reproduce a specific string — it is trained to
exhibit correct clinical behavior (see multitask_training_plan.md §9.2).

Dataset B — Symptom collection (interactive simulation):
  • end_symptoms_rate     : [END_SYMPTOMS] emitted by end of collection
  • premature_diag_rate   : [DIAGNOSIS] appears before collection is done
  • single_question_rate  : one question per assistant turn
  • correct_flow_rate     : end_symptoms AND no premature diagnosis

Dataset C — RAG QA (single-shot):
  • think_block_rate      : <think>...</think> present
  • disclaimer_rate       : response mentions "dermatologist"
  • context_grounded_rate : response re-uses ≥1 content word from [CONTEXT]
  • hallucination_rate    : response introduces medical claims absent from context

Dataset A — Visual alignment (fine-tuned: real visual tokens; zero-shot: text-serialized DenseNet):
  • think_block_rate      : <think>...</think> present
  • diagnosis_tag_rate    : [DIAGNOSIS] present
  • condition_match_rate  : primary condition from DenseNet appears in response
  • disclaimer_rate       : response mentions "dermatologist"

Dataset D — Joint consultation (fine-tuned: real visual tokens; zero-shot: text-serialized DenseNet):
  Same as A, plus:
  • symptoms_referenced_rate : <think> block references ≥1 symptom term from prior dialogue

Usage (from project root):
    python unified_training/evaluate.py
    python unified_training/evaluate.py --max-samples 5 --dataset b
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

def has_end_symptoms(text: str) -> bool:
    return "[END_SYMPTOMS]" in text

def has_diagnosis_tag(text: str) -> bool:
    return "[DIAGNOSIS]" in text

def has_think_block(text: str) -> bool:
    return bool(re.search(r"<think>.*?</think>", text, re.DOTALL))

def extract_think_block(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    return m.group(1) if m else ""

def has_disclaimer(text: str) -> bool:
    return bool(re.search(r"dermatologist", text, re.IGNORECASE))

def is_single_question(text: str) -> bool:
    """Exactly one '?' per turn — the one-question-at-a-time protocol."""
    return text.count("?") == 1

def condition_in_response(response: str, primary: str) -> bool:
    """Check the primary DenseNet condition appears in the response."""
    return primary.lower() in response.lower()

def symptoms_referenced(think_text: str, symptom_words: set[str]) -> bool:
    """At least one symptom word from the user dialogue appears in <think>."""
    if not symptom_words:
        return False
    think_words = set(re.findall(r"[a-zA-Z]{3,}", think_text.lower()))
    return bool(think_words & symptom_words)

def extract_context_words(messages: list) -> set[str]:
    """Content words (≥4 chars) from [CONTEXT] blocks."""
    stop = {"the", "a", "an", "is", "are", "was", "were", "of", "in", "to",
            "and", "or", "but", "it", "this", "that", "for", "with", "on",
            "at", "by", "from", "as", "be", "been", "have", "has", "can",
            "do", "does", "not", "also", "may", "more", "its", "their"}
    words: set[str] = set()
    for m in messages:
        if "[CONTEXT]" in m.get("content", ""):
            for w in re.findall(r"[a-zA-Z]{4,}", m["content"].lower()):
                if w not in stop:
                    words.add(w)
    return words

def is_context_grounded(response: str, context_words: set[str]) -> bool:
    if not context_words:
        return True
    return bool(set(re.findall(r"[a-zA-Z]{4,}", response.lower())) & context_words)

_MEDICAL_CLAIM_RE = re.compile(
    r"\b(prescri|medication|drug|dosage|treatment|cream|antibiotic|"
    r"steroid|cortisone|therapy|surgery|biopsy)\b",
    re.IGNORECASE,
)

def has_unsupported_medical_claims(response: str, context_words: set[str]) -> bool:
    claim_words = set(m.group(0).lower() for m in _MEDICAL_CLAIM_RE.finditer(response))
    if not claim_words:
        return False
    return not claim_words.issubset(context_words)


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
        if msg["content"] == "":   # skip image placeholder
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
    """Generate using inputs_embeds (needed when visual tokens are injected)."""
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
    # With inputs_embeds, generate() returns only newly generated token IDs
    return tokenizer.decode(out[0], skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Visual token injection for A/D (fine-tuned path)
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_visual_embeds(
    model, tokenizer, messages: list, image_msg_index: int,
    image_path: str, feat_extractor, projection,
    device: torch.device, num_visual_tokens: int,
) -> torch.Tensor:
    """Build combined text+visual embeddings for one A/D sample.

    Layout: [pre-image text tokens] + [N visual tokens] + [assistant header tokens]

    The empty user message at image_msg_index marks where visual tokens go.
    Messages after image_msg_index (i.e. the reference assistant turn) are
    excluded — we're building the prompt for generation, not training.
    """
    pre_image_msgs  = messages[:image_msg_index]   # everything before the image slot
    # Build text token IDs: pre-image messages + assistant header primer
    text_ids = _chat_ids(pre_image_msgs, tokenizer, prime_assistant=True)
    insert_pos = len(text_ids) - len(  # right before the assistant header
        tokenizer.encode("<|start_header_id|>assistant<|end_header_id|>\n\n",
                         add_special_tokens=False)
    )
    # Keep insert_pos at the boundary right after pre-image text (before asst header)
    # We split: pre_ids + [visual] + asst_header_ids
    asst_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    pre_ids = text_ids[: len(text_ids) - len(asst_header_ids)]

    pre_tensor   = torch.tensor([pre_ids],        dtype=torch.long, device=device)
    asst_tensor  = torch.tensor([asst_header_ids], dtype=torch.long, device=device)

    embed_fn     = model.get_input_embeddings()
    pre_embeds   = embed_fn(pre_tensor)    # (1, pre_len, D)
    asst_embeds  = embed_fn(asst_tensor)   # (1, asst_len, D)

    # Visual tokens
    image = Image.open(image_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    vis_feat   = feat_extractor(img_tensor)                        # (1, densenet_dim)
    vis_tokens = projection(vis_feat.to(torch.bfloat16))           # (1, N, llm_dim)
    vis_tokens = vis_tokens.to(pre_embeds.dtype)

    combined = torch.cat([pre_embeds, vis_tokens, asst_embeds], dim=1)  # (1, L+N+H, D)
    return combined


# ---------------------------------------------------------------------------
# Zero-shot text serialization for A/D (no visual tokens)
# ---------------------------------------------------------------------------

def serialize_densenet(densenet_output: dict) -> str:
    """Convert DenseNet predictions to a text string for zero-shot baseline."""
    primary  = densenet_output.get("primary", "unknown")
    conf     = densenet_output.get("confidence", 0.0)
    secondary = densenet_output.get("secondary", "")
    conf2    = densenet_output.get("confidence_2", 0.0)
    morphology = ", ".join(densenet_output.get("morphology", []))
    location = densenet_output.get("location", "")

    lines = [
        "[VISUAL_INPUT]",
        f"DenseNet prediction: {primary} ({conf:.0%} confidence)",
    ]
    if secondary:
        lines.append(f"Secondary: {secondary} ({conf2:.0%})")
    if morphology:
        lines.append(f"Morphology: {morphology}")
    if location:
        lines.append(f"Location: {location}")
    lines.append("[/VISUAL_INPUT]")
    return "\n".join(lines)


def build_zeroshot_messages_visual(messages: list, image_msg_index: int,
                                    densenet_output: dict) -> list:
    """Replace the image placeholder with a text-serialized DenseNet message."""
    result = list(messages[:image_msg_index])
    result.append({"role": "user", "content": serialize_densenet(densenet_output)})
    # Skip the reference assistant turn (index image_msg_index+1 if present)
    return result


# ---------------------------------------------------------------------------
# Symptom words for D (what the user described in the dialogue)
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
# Dataset B — interactive simulation
# ---------------------------------------------------------------------------

def evaluate_b_sample(model, tokenizer, messages: list,
                       max_new_tokens: int, device: torch.device) -> dict:
    context = [messages[0]]   # system prompt
    generated: list[str] = []
    premature_diag  = False
    end_symptoms_turn: int | None = None
    single_q_flags: list[bool] = []

    for msg in messages[1:]:
        if msg["role"] == "user":
            context.append(msg)
        else:
            ids      = torch.tensor([_chat_ids(context, tokenizer)], dtype=torch.long)
            response = generate_from_ids(model, tokenizer, ids, max_new_tokens, device)
            generated.append(response)

            if has_end_symptoms(response) and end_symptoms_turn is None:
                end_symptoms_turn = len(generated)
            if has_diagnosis_tag(response) and end_symptoms_turn is None:
                premature_diag = True
            single_q_flags.append(is_single_question(response))

            context.append({"role": "assistant", "content": response})

    n = len(generated)
    return {
        "end_symptoms_emitted":   end_symptoms_turn is not None,
        "end_symptoms_turn":      end_symptoms_turn,
        "premature_diagnosis":    premature_diag,
        "correct_flow":           (end_symptoms_turn is not None) and (not premature_diag),
        "total_turns":            n,
        "single_question_rate":   sum(single_q_flags) / max(n, 1),
        "last_response":          generated[-1] if generated else "",
    }


# ---------------------------------------------------------------------------
# Dataset C — single-shot RAG QA
# ---------------------------------------------------------------------------

def evaluate_c_sample(model, tokenizer, messages: list,
                       max_new_tokens: int, device: torch.device) -> dict:
    last_asst_idx = max(
        (i for i, m in enumerate(messages) if m["role"] == "assistant"),
        default=None,
    )
    if last_asst_idx is None:
        return {}

    input_msgs    = messages[:last_asst_idx]
    context_words = extract_context_words(input_msgs)
    ids           = torch.tensor([_chat_ids(input_msgs, tokenizer)], dtype=torch.long)
    response      = generate_from_ids(model, tokenizer, ids, max_new_tokens, device)

    return {
        "has_think_block":     has_think_block(response),
        "has_disclaimer":      has_disclaimer(response),
        "context_grounded":    is_context_grounded(response, context_words),
        "hallucination_flag":  has_unsupported_medical_claims(response, context_words),
        "response":            response,
        "reference":           messages[last_asst_idx]["content"],
    }


# ---------------------------------------------------------------------------
# Dataset A — visual alignment (single-shot)
# ---------------------------------------------------------------------------

def evaluate_a_sample_finetuned(
    model, tokenizer, messages: list, image_msg_index: int,
    image_path: str, densenet_output: dict,
    feat_extractor, projection,
    max_new_tokens: int, device: torch.device,
) -> dict:
    embeds   = build_visual_embeds(
        model, tokenizer, messages, image_msg_index,
        image_path, feat_extractor, projection, device,
        projection.num_tokens,
    )
    response = generate_from_embeds(model, tokenizer, embeds, max_new_tokens)
    primary  = densenet_output.get("primary", "")

    return {
        "has_think_block":     has_think_block(response),
        "has_diagnosis_tag":   has_diagnosis_tag(response),
        "condition_match":     condition_in_response(response, primary),
        "has_disclaimer":      has_disclaimer(response),
        "response":            response,
        "primary_condition":   primary,
    }


def evaluate_a_sample_zeroshot(
    model, tokenizer, messages: list, image_msg_index: int,
    densenet_output: dict, max_new_tokens: int, device: torch.device,
) -> dict:
    input_msgs = build_zeroshot_messages_visual(messages, image_msg_index, densenet_output)
    ids        = torch.tensor([_chat_ids(input_msgs, tokenizer)], dtype=torch.long)
    response   = generate_from_ids(model, tokenizer, ids, max_new_tokens, device)
    primary    = densenet_output.get("primary", "")

    return {
        "has_think_block":     has_think_block(response),
        "has_diagnosis_tag":   has_diagnosis_tag(response),
        "condition_match":     condition_in_response(response, primary),
        "has_disclaimer":      has_disclaimer(response),
        "response":            response,
        "primary_condition":   primary,
    }


# ---------------------------------------------------------------------------
# Dataset D — joint consultation (A metrics + symptoms_referenced)
# ---------------------------------------------------------------------------

def evaluate_d_sample_finetuned(
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


def evaluate_d_sample_zeroshot(
    model, tokenizer, messages: list, image_msg_index: int,
    densenet_output: dict, max_new_tokens: int, device: torch.device,
) -> dict:
    input_msgs    = build_zeroshot_messages_visual(messages, image_msg_index, densenet_output)
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
# Aggregators
# ---------------------------------------------------------------------------

def agg_b(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "n": n,
        "end_symptoms_rate":    round(sum(r["end_symptoms_emitted"] for r in results) / n, 3),
        "premature_diag_rate":  round(sum(r["premature_diagnosis"]  for r in results) / n, 3),
        "correct_flow_rate":    round(sum(r["correct_flow"]         for r in results) / n, 3),
        "single_question_rate": round(sum(r["single_question_rate"] for r in results) / n, 3),
    }

def agg_c(results: list[dict]) -> dict:
    n = len(results)
    if n == 0:
        return {}
    return {
        "n": n,
        "think_block_rate":      round(sum(r["has_think_block"]    for r in results) / n, 3),
        "disclaimer_rate":       round(sum(r["has_disclaimer"]     for r in results) / n, 3),
        "context_grounded_rate": round(sum(r["context_grounded"]   for r in results) / n, 3),
        "hallucination_rate":    round(sum(r["hallucination_flag"] for r in results) / n, 3),
    }

def agg_ad(results: list[dict], include_symptoms: bool = False) -> dict:
    n = len(results)
    if n == 0:
        return {}
    out = {
        "n": n,
        "think_block_rate":    round(sum(r["has_think_block"]   for r in results) / n, 3),
        "diagnosis_tag_rate":  round(sum(r["has_diagnosis_tag"] for r in results) / n, 3),
        "condition_match_rate":round(sum(r["condition_match"]   for r in results) / n, 3),
        "disclaimer_rate":     round(sum(r["has_disclaimer"]    for r in results) / n, 3),
    }
    if include_symptoms:
        out["symptoms_referenced_rate"] = round(
            sum(r["symptoms_referenced"] for r in results) / n, 3
        )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Behavioral evaluation: fine-tuned vs zero-shot")
    parser.add_argument("--max-samples",    type=int, default=None,
                        help="Cap samples per dataset (default: all)")
    parser.add_argument("--dataset",        choices=["a", "b", "c", "d", "all"], default="all")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--skip-zeroshot",  action="store_true")
    args = parser.parse_args()

    config = Phase4Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Resolve which datasets to run
    run_ds = {"a", "b", "c", "d"} if args.dataset == "all" else {args.dataset}

    ds_paths = {
        "A": os.path.join(_project_root, "datasets/dataset_a/validation.jsonl"),
        "B": os.path.join(_project_root, "datasets/dataset_b/test.jsonl"),
        "C": os.path.join(_project_root, "datasets/dataset_c/test.jsonl"),
        "D": os.path.join(_project_root, "datasets/dataset_d/validation.jsonl"),
    }
    active = {k: v for k, v in ds_paths.items() if k.lower() in run_ds}

    all_samples = {tag: load_jsonl(path, args.max_samples) for tag, path in active.items()}
    for tag, samples in all_samples.items():
        logger.info(f"Dataset {tag}: {len(samples)} samples")

    all_results: dict = {}
    summary: dict     = {}

    images_root = os.path.join(_project_root, config.images_dir)

    # ---- Fine-tuned ----
    ft_model, ft_tok, ft_proj, ft_feat = load_finetuned(config, device)
    all_results["finetuned"] = {}

    for ds_tag, samples in all_samples.items():
        ds_results = []
        pbar = tqdm(enumerate(samples), total=len(samples),
                    desc=f"finetuned/{ds_tag}", unit="sample")
        for i, item in pbar:
            msgs = item["messages"]
            if ds_tag == "B":
                r = evaluate_b_sample(ft_model, ft_tok, msgs,
                                      args.max_new_tokens, device)
            elif ds_tag == "C":
                r = evaluate_c_sample(ft_model, ft_tok, msgs,
                                      args.max_new_tokens, device)
            elif ds_tag == "A":
                r = evaluate_a_sample_finetuned(
                    ft_model, ft_tok, msgs,
                    item["image_msg_index"],
                    resolve_image_path(item, images_root),
                    item["DenseNet_output"],
                    ft_feat, ft_proj,
                    args.max_new_tokens, device,
                )
            else:  # D
                r = evaluate_d_sample_finetuned(
                    ft_model, ft_tok, msgs,
                    item["image_msg_index"],
                    resolve_image_path(item, images_root),
                    item["DenseNet_output"],
                    ft_feat, ft_proj,
                    args.max_new_tokens, device,
                )
            r["sample_idx"] = i
            ds_results.append(r)
            pbar.set_postfix(sample=i + 1)

        all_results["finetuned"][ds_tag] = ds_results
        key = f"finetuned_{ds_tag}"
        if ds_tag == "B":
            summary[key] = agg_b(ds_results)
        elif ds_tag == "C":
            summary[key] = agg_c(ds_results)
        elif ds_tag == "A":
            summary[key] = agg_ad(ds_results, include_symptoms=False)
        else:
            summary[key] = agg_ad(ds_results, include_symptoms=True)
        logger.info(f"  [{key}] {summary[key]}")

    del ft_model, ft_tok, ft_proj, ft_feat
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---- Zero-shot ----
    if not args.skip_zeroshot:
        zs_model, zs_tok = load_zeroshot(config, device)
        all_results["zeroshot"] = {}

        for ds_tag, samples in all_samples.items():
            ds_results = []
            pbar = tqdm(enumerate(samples), total=len(samples),
                        desc=f"zeroshot/{ds_tag}", unit="sample")
            for i, item in pbar:
                msgs = item["messages"]
                if ds_tag == "B":
                    r = evaluate_b_sample(zs_model, zs_tok, msgs,
                                          args.max_new_tokens, device)
                elif ds_tag == "C":
                    r = evaluate_c_sample(zs_model, zs_tok, msgs,
                                          args.max_new_tokens, device)
                elif ds_tag == "A":
                    r = evaluate_a_sample_zeroshot(
                        zs_model, zs_tok, msgs,
                        item["image_msg_index"],
                        item["DenseNet_output"],
                        args.max_new_tokens, device,
                    )
                else:  # D
                    r = evaluate_d_sample_zeroshot(
                        zs_model, zs_tok, msgs,
                        item["image_msg_index"],
                        item["DenseNet_output"],
                        args.max_new_tokens, device,
                    )
                r["sample_idx"] = i
                ds_results.append(r)
                pbar.set_postfix(sample=i + 1)

            all_results["zeroshot"][ds_tag] = ds_results
            key = f"zeroshot_{ds_tag}"
            if ds_tag == "B":
                summary[key] = agg_b(ds_results)
            elif ds_tag == "C":
                summary[key] = agg_c(ds_results)
            elif ds_tag == "A":
                summary[key] = agg_ad(ds_results, include_symptoms=False)
            else:
                summary[key] = agg_ad(ds_results, include_symptoms=True)
            logger.info(f"  [{key}] {summary[key]}")

        del zs_model, zs_tok
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ---- Print summary ----
    logger.info("\n" + "=" * 72)
    logger.info("BEHAVIORAL EVALUATION SUMMARY")
    logger.info("=" * 72)

    def delta(a, b): s = a - b; return f"{'+'if s>=0 else ''}{s:.1%}"

    if "B" in all_samples:
        logger.info("\nDataset B — Symptom Collection")
        logger.info(f"  {'Model':<14} {'EndSymptoms':>12} {'PrematureDx':>12} {'CorrectFlow':>12} {'SingleQ/turn':>13}")
        logger.info("  " + "-" * 56)
        for tag in ("finetuned", "zeroshot"):
            k = f"{tag}_B"
            if k in summary:
                m = summary[k]
                logger.info(f"  {tag:<14} {m['end_symptoms_rate']:>12.1%} {m['premature_diag_rate']:>12.1%} {m['correct_flow_rate']:>12.1%} {m['single_question_rate']:>13.1%}")
        if "finetuned_B" in summary and "zeroshot_B" in summary:
            ft, zs = summary["finetuned_B"], summary["zeroshot_B"]
            logger.info("  " + "-" * 56)
            logger.info(f"  {'delta (ft-zs)':<14} {delta(ft['end_symptoms_rate'],zs['end_symptoms_rate']):>12} {delta(ft['premature_diag_rate'],zs['premature_diag_rate']):>12} {delta(ft['correct_flow_rate'],zs['correct_flow_rate']):>12} {delta(ft['single_question_rate'],zs['single_question_rate']):>13}")

    if "C" in all_samples:
        logger.info("\nDataset C — RAG QA")
        logger.info(f"  {'Model':<14} {'ThinkBlock':>11} {'Disclaimer':>11} {'Grounded':>10} {'Hallucin.':>10}")
        logger.info("  " + "-" * 50)
        for tag in ("finetuned", "zeroshot"):
            k = f"{tag}_C"
            if k in summary:
                m = summary[k]
                logger.info(f"  {tag:<14} {m['think_block_rate']:>11.1%} {m['disclaimer_rate']:>11.1%} {m['context_grounded_rate']:>10.1%} {m['hallucination_rate']:>10.1%}")
        if "finetuned_C" in summary and "zeroshot_C" in summary:
            ft, zs = summary["finetuned_C"], summary["zeroshot_C"]
            logger.info("  " + "-" * 50)
            logger.info(f"  {'delta (ft-zs)':<14} {delta(ft['think_block_rate'],zs['think_block_rate']):>11} {delta(ft['disclaimer_rate'],zs['disclaimer_rate']):>11} {delta(ft['context_grounded_rate'],zs['context_grounded_rate']):>10} {delta(ft['hallucination_rate'],zs['hallucination_rate']):>10}")

    if "A" in all_samples:
        logger.info("\nDataset A — Visual Alignment  [zero-shot uses text-serialized DenseNet]")
        logger.info(f"  {'Model':<14} {'ThinkBlock':>11} {'Diagnosis':>10} {'CondMatch':>10} {'Disclaimer':>11}")
        logger.info("  " + "-" * 52)
        for tag in ("finetuned", "zeroshot"):
            k = f"{tag}_A"
            if k in summary:
                m = summary[k]
                logger.info(f"  {tag:<14} {m['think_block_rate']:>11.1%} {m['diagnosis_tag_rate']:>10.1%} {m['condition_match_rate']:>10.1%} {m['disclaimer_rate']:>11.1%}")
        if "finetuned_A" in summary and "zeroshot_A" in summary:
            ft, zs = summary["finetuned_A"], summary["zeroshot_A"]
            logger.info("  " + "-" * 52)
            logger.info(f"  {'delta (ft-zs)':<14} {delta(ft['think_block_rate'],zs['think_block_rate']):>11} {delta(ft['diagnosis_tag_rate'],zs['diagnosis_tag_rate']):>10} {delta(ft['condition_match_rate'],zs['condition_match_rate']):>10} {delta(ft['disclaimer_rate'],zs['disclaimer_rate']):>11}")

    if "D" in all_samples:
        logger.info("\nDataset D — Joint Consultation  [zero-shot uses text-serialized DenseNet]")
        logger.info(f"  {'Model':<14} {'ThinkBlock':>11} {'Diagnosis':>10} {'CondMatch':>10} {'Disclaimer':>11} {'SymptomsRef':>12}")
        logger.info("  " + "-" * 62)
        for tag in ("finetuned", "zeroshot"):
            k = f"{tag}_D"
            if k in summary:
                m = summary[k]
                logger.info(f"  {tag:<14} {m['think_block_rate']:>11.1%} {m['diagnosis_tag_rate']:>10.1%} {m['condition_match_rate']:>10.1%} {m['disclaimer_rate']:>11.1%} {m['symptoms_referenced_rate']:>12.1%}")
        if "finetuned_D" in summary and "zeroshot_D" in summary:
            ft, zs = summary["finetuned_D"], summary["zeroshot_D"]
            logger.info("  " + "-" * 62)
            logger.info(f"  {'delta (ft-zs)':<14} {delta(ft['think_block_rate'],zs['think_block_rate']):>11} {delta(ft['diagnosis_tag_rate'],zs['diagnosis_tag_rate']):>10} {delta(ft['condition_match_rate'],zs['condition_match_rate']):>10} {delta(ft['disclaimer_rate'],zs['disclaimer_rate']):>11} {delta(ft['symptoms_referenced_rate'],zs['symptoms_referenced_rate']):>12}")

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
