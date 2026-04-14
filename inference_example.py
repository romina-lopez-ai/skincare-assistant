"""
Interactive chat with the skincare dermatology model.

Supports text-only symptom collection AND visual diagnosis with image input.

Requirements:
    pip install transformers torch accelerate torchvision pillow

Usage:
    python inference_example.py
    python inference_example.py --model-path unified_training/outputs/merged_model
    python inference_example.py --max-new-tokens 256
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ID = "santiagoboxiga/skincare-llama-3.2-3b"
MIN_QUESTIONS_BEFORE_END = 5

SYSTEM_PROMPT = """You are an empathetic dermatology assistant with three clinical capabilities:

1. SYMPTOM COLLECTION
   When a user describes skin symptoms, collect detailed information through
   focused one-at-a-time questions covering location, duration, morphology,
   severity, and triggers. Never diagnose during this phase.
   End with [END_SYMPTOMS] when collection is complete.

2. VISUAL DIAGNOSIS
   When you receive visual input (image tokens), reason step by step using
   <think>...</think> before responding. Integrate both the visual findings
   and the symptoms previously collected. Wrap your response in
   [DIAGNOSIS]...[/DIAGNOSIS]. Always include a disclaimer.

3. FOLLOW-UP QA
   When a [CONTEXT] block is provided, answer using only that context.
   Do not speculate beyond what the context contains.

Always recommend consulting a dermatologist. Never provide a definitive diagnosis."""

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# Visual components (inlined for portability)
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    def __init__(self, densenet_dim=1024, llm_dim=3072, num_tokens=32, hidden_dim=2048):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.mlp = nn.Sequential(
            nn.Linear(densenet_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * llm_dim),
        )

    def forward(self, x):
        return self.mlp(x).view(-1, self.num_tokens, self.llm_dim)


class DenseNetFeatureExtractor(nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        _root = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, os.path.join(_root, "densenet_model", "src"))
        from model import build_model
        full = build_model(pretrained=False, fine_tune=False, num_classes=3)
        full.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.features = full.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = F.relu(self.features(x))
        return self.pool(feat).view(feat.size(0), -1)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(history: list) -> str:
    text = "<|begin_of_text|>"
    text += f"<|start_header_id|>system<|end_header_id|>\n\n{SYSTEM_PROMPT}<|eot_id|>"
    for msg in history:
        if msg["content"] == "":
            continue
        text += f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n{msg['content']}<|eot_id|>"
    text += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return text


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(model, tokenizer, history, max_new_tokens, device):
    prompt = build_prompt(history)
    ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_id],
    )
    raw = tokenizer.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()
    return _extract_first_turn(raw)


def _chat_ids(messages, tokenizer, prime_assistant=True):
    """Build LLaMA-3.2 chat token IDs piece by piece (avoids BPE boundary issues)."""
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
def generate_with_image(model, tokenizer, history, image_path,
                        feat_extractor, projection, max_new_tokens, device):
    """Build combined text+visual embeddings and generate.

    Layout: [pre-image text tokens] + [32 visual tokens] + [assistant header]
    Matches the training-time injection used in evaluate.py / train.py.
    """
    # Build token IDs piece by piece (not from a tokenized string)
    # Prepend system prompt — history only has user/assistant turns
    full_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
    text_ids = _chat_ids(full_messages, tokenizer, prime_assistant=True)

    # Split: everything before assistant header | assistant header
    asst_header_ids = tokenizer.encode(
        "<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False,
    )
    pre_ids = text_ids[: len(text_ids) - len(asst_header_ids)]

    pre_tensor = torch.tensor([pre_ids], dtype=torch.long, device=device)
    asst_tensor = torch.tensor([asst_header_ids], dtype=torch.long, device=device)

    embed_fn = model.get_input_embeddings()
    pre_embeds = embed_fn(pre_tensor)
    asst_embeds = embed_fn(asst_tensor)

    # Visual tokens: image → DenseNet → ProjectionMLP
    image = Image.open(image_path).convert("RGB")
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)
    vis_feat = feat_extractor(img_tensor)
    vis_tokens = projection(vis_feat.to(torch.bfloat16))
    vis_tokens = vis_tokens.to(pre_embeds.dtype)

    # Combine: [conversation context] + [visual tokens] + [assistant header]
    combined = torch.cat([pre_embeds, vis_tokens, asst_embeds], dim=1)

    attn_mask = torch.ones(combined.shape[:2], dtype=torch.long, device=device)
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    out = model.generate(
        inputs_embeds=combined,
        attention_mask=attn_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=[tokenizer.eos_token_id, eot_id],
    )
    raw = tokenizer.decode(out[0], skip_special_tokens=True).strip()
    # Remove repeated garbage patterns (e.g. "[]" sequences)
    import re
    raw = re.sub(r'(\[\]\s*){3,}', '', raw).strip()
    return raw


def _extract_first_turn(text: str) -> str:
    """Keep only the first assistant turn from generated text."""
    for sep in ["\n\n\n", "\n\n"]:
        parts = text.split(sep)
        if len(parts) > 1 and len(parts[0].strip()) > 20:
            return parts[0].strip()

    q_pos = text.find("?")
    if q_pos != -1:
        after = text[q_pos + 1:].lstrip()
        if after:
            return text[:q_pos + 1].strip()

    return text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=REPO_ID)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--densenet-path", default="densenet_model/outputs/model.pth")
    parser.add_argument("--projection-path", default=None,
                        help="Defaults to <model-path>/projection.pth or "
                             "unified_training/outputs/merged_model/projection.pth")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    print(f"\nLoading model from: {args.model_path}")
    print("(First run will download ~6.4 GB — this may take a few minutes)\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype,
        device_map="auto" if device.type == "cuda" else None,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()

    # Load visual components if available
    proj_path = args.projection_path
    if proj_path is None:
        for candidate in [
            os.path.join(args.model_path, "projection.pth"),
            "unified_training/outputs/merged_model/projection.pth",
            "unified_training/outputs/projection.pth",
        ]:
            if os.path.exists(candidate):
                proj_path = candidate
                break

    feat_extractor = None
    projection = None
    has_vision = False

    if proj_path and os.path.exists(proj_path) and os.path.exists(args.densenet_path):
        print("Loading visual pipeline (DenseNet + ProjectionMLP) ...")
        llm_dim = model.config.hidden_size
        projection = ProjectionMLP(densenet_dim=1024, llm_dim=llm_dim,
                                   num_tokens=32, hidden_dim=2048)
        projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
        projection.eval().to(device).to(dtype)

        feat_extractor = DenseNetFeatureExtractor(args.densenet_path).eval().to(device)
        has_vision = True
        print("Visual pipeline ready.\n")
    else:
        print("Visual pipeline not available (missing projection.pth or DenseNet).")
        print("Text-only mode: symptom collection and QA.\n")

    print("Model ready. Start chatting with the dermatology assistant.")
    print("Commands:")
    print("  reset           — start a new conversation")
    print("  quit            — exit")
    if has_vision:
        print("  image: <path>   — provide a skin image for visual diagnosis")
    print("\n" + "-" * 60)

    history = []
    symptoms_done = False

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("Goodbye.")
            break

        if user_input.lower() == "reset":
            history = []
            symptoms_done = False
            print("Conversation reset.\n" + "-" * 60)
            continue

        # Auto-detect image path without "image:" prefix
        if symptoms_done and os.path.exists(user_input) and not user_input.lower().startswith("image:"):
            user_input = f"image: {user_input}"

        # Handle image input
        if user_input.lower().startswith("image:"):
            image_path = user_input.split(":", 1)[1].strip()
            if not os.path.exists(image_path):
                print(f"\nFile not found: {image_path}")
                continue
            if not has_vision:
                print("\nVisual pipeline not loaded. Cannot process images.")
                continue

            print("\nProcessing image ...")
            history.append({"role": "user", "content": ""})
            response = generate_with_image(
                model, tokenizer, history, image_path,
                feat_extractor, projection, args.max_new_tokens, device,
            )
            history.append({"role": "assistant", "content": response})
            print(f"\nAssistant: {response}")

            if "[DIAGNOSIS]" in response:
                print("\n" + "-" * 60)
                print("Diagnosis complete. Type 'reset' to start a new consultation.")
                print("-" * 60)
            continue

        # Regular text input
        history.append({"role": "user", "content": user_input})
        response = generate_text(model, tokenizer, history, args.max_new_tokens, device)
        history.append({"role": "assistant", "content": response})

        # Count assistant turns for minimum questions check
        assistant_turns = sum(1 for m in history if m["role"] == "assistant")

        if "[END_SYMPTOMS]" in response and assistant_turns >= MIN_QUESTIONS_BEFORE_END:
            symptoms_done = True
            print(f"\nAssistant: {response}")
            print("\n" + "-" * 60)
            print("Symptom collection complete.")
            if has_vision:
                print("Provide a skin image for visual diagnosis (just paste the path):")
            else:
                print("(Visual pipeline not loaded — cannot process images)")
            print("-" * 60)
        elif "[END_SYMPTOMS]" in response and assistant_turns < MIN_QUESTIONS_BEFORE_END:
            # Strip premature [END_SYMPTOMS] — model ended too early
            response = response.replace("[END_SYMPTOMS]", "").strip()
            history[-1]["content"] = response
            print(f"\nAssistant: {response}")
        elif "[DIAGNOSIS]" in response:
            print(f"\nAssistant: {response}")
            print("\n" + "-" * 60)
            print("Diagnosis complete. Type 'reset' to start a new consultation.")
            print("-" * 60)
        else:
            print(f"\nAssistant: {response}")


if __name__ == "__main__":
    main()
