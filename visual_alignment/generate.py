"""Inference script — generate clinical text from a single image."""

import argparse
import os
import sys

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from config import Phase2Config
from projection import ProjectionMLP
from dataset import IMAGE_TRANSFORM


def load_pipeline(config, device):
    """Load all components: DenseNet extractor, projection, SLM + LoRA."""

    # --- DenseNet feature extractor ---
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "densenet_model", "src"))
    from model import build_model
    import torch.nn.functional as F

    class _Extractor(torch.nn.Module):
        def __init__(self, checkpoint_path):
            super().__init__()
            full = build_model(pretrained=False, fine_tune=False, num_classes=3)
            full.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
            self.features = full.features
            self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
            for p in self.parameters():
                p.requires_grad = False

        def forward(self, x):
            feat = F.relu(self.features(x))
            return self.pool(feat).view(feat.size(0), -1)

    feat_extractor = _Extractor(config.densenet_checkpoint).eval().to(device)

    # --- SLM with 4-bit + LoRA adapters ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_slm = AutoModelForCausalLM.from_pretrained(
        config.slm_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=config.hf_token or None,
    )
    lora_path = os.path.join(config.output_dir, "lora_adapters")
    slm = PeftModel.from_pretrained(base_slm, lora_path)
    slm.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        config.slm_model_name, token=config.hf_token or None,
    )

    # --- Projection MLP ---
    llm_dim = slm.config.hidden_size
    projection = ProjectionMLP(
        densenet_dim=config.densenet_dim,
        llm_dim=llm_dim,
        num_tokens=config.num_visual_tokens,
        hidden_dim=config.projection_hidden_dim,
    )
    proj_path = os.path.join(config.output_dir, "projection.pth")
    projection.load_state_dict(torch.load(proj_path, map_location="cpu"))
    projection.eval().to(device).to(torch.bfloat16)

    return feat_extractor, projection, slm, tokenizer


@torch.no_grad()
def generate(image_path: str, feat_extractor, projection, slm, tokenizer,
             config, device):
    """Run full pipeline on a single image and return generated text."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(device)

    visual_features = feat_extractor(image_tensor)
    visual_tokens = projection(visual_features.to(torch.bfloat16))  # (1, N, llm_dim)

    generated_ids = []
    current_embeds = visual_tokens
    past_key_values = None

    for _ in range(config.max_gen_length):
        outputs = slm(inputs_embeds=current_embeds, past_key_values=past_key_values, use_cache=True)
        next_logits = outputs.logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1)

        if next_id.item() == tokenizer.eos_token_id:
            break

        generated_ids.append(next_id.item())
        past_key_values = outputs.past_key_values
        current_embeds = slm.get_input_embeddings()(next_id).unsqueeze(1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 2 inference")
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--model-name", default=None, help="Override SLM model name")
    args = parser.parse_args()

    config = Phase2Config()
    if args.model_name:
        config.slm_model_name = args.model_name

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_extractor, projection, slm, tokenizer = load_pipeline(config, device)

    text = generate(args.image_path, feat_extractor, projection, slm, tokenizer,
                    config, device)

    print("\n" + "=" * 60)
    print("Generated Clinical Note")
    print("=" * 60)
    print(text)


if __name__ == "__main__":
    main()
