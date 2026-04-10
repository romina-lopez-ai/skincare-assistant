"""
Merge Phase 4 LoRA adapters into base Llama 3.2-3B and save a standalone model.

Output: unified_training/outputs/merged_model/
  - Full model weights (safetensors)
  - Tokenizer
  - config.json

Note: projection.pth (visual alignment MLP) is NOT merged here — it is a separate
      component loaded independently at inference time alongside the merged LLM.

Usage:
    python unified_training/merge_model.py
    python unified_training/merge_model.py --output-dir unified_training/outputs/merged_model
"""

import argparse
import os
import shutil

import torch
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


def merge(lora_adapter_dir: str, output_dir: str, hf_token: str) -> None:
    base_model_name = "meta-llama/Llama-3.2-3B"

    print(f"Loading base model: {base_model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",   # keep on CPU — merge does not need GPU
        token=hf_token,
    )

    print(f"Loading LoRA adapters from: {lora_adapter_dir}")
    model = PeftModel.from_pretrained(model, lora_adapter_dir)

    print("Merging LoRA weights into base model …")
    model = model.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)

    print(f"Saving merged model to: {output_dir}")
    model.save_pretrained(output_dir, safe_serialization=True)

    print("Saving tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=hf_token)
    tokenizer.save_pretrained(output_dir)

    # Copy projection checkpoint alongside so inference can find everything in one place
    projection_src = os.path.join(
        os.path.dirname(lora_adapter_dir), "projection.pth"
    )
    if os.path.exists(projection_src):
        projection_dst = os.path.join(output_dir, "projection.pth")
        shutil.copy2(projection_src, projection_dst)
        print(f"Copied projection.pth → {projection_dst}")
    else:
        print("projection.pth not found next to lora_adapters — skipping copy.")

    print("\nDone. Merged model saved to:", output_dir)
    print("Files:")
    for f in sorted(os.listdir(output_dir)):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"  {f}  ({size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into base LLM")
    parser.add_argument(
        "--lora-dir",
        default="unified_training/outputs/lora_adapters",
        help="Path to the PEFT adapter directory",
    )
    parser.add_argument(
        "--output-dir",
        default="unified_training/outputs/merged_model",
        help="Where to save the merged model",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token (defaults to HF_TOKEN env var)",
    )
    args = parser.parse_args()

    merge(
        lora_adapter_dir=args.lora_dir,
        output_dir=args.output_dir,
        hf_token=args.hf_token,
    )


if __name__ == "__main__":
    main()
