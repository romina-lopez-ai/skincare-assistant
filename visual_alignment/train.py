"""Phase 2 — Visual Alignment Training (Projection MLP + LoRA).

Image -> frozen DenseNet -> 1024-dim -> ProjectionMLP -> visual tokens ->
LLaMA-3.2-3B (LoRA) -> generates truncated clinical note.
"""

import json
import os
import sys
import logging
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from torchvision import models

from config import Phase2Config
from projection import ProjectionMLP
from dataset import VisualAlignmentDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DenseNet feature extractor (frozen)
# ---------------------------------------------------------------------------

class DenseNetFeatureExtractor(torch.nn.Module):
    """Loads a fine-tuned DenseNet-121 and extracts 1024-dim features."""

    def __init__(self, checkpoint_path: str):
        super().__init__()
        # Reconstruct the same architecture used during DenseNet training
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "densenet_model", "src"))
        from model import build_model

        full_model = build_model(pretrained=False, fine_tune=False, num_classes=3)
        full_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

        # Keep only the feature backbone (discard the classifier)
        self.features = full_model.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 3, 224, 224) -> (B, 1024)"""
        feat = self.features(x)
        feat = F.relu(feat)  # DenseNet applies ReLU after features
        feat = self.pool(feat)
        return feat.view(feat.size(0), -1)


# ---------------------------------------------------------------------------
# Generation helper (qualitative check during training)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_samples(slm, projection, feat_extractor, dataset, tokenizer,
                     config, device, n=3):
    """Generate text from visual tokens for n random validation samples."""
    slm.eval()
    projection.eval()

    indices = random.sample(range(len(dataset)), min(n, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        image = sample["image"].unsqueeze(0).to(device)

        visual_features = feat_extractor(image)
        visual_tokens = projection(visual_features.to(torch.bfloat16))  # (1, N, llm_dim)

        # Autoregressive generation from visual tokens
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

        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        logger.info(f"\n--- Val sample {idx} ---")
        logger.info(f"Ground truth: {sample['text'][:300]}...")
        logger.info(f"Generated:    {generated_text[:300]}")

    slm.train()
    projection.train()


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(projection, slm, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(projection.state_dict(), os.path.join(output_dir, "projection.pth"))
    slm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
    logger.info(f"Checkpoint saved to {output_dir}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    config = Phase2Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        config.slm_model_name, token=config.hf_token or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- SLM with 4-bit quantization ----
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    slm = AutoModelForCausalLM.from_pretrained(
        config.slm_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=config.hf_token or None,
    )

    # ---- LoRA ----
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        task_type="CAUSAL_LM",
    )
    slm = get_peft_model(slm, lora_config)
    slm.print_trainable_parameters()

    # ---- Frozen DenseNet ----
    feat_extractor = DenseNetFeatureExtractor(config.densenet_checkpoint)
    feat_extractor.eval().to(device)

    # ---- Projection MLP ----
    llm_dim = slm.config.hidden_size  # 3072 for LLaMA-3.2-3B
    projection = ProjectionMLP(
        densenet_dim=config.densenet_dim,
        llm_dim=llm_dim,
        num_tokens=config.num_visual_tokens,
        hidden_dim=config.projection_hidden_dim,
    ).to(device).to(torch.bfloat16)
    logger.info(f"Projection params: {sum(p.numel() for p in projection.parameters()):,}")

    # ---- Datasets ----
    train_ds = VisualAlignmentDataset(
        config.train_jsonl, config.images_dir, tokenizer, config.max_length,
    )
    val_ds = VisualAlignmentDataset(
        config.val_jsonl, config.images_dir, tokenizer, config.max_length,
    )
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True,
    )

    # ---- Optimizer (two param groups) ----
    lora_params = [p for p in slm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": projection.parameters(), "lr": config.lr_projection},
        {"params": lora_params, "lr": config.lr_lora},
    ], weight_decay=config.weight_decay)

    num_steps = (len(train_loader) * config.epochs) // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_steps),
        num_training_steps=num_steps,
    )

    # ---- Training ----
    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []

    for epoch in range(config.epochs):
        slm.train()
        projection.train()
        total_loss = 0.0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # 1) DenseNet features (frozen)
            with torch.no_grad():
                visual_features = feat_extractor(images)  # (B, 1024)

            # 2) Project to visual tokens
            visual_tokens = projection(visual_features.to(torch.bfloat16))  # (B, N, llm_dim)

            # 3) Get text embeddings from SLM embedding layer
            text_embeds = slm.get_input_embeddings()(input_ids)  # (B, seq, llm_dim)

            # 4) Concatenate [visual_tokens | text_embeds]
            visual_tokens = visual_tokens.to(text_embeds.dtype)
            combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

            # 5) Build attention mask for visual positions
            B, N = images.size(0), config.num_visual_tokens
            visual_attn = torch.ones(B, N, device=device, dtype=attention_mask.dtype)
            combined_attn = torch.cat([visual_attn, attention_mask], dim=1)

            # 6) Build labels: -100 for visual tokens, input_ids for text
            visual_labels = torch.full((B, N), -100, device=device, dtype=input_ids.dtype)
            text_labels = input_ids.clone()
            text_labels[attention_mask == 0] = -100
            combined_labels = torch.cat([visual_labels, text_labels], dim=1)

            # 7) Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.bf16):
                outputs = slm(
                    inputs_embeds=combined_embeds,
                    attention_mask=combined_attn,
                    labels=combined_labels,
                )
                loss = outputs.loss / config.grad_accum_steps

            loss.backward()

            if (step + 1) % config.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(projection.parameters()) + lora_params,
                    max_norm=config.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * config.grad_accum_steps

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        val_loss = validate(slm, projection, feat_extractor, val_loader, config, device)

        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "lr_proj": optimizer.param_groups[0]["lr"],
            "lr_lora": optimizer.param_groups[1]["lr"],
        }
        training_log.append(epoch_info)
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Generate samples for inspection
        generate_samples(slm, projection, feat_extractor, val_ds, tokenizer,
                         config, device, n=config.num_gen_samples)

        # Checkpoint on best val loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(projection, slm, config.output_dir)
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    # Save training log
    os.makedirs(config.output_dir, exist_ok=True)
    with open(os.path.join(config.output_dir, "training_log.json"), "w") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def validate(slm, projection, feat_extractor, val_loader, config, device):
    slm.eval()
    projection.eval()
    total_loss = 0.0

    for batch in val_loader:
        images = batch["image"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        visual_features = feat_extractor(images)
        visual_tokens = projection(visual_features.to(torch.bfloat16))
        text_embeds = slm.get_input_embeddings()(input_ids)
        visual_tokens = visual_tokens.to(text_embeds.dtype)
        combined_embeds = torch.cat([visual_tokens, text_embeds], dim=1)

        B, N = images.size(0), config.num_visual_tokens
        visual_attn = torch.ones(B, N, device=device, dtype=attention_mask.dtype)
        combined_attn = torch.cat([visual_attn, attention_mask], dim=1)

        visual_labels = torch.full((B, N), -100, device=device, dtype=input_ids.dtype)
        text_labels = input_ids.clone()
        text_labels[attention_mask == 0] = -100
        combined_labels = torch.cat([visual_labels, text_labels], dim=1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config.bf16):
            outputs = slm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attn,
                labels=combined_labels,
            )
        total_loss += outputs.loss.item()

    return total_loss / len(val_loader)


if __name__ == "__main__":
    main()
