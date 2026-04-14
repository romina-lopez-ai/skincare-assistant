"""Phase 4 — Unified Multitask Training (A + B + C + D).

Initializes from Phase 2 checkpoints (projection + LoRA) and trains on all
four dataset types simultaneously with weighted sampling.

Image datasets (A, D): visual tokens spliced at image_msg_index position.
Text datasets (B, C): standard causal LM forward pass.
"""

import json
import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_cosine_schedule_with_warmup,
)
from peft import PeftModel

# Add this directory and parent to sys.path for direct execution
_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_this_dir, "..")
for _p in (_this_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from visual_alignment.projection import ProjectionMLP
from unified_training.config import Phase4Config
from unified_training.dataset import build_datasets, multitask_collate_fn

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DenseNet feature extractor (frozen) — same as Phase 2
# ---------------------------------------------------------------------------

class DenseNetFeatureExtractor(torch.nn.Module):
    def __init__(self, checkpoint_path):
        super().__init__()
        sys.path.insert(
            0,
            os.path.join(os.path.dirname(__file__), "..", "densenet_model", "src"),
        )
        from model import build_model

        full_model = build_model(pretrained=False, fine_tune=False, num_classes=3)
        full_model.load_state_dict(
            torch.load(checkpoint_path, map_location="cpu")
        )
        self.features = full_model.features
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        feat = F.relu(self.features(x))
        feat = self.pool(feat)
        return feat.view(feat.size(0), -1)


# ---------------------------------------------------------------------------
# Visual token injection — splice into text embeddings
# ---------------------------------------------------------------------------

def build_combined_embeddings(
    text_embeds, attention_mask, labels,
    visual_tokens, has_image, insert_positions, image_batch_indices,
    num_visual_tokens, device,
):
    """Splice visual tokens into text embeddings at the correct positions.

    For image samples: insert N visual tokens at insert_pos, shifting
    subsequent text tokens right.
    For text-only samples: pad with N zeros at the end (attn=0, label=-100).

    Returns combined_embeds, combined_attn, combined_labels — all shape
    (B, max_text_len + N, dim).
    """
    B, max_text_len, dim = text_embeds.shape
    N = num_visual_tokens
    combined_len = max_text_len + N

    combined_embeds = torch.zeros(B, combined_len, dim, device=device, dtype=text_embeds.dtype)
    combined_attn = torch.zeros(B, combined_len, device=device, dtype=attention_mask.dtype)
    combined_labels = torch.full((B, combined_len), -100, device=device, dtype=labels.dtype)

    # Map batch index -> index into visual_tokens tensor
    img_map = {}
    if image_batch_indices is not None:
        for vt_idx, batch_idx in enumerate(image_batch_indices.tolist()):
            img_map[batch_idx] = vt_idx

    for i in range(B):
        if has_image[i] and i in img_map:
            pos = insert_positions[i].item()
            vt = visual_tokens[img_map[i]]  # (N, dim)

            # Before insert point
            combined_embeds[i, :pos] = text_embeds[i, :pos]
            combined_attn[i, :pos] = attention_mask[i, :pos]
            combined_labels[i, :pos] = labels[i, :pos]

            # Visual tokens
            combined_embeds[i, pos : pos + N] = vt
            combined_attn[i, pos : pos + N] = 1
            combined_labels[i, pos : pos + N] = -100

            # After insert point
            remaining = min(max_text_len - pos, combined_len - pos - N)
            if remaining > 0:
                combined_embeds[i, pos + N : pos + N + remaining] = (
                    text_embeds[i, pos : pos + remaining]
                )
                combined_attn[i, pos + N : pos + N + remaining] = (
                    attention_mask[i, pos : pos + remaining]
                )
                combined_labels[i, pos + N : pos + N + remaining] = (
                    labels[i, pos : pos + remaining]
                )
        else:
            # Text-only: copy text, pad end with zeros
            combined_embeds[i, :max_text_len] = text_embeds[i]
            combined_attn[i, :max_text_len] = attention_mask[i]
            combined_labels[i, :max_text_len] = labels[i]

    return combined_embeds, combined_attn, combined_labels


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_single(
    slm, projection, feat_extractor, val_loader,
    config, device, num_visual_tokens,
):
    """Run validation on a single dataset loader. Returns average loss."""
    slm.eval()
    projection.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in val_loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels_batch = batch["labels"].to(device)
        has_image = batch["has_image"]
        insert_positions = batch["insert_positions"]
        images = batch["images"]
        image_batch_indices = batch["image_batch_indices"]

        text_embeds = slm.get_input_embeddings()(input_ids)

        visual_tokens = None
        if images is not None:
            images_dev = images.to(device)
            vis_feat = feat_extractor(images_dev)
            visual_tokens = projection(vis_feat.to(torch.bfloat16))
            visual_tokens = visual_tokens.to(text_embeds.dtype)

        combined_embeds, combined_attn, combined_labels = (
            build_combined_embeddings(
                text_embeds, attn_mask, labels_batch,
                visual_tokens, has_image, insert_positions,
                image_batch_indices, num_visual_tokens, device,
            )
        )

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
            enabled=config.bf16 and device.type == "cuda",
        ):
            outputs = slm(
                inputs_embeds=combined_embeds,
                attention_mask=combined_attn,
                labels=combined_labels,
            )
        total_loss += outputs.loss.item()
        num_batches += 1

    slm.train()
    projection.train()
    return total_loss / max(num_batches, 1)


def validate_all(
    slm, projection, feat_extractor, val_loaders, config, device,
):
    """Validate on each dataset and compute weighted combined loss."""
    losses = {}
    weights = {"A": config.mix_a, "B": config.mix_b, "C": config.mix_c, "D": config.mix_d}

    for ds_type, loader in val_loaders.items():
        losses[ds_type] = validate_single(
            slm, projection, feat_extractor, loader,
            config, device, config.num_visual_tokens,
        )

    combined = sum(weights[t] * losses[t] for t in losses)
    losses["combined"] = combined
    return losses


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(projection, slm, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(
        projection.state_dict(), os.path.join(output_dir, "projection.pth")
    )
    slm.save_pretrained(os.path.join(output_dir, "lora_adapters"))
    logger.info(f"Best model checkpoint saved to {output_dir}")


def save_resume_state(projection, slm, optimizer, scheduler,
                      epoch, best_val_loss, patience_counter,
                      training_log, output_dir):
    """Save full training state after every epoch for crash recovery."""
    resume_dir = os.path.join(output_dir, "resume")
    os.makedirs(resume_dir, exist_ok=True)

    torch.save(projection.state_dict(), os.path.join(resume_dir, "projection.pth"))
    slm.save_pretrained(os.path.join(resume_dir, "lora_adapters"))
    torch.save(optimizer.state_dict(), os.path.join(resume_dir, "optimizer.pt"))
    torch.save(scheduler.state_dict(), os.path.join(resume_dir, "scheduler.pt"))

    state = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "patience_counter": patience_counter,
        "training_log": training_log,
    }
    with open(os.path.join(resume_dir, "training_state.json"), "w") as f:
        json.dump(state, f, indent=2)

    logger.info(f"Resume state saved at epoch {epoch + 1}")


def load_resume_state(output_dir):
    """Return resume state dict if a previous run exists, else None."""
    state_path = os.path.join(output_dir, "resume", "training_state.json")
    if not os.path.exists(state_path):
        return None
    with open(state_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = Phase4Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(
        config.slm_model_name, token=config.hf_token or None,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ---- SLM + Phase 2 LoRA ----
    if torch.cuda.is_available():
        logger.info("Loading SLM with 4-bit quantization.")
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
    else:
        logger.warning("No CUDA — loading SLM without quantization on CPU.")
        base_slm = AutoModelForCausalLM.from_pretrained(
            config.slm_model_name, token=config.hf_token or None,
        )
        base_slm.to(device)

    # Load LoRA — from resume checkpoint if available, else Phase 2
    resume_state = load_resume_state(config.output_dir)
    lora_source = (
        os.path.join(config.output_dir, "resume", "lora_adapters")
        if resume_state else config.phase2_lora
    )
    logger.info(f"Loading LoRA from {lora_source}")
    slm = PeftModel.from_pretrained(
        base_slm, lora_source, is_trainable=True,
    )
    slm.print_trainable_parameters()

    # ---- Frozen DenseNet ----
    feat_extractor = DenseNetFeatureExtractor(config.densenet_checkpoint)
    feat_extractor.eval().to(device)

    # ---- Projection MLP ----
    llm_dim = slm.config.hidden_size
    projection = ProjectionMLP(
        densenet_dim=config.densenet_dim,
        llm_dim=llm_dim,
        num_tokens=config.num_visual_tokens,
        hidden_dim=config.projection_hidden_dim,
    ).to(device).to(torch.bfloat16)

    # Load projection from resume checkpoint if available, else Phase 2
    proj_source = (
        os.path.join(config.output_dir, "resume", "projection.pth")
        if resume_state else config.phase2_projection
    )
    logger.info(f"Loading projection from {proj_source}")
    proj_state = torch.load(proj_source, map_location="cpu")
    projection.load_state_dict(proj_state)
    logger.info(
        f"Projection params: {sum(p.numel() for p in projection.parameters()):,}"
    )

    # ---- Datasets ----
    train_dataset, val_datasets, train_sampler = build_datasets(
        config, tokenizer,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        sampler=train_sampler,
        collate_fn=multitask_collate_fn,
        num_workers=2,
        pin_memory=True,
    )
    val_loaders = {
        ds_type: DataLoader(
            ds,
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=multitask_collate_fn,
            num_workers=2,
            pin_memory=True,
        )
        for ds_type, ds in val_datasets.items()
    }

    # ---- Optimizer (two param groups) ----
    lora_params = [p for p in slm.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [
            {"params": projection.parameters(), "lr": config.lr_projection},
            {"params": lora_params, "lr": config.lr_lora},
        ],
        weight_decay=config.weight_decay,
    )

    # ---- Restore resume state if available ----
    start_epoch = 0
    best_val_loss = float("inf")
    patience_counter = 0
    training_log = []

    if resume_state:
        resume_dir = os.path.join(config.output_dir, "resume")
        optimizer.load_state_dict(
            torch.load(os.path.join(resume_dir, "optimizer.pt"), map_location=device)
        )
        start_epoch = resume_state["epoch"] + 1
        best_val_loss = resume_state["best_val_loss"]
        patience_counter = resume_state["patience_counter"]
        training_log = resume_state["training_log"]
        logger.info(
            f"Resumed from epoch {resume_state['epoch'] + 1} "
            f"(best val loss: {best_val_loss:.4f}, starting at epoch {start_epoch + 1})"
        )

    # Build scheduler after resume so it covers only the remaining epochs.
    # This ensures the cosine schedule is fresh when continuing a completed run.
    remaining_epochs = config.epochs - start_epoch
    num_steps = (len(train_loader) * remaining_epochs) // config.grad_accum_steps
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * num_steps),
        num_training_steps=num_steps,
    )

    # ---- Training loop ----

    for epoch in range(start_epoch, config.epochs):
        slm.train()
        projection.train()
        total_loss = 0.0
        optimizer.zero_grad()

        # Track per-dataset loss within epoch
        epoch_ds_loss = {"A": 0.0, "B": 0.0, "C": 0.0, "D": 0.0}
        epoch_ds_count = {"A": 0, "B": 0, "C": 0, "D": 0}

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels_batch = batch["labels"].to(device)
            has_image = batch["has_image"]
            insert_positions = batch["insert_positions"]
            images = batch["images"]
            image_batch_indices = batch["image_batch_indices"]

            # 1) Text embeddings
            text_embeds = slm.get_input_embeddings()(input_ids)

            # 2) Visual tokens for image samples
            visual_tokens = None
            if images is not None:
                images_dev = images.to(device)
                with torch.no_grad():
                    vis_feat = feat_extractor(images_dev)
                visual_tokens = projection(vis_feat.to(torch.bfloat16))
                visual_tokens = visual_tokens.to(text_embeds.dtype)

            # 3) Splice visual tokens into embeddings
            combined_embeds, combined_attn, combined_labels = (
                build_combined_embeddings(
                    text_embeds, attn_mask, labels_batch,
                    visual_tokens, has_image, insert_positions,
                    image_batch_indices, config.num_visual_tokens, device,
                )
            )

            # 4) Forward pass
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16,
                enabled=config.bf16 and device.type == "cuda",
            ):
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

            step_loss = outputs.loss.item()
            total_loss += step_loss

            # Track per-dataset training loss
            for ds_type in batch["dataset_types"]:
                epoch_ds_loss[ds_type] += step_loss
                epoch_ds_count[ds_type] += 1

        avg_train_loss = total_loss / len(train_loader)

        # ---- Validation ----
        val_losses = validate_all(
            slm, projection, feat_extractor, val_loaders, config, device,
        )

        # Log
        epoch_info = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "train_loss_per_dataset": {
                t: round(epoch_ds_loss[t] / max(epoch_ds_count[t], 1), 4)
                for t in "ABCD"
            },
            "val_loss_A": round(val_losses["A"], 4),
            "val_loss_B": round(val_losses["B"], 4),
            "val_loss_C": round(val_losses["C"], 4),
            "val_loss_D": round(val_losses["D"], 4),
            "val_loss_combined": round(val_losses["combined"], 4),
            "lr_proj": optimizer.param_groups[0]["lr"],
            "lr_lora": optimizer.param_groups[1]["lr"],
        }
        training_log.append(epoch_info)

        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val A: {val_losses['A']:.4f} B: {val_losses['B']:.4f} "
            f"C: {val_losses['C']:.4f} D: {val_losses['D']:.4f} | "
            f"Combined: {val_losses['combined']:.4f}"
        )

        # Checkpoint on best combined val loss
        if val_losses["combined"] < best_val_loss:
            best_val_loss = val_losses["combined"]
            patience_counter = 0
            save_checkpoint(projection, slm, config.output_dir)
        else:
            patience_counter += 1

        # Save full training state every epoch for crash recovery
        save_resume_state(
            projection, slm, optimizer, scheduler,
            epoch, best_val_loss, patience_counter,
            training_log, config.output_dir,
        )

        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save training log
    os.makedirs(config.output_dir, exist_ok=True)
    log_path = os.path.join(config.output_dir, "training_log.json")
    with open(log_path, "w") as f:
        json.dump(training_log, f, indent=2)

    # Clean up resume state so re-runs start fresh from Phase 2 weights
    resume_marker = os.path.join(config.output_dir, "resume", "training_state.json")
    if os.path.exists(resume_marker):
        os.remove(resume_marker)
        logger.info("Cleared resume state (training completed successfully)")

    logger.info(f"Training complete. Best combined val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
