"""Unified dataset for Phase 4 multitask training.

Handles all 4 dataset types (A/B/C/D) with proper tokenization,
assistant-only loss masking, and image loading for visual datasets.
"""

import json
import logging
import os

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset, WeightedRandomSampler
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IGNORE_INDEX = -100


class SingleTaskDataset(Dataset):
    """Loads one JSONL dataset and returns tokenized conversations.

    For image datasets (A, D): loads the image and records where visual
    tokens should be inserted. The empty user message at image_msg_index
    is skipped during tokenization.

    For text datasets (B, C): no image handling.
    """

    def __init__(self, jsonl_path, dataset_type, tokenizer, max_length,
                 images_root=None):
        self.dataset_type = dataset_type
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.images_root = images_root
        self.has_images = dataset_type in ("A", "D")

        # Cache special token IDs
        self._start_header = tokenizer.encode(
            "<|start_header_id|>", add_special_tokens=False
        )
        self._end_header = tokenizer.encode(
            "<|end_header_id|>", add_special_tokens=False
        )
        self._eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        self._newlines = tokenizer.encode("\n\n", add_special_tokens=False)
        self._bos = [tokenizer.bos_token_id]

        # Pre-tokenize role names
        self._role_ids = {
            "system": tokenizer.encode("system", add_special_tokens=False),
            "user": tokenizer.encode("user", add_special_tokens=False),
            "assistant": tokenizer.encode("assistant", add_special_tokens=False),
        }

        self.samples = self._load(jsonl_path)
        logger.info(
            f"Dataset {dataset_type}: loaded {len(self.samples)} samples "
            f"from {jsonl_path}"
        )

    def _load(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                sample = {"messages": item["messages"]}

                if self.has_images:
                    img_rel = item["image_path"].replace("images/", "", 1)
                    sample["image_path"] = os.path.join(self.images_root, img_rel)
                    sample["image_msg_index"] = item["image_msg_index"]

                samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        messages = sample["messages"]
        image_msg_index = sample.get("image_msg_index", -1)

        tokenized = self._tokenize_conversation(messages, image_msg_index)

        result = {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
            "insert_pos": tokenized["insert_pos"],
            "has_image": self.has_images,
            "dataset_type": self.dataset_type,
        }

        if self.has_images:
            image = Image.open(sample["image_path"]).convert("RGB")
            result["image"] = IMAGE_TRANSFORM(image)
        else:
            result["image"] = None

        return result

    def _tokenize_conversation(self, messages, image_msg_index):
        """Tokenize multi-turn conversation with assistant-only loss labels.

        Builds the LLaMA 3.2 chat format token-by-token:
          BOS + for each message:
            <|start_header_id|> role <|end_header_id|> \\n\\n content <|eot_id|>

        For image messages (empty user msg at image_msg_index): the message
        is skipped and its position is recorded as insert_pos.

        Labels = IGNORE_INDEX everywhere except assistant content + eot.
        """
        all_ids = list(self._bos)
        all_labels = [IGNORE_INDEX]
        insert_pos = -1

        for msg_idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            # Image placeholder — skip, record position
            if msg_idx == image_msg_index and content == "":
                insert_pos = len(all_ids)
                continue

            # Build header: <|start_header_id|> role <|end_header_id|> \n\n
            header_ids = (
                self._start_header
                + self._role_ids[role]
                + self._end_header
                + self._newlines
            )

            # Content + eot
            content_ids = self.tokenizer.encode(
                content, add_special_tokens=False
            )
            eot_ids = list(self._eot)

            if role == "assistant":
                # Header: masked. Content + eot: train.
                all_ids.extend(header_ids)
                all_labels.extend([IGNORE_INDEX] * len(header_ids))
                body = content_ids + eot_ids
                all_ids.extend(body)
                all_labels.extend(body)
            else:
                # System/user: all masked
                segment = header_ids + content_ids + eot_ids
                all_ids.extend(segment)
                all_labels.extend([IGNORE_INDEX] * len(segment))

        # Truncate to max_length
        all_ids = all_ids[: self.max_length]
        all_labels = all_labels[: self.max_length]

        # Adjust insert_pos if it would fall beyond truncated length
        if insert_pos >= len(all_ids):
            insert_pos = len(all_ids)

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "attention_mask": torch.ones(len(all_ids), dtype=torch.long),
            "insert_pos": insert_pos,
        }


def build_datasets(config, tokenizer):
    """Build train and val datasets for all 4 tasks.

    Returns:
        train_dataset: ConcatDataset of all 4 training sets
        val_datasets: dict of {"A": dataset, "B": ..., "C": ..., "D": ...}
        train_sampler: WeightedRandomSampler for mix ratios
    """
    ds_configs = [
        ("A", config.dataset_a_train, config.dataset_a_val, True),
        ("B", config.dataset_b_train, config.dataset_b_val, False),
        ("C", config.dataset_c_train, config.dataset_c_val, False),
        ("D", config.dataset_d_train, config.dataset_d_val, True),
    ]

    train_datasets = {}
    val_datasets = {}

    for ds_type, train_path, val_path, has_img in ds_configs:
        img_root = config.images_dir if has_img else None
        train_datasets[ds_type] = SingleTaskDataset(
            train_path, ds_type, tokenizer, config.max_length, img_root,
        )
        val_datasets[ds_type] = SingleTaskDataset(
            val_path, ds_type, tokenizer, config.max_length, img_root,
        )

    # Build weighted sampler
    target_ratios = {
        "A": config.mix_a,
        "B": config.mix_b,
        "C": config.mix_c,
        "D": config.mix_d,
    }
    sample_weights = []
    for ds_type in ["A", "B", "C", "D"]:
        ds = train_datasets[ds_type]
        weight = target_ratios[ds_type] / len(ds)
        sample_weights.extend([weight] * len(ds))

    combined_train = ConcatDataset(
        [train_datasets[t] for t in ["A", "B", "C", "D"]]
    )
    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(combined_train),
        replacement=True,
    )

    return combined_train, val_datasets, sampler


def multitask_collate_fn(batch):
    """Collate mixed-type samples into a padded batch.

    Handles both image and text-only samples. Images are stacked
    separately; image_indices tracks which batch positions have images.
    """
    # Find max text length in batch
    max_len = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)

    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    insert_positions = torch.full((B,), -1, dtype=torch.long)
    has_image = torch.zeros(B, dtype=torch.bool)
    dataset_types = []

    images = []
    image_batch_indices = []

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        insert_positions[i] = b["insert_pos"]
        has_image[i] = b["has_image"]
        dataset_types.append(b["dataset_type"])

        if b["has_image"] and b["image"] is not None:
            images.append(b["image"])
            image_batch_indices.append(i)

    images_tensor = torch.stack(images) if images else None
    image_batch_indices = (
        torch.tensor(image_batch_indices, dtype=torch.long) if images else None
    )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "insert_positions": insert_positions,
        "has_image": has_image,
        "images": images_tensor,
        "image_batch_indices": image_batch_indices,
        "dataset_types": dataset_types,
    }
