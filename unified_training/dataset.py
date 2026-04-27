"""Dataset for Phase 4 training with dataset F.

Handles image-based samples with patient symptoms and visual tokens.
Each sample has: system prompt, user message with symptoms, empty image
placeholder at image_msg_index, and assistant diagnosis response.
"""

import json
import logging
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

IGNORE_INDEX = -100


class DatasetF(Dataset):
    """Loads dataset F JSONL and returns tokenized conversations with images.

    Each sample includes an image and records where visual tokens should be
    inserted. The empty user message at image_msg_index is skipped during
    tokenization.
    """

    def __init__(self, jsonl_path, tokenizer, max_length, images_root):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.images_root = images_root

        self._start_header = tokenizer.encode(
            "<|start_header_id|>", add_special_tokens=False
        )
        self._end_header = tokenizer.encode(
            "<|end_header_id|>", add_special_tokens=False
        )
        self._eot = tokenizer.encode("<|eot_id|>", add_special_tokens=False)
        self._newlines = tokenizer.encode("\n\n", add_special_tokens=False)
        self._bos = [tokenizer.bos_token_id]

        self._role_ids = {
            "system": tokenizer.encode("system", add_special_tokens=False),
            "user": tokenizer.encode("user", add_special_tokens=False),
            "assistant": tokenizer.encode("assistant", add_special_tokens=False),
        }

        self.samples = self._load(jsonl_path)
        logger.info(f"Dataset F: loaded {len(self.samples)} samples from {jsonl_path}")

    def _load(self, path):
        samples = []
        with open(path) as f:
            for line in f:
                item = json.loads(line)
                img_rel = item["image_path"].replace("images/", "", 1)
                samples.append({
                    "messages": item["messages"],
                    "image_path": os.path.join(self.images_root, img_rel),
                    "image_msg_index": item["image_msg_index"],
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokenized = self._tokenize_conversation(
            sample["messages"], sample["image_msg_index"]
        )

        image = Image.open(sample["image_path"]).convert("RGB")

        return {
            "input_ids": tokenized["input_ids"],
            "labels": tokenized["labels"],
            "attention_mask": tokenized["attention_mask"],
            "insert_pos": tokenized["insert_pos"],
            "image": IMAGE_TRANSFORM(image),
        }

    def _tokenize_conversation(self, messages, image_msg_index):
        """Tokenize multi-turn conversation with assistant-only loss labels.

        Builds the LLaMA 3.2 chat format token-by-token:
          BOS + for each message:
            <|start_header_id|> role <|end_header_id|> \\n\\n content <|eot_id|>

        The empty user message at image_msg_index is skipped and its position
        is recorded as insert_pos for visual token injection.

        Labels = IGNORE_INDEX everywhere except assistant content + eot.
        """
        all_ids = list(self._bos)
        all_labels = [IGNORE_INDEX]
        insert_pos = -1

        for msg_idx, msg in enumerate(messages):
            role = msg["role"]
            content = msg["content"]

            if msg_idx == image_msg_index and content == "":
                insert_pos = len(all_ids)
                continue

            header_ids = (
                self._start_header
                + self._role_ids[role]
                + self._end_header
                + self._newlines
            )

            content_ids = self.tokenizer.encode(
                content, add_special_tokens=False
            )
            eot_ids = list(self._eot)

            if role == "assistant":
                all_ids.extend(header_ids)
                all_labels.extend([IGNORE_INDEX] * len(header_ids))
                body = content_ids + eot_ids
                all_ids.extend(body)
                all_labels.extend(body)
            else:
                segment = header_ids + content_ids + eot_ids
                all_ids.extend(segment)
                all_labels.extend([IGNORE_INDEX] * len(segment))

        all_ids = all_ids[: self.max_length]
        all_labels = all_labels[: self.max_length]

        if insert_pos >= len(all_ids):
            insert_pos = len(all_ids)

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
            "attention_mask": torch.ones(len(all_ids), dtype=torch.long),
            "insert_pos": insert_pos,
        }


def build_datasets(config, tokenizer):
    """Build train and val datasets for dataset F.

    Returns:
        train_dataset: DatasetF for training
        val_dataset: DatasetF for validation
    """
    train_dataset = DatasetF(
        config.dataset_train, tokenizer, config.max_length, config.images_dir,
    )
    val_dataset = DatasetF(
        config.dataset_val, tokenizer, config.max_length, config.images_dir,
    )
    return train_dataset, val_dataset


def collate_fn(batch):
    """Collate image samples into a padded batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)

    input_ids = torch.zeros(B, max_len, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    attention_mask = torch.zeros(B, max_len, dtype=torch.long)
    insert_positions = torch.full((B,), -1, dtype=torch.long)
    images = []

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        labels[i, :seq_len] = b["labels"]
        attention_mask[i, :seq_len] = b["attention_mask"]
        insert_positions[i] = b["insert_pos"]
        images.append(b["image"])

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "insert_positions": insert_positions,
        "images": torch.stack(images),
    }
