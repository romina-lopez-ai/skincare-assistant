"""Custom Dataset for Phase 2 visual alignment training.

Supports Dataset A format: messages array with system/user/assistant roles.
The assistant message contains <think> + [DIAGNOSIS] blocks.
"""

import json
import os
import logging

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet normalization (same as DenseNet training)
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class VisualAlignmentDataset(Dataset):
    """Loads image + assistant response pairs for visual-language alignment.

    Reads Dataset A format (messages array). The training target is the
    assistant message containing <think> + [DIAGNOSIS].

    Each sample returns:
        image: (3, 224, 224) tensor
        input_ids: (seq_len,) tokenized assistant content
        attention_mask: (seq_len,) attention mask
    """

    def __init__(self, jsonl_path: str, images_root: str, tokenizer,
                 max_length: int = 512, transform=None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.transform = transform or IMAGE_TRANSFORM
        self.samples = []  # (image_path, assistant_text)

        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line)
                img_rel = item["image_path"].replace("images/", "", 1)
                img_path = os.path.join(images_root, img_rel)

                # Extract assistant content from messages
                assistant_text = ""
                for msg in item["messages"]:
                    if msg["role"] == "assistant":
                        assistant_text = msg["content"]
                        break

                if assistant_text:
                    self.samples.append((img_path, assistant_text))

        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, note_text = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        tokens = self.tokenizer(
            note_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)

        return {
            "image": image,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "text": note_text,
        }


def collate_fn(batch):
    """Pad variable-length token sequences to the longest in the batch."""
    images = torch.stack([b["image"] for b in batch])

    # Find max length in this batch
    max_len = max(b["input_ids"].size(0) for b in batch)

    # Right-pad input_ids with pad_token_id=0, attention_mask with 0
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, b in enumerate(batch):
        seq_len = b["input_ids"].size(0)
        input_ids[i, :seq_len] = b["input_ids"]
        attention_mask[i, :seq_len] = b["attention_mask"]

    texts = [b["text"] for b in batch]

    return {
        "image": images,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "text": texts,
    }
