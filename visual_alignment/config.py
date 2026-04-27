import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Phase2Config:
    # Paths (relative to project root)
    densenet_checkpoint: str = "densenet_model/outputs/model.pth"
    train_jsonl: str = "datasets/dataset_a/train.jsonl"
    val_jsonl: str = "datasets/dataset_a/validation.jsonl"
    images_dir: str = "datasets/images_dataset"
    output_dir: str = "visual_alignment/outputs"

    # SLM
    slm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_token: str = ""

    # Architecture
    num_visual_tokens: int = 32
    projection_hidden_dim: int = 2048
    densenet_dim: int = 1024

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training
    batch_size: int = 4
    grad_accum_steps: int = 4
    epochs: int = 15
    lr_projection: float = 2e-4
    lr_lora: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    max_length: int = 768
    patience: int = 3

    # Generation / eval
    max_gen_length: int = 256
    num_gen_samples: int = 4

    # Device
    bf16: bool = True

    def __post_init__(self):
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN", "")
