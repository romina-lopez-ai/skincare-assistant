import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Phase4Config:
    # Dataset paths
    dataset_a_train: str = "datasets/dataset_a/train.jsonl"
    dataset_a_val: str = "datasets/dataset_a/validation.jsonl"
    dataset_b_train: str = "datasets/dataset_b/train.jsonl"
    dataset_b_val: str = "datasets/dataset_b/validation.jsonl"
    dataset_c_train: str = "datasets/dataset_c/train.jsonl"
    dataset_c_val: str = "datasets/dataset_c/validation.jsonl"
    dataset_d_train: str = "datasets/dataset_d/train.jsonl"
    dataset_d_val: str = "datasets/dataset_d/validation.jsonl"
    images_dir: str = "datasets/images_dataset"

    # Phase 2 checkpoints
    phase2_projection: str = "visual_alignment/outputs/projection.pth"
    phase2_lora: str = "visual_alignment/outputs/lora_adapters"
    densenet_checkpoint: str = "densenet_model/outputs/model.pth"

    # Output
    output_dir: str = "unified_training/outputs"

    # SLM
    slm_model_name: str = "meta-llama/Llama-3.2-3B"
    hf_token: str = ""

    # Architecture (must match Phase 2)
    num_visual_tokens: int = 32
    projection_hidden_dim: int = 2048
    densenet_dim: int = 1024

    # LoRA (must match Phase 2)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training — lower LRs than Phase 2 to avoid forgetting
    # RTX 4070 12GB : batch_size=2,  grad_accum=16 → effective batch=32
    # A100 40GB     : batch_size=8,  grad_accum=4  → effective batch=32
    batch_size: int = 2
    grad_accum_steps: int = 16
    epochs: int = 15
    lr_projection: float = 1e-5
    lr_lora: float = 1e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.1
    max_length: int = 1536
    patience: int = 3

    # Mix ratios
    mix_a: float = 0.25
    mix_b: float = 0.25
    mix_c: float = 0.15
    mix_d: float = 0.35

    # Generation / eval
    max_gen_length: int = 256
    num_gen_samples: int = 2

    # Device
    bf16: bool = True

    def __post_init__(self):
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN", "")
