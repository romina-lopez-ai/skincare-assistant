import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ProjectionEvalConfig:
    dataset_train: str = "datasets/dataset_d/train.jsonl"
    dataset_val: str = "datasets/dataset_d/validation.jsonl"
    images_dir: str = "datasets/images_dataset"

    phase2_projection: str = "visual_alignment/outputs/projection.pth"
    phase4_projection: str = "unified_training/outputs/projection.pth"
    densenet_checkpoint: str = "densenet_model/outputs/model.pth"

    slm_model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_token: str = ""

    num_visual_tokens: int = 32
    projection_hidden_dim: int = 2048
    densenet_dim: int = 1024

    output_dir: str = "evaluation_metrics/outputs"

    condition_labels: tuple = ("acne", "eczema", "psoriasis")

    bf16: bool = True

    def __post_init__(self):
        if not self.hf_token:
            self.hf_token = os.environ.get("HF_TOKEN", "")
