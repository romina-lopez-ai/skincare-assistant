import torch
import torch.nn as nn


class ProjectionMLP(nn.Module):
    """Projects DenseNet features into SLM token embedding space.

    Linear(densenet_dim, hidden_dim) -> GELU -> Linear(hidden_dim, num_tokens * llm_dim)
    Output is reshaped to (batch, num_tokens, llm_dim).
    """

    def __init__(self, densenet_dim: int = 1024, llm_dim: int = 3072,
                 num_tokens: int = 32, hidden_dim: int = 2048):
        super().__init__()
        self.num_tokens = num_tokens
        self.llm_dim = llm_dim
        self.mlp = nn.Sequential(
            nn.Linear(densenet_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_tokens * llm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, densenet_dim) -> (B, num_tokens, llm_dim)"""
        projected = self.mlp(x)
        return projected.view(-1, self.num_tokens, self.llm_dim)
