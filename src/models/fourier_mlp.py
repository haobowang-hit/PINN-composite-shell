from __future__ import annotations

import torch
from torch import nn

from src.models.mlp import MLP


class FourierMLP(nn.Module):
    def __init__(self, num_features: int = 8, **mlp_kwargs) -> None:
        super().__init__()
        self.register_buffer("freq", 2.0 ** torch.arange(num_features).float())
        self.mlp = MLP(in_dim=2 * num_features + 1, **mlp_kwargs)

    def forward(self, sbar: torch.Tensor) -> torch.Tensor:
        z = torch.pi * sbar * self.freq.reshape(1, -1)
        return self.mlp(torch.cat([sbar, torch.sin(z), torch.cos(z)], dim=1))

