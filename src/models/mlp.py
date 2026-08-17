from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int = 1,
        out_dim: int = 3,
        hidden_layers: int = 4,
        hidden_dim: int = 64,
        activation: str = "tanh",
    ) -> None:
        super().__init__()
        acts = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU}
        if activation not in acts:
            raise ValueError(f"Unsupported activation: {activation}")
        layers: list[nn.Module] = []
        width_in = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(width_in, hidden_dim))
            layers.append(acts[activation]())
            width_in = hidden_dim
        layers.append(nn.Linear(width_in, out_dim))
        self.net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, sbar: torch.Tensor) -> torch.Tensor:
        return self.net(sbar)

