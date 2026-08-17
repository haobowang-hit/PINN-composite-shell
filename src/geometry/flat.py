from __future__ import annotations

import torch


def flat_reference(s: torch.Tensor, length: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    x = s
    y = torch.zeros_like(s)
    theta = torch.zeros_like(s)
    kappa0 = torch.zeros_like(s)
    return torch.cat([x, y, theta], dim=1), kappa0

