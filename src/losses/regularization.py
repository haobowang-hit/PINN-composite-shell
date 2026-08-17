from __future__ import annotations

import torch

from src.physics.strain import derivative


def smoothness_regularization(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    dq = derivative(q, s)
    ddq = derivative(dq, s)
    return ddq.pow(2).mean()

