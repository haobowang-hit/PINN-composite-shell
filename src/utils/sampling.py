from __future__ import annotations

import torch


def sample_arc_length(n: int, length: float, mode: str, device: torch.device) -> torch.Tensor:
    if mode == "sobol":
        engine = torch.quasirandom.SobolEngine(dimension=1, scramble=True)
        s = engine.draw(n).to(device)
    elif mode == "uniform":
        s = torch.linspace(0.0, 1.0, n, device=device).reshape(-1, 1)
    else:
        raise ValueError(f"Unknown sampling mode: {mode}")
    return (length * s).requires_grad_(True)


def fixed_grid(n: int, length: float, device: torch.device) -> torch.Tensor:
    return torch.linspace(0.0, length, n, device=device).reshape(-1, 1).requires_grad_(True)

