from __future__ import annotations

import torch

from src.physics.energy import energy_density, integrate_trapezoid


def total_potential(
    q: torch.Tensor,
    s: torch.Tensor,
    kappa0: torch.Tensor,
    material: dict,
    loads: dict | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    density, strains = energy_density(q, s, kappa0, material)
    energy = integrate_trapezoid(density, s)
    work = torch.zeros((), dtype=q.dtype, device=q.device)
    if loads and "tip_force" in loads:
        fx = float(loads["tip_force"].get("fx", 0.0))
        fy = float(loads["tip_force"].get("fy", 0.0))
        work = work + fx * q[-1, 0] + fy * q[-1, 1]
    return energy - work, strains

