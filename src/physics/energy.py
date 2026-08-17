from __future__ import annotations

import torch

from src.physics.strain import reissner_strains


def energy_density(
    q: torch.Tensor,
    s: torch.Tensor,
    kappa0: torch.Tensor,
    material: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    strains = reissner_strains(q, s, kappa0)
    ea = float(material["EA"])
    kga = float(material["kGA"])
    ei = float(material["EI"])
    density = 0.5 * ea * strains["epsilon"] ** 2
    density = density + 0.5 * kga * strains["gamma"] ** 2
    density = density + 0.5 * ei * strains["bend"] ** 2
    return density, strains


def integrate_trapezoid(values: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(s.reshape(-1))
    xs = s.reshape(-1)[order]
    ys = values.reshape(-1)[order]
    return torch.trapz(ys, xs)

