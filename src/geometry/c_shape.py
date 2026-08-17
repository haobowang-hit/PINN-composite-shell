from __future__ import annotations

import math

import torch


def c_shape_reference(
    s: torch.Tensor,
    length: float = 1.0,
    angle_deg: float = 220.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    alpha = math.radians(angle_deg)
    radius = length / alpha
    phi0 = -0.5 * alpha
    phi = phi0 + alpha * s / length
    x = radius * (torch.sin(phi) - math.sin(phi0))
    y = -radius * (torch.cos(phi) - math.cos(phi0))
    theta = phi
    kappa0 = torch.full_like(s, 1.0 / radius)
    return torch.cat([x, y, theta], dim=1), kappa0

