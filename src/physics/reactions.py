from __future__ import annotations

import torch

from src.physics.strong_form import analytic_internal


def end_reaction(q: torch.Tensor, s: torch.Tensor, kappa0: torch.Tensor, material: dict, end: str) -> dict[str, float]:
    internal = analytic_internal(q, s, kappa0, material)
    idx = 0 if end == "left" else -1
    sign = -1.0 if end == "left" else 1.0
    force = sign * internal["force"][idx].detach().cpu()
    moment = sign * internal["moment"][idx].detach().cpu()
    return {"Rx": float(force[0]), "Ry": float(force[1]), "M": float(moment.reshape(-1)[0])}

