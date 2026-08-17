from __future__ import annotations

import torch


def boundary_penalty(q_at_points: dict[str, torch.Tensor], constraints: list[dict]) -> torch.Tensor:
    if not constraints:
        any_q = next(iter(q_at_points.values()))
        return torch.zeros((), dtype=any_q.dtype, device=any_q.device)
    loss = None
    for bc in constraints:
        pred = q_at_points[str(bc["s"])]
        target = torch.tensor(bc["value"], dtype=pred.dtype, device=pred.device).reshape(1, -1)
        mask = torch.tensor(bc.get("mask", [1, 1, 1]), dtype=pred.dtype, device=pred.device).reshape(1, -1)
        term = ((pred - target) * mask).pow(2).mean()
        loss = term if loss is None else loss + term
    return loss


def natural_tip_force_penalty(force: torch.Tensor, target: dict, weight_mask: tuple[int, int] = (1, 1)) -> torch.Tensor:
    desired = torch.tensor(
        [float(target.get("fx", 0.0)), float(target.get("fy", 0.0))],
        dtype=force.dtype,
        device=force.device,
    ).reshape(1, 2)
    mask = torch.tensor(weight_mask, dtype=force.dtype, device=force.device).reshape(1, 2)
    return ((force[-1:] - desired) * mask).pow(2).mean()

