from __future__ import annotations

import torch

from src.losses.plate_boundary import _bc_value_mask


def _constraint_target(case, bc: dict, eta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    value = torch.tensor(bc["value"], dtype=eta.dtype, device=eta.device).reshape(1, -1)
    mask = torch.tensor(bc.get("mask", [1, 1, 0]), dtype=eta.dtype, device=eta.device).reshape(1, -1)
    plane = case.geometry.get("fem_plane", "xy")
    lock_width = bool(bc.get("lock_width", True))
    if plane == "xz":
        target = torch.cat(
            [
                value[:, 0:1].expand_as(eta),
                eta,
                value[:, 1:2].expand_as(eta),
            ],
            dim=1,
        )
        pos_mask = torch.cat(
            [
                mask[:, 0:1].expand_as(eta),
                torch.ones_like(eta) if lock_width else torch.zeros_like(eta),
                mask[:, 1:2].expand_as(eta),
            ],
            dim=1,
        )
    else:
        target = torch.cat(
            [
                value[:, 0:1].expand_as(eta),
                value[:, 1:2].expand_as(eta),
                eta,
            ],
            dim=1,
        )
        pos_mask = torch.cat(
            [
                mask[:, 0:1].expand_as(eta),
                mask[:, 1:2].expand_as(eta),
                torch.ones_like(eta) if lock_width else torch.zeros_like(eta),
            ],
            dim=1,
        )
    theta = value[:, 2:3] if value.shape[1] >= 3 and mask[0, 2] > 0 else None
    return target, pos_mask, theta


def surface_boundary_penalty(model, case, n_width: int = 21) -> torch.Tensor:
    edge_constraints = case.geometry.get("edge_constraints", [])
    if edge_constraints:
        return surface_edge_boundary_penalty(model, case, edge_constraints, n_width)
    if not case.constraints:
        device = next(model.parameters()).device
        return torch.zeros((), dtype=torch.float32, device=device)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    width = float(case.geometry.get("width", 1.0))
    eta = torch.linspace(-0.5 * width, 0.5 * width, n_width, device=device, dtype=dtype).reshape(-1, 1)
    total = torch.zeros((), dtype=dtype, device=device)
    for bc in case.constraints:
        s_val = float(bc["s"]) * float(case.length)
        s = torch.full_like(eta, s_val)
        xi = torch.cat([s, eta], dim=1).requires_grad_(True)
        r, _, q, _, _ = model(xi)
        target, pos_mask, theta = _constraint_target(case, bc, eta)
        total = total + ((r - target) * pos_mask).pow(2).mean()
        if theta is not None:
            theta_index = int(bc.get("theta_index", 0))
            total = total + (q[:, 3 + theta_index : 4 + theta_index] - theta).pow(2).mean()
    return total


def surface_edge_boundary_penalty(model, case, edge_constraints: list[dict], n_width: int = 21) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    width = float(case.geometry.get("width", 1.0))
    eta = torch.linspace(-0.5 * width, 0.5 * width, n_width, device=device, dtype=dtype).reshape(-1, 1)
    total = torch.zeros((), dtype=dtype, device=device)
    load_factor = float(getattr(case, "current_load_factor", 1.0))
    for bc in edge_constraints:
        s_val = float(bc["s"]) * float(case.length)
        s = torch.full_like(eta, s_val)
        xi = torch.cat([s, eta], dim=1).requires_grad_(True)
        _, _, q, _, _ = model(xi)
        value, mask = _bc_value_mask(bc, dtype, device, load_factor)
        total = total + ((q - value) * mask).pow(2).mean()
    return total / max(len(edge_constraints), 1)
