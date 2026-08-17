from __future__ import annotations

import torch


DOF_INDEX = {
    "U1": 0,
    "U2": 1,
    "U3": 2,
    "UR1": 3,
    "UR2": 4,
}


def _bc_value_mask(
    bc: dict,
    dtype: torch.dtype,
    device: torch.device,
    load_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "dofs" in bc:
        value = torch.zeros(1, 5, dtype=dtype, device=device)
        mask = torch.zeros(1, 5, dtype=dtype, device=device)
        unsupported = sorted(set(bc["dofs"]) - set(DOF_INDEX))
        if unsupported:
            raise ValueError(
                "plate5d supports U1/U2/U3/UR1/UR2 only; unsupported Abaqus DOF(s): "
                + ", ".join(unsupported)
            )
        for name, val in bc["dofs"].items():
            idx = DOF_INDEX[name]
            value[0, idx] = float(val) * load_factor
            mask[0, idx] = 1.0
        return value, mask
    value = torch.tensor(bc["value"], dtype=dtype, device=device).reshape(1, 5)
    value = value * load_factor
    mask = torch.tensor(bc.get("mask", [1, 1, 1, 1, 1]), dtype=dtype, device=device).reshape(1, 5)
    return value, mask


def plate_edge_boundary_penalty(model, case, n_width: int = 41) -> torch.Tensor:
    edge_constraints = case.geometry.get("edge_constraints", [])
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    if not edge_constraints:
        return torch.zeros((), dtype=dtype, device=device)

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
