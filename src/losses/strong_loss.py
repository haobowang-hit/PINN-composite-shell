from __future__ import annotations

import torch

from src.losses.boundary_loss import boundary_penalty, natural_tip_force_penalty
from src.losses.plate_boundary import plate_edge_boundary_penalty
from src.losses.regularization import smoothness_regularization
from src.losses.surface_boundary import surface_boundary_penalty
from src.losses.weak_loss import contact_penalty
from src.physics.plate5 import plate5_fields
from src.physics.shell_strong import shell_euler_lagrange_residual
from src.physics.strong_form import analytic_internal, euler_lagrange_residual
from src.physics.surface_shell import surface_shell_fields


def strong_loss(model, s: torch.Tensor, case, weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    if getattr(case, "analysis_dim", "centerline1d") in {"shell2d", "plate5d"}:
        analysis_dim = getattr(case, "analysis_dim", "shell2d")
        residual = shell_euler_lagrange_residual(model, s, case)
        fields = plate5_fields(model, s, case) if analysis_dim == "plate5d" else surface_shell_fields(model, s, case)
        if analysis_dim == "plate5d":
            bc = plate_edge_boundary_penalty(model, case, int(case.training.get("bc_width_points", 41)))
        else:
            bc = surface_boundary_penalty(model, case, int(case.training.get("bc_width_points", 21)))
        contact = contact_penalty(fields, case)
        gauge = torch.zeros((), dtype=residual.dtype, device=residual.device)
        for item in case.loss_weights.get("gauge", []):
            if item.get("type") == "mean_dof":
                dof = int(item["dof"])
                target = float(item.get("value", 0.0))
                gauge = gauge + (fields["q"][:, dof].mean() - target) ** 2
        residual_loss = residual.pow(2).mean()
        total = (
            residual_loss
            + float(weights["bc"]) * bc
            + float(weights.get("contact", weights["bc"])) * contact
            + float(weights.get("gauge_weight", weights["bc"])) * gauge
        )
        logs = {
            "loss": float(total.detach().cpu()),
            "residual": float(residual_loss.detach().cpu()),
            "bc": float(bc.detach().cpu()),
            "contact": float(contact.detach().cpu()),
            "gauge": float(gauge.detach().cpu()),
            "membrane_rms": float(fields["membrane"].pow(2).mean().sqrt().detach().cpu()),
            "bending_rms": float(fields["bending"].pow(2).mean().sqrt().detach().cpu()),
            "shear_rms": float(fields["shear"].pow(2).mean().sqrt().detach().cpu()),
            "s11_rms": float(fields["stress"][:, 0].pow(2).mean().sqrt().detach().cpu()),
        }
        return total, logs
    q, kappa0 = model(s)
    residual = euler_lagrange_residual(q, s, kappa0, case.material)
    cs = case.constraint_s_tensor(s.device)
    q_bc, _ = model(cs)
    q_points = {str(c["s"]): q_bc[i : i + 1] for i, c in enumerate(case.constraints)}
    bc = boundary_penalty(q_points, case.constraints)
    nbc = torch.zeros((), dtype=q.dtype, device=q.device)
    if case.loads and "tip_force" in case.loads:
        internal = analytic_internal(q, s, kappa0, case.material)
        nbc = natural_tip_force_penalty(internal["force"], case.loads["tip_force"])
    reg = smoothness_regularization(q, s)
    residual_loss = residual.pow(2).mean()
    total = residual_loss + float(weights["bc"]) * bc + float(weights.get("nbc", weights["bc"])) * nbc + float(weights["reg"]) * reg
    logs = {
        "loss": float(total.detach().cpu()),
        "residual": float(residual_loss.detach().cpu()),
        "bc": float(bc.detach().cpu()),
        "nbc": float(nbc.detach().cpu()),
        "reg": float(reg.detach().cpu()),
    }
    return total, logs
