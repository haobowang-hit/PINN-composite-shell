from __future__ import annotations

import torch

from src.losses.boundary_loss import boundary_penalty
from src.losses.plate_boundary import plate_edge_boundary_penalty
from src.losses.regularization import smoothness_regularization
from src.losses.surface_boundary import surface_boundary_penalty
from src.physics.plate5 import plate5_energy
from src.physics.surface_shell import surface_energy
from src.physics.weak_form import total_potential


def contact_penalty(fields: dict[str, torch.Tensor], case) -> torch.Tensor:
    contact = case.loads.get("contact", {}) if getattr(case, "loads", None) else {}
    if not contact:
        return torch.zeros((), dtype=fields["r"].dtype, device=fields["r"].device)
    axis = int(contact.get("axis", 1))
    coord = fields["r"][:, axis : axis + 1]
    loss = torch.zeros((), dtype=coord.dtype, device=coord.device)

    load_factor = float(getattr(case, "current_load_factor", 1.0))
    min_value = contact.get("min")
    max_value = contact.get("max")
    if min_value is not None and "initial_min" in contact:
        min_value = float(contact["initial_min"]) + load_factor * (float(min_value) - float(contact["initial_min"]))
    if max_value is not None and "initial_max" in contact:
        max_value = float(contact["initial_max"]) + load_factor * (float(max_value) - float(contact["initial_max"]))

    if min_value is not None:
        loss = loss + torch.relu(float(min_value) - coord).pow(2).mean()
    if max_value is not None:
        loss = loss + torch.relu(coord - float(max_value)).pow(2).mean()
    return loss


def weak_loss(model, s: torch.Tensor, case, weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    if getattr(case, "analysis_dim", "centerline1d") == "plate5d":
        potential, fields = plate5_energy(model, s, case)
        bc = plate_edge_boundary_penalty(model, case, int(case.training.get("bc_width_points", 41)))
        gauge = torch.zeros((), dtype=potential.dtype, device=potential.device)
        for item in case.loss_weights.get("gauge", []):
            if item.get("type") == "mean_dof":
                dof = int(item["dof"])
                target = float(item.get("value", 0.0))
                gauge = gauge + (fields["q"][:, dof].mean() - target) ** 2
        total = potential + float(weights["bc"]) * bc + float(weights.get("gauge_weight", weights["bc"])) * gauge
        logs = {
            "loss": float(total.detach().cpu()),
            "potential": float(potential.detach().cpu()),
            "membrane_energy": float(fields["membrane_energy"].mean().detach().cpu()),
            "bending_energy": float(fields["bending_energy"].mean().detach().cpu()),
            "shear_energy": float(fields["shear_energy"].mean().detach().cpu()),
            "bc": float(bc.detach().cpu()),
            "gauge": float(gauge.detach().cpu()),
            "membrane_rms": float(fields["membrane"].pow(2).mean().sqrt().detach().cpu()),
            "bending_rms": float(fields["bending"].pow(2).mean().sqrt().detach().cpu()),
            "shear_rms": float(fields["shear"].pow(2).mean().sqrt().detach().cpu()),
            "s11_rms": float(fields["stress"][:, 0].pow(2).mean().sqrt().detach().cpu()),
        }
        return total, logs

    if getattr(case, "analysis_dim", "centerline1d") == "shell2d":
        potential, fields = surface_energy(model, s, case)
        bc = surface_boundary_penalty(model, case, int(case.training.get("bc_width_points", 21)))
        contact = contact_penalty(fields, case)
        gauge = torch.zeros((), dtype=potential.dtype, device=potential.device)
        for item in case.loss_weights.get("gauge", []):
            if item.get("type") == "mean_dof":
                dof = int(item["dof"])
                target = float(item.get("value", 0.0))
                gauge = gauge + (fields["q"][:, dof].mean() - target) ** 2
        reg = torch.zeros((), dtype=potential.dtype, device=potential.device)
        total = (
            potential
            + float(weights["bc"]) * bc
            + float(weights.get("contact", weights["bc"])) * contact
            + float(weights.get("gauge_weight", weights["bc"])) * gauge
            + float(weights.get("reg", 0.0)) * reg
        )
        logs = {
            "loss": float(total.detach().cpu()),
            "potential": float(potential.detach().cpu()),
            "bc": float(bc.detach().cpu()),
            "contact": float(contact.detach().cpu()),
            "gauge": float(gauge.detach().cpu()),
            "reg": float(reg.detach().cpu()),
            "membrane_rms": float(fields["membrane"].pow(2).mean().sqrt().detach().cpu()),
            "bending_rms": float(fields["bending"].pow(2).mean().sqrt().detach().cpu()),
            "shear_rms": float(fields["shear"].pow(2).mean().sqrt().detach().cpu()),
            "s11_rms": float(fields["stress"][:, 0].pow(2).mean().sqrt().detach().cpu()),
        }
        return total, logs

    q, kappa0 = model(s)
    potential, strains = total_potential(q, s, kappa0, case.material, case.loads)
    cs = case.constraint_s_tensor(s.device)
    q_bc, _ = model(cs)
    q_points = {str(c["s"]): q_bc[i : i + 1] for i, c in enumerate(case.constraints)}
    bc = boundary_penalty(q_points, case.constraints)
    reg = smoothness_regularization(q, s)
    total = potential + float(weights["bc"]) * bc + float(weights["reg"]) * reg
    logs = {
        "loss": float(total.detach().cpu()),
        "potential": float(potential.detach().cpu()),
        "bc": float(bc.detach().cpu()),
        "reg": float(reg.detach().cpu()),
        "eps_rms": float(strains["epsilon"].pow(2).mean().sqrt().detach().cpu()),
        "gamma_rms": float(strains["gamma"].pow(2).mean().sqrt().detach().cpu()),
    }
    return total, logs
