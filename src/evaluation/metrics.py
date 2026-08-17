from __future__ import annotations

import torch

from src.losses.boundary_loss import boundary_penalty
from src.physics.energy import energy_density, integrate_trapezoid
from src.physics.strong_form import analytic_internal, euler_lagrange_residual


def evaluate_model(model, case, s: torch.Tensor) -> tuple[dict, dict[str, torch.Tensor]]:
    q, kappa0 = model(s)
    q0, _ = case.reference(s)
    density, strains = energy_density(q, s, kappa0, case.material)
    internal = analytic_internal(q, s, kappa0, case.material)
    residual = euler_lagrange_residual(q, s, kappa0, case.material)
    cs = case.constraint_s_tensor(s.device)
    q_bc, _ = model(cs)
    q_points = {str(c["s"]): q_bc[i : i + 1] for i, c in enumerate(case.constraints)}
    bc = boundary_penalty(q_points, case.constraints)
    metrics = {
        "energy": float(integrate_trapezoid(density, s).detach().cpu()),
        "strong_residual_rms": float(residual.pow(2).mean().sqrt().detach().cpu()),
        "boundary_mse": float(bc.detach().cpu()),
        "tip_x": float(q[-1, 0].detach().cpu()),
        "tip_y": float(q[-1, 1].detach().cpu()),
        "tip_theta": float(q[-1, 2].detach().cpu()),
        "eps_rms": float(strains["epsilon"].pow(2).mean().sqrt().detach().cpu()),
        "gamma_rms": float(strains["gamma"].pow(2).mean().sqrt().detach().cpu()),
        "bend_rms": float(strains["bend"].pow(2).mean().sqrt().detach().cpu()),
        "moment_abs_max": float(internal["moment"].abs().max().detach().cpu()),
        "axial_force_abs_max": float(internal["N"].abs().max().detach().cpu()),
        "shear_force_abs_max": float(internal["Q"].abs().max().detach().cpu()),
    }
    arrays = {"q": q, "q0": q0, "kappa0": kappa0, "residual": residual, **strains, **internal}
    return metrics, arrays
