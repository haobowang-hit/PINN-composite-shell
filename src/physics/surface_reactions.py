from __future__ import annotations

import torch

from src.physics.plate5 import plate5_fields
from src.physics.surface_shell import surface_shell_fields


def _trapz_width(values: torch.Tensor, eta: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(eta.reshape(-1))
    return torch.trapz(values.reshape(-1)[order], eta.reshape(-1)[order])


def mechanics_snapshot(model, case, device: torch.device, load_factor: float, width_points: int = 101) -> dict[str, float]:
    """Evaluate energy components and generalized edge actions.

    The edge quantities are thickness-integrated shell resultants integrated
    across the strip width.  They are intended for equilibrium-path and FEM
    reaction checks; signs follow the local shell convention and magnitudes
    are also exported for convention-independent comparisons.
    """
    width = float(case.geometry.get("width", 1.0))
    length = float(case.length)
    ns = max(81, int(case.training.get("eval_points", 161)))
    nw = max(21, int(case.training.get("export_width_points", 25)))
    s = torch.linspace(0.0, length, ns, device=device)
    eta = torch.linspace(-0.5 * width, 0.5 * width, nw, device=device)
    ss, ee = torch.meshgrid(s, eta, indexing="ij")
    xi = torch.stack([ss.reshape(-1), ee.reshape(-1)], dim=1).requires_grad_(True)
    fields = plate5_fields(model, xi, case) if case.analysis_dim == "plate5d" else surface_shell_fields(model, xi, case)
    area = length * width
    em = fields["membrane_energy"].mean() * area
    eb = fields["bending_energy"].mean() * area
    es = fields["shear_energy"].mean() * area

    eta_edge = torch.linspace(-0.5 * width, 0.5 * width, width_points, device=device)
    out: dict[str, float] = {
        "load_factor": float(load_factor),
        "energy_total": float((em + eb + es).detach().cpu()),
        "energy_membrane": float(em.detach().cpu()),
        "energy_bending": float(eb.detach().cpu()),
        "energy_shear": float(es.detach().cpu()),
    }
    for edge_name, edge_s, sign in [("left", 0.0, -1.0), ("right", length, 1.0)]:
        xi_edge = torch.stack([torch.full_like(eta_edge, edge_s), eta_edge], dim=1).requires_grad_(True)
        edge = plate5_fields(model, xi_edge, case) if case.analysis_dim == "plate5d" else surface_shell_fields(model, xi_edge, case)
        n11 = sign * _trapz_width(edge["membrane_resultant"][:, 0], eta_edge)
        n12 = sign * _trapz_width(edge["membrane_resultant"][:, 2], eta_edge)
        q1 = sign * _trapz_width(edge["shear_resultant"][:, 0], eta_edge)
        m11 = sign * _trapz_width(edge["moment"][:, 0], eta_edge)
        force = torch.sqrt(n11 * n11 + n12 * n12 + q1 * q1)
        out[f"{edge_name}_N11"] = float(n11.detach().cpu())
        out[f"{edge_name}_N12"] = float(n12.detach().cpu())
        out[f"{edge_name}_Q1"] = float(q1.detach().cpu())
        out[f"{edge_name}_M11"] = float(m11.detach().cpu())
        out[f"{edge_name}_force_magnitude"] = float(force.detach().cpu())
    return out
