from __future__ import annotations

import torch

from src.physics.strain import derivative, grad_scalar


def analytic_internal(q: torch.Tensor, s: torch.Tensor, kappa0: torch.Tensor, material: dict) -> dict[str, torch.Tensor]:
    dq = derivative(q, s)
    theta = q[:, 2:3]
    d1 = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
    d2 = torch.cat([-torch.sin(theta), torch.cos(theta)], dim=1)
    tangent = dq[:, 0:2]
    eps = (tangent * d1).sum(dim=1, keepdim=True) - 1.0
    gam = (tangent * d2).sum(dim=1, keepdim=True)
    kap = dq[:, 2:3]
    n = float(material["EA"]) * eps
    qshear = float(material["kGA"]) * gam
    moment = float(material["EI"]) * (kap - kappa0)
    force = n * d1 + qshear * d2
    return {
        "force": force,
        "moment": moment,
        "N": n,
        "Q": qshear,
        "epsilon": eps,
        "gamma": gam,
        "kappa": kap,
        "bend": kap - kappa0,
    }


def euler_lagrange_residual(
    q: torch.Tensor,
    s: torch.Tensor,
    kappa0: torch.Tensor,
    material: dict,
) -> torch.Tensor:
    internal = analytic_internal(q, s, kappa0, material)
    force = internal["force"]
    moment = internal["moment"]
    dforce = torch.cat([grad_scalar(force[:, i : i + 1], s) for i in range(2)], dim=1)
    dm = grad_scalar(moment, s)
    theta = q[:, 2:3]
    d1 = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
    d2 = torch.cat([-torch.sin(theta), torch.cos(theta)], dim=1)
    n = float(material["EA"]) * internal["epsilon"]
    qshear = float(material["kGA"]) * internal["gamma"]
    dwdtheta = n * (internal["force"][:, 0:1] * 0.0)
    dwdtheta = n * (derivative(q, s)[:, 0:2] * d2).sum(dim=1, keepdim=True)
    dwdtheta = dwdtheta - qshear * (derivative(q, s)[:, 0:2] * d1).sum(dim=1, keepdim=True)
    r_theta = dm - dwdtheta
    return torch.cat([dforce, r_theta], dim=1)
