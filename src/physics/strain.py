from __future__ import annotations

import torch


def grad_scalar(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=False,
    )[0]


def derivative(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    cols = [grad_scalar(q[:, i : i + 1], s) for i in range(q.shape[1])]
    return torch.cat(cols, dim=1)


def reissner_strains(q: torch.Tensor, s: torch.Tensor, kappa0: torch.Tensor) -> dict[str, torch.Tensor]:
    dq = derivative(q, s)
    theta = q[:, 2:3]
    d1 = torch.cat([torch.cos(theta), torch.sin(theta)], dim=1)
    d2 = torch.cat([-torch.sin(theta), torch.cos(theta)], dim=1)
    tangent = dq[:, 0:2]
    epsilon = (tangent * d1).sum(dim=1, keepdim=True) - 1.0
    gamma = (tangent * d2).sum(dim=1, keepdim=True)
    kappa = dq[:, 2:3]
    return {"dq": dq, "epsilon": epsilon, "gamma": gamma, "kappa": kappa, "bend": kappa - kappa0}

