from __future__ import annotations

import torch
from torch.func import grad, jvp, vmap

from src.physics.plate5 import laminate_stiffness


def _autograd_grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    value = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(x) if value is None else value


def _network_gradient(q: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    columns = []
    for component in range(q.shape[1]):
        columns.append(_autograd_grad(q[:, component : component + 1], xi))
    return torch.stack(columns, dim=1)


def _tensor_energy(strain: torch.Tensor, resultant: torch.Tensor) -> torch.Tensor:
    return 0.5 * (
        strain[0] * resultant[0]
        + strain[1] * resultant[1]
        + 2.0 * strain[2] * resultant[2]
    )


def _plate_density(
    q: torch.Tensor,
    p: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    As: torch.Tensor,
) -> torch.Tensor:
    ux = p[0:3, 0]
    uy = p[0:3, 1]
    membrane = torch.stack(
        [
            ux[0] + 0.5 * torch.dot(ux, ux),
            uy[1] + 0.5 * torch.dot(uy, uy),
            0.5 * (uy[0] + ux[1] + torch.dot(ux, uy)),
        ]
    )
    bending = torch.stack(
        [
            p[4, 0],
            -p[3, 1],
            0.5 * (p[4, 1] - p[3, 0]),
        ]
    )
    shear = torch.stack([p[2, 0] + q[4], p[2, 1] - q[3]])
    N = A @ membrane + B @ bending
    M = B @ membrane + D @ bending
    Q = As @ shear
    return _tensor_energy(membrane, N) + _tensor_energy(bending, M) + 0.5 * torch.dot(shear, Q)


def _surface_density(
    q: torch.Tensor,
    p: torch.Tensor,
    R_s: torch.Tensor,
    R_e: torch.Tensor,
    N0: torch.Tensor,
    e1_s: torch.Tensor,
    e1_e: torch.Tensor,
    e2_s: torch.Tensor,
    e2_e: torch.Tensor,
    N0_s: torch.Tensor,
    N0_e: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    D: torch.Tensor,
    As: torch.Tensor,
) -> torch.Tensor:
    A11 = torch.dot(R_s, R_s).clamp_min(1.0e-12)
    A22 = torch.dot(R_e, R_e).clamp_min(1.0e-12)
    A12 = torch.dot(R_s, R_e)
    inv_l1 = torch.rsqrt(A11)
    inv_l2 = torch.rsqrt(A22)
    e1 = R_s * inv_l1
    e2 = R_e * inv_l2
    r_s = R_s + p[0:3, 0]
    r_e = R_e + p[0:3, 1]
    membrane = torch.stack(
        [
            0.5 * (torch.dot(r_s, r_s) - A11) / A11,
            0.5 * (torch.dot(r_e, r_e) - A22) / A22,
            0.5 * (torch.dot(r_s, r_e) - A12) * inv_l1 * inv_l2,
        ]
    )

    def director_fn(theta: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor, normal: torch.Tensor) -> torch.Tensor:
        rotvec = theta[0] * t1 + theta[1] * t2
        # ``norm(rotvec)`` has an undefined second derivative at zero.  The
        # strong residual differentiates the energy twice, so use a clamped
        # squared norm to retain the removable Rodrigues limit without NaNs.
        phi = torch.sqrt(torch.dot(rotvec, rotvec).clamp_min(1.0e-12))
        director = torch.cos(phi) * normal + torch.sinc(phi / torch.pi) * torch.cross(rotvec, normal, dim=0)
        return director / torch.linalg.norm(director).clamp_min(1.0e-12)

    theta = q[3:5]
    director = director_fn(theta, e1, e2, N0)
    _, d_s = jvp(
        director_fn,
        (theta, e1, e2, N0),
        (p[3:5, 0], e1_s, e2_s, N0_s),
    )
    _, d_e = jvp(
        director_fn,
        (theta, e1, e2, N0),
        (p[3:5, 1], e1_e, e2_e, N0_e),
    )
    B11 = -torch.dot(R_s, N0_s) / A11
    B22 = -torch.dot(R_e, N0_e) / A22
    B12 = -0.5 * (torch.dot(R_s, N0_e) + torch.dot(R_e, N0_s)) * inv_l1 * inv_l2
    b11 = -torch.dot(r_s, d_s) / A11
    b22 = -torch.dot(r_e, d_e) / A22
    b12 = -0.5 * (torch.dot(r_s, d_e) + torch.dot(r_e, d_s)) * inv_l1 * inv_l2
    bending = torch.stack([b11 - B11, b22 - B22, b12 - B12])
    shear = torch.stack([torch.dot(r_s, director) * inv_l1, torch.dot(r_e, director) * inv_l2])
    N = A @ membrane + B @ bending
    M = B @ membrane + D @ bending
    Q = As @ shear
    return _tensor_energy(membrane, N) + _tensor_energy(bending, M) + 0.5 * torch.dot(shear, Q)


def _reference_basis(model, xi: torch.Tensor) -> tuple[torch.Tensor, ...]:
    r0, _, _ = model.reference_surface(xi)
    columns = []
    for component in range(3):
        columns.append(_autograd_grad(r0[:, component : component + 1], xi))
    jac = torch.stack(columns, dim=1)
    R_s = jac[:, :, 0]
    R_e = jac[:, :, 1]
    N0 = torch.cross(R_s, R_e, dim=1)
    N0 = N0 / torch.linalg.norm(N0, dim=1, keepdim=True).clamp_min(1.0e-12)
    inv_l1 = torch.rsqrt((R_s * R_s).sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    inv_l2 = torch.rsqrt((R_e * R_e).sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    e1 = R_s * inv_l1
    e2 = R_e * inv_l2
    e1_grad = _network_gradient(e1, xi)
    e2_grad = _network_gradient(e2, xi)
    n_grad = _network_gradient(N0, xi)
    return (
        R_s,
        R_e,
        N0,
        e1_grad[:, :, 0],
        e1_grad[:, :, 1],
        e2_grad[:, :, 0],
        e2_grad[:, :, 1],
        n_grad[:, :, 0],
        n_grad[:, :, 1],
    )


def shell_euler_lagrange_residual(model, xi: torch.Tensor, case) -> torch.Tensor:
    """Euler-Lagrange residual of the same pointwise energy used by the weak form."""
    if not xi.requires_grad:
        xi = xi.clone().detach().requires_grad_(True)
    _, _, q, _, _ = model(xi)
    p = _network_gradient(q, xi)
    stiff = laminate_stiffness(case, xi.device, xi.dtype)
    A, B, D, As = stiff["A"], stiff["B"], stiff["D"], stiff["As"]

    if getattr(case, "analysis_dim", "centerline1d") == "plate5d":
        density = _plate_density
        dW_dq = vmap(grad(density, argnums=0), in_dims=(0, 0, None, None, None, None))(q, p, A, B, D, As)
        dW_dp = vmap(grad(density, argnums=1), in_dims=(0, 0, None, None, None, None))(q, p, A, B, D, As)
    else:
        reference = _reference_basis(model, xi)
        density = _surface_density
        in_dims = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, None, None, None, None)
        args = (q, p, *reference, A, B, D, As)
        dW_dq = vmap(grad(density, argnums=0), in_dims=in_dims)(*args)
        dW_dp = vmap(grad(density, argnums=1), in_dims=in_dims)(*args)

    divergence = []
    for component in range(q.shape[1]):
        term = torch.zeros_like(q[:, component : component + 1])
        for alpha in range(2):
            derivative = _autograd_grad(dW_dp[:, component, alpha : alpha + 1], xi)
            term = term + derivative[:, alpha : alpha + 1]
        divergence.append(term)
    return dW_dq - torch.cat(divergence, dim=1)
