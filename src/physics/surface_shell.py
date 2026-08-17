from __future__ import annotations

import torch

from src.physics.plate5 import laminate_stiffness


def _grad(y: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
    if not y.requires_grad:
        return torch.zeros_like(xi)
    g = torch.autograd.grad(
        y,
        xi,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    if g is None:
        return torch.zeros_like(xi)
    return g


def _vec_grad(v: torch.Tensor, xi: torch.Tensor, col: int) -> torch.Tensor:
    return torch.cat([_grad(v[:, i : i + 1], xi)[:, col : col + 1] for i in range(v.shape[1])], dim=1)


def _surface_basis(r: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r_s = _vec_grad(r, xi, 0)
    r_e = _vec_grad(r, xi, 1)
    normal = torch.cross(r_s, r_e, dim=1)
    normal = normal / torch.linalg.norm(normal, dim=1, keepdim=True).clamp_min(1.0e-12)
    return r_s, r_e, normal


def _voigt_metric(a11: torch.Tensor, a22: torch.Tensor, a12: torch.Tensor) -> torch.Tensor:
    return torch.cat([a11, a22, a12], dim=1)


def _matmul_c(voigt: torch.Tensor, young: float, nu: float) -> torch.Tensor:
    c = young / (1.0 - nu * nu)
    out11 = c * (voigt[:, 0:1] + nu * voigt[:, 1:2])
    out22 = c * (nu * voigt[:, 0:1] + voigt[:, 1:2])
    out12 = (young / (1.0 + nu)) * voigt[:, 2:3]
    return torch.cat([out11, out22, out12], dim=1)


def _tensor_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, 0:1] * b[:, 0:1] + a[:, 1:2] * b[:, 1:2] + 2.0 * a[:, 2:3] * b[:, 2:3]


def shell_material(case) -> dict[str, float]:
    geom = case.geometry
    mat = dict(case.material)
    thickness = float(mat.get("thickness", geom.get("thickness", 0.4)))
    nu = float(mat.get("nu", 0.3))
    if "E" in mat:
        young = float(mat["E"])
    else:
        width = float(geom.get("width", 1.0))
        young = float(mat.get("EA", 1.0)) / max(width * thickness, 1.0e-12)
    return {"E": young, "nu": nu, "thickness": thickness}


def surface_shell_fields(model, xi: torch.Tensor, case) -> dict[str, torch.Tensor]:
    if not xi.requires_grad:
        xi = xi.clone().detach().requires_grad_(True)
    r, r0, q, q0, kappa0 = model(xi)
    u = q[:, 0:3]
    theta = q[:, 3:5]
    r_s, r_e, n = _surface_basis(r, xi)
    R_s, R_e, N0 = _surface_basis(r0, xi)
    u_s = _vec_grad(u, xi, 0)
    u_e = _vec_grad(u, xi, 1)
    A11 = (R_s * R_s).sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    A22 = (R_e * R_e).sum(dim=1, keepdim=True).clamp_min(1.0e-12)
    A12 = (R_s * R_e).sum(dim=1, keepdim=True)
    inv_l1 = torch.rsqrt(A11)
    inv_l2 = torch.rsqrt(A22)
    e1 = R_s * inv_l1
    e2 = R_e * inv_l2

    a11 = (r_s * r_s).sum(dim=1, keepdim=True)
    a22 = (r_e * r_e).sum(dim=1, keepdim=True)
    a12 = (r_s * r_e).sum(dim=1, keepdim=True)
    # Green-Lagrange membrane strain in the local orthonormal surface basis.
    # This is essential for C/Omega flattening: a linear strain lets the
    # network change arc length during large rotations without enough penalty.
    e11 = 0.5 * (a11 - A11) / A11
    e22 = 0.5 * (a22 - A22) / A22
    e12 = 0.5 * (a12 - A12) * inv_l1 * inv_l2
    membrane = _voigt_metric(e11, e22, e12)

    # Rotate the reference director by the two tangent-plane rotations.  The
    # exact Rodrigues map is required for flattening: a linearized director is
    # not unit length and corrupts both curvature and transverse shear at the
    # large rotations encountered by the C and half-Omega shells.
    rotvec = theta[:, 0:1] * e1 + theta[:, 1:2] * e2
    # Keep the Rodrigues map twice differentiable at zero rotation.  This is
    # immaterial to the field values but essential when the strong form takes
    # another derivative of the energy density.
    phi = torch.sqrt((rotvec * rotvec).sum(dim=1, keepdim=True).clamp_min(1.0e-12))
    sinc_phi = torch.sinc(phi / torch.pi)  # sin(phi) / phi, stable at zero
    director = torch.cos(phi) * N0 + sinc_phi * torch.cross(rotvec, N0, dim=1)
    director = director / torch.linalg.norm(director, dim=1, keepdim=True).clamp_min(1.0e-12)

    d_s = _vec_grad(director, xi, 0)
    d_e = _vec_grad(director, xi, 1)
    N0_s = _vec_grad(N0, xi, 0)
    N0_e = _vec_grad(N0, xi, 1)

    B11 = -(R_s * N0_s).sum(dim=1, keepdim=True) / A11
    B22 = -(R_e * N0_e).sum(dim=1, keepdim=True) / A22
    B12 = -0.5 * ((R_s * N0_e).sum(dim=1, keepdim=True) + (R_e * N0_s).sum(dim=1, keepdim=True)) * inv_l1 * inv_l2
    b11 = -(r_s * d_s).sum(dim=1, keepdim=True) / A11
    b22 = -(r_e * d_e).sum(dim=1, keepdim=True) / A22
    b12 = -0.5 * ((r_s * d_e).sum(dim=1, keepdim=True) + (r_e * d_s).sum(dim=1, keepdim=True)) * inv_l1 * inv_l2
    k11 = b11 - B11
    k22 = b22 - B22
    k12 = b12 - B12
    bending = _voigt_metric(k11, k22, k12)

    # Director/tangent orthogonality; identically zero in the Kirchhoff limit.
    shear1 = (r_s * director).sum(dim=1, keepdim=True) * inv_l1
    shear2 = (r_e * director).sum(dim=1, keepdim=True) * inv_l2
    shear = torch.cat([shear1, shear2], dim=1)

    stiff = laminate_stiffness(case, xi.device, xi.dtype)
    membrane_resultant = membrane @ stiff["A"].T + bending @ stiff["B"].T
    moment = membrane @ stiff["B"].T + bending @ stiff["D"].T
    shear_resultant = shear @ stiff["As"].T
    stress = membrane_resultant / max(float(stiff["thickness"]), 1.0e-12)
    membrane_energy = 0.5 * _tensor_inner(membrane, membrane_resultant)
    bending_energy = 0.5 * _tensor_inner(bending, moment)
    shear_energy = 0.5 * (shear * shear_resultant).sum(dim=1, keepdim=True)
    density = membrane_energy + bending_energy + shear_energy

    return {
        "r": r,
        "r0": r0,
        "q": q,
        "q0": q0,
        "kappa0": kappa0,
        "normal": n,
        "director": director,
        "membrane": membrane,
        "bending": bending,
        "shear": shear,
        "stress": stress,
        "membrane_resultant": membrane_resultant,
        "shear_resultant": shear_resultant,
        "moment": moment,
        "membrane_energy": membrane_energy,
        "bending_energy": bending_energy,
        "shear_energy": shear_energy,
        "density": density,
    }


def surface_energy(model, xi: torch.Tensor, case) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fields = surface_shell_fields(model, xi, case)
    area = float(case.length) * float(case.geometry.get("width", 1.0))
    return fields["density"].mean() * area, fields
