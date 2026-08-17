from __future__ import annotations

import torch


def _grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    g = torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(x) if g is None else g


def _d(v: torch.Tensor, xi: torch.Tensor, component: int, coord: int) -> torch.Tensor:
    return _grad(v[:, component : component + 1], xi)[:, coord : coord + 1]


def _tensor_inner(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a[:, 0:1] * b[:, 0:1] + a[:, 1:2] * b[:, 1:2] + 2.0 * a[:, 2:3] * b[:, 2:3]


def _orthotropic_q(e1: float, e2: float, nu12: float, g12: float) -> torch.Tensor:
    nu21 = nu12 * e2 / e1
    denom = 1.0 - nu12 * nu21
    return torch.tensor(
        [
            [e1 / denom, nu12 * e2 / denom, 0.0],
            [nu12 * e2 / denom, e2 / denom, 0.0],
            [0.0, 0.0, g12],
        ],
        dtype=torch.float32,
    )


def _transform_q(q: torch.Tensor, angle_deg: float) -> torch.Tensor:
    # Standard transformed reduced stiffness for engineering shear gamma12.
    import math

    m = math.cos(math.radians(angle_deg))
    n = math.sin(math.radians(angle_deg))
    q11, q12, q22, q66 = q[0, 0], q[0, 1], q[1, 1], q[2, 2]
    m2, n2 = m * m, n * n
    m4, n4 = m2 * m2, n2 * n2
    qbar11 = q11 * m4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * n4
    qbar22 = q11 * n4 + 2.0 * (q12 + 2.0 * q66) * m2 * n2 + q22 * m4
    qbar12 = (q11 + q22 - 4.0 * q66) * m2 * n2 + q12 * (m4 + n4)
    qbar16 = (q11 - q12 - 2.0 * q66) * m * m2 * n - (q22 - q12 - 2.0 * q66) * m * n * n2
    qbar26 = (q11 - q12 - 2.0 * q66) * m * n * n2 - (q22 - q12 - 2.0 * q66) * m * m2 * n
    qbar66 = (q11 + q22 - 2.0 * q12 - 2.0 * q66) * m2 * n2 + q66 * (m4 + n4)
    return torch.stack(
        [
            torch.stack([qbar11, qbar12, qbar16]),
            torch.stack([qbar12, qbar22, qbar26]),
            torch.stack([qbar16, qbar26, qbar66]),
        ]
    )


def laminate_stiffness(case, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor | float]:
    mat = case.material
    geom = case.geometry
    thickness = float(mat.get("thickness", geom.get("thickness", 0.4)))
    shear_factor = float(mat.get("shear_factor", 5.0 / 6.0))
    if "laminate" not in mat:
        e = float(mat.get("E", 100.0))
        nu = float(mat.get("nu", 0.3))
        g = e / (2.0 * (1.0 + nu))
        c = e / (1.0 - nu * nu)
        qbar = torch.tensor([[c, nu * c, 0.0], [nu * c, c, 0.0], [0.0, 0.0, g]], device=device, dtype=dtype)
        return {
            "A": thickness * qbar,
            "B": torch.zeros(3, 3, device=device, dtype=dtype),
            "D": (thickness**3 / 12.0) * qbar,
            "As": shear_factor * thickness * torch.diag(torch.tensor([g, g], device=device, dtype=dtype)),
            "thickness": thickness,
        }

    lam = mat["laminate"]
    e1 = float(lam["E1"])
    e2 = float(lam["E2"])
    nu12 = float(lam["nu12"])
    g12 = float(lam["G12"])
    g13 = float(lam.get("G13", g12))
    g23 = float(lam.get("G23", g12))
    plies = lam["plies"]
    total_t = sum(float(p["thickness"]) for p in plies)
    z = -0.5 * total_t
    A = torch.zeros(3, 3, device=device, dtype=dtype)
    B = torch.zeros(3, 3, device=device, dtype=dtype)
    D = torch.zeros(3, 3, device=device, dtype=dtype)
    As = torch.zeros(2, 2, device=device, dtype=dtype)
    q_local = _orthotropic_q(e1, e2, nu12, g12).to(device=device, dtype=dtype)
    for ply in plies:
        t = float(ply["thickness"])
        angle = float(ply["angle"])
        z0, z1 = z, z + t
        qbar = _transform_q(q_local, angle).to(device=device, dtype=dtype)
        A = A + qbar * (z1 - z0)
        B = B + 0.5 * qbar * (z1 * z1 - z0 * z0)
        D = D + (1.0 / 3.0) * qbar * (z1**3 - z0**3)
        if abs((angle % 180.0) - 90.0) < 1.0e-8:
            gx, gy = g23, g13
        else:
            gx, gy = g13, g23
        As = As + shear_factor * t * torch.diag(torch.tensor([gx, gy], device=device, dtype=dtype))
        z = z1
    return {"A": A, "B": B, "D": D, "As": As, "thickness": total_t}


def _matvec(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return vec @ mat.T


def plate5_fields(model, xi: torch.Tensor, case) -> dict[str, torch.Tensor]:
    if not xi.requires_grad:
        xi = xi.clone().detach().requires_grad_(True)
    r, r0, q, q0, kappa0 = model(xi)
    u = q[:, 0:3]
    thx = q[:, 3:4]
    thy = q[:, 4:5]

    u_x = torch.cat([_d(u, xi, i, 0) for i in range(3)], dim=1)
    u_y = torch.cat([_d(u, xi, i, 1) for i in range(3)], dim=1)

    # Green-Lagrange membrane strain on a flat reference midsurface.
    e11 = u_x[:, 0:1] + 0.5 * (u_x * u_x).sum(dim=1, keepdim=True)
    e22 = u_y[:, 1:2] + 0.5 * (u_y * u_y).sum(dim=1, keepdim=True)
    e12 = 0.5 * (u_y[:, 0:1] + u_x[:, 1:2] + (u_x * u_y).sum(dim=1, keepdim=True))
    membrane = torch.cat([e11, e22, e12], dim=1)

    # theta1 = rotation about global X, theta2 = rotation about global Y.
    # Sign convention follows plate bending energy; signs do not affect energy,
    # but are exported consistently as shell curvature resultants.
    k11 = _grad(thy, xi)[:, 0:1]
    k22 = -_grad(thx, xi)[:, 1:2]
    k12 = 0.5 * (_grad(thy, xi)[:, 1:2] - _grad(thx, xi)[:, 0:1])
    bending = torch.cat([k11, k22, k12], dim=1)

    g1 = u_x[:, 2:3] + thy
    g2 = u_y[:, 2:3] - thx
    shear = torch.cat([g1, g2], dim=1)

    stiff = laminate_stiffness(case, xi.device, xi.dtype)
    membrane_resultant = _matvec(stiff["A"], membrane) + _matvec(stiff["B"], bending)
    moment = _matvec(stiff["B"], membrane) + _matvec(stiff["D"], bending)
    shear_resultant = shear @ stiff["As"].T
    # Export an equivalent midplane stress for comparison with shell output.
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
        "membrane": membrane,
        "bending": bending,
        "shear": shear,
        "stress": stress,
        "membrane_resultant": membrane_resultant,
        "moment": moment,
        "shear_resultant": shear_resultant,
        "membrane_energy": membrane_energy,
        "bending_energy": bending_energy,
        "shear_energy": shear_energy,
        "density": density,
    }


def plate5_energy(model, xi: torch.Tensor, case) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fields = plate5_fields(model, xi, case)
    area = float(case.length) * float(case.geometry.get("width", 1.0))
    return fields["density"].mean() * area, fields
