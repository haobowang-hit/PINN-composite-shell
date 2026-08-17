from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from torch import nn

from src.geometry import HalfOmegaReference, OmegaReference, c_shape_reference, flat_reference
from src.models.fourier_mlp import FourierMLP
from src.models.mlp import MLP
from src.models.siren import Siren
from src.utils.io import load_yaml


ReferenceFn = Callable[[torch.Tensor], tuple[torch.Tensor, torch.Tensor]]


@dataclass
class CaseDefinition:
    case_id: str
    name: str
    length: float
    material: dict
    constraints: list[dict]
    loads: dict
    training: dict
    loss_weights: dict
    model_config: dict
    geometry: dict
    reference: ReferenceFn
    output_dir: str
    analysis_dim: str = "centerline1d"

    def constraint_s_tensor(self, device: torch.device) -> torch.Tensor:
        values = [float(c["s"]) * self.length for c in self.constraints]
        return torch.tensor(values, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)


class StripPINN(nn.Module):
    def __init__(self, case: CaseDefinition) -> None:
        super().__init__()
        cfg = case.model_config
        kind = cfg.get("type", "mlp")
        kwargs = {
            "out_dim": 3,
            "hidden_layers": int(cfg.get("hidden_layers", 4)),
            "hidden_dim": int(cfg.get("hidden_dim", 64)),
            "activation": cfg.get("activation", "tanh"),
        }
        if kind == "mlp":
            self.net = MLP(in_dim=1, **kwargs)
        elif kind == "fourier_mlp":
            self.net = FourierMLP(**kwargs)
        elif kind == "siren":
            self.net = Siren(in_dim=1, **kwargs)
        else:
            raise ValueError(f"Unknown model type: {kind}")
        self.case = case
        self.scale = torch.tensor(cfg.get("output_scale", [0.25, 0.25, 0.5]), dtype=torch.float32)

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q0, kappa0 = self.case.reference(s)
        scale = self.scale.to(device=s.device, dtype=s.dtype).reshape(1, 3)
        sbar = 2.0 * s / self.case.length - 1.0
        delta = self.net(sbar) * scale
        return q0 + delta, kappa0


class ShellSurfacePINN(nn.Module):
    """2D developable strip surface model.

    Input coordinates are physical ``(s, eta)`` where ``s`` is arc length and
    ``eta`` is the width coordinate. The network predicts the five shell
    unknowns ``(u1, u2, u3, theta1, theta2)`` in global-displacement/local-
    rotation form.
    """

    def __init__(self, case: CaseDefinition) -> None:
        super().__init__()
        cfg = case.model_config
        kind = cfg.get("type", "mlp")
        kwargs = {
            "out_dim": 5,
            "hidden_layers": int(cfg.get("hidden_layers", 4)),
            "hidden_dim": int(cfg.get("hidden_dim", 64)),
            "activation": cfg.get("activation", "tanh"),
        }
        if kind != "mlp":
            raise ValueError("shell2d currently supports model.type: mlp")
        self.net = MLP(in_dim=2, **kwargs)
        self.case = case
        scale = cfg.get("output_scale", [1.0, 1.0, 1.0, 0.2, 0.2])
        if len(scale) < 5:
            scale = list(scale) + [0.2] * (5 - len(scale))
        self.scale = torch.tensor(scale[:5], dtype=torch.float32)

    def reference_surface(self, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = xi[:, 0:1]
        eta = xi[:, 1:2]
        q0, kappa0 = self.case.reference(s)
        plane = self.case.geometry.get("fem_plane", "xy")
        if plane == "xz":
            r0 = torch.cat([q0[:, 0:1], eta, q0[:, 1:2]], dim=1)
        else:
            r0 = torch.cat([q0[:, 0:1], q0[:, 1:2], eta], dim=1)
        return r0, q0, kappa0

    def _apply_hard_edge_constraints(self, q_raw: torch.Tensor, sbar: torch.Tensor) -> torch.Tensor:
        """Embed prescribed edge values by one-dimensional Lagrange factors.

        The construction is applied component by component, so translations
        may be fixed at several sections while the two rotations remain free
        except where explicitly prescribed.
        """
        if not bool(self.case.geometry.get("hard_essential_constraints", False)):
            return q_raw
        constraints = self.case.geometry.get("edge_constraints", [])
        if not constraints:
            return q_raw
        names = ("U1", "U2", "U3", "UR1", "UR2")
        load_factor = float(getattr(self.case, "current_load_factor", 1.0))
        columns = []
        for j, name in enumerate(names):
            points: list[tuple[float, float]] = []
            for bc in constraints:
                dofs = bc.get("dofs", {})
                if name in dofs:
                    points.append((2.0 * float(bc["s"]) - 1.0, load_factor * float(dofs[name])))
            if not points:
                columns.append(q_raw[:, j : j + 1])
                continue
            particular = torch.zeros_like(sbar)
            vanishing = torch.ones_like(sbar)
            for i, (si, yi) in enumerate(points):
                basis = torch.ones_like(sbar)
                for k, (sk, _) in enumerate(points):
                    if k != i:
                        basis = basis * (sbar - sk) / (si - sk)
                particular = particular + yi * basis
                vanishing = vanishing * (sbar - si)
            columns.append(particular + vanishing * q_raw[:, j : j + 1])
        return torch.cat(columns, dim=1)

    def forward(self, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r0, q0, kappa0 = self.reference_surface(xi)
        width = float(self.case.geometry.get("width", 1.0))
        sbar = 2.0 * xi[:, 0:1] / self.case.length - 1.0
        ebar = 2.0 * xi[:, 1:2] / width
        scale = self.scale.to(device=xi.device, dtype=xi.dtype).reshape(1, 5)
        q_raw = self.net(torch.cat([sbar, ebar], dim=1)) * scale
        q = self._apply_hard_edge_constraints(q_raw, sbar)
        u = q[:, 0:3]
        theta = q[:, 3:5]
        return r0 + u, r0, q, q0, kappa0


def load_case(path: str | Path, device: torch.device) -> CaseDefinition:
    cfg = load_yaml(path)
    geom = cfg["geometry"]
    length = float(geom.get("length", 1.0))
    gtype = geom["type"]
    if gtype == "flat":
        ref = lambda s: flat_reference(s, length=length)
    elif gtype == "c_shape":
        angle = float(geom.get("angle_deg", 220.0))
        ref = lambda s: c_shape_reference(s, length=length, angle_deg=angle)
    elif gtype == "omega":
        omega = OmegaReference(length=length, height=float(geom.get("height", 0.55)), device=device)
        ref = omega
    elif gtype == "half_omega":
        omega = HalfOmegaReference(
            radius=float(geom.get("radius", 5.0)),
            leg_length=float(geom.get("leg_length", 2.0)),
            device=device,
        )
        length = float(geom.get("length", omega.length))
        ref = omega
    else:
        raise ValueError(f"Unknown geometry type: {gtype}")
    return CaseDefinition(
        case_id=cfg["case_id"],
        name=cfg["name"],
        length=length,
        material=cfg["material"],
        constraints=cfg.get("constraints", []),
        loads=cfg.get("loads", {}),
        training=cfg["training"],
        loss_weights=cfg["loss_weights"],
        model_config=cfg["model"],
        geometry=geom,
        reference=ref,
        output_dir=cfg.get("output_dir", f"results/raw/{cfg['case_id']}"),
        analysis_dim=cfg.get("analysis_dim", "centerline1d"),
    )
