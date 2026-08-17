from __future__ import annotations

import numpy as np
import torch

from src.geometry.spline_utils import resample_polyline


def omega_table(length: float = 1.0, height: float = 0.55, n: int = 2001) -> dict[str, np.ndarray]:
    points = np.array(
        [
            [-0.50 * length, 0.0],
            [-0.35 * length, 0.35 * height],
            [-0.15 * length, height],
            [0.0, height],
            [0.15 * length, height],
            [0.35 * length, 0.35 * height],
            [0.50 * length, 0.0],
        ]
    )
    table = resample_polyline(points, n=n)
    table["s"] = table["s"] * length
    table["length"] = length
    return table


def half_omega_points(radius: float = 5.0, leg_length: float = 2.0, n: int = 2001) -> np.ndarray:
    a = radius * np.sin(np.pi / 3.0)
    line_n = max(8, int(0.08 * n))
    side_n = max(16, int(0.20 * n))
    center_n = max(32, n - 2 * line_n - 2 * side_n)

    points: list[tuple[float, float]] = []

    def append_line(p0, p1, count):
        start = 1 if points else 0
        for i in range(start, count + 1):
            t = i / count
            points.append((p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])))

    def append_arc(center, angle0, angle1, count):
        start = 1 if points else 0
        for i in range(start, count + 1):
            t = i / count
            angle = angle0 + t * (angle1 - angle0)
            points.append((center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)))

    append_line((-2.0 * a - leg_length, 0.0), (-2.0 * a, 0.0), line_n)
    append_arc((-2.0 * a, radius), -0.5 * np.pi, -np.pi / 6.0, side_n)
    append_arc((0.0, 0.0), 5.0 * np.pi / 6.0, np.pi / 6.0, center_n)
    append_arc((2.0 * a, radius), -5.0 * np.pi / 6.0, -0.5 * np.pi, side_n)
    append_line((2.0 * a, 0.0), (2.0 * a + leg_length, 0.0), line_n)
    return np.asarray(points, dtype=float)


def half_omega_table(radius: float = 5.0, leg_length: float = 2.0, n: int = 2001) -> dict[str, np.ndarray]:
    points = half_omega_points(radius=radius, leg_length=leg_length, n=n)
    table = resample_polyline(points, n=n)
    arc_length = np.zeros_like(table["s"])
    arc_length[:] = table["s"] * _polyline_length(points)
    table["s"] = arc_length
    table["length"] = float(arc_length[-1])
    return table


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def torch_interp(s: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    s1 = s.reshape(-1)
    idx = torch.searchsorted(xp, s1).clamp(1, xp.numel() - 1)
    x0, x1 = xp[idx - 1], xp[idx]
    y0, y1 = fp[idx - 1], fp[idx]
    w = (s1 - x0) / torch.clamp(x1 - x0, min=1e-12)
    return (y0 + w * (y1 - y0)).reshape(-1, 1)


class OmegaReference:
    def __init__(self, length: float = 1.0, height: float = 0.55, device: torch.device | None = None) -> None:
        table = omega_table(length=length, height=height)
        device = device or torch.device("cpu")
        self.s = torch.tensor(table["s"], dtype=torch.float32, device=device)
        self.x = torch.tensor(table["x"], dtype=torch.float32, device=device)
        self.y = torch.tensor(table["y"], dtype=torch.float32, device=device)
        self.theta = torch.tensor(table["theta"], dtype=torch.float32, device=device)
        self.kappa = torch.tensor(table["kappa"], dtype=torch.float32, device=device)

    def __call__(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q0 = torch.cat(
            [
                torch_interp(s, self.s, self.x),
                torch_interp(s, self.s, self.y),
                torch_interp(s, self.s, self.theta),
            ],
            dim=1,
        )
        kappa0 = torch_interp(s, self.s, self.kappa)
        return q0, kappa0


class HalfOmegaReference:
    def __init__(self, radius: float = 5.0, leg_length: float = 2.0, device: torch.device | None = None) -> None:
        table = half_omega_table(radius=radius, leg_length=leg_length)
        device = device or torch.device("cpu")
        self.length = float(table["length"])
        self.s = torch.tensor(table["s"], dtype=torch.float32, device=device)
        self.x = torch.tensor(table["x"], dtype=torch.float32, device=device)
        self.y = torch.tensor(table["y"], dtype=torch.float32, device=device)
        self.theta = torch.tensor(table["theta"], dtype=torch.float32, device=device)
        self.kappa = torch.tensor(table["kappa"], dtype=torch.float32, device=device)

    def __call__(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q0 = torch.cat(
            [
                torch_interp(s, self.s, self.x),
                torch_interp(s, self.s, self.y),
                torch_interp(s, self.s, self.theta),
            ],
            dim=1,
        )
        kappa0 = torch_interp(s, self.s, self.kappa)
        return q0, kappa0
