from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def resample_polyline(points: np.ndarray, n: int = 2001) -> dict[str, np.ndarray]:
    points = np.asarray(points, dtype=float)
    chord = np.linalg.norm(np.diff(points, axis=0), axis=1)
    t = np.concatenate([[0.0], np.cumsum(chord)])
    t = t / t[-1]
    cs_x = CubicSpline(t, points[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, points[:, 1], bc_type="natural")

    dense_t = np.linspace(0.0, 1.0, max(10 * n, 1000))
    dx = cs_x(dense_t, 1)
    dy = cs_y(dense_t, 1)
    speed = np.sqrt(dx * dx + dy * dy)
    arc = np.concatenate([[0.0], np.cumsum(0.5 * (speed[:-1] + speed[1:]) * np.diff(dense_t))])
    arc = arc / arc[-1]

    target = np.linspace(0.0, 1.0, n)
    t_resampled = np.interp(target, arc, dense_t)
    x = cs_x(t_resampled)
    y = cs_y(t_resampled)
    dx = cs_x(t_resampled, 1)
    dy = cs_y(t_resampled, 1)
    ddx = cs_x(t_resampled, 2)
    ddy = cs_y(t_resampled, 2)
    theta = np.unwrap(np.arctan2(dy, dx))
    ds_dt = np.sqrt(dx * dx + dy * dy)
    kappa = (dx * ddy - dy * ddx) / np.maximum(ds_dt**3, 1e-12)
    length = float(arc[-1])
    return {"s": target, "x": x, "y": y, "theta": theta, "kappa": kappa, "length": length}

