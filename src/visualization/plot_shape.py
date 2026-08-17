from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_shape(shape_csv: str | Path, out_path: str | Path, title: str) -> None:
    df = pd.read_csv(shape_csv)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(df["x0"], df["y0"], "--", color="0.55", lw=1.5, label="initial")
    ax.plot(df["x"], df["y"], color="#1f77b4", lw=2.0, label="PINN")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

