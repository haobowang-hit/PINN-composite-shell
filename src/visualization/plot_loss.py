from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_loss(history_csv: str | Path, out_path: str | Path, title: str) -> None:
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.semilogy(df["step"], df["loss"].clip(lower=1e-16), lw=1.8)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title(title)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

