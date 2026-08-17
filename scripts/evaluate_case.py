from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Case id prefix, e.g. case01_flat_bending")
    args = parser.parse_args()
    rows = []
    root = Path("results/raw") / args.case
    for path in root.glob("*/metrics.json"):
        with path.open("r", encoding="utf-8") as f:
            row = json.load(f)
        row["run_dir"] = str(path.parent)
        rows.append(row)
    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"no metrics found under {root}")
        return
    out = out_dir / f"{args.case}_metrics.csv"
    pd.DataFrame(rows).sort_values(["form", "seed"]).to_csv(out, index=False)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()

