from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    "configs/case01_flat_bending.yaml",
    "configs/case02_c_flattening.yaml",
    "configs/case03_omega_flattening.yaml",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forms", nargs="+", default=["strong", "weak"], choices=["strong", "weak"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    for config in CONFIGS:
        for form in args.forms:
            for seed in args.seeds:
                cmd = [sys.executable, str(root / "scripts" / "run_case.py"), "--config", str(root / config), "--form", form, "--seed", str(seed)]
                if args.device:
                    cmd += ["--device", args.device]
                subprocess.run(cmd, check=True, cwd=root)


if __name__ == "__main__":
    main()

