from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.training.cases import load_case
from src.training.trainer import train_case
from src.utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--form", choices=["strong", "weak"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action="store_true", help="Resume from run-dir/model.pt if it exists")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    case = load_case(args.config, device)
    run_dir = train_case(case, args.form, args.seed, device, resume=args.resume)
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
