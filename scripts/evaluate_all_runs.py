from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CASE_CONFIGS = {
    "case01_flat_bending": ROOT / "configs" / "case01_flat_bending.yaml",
    "case02_c_flattening": ROOT / "configs" / "case02_c_flattening.yaml",
    "case03_omega_flattening": ROOT / "configs" / "case03_omega_flattening.yaml",
}
RUN_RE = re.compile(r"^(strong|weak)_seed(\d+)$")

RUN_COLUMNS = [
    "case_id",
    "form",
    "seed",
    "run_dir",
    "has_model",
    "has_prediction",
    "has_comparison",
    "status",
    "error_message",
    "runtime_sec",
    "energy",
    "membrane_rms",
    "bending_rms",
    "shear_rms",
    "s11_rms",
    "s22_rms",
    "s12_rms",
    "centerline_relative_l2",
    "centerline_displacement_relative_l2",
    "centerline_normalized_rmse",
    "centerline_rmse",
    "max_point_error",
    "surface_displacement_vector_relative_l2",
    "surface_displacement_vector_rmse",
    "surface_LE11_relative_l2",
    "surface_LE22_relative_l2",
    "surface_LE12_relative_l2",
    "surface_S11_relative_l2",
    "surface_S22_relative_l2",
    "surface_S12_relative_l2",
    "surface_LE11_rmse",
    "surface_LE22_rmse",
    "surface_LE12_rmse",
    "surface_S11_rmse",
    "surface_S22_rmse",
    "surface_S12_rmse",
]

SUMMARY_KEYS = [
    "runtime_sec",
    "centerline_relative_l2",
    "centerline_displacement_relative_l2",
    "centerline_normalized_rmse",
    "surface_displacement_vector_relative_l2",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_int_set(spec: str | None) -> set[int] | None:
    if not spec:
        return None
    values: set[int] = set()
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            a, b = token.split("-", 1)
            values.update(range(int(a), int(b) + 1))
        else:
            values.add(int(token))
    return values


def discover_runs(raw_root: Path, cases: set[str] | None, forms: set[str] | None, seeds: set[int] | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for case_dir in sorted(raw_root.glob("case*")):
        if not case_dir.is_dir():
            continue
        case_id = case_dir.name
        if cases and case_id not in cases:
            continue
        for run_dir in sorted(case_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            match = RUN_RE.match(run_dir.name)
            if not match:
                continue
            form = match.group(1)
            seed = int(match.group(2))
            if forms and form not in forms:
                continue
            if seeds and seed not in seeds:
                continue
            runs.append({"case_id": case_id, "form": form, "seed": seed, "run_dir": run_dir})
    return runs


def has_prediction(run_dir: Path) -> bool:
    return (run_dir / "predicted_surface.csv").exists() or (run_dir / "predicted_shape.csv").exists()


def run_command(cmd: list[str], dry_run: bool) -> None:
    print(" ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def export_prediction(case_id: str, run_dir: Path, device: str, python_exe: str, dry_run: bool) -> None:
    config = CASE_CONFIGS.get(case_id)
    if config is None:
        raise FileNotFoundError(f"No config registered for {case_id}")
    run_command(
        [
            python_exe,
            "scripts/export_case.py",
            "--config",
            str(config.relative_to(ROOT)),
            "--run-dir",
            str(run_dir.relative_to(ROOT)),
            "--device",
            device,
        ],
        dry_run,
    )


def compare_run(case_id: str, run_dir: Path, fem_root: Path, python_exe: str, dry_run: bool) -> None:
    run_command(
        [
            python_exe,
            "scripts/compare_fem_pinn.py",
            "--case",
            case_id,
            "--run-dir",
            str(run_dir.relative_to(ROOT)),
            "--fem-root",
            str(fem_root),
        ],
        dry_run,
    )


def row_for_run(case_id: str, form: str, seed: int, run_dir: Path, status: str, error_message: str = "") -> dict[str, Any]:
    metrics = load_json(run_dir / "metrics.json")
    comparison = load_json(run_dir / "fem_compare" / "comparison_metrics.json")
    row: dict[str, Any] = {
        "case_id": case_id,
        "form": form,
        "seed": seed,
        "run_dir": str(run_dir.relative_to(ROOT)),
        "has_model": (run_dir / "model.pt").exists(),
        "has_prediction": has_prediction(run_dir),
        "has_comparison": bool(comparison),
        "status": status,
        "error_message": error_message,
    }
    row.update(metrics)
    row.update(comparison)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        columns = seen
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["case_id"]), str(row["form"]))].append(row)
    summary: list[dict[str, Any]] = []
    for (case_id, form), group in sorted(groups.items()):
        out: dict[str, Any] = {
            "case_id": case_id,
            "form": form,
            "runs": len(group),
            "comparisons": sum(1 for row in group if row.get("has_comparison")),
            "successful": sum(1 for row in group if row.get("status") == "ok"),
        }
        for key in SUMMARY_KEYS:
            vals = [v for row in group if (v := as_float(row.get(key))) is not None]
            if vals:
                out[f"{key}_mean"] = mean(vals)
                out[f"{key}_std"] = pstdev(vals) if len(vals) > 1 else 0.0
                out[f"{key}_min"] = min(vals)
                out[f"{key}_max"] = max(vals)
        summary.append(out)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="results/raw")
    parser.add_argument("--fem-root", default="fem/results")
    parser.add_argument("--out-dir", default="results/evaluation")
    parser.add_argument("--case", action="append", help="Case id. Can be passed more than once.")
    parser.add_argument("--form", action="append", choices=["strong", "weak"], help="Form filter. Can be passed more than once.")
    parser.add_argument("--seeds", default=None, help="Seed filter such as 0,1,2 or 0-10.")
    parser.add_argument("--device", default="cpu", help="Device used only when export_case.py must be called.")
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable used for export_case.py and compare_fem_pinn.py.")
    parser.add_argument("--force", action="store_true", help="Re-run comparison even if fem_compare/comparison_metrics.json exists.")
    parser.add_argument("--refresh-export", action="store_true", help="Re-export prediction fields from model.pt before comparison.")
    parser.add_argument("--skip-compare", action="store_true", help="Only write summary tables from existing metrics/comparisons.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    fem_root = Path(args.fem_root)
    if not raw_root.is_absolute():
        raw_root = ROOT / raw_root
    cases = set(args.case) if args.case else None
    forms = set(args.form) if args.form else None
    seeds = parse_int_set(args.seeds)
    runs = discover_runs(raw_root, cases, forms, seeds)
    print(f"discovered runs: {len(runs)}", flush=True)

    rows: list[dict[str, Any]] = []
    for item in runs:
        case_id = item["case_id"]
        form = item["form"]
        seed = item["seed"]
        run_dir = item["run_dir"]
        status = "ok"
        error_message = ""
        try:
            if not args.skip_compare:
                if args.refresh_export or not has_prediction(run_dir):
                    export_prediction(case_id, run_dir, args.device, args.python_exe, args.dry_run)
                comparison_path = run_dir / "fem_compare" / "comparison_metrics.json"
                if args.force or not comparison_path.exists():
                    compare_run(case_id, run_dir, fem_root, args.python_exe, args.dry_run)
        except Exception as exc:
            status = "failed"
            error_message = str(exc)
            print(f"failed: {run_dir}: {error_message}", flush=True)
        rows.append(row_for_run(case_id, form, seed, run_dir, status, error_message))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    run_csv = out_dir / "all_run_metrics.csv"
    summary_csv = out_dir / "summary_by_case_form.csv"
    write_csv(run_csv, rows, RUN_COLUMNS)
    write_csv(summary_csv, summarize(rows))
    with (out_dir / "all_run_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print(f"saved: {run_csv.relative_to(ROOT)}")
    print(f"saved: {summary_csv.relative_to(ROOT)}")
    print(f"saved: {(out_dir / 'all_run_metrics.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
