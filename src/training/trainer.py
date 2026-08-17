from __future__ import annotations

import time
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm import trange

from src.evaluation.metrics import evaluate_model
from src.losses import strong_loss, weak_loss
from src.physics.plate5 import plate5_fields
from src.physics.reactions import end_reaction
from src.physics.surface_shell import surface_shell_fields
from src.physics.surface_reactions import mechanics_snapshot
from src.training.cases import CaseDefinition, ShellSurfacePINN, StripPINN
from src.utils.io import save_json, save_yaml
from src.utils.sampling import fixed_grid, sample_arc_length
from src.visualization import plot_loss, plot_shape


def loss_fn_for(form: str):
    if form == "weak":
        return weak_loss
    if form == "strong":
        return strong_loss
    raise ValueError("--form must be 'weak' or 'strong'")


def export_pinn_fem_format(case: CaseDefinition, shape: pd.DataFrame, run_dir: Path) -> None:
    width = float(case.geometry.get("width", 1.0))
    width_points = int(case.training.get("export_width_points", 1))
    plane = case.geometry.get("fem_plane", "xy")
    z_values = [0.0] if width_points <= 1 else [
        -0.5 * width + width * i / float(width_points - 1) for i in range(width_points)
    ]
    rows = []
    node_label = 1
    for _, row in shape.iterrows():
        for zeta in z_values:
            if plane == "xz":
                x0, y0, z0 = float(row["x0"]), zeta, float(row["y0"])
                x, y, z = float(row["x"]), zeta, float(row["y"])
            else:
                x0, y0, z0 = float(row["x0"]), float(row["y0"]), zeta
                x, y, z = float(row["x"]), float(row["y"]), zeta
            rows.append(
                {
                    "Instance": "PINN_CENTERLINE_EXTRUDED",
                    "NodeLabel": node_label,
                    "X0": x0,
                    "Y0": y0,
                    "Z0": z0,
                    "U1": x - x0,
                    "U2": y - y0,
                    "U3": z - z0,
                    "X": x,
                    "Y": y,
                    "Z": z,
                    "RF1": 0.0,
                    "RF2": 0.0,
                    "RF3": 0.0,
                    "RM1": 0.0,
                    "RM2": 0.0,
                    "RM3": 0.0,
                    "LE11": 0.0,
                    "LE22": 0.0,
                    "LE33": 0.0,
                    "LE12": 0.0,
                    "LE13": 0.0,
                    "LE23": 0.0,
                    "S11": 0.0,
                    "S22": 0.0,
                    "S33": 0.0,
                    "S12": 0.0,
                    "S13": 0.0,
                    "S23": 0.0,
                }
            )
            node_label += 1
    pd.DataFrame(rows).to_csv(run_dir / "predicted_fem_format.csv", index=False)


def sample_surface_points(n: int, case: CaseDefinition, device: torch.device) -> torch.Tensor:
    width = float(case.geometry.get("width", 1.0))
    s = torch.rand(n, 1, device=device) * float(case.length)
    eta = (torch.rand(n, 1, device=device) - 0.5) * width
    return torch.cat([s, eta], dim=1).requires_grad_(True)


def surface_eval_grid(case: CaseDefinition, device: torch.device) -> torch.Tensor:
    ns = int(case.training.get("eval_points", 401))
    nw = int(case.training.get("export_width_points", 21))
    width = float(case.geometry.get("width", 1.0))
    s = torch.linspace(0.0, float(case.length), ns, device=device)
    eta = torch.linspace(-0.5 * width, 0.5 * width, nw, device=device)
    ss, ee = torch.meshgrid(s, eta, indexing="ij")
    return torch.stack([ss.reshape(-1), ee.reshape(-1)], dim=1).requires_grad_(True)


def shell_fields_for_case(case: CaseDefinition, model: ShellSurfacePINN, xi: torch.Tensor) -> dict:
    if getattr(case, "analysis_dim", "centerline1d") == "plate5d":
        return plate5_fields(model, xi, case)
    return surface_shell_fields(model, xi, case)


def validation_snapshot(case: CaseDefinition, model: ShellSurfacePINN, device: torch.device, step: int, form: str, load_factor: float) -> dict:
    """Offline FEM validation for plotting only.

    This function is intentionally opt-in through ``training.validation_interval``.
    It does not contribute to the loss, does not back-propagate, and does not
    influence optimizer steps.
    """
    if getattr(case, "analysis_dim", "centerline1d") not in {"shell2d", "plate5d"}:
        return {}
    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    try:
        from compare_fem_pinn import find_final_fem_csv, resample_fem_surface_to_pinn
    except Exception as exc:
        return {"step": step, "form": form, "load_factor": load_factor, "validation_error": str(exc)}

    was_training = model.training
    model.eval()
    try:
        fem_csv = find_final_fem_csv(case.case_id, root / "fem" / "results")
        fem_df = pd.read_csv(fem_csv)
        xi = surface_eval_grid(case, device)
        with torch.no_grad():
            r, r0, q, _, _ = model(xi)
        xi_cpu = xi.detach().cpu()
        r_cpu = r.detach().cpu()
        r0_cpu = r0.detach().cpu()
        q_cpu = q.detach().cpu()
        pinn = pd.DataFrame(
            {
                "s": xi_cpu[:, 0].numpy(),
                "eta": xi_cpu[:, 1].numpy(),
                "X0": r0_cpu[:, 0].numpy(),
                "Y0": r0_cpu[:, 1].numpy(),
                "Z0": r0_cpu[:, 2].numpy(),
                "X": r_cpu[:, 0].numpy(),
                "Y": r_cpu[:, 1].numpy(),
                "Z": r_cpu[:, 2].numpy(),
                "U1": (r_cpu[:, 0] - r0_cpu[:, 0]).numpy(),
                "U2": (r_cpu[:, 1] - r0_cpu[:, 1]).numpy(),
                "U3": (r_cpu[:, 2] - r0_cpu[:, 2]).numpy(),
                "theta1": q_cpu[:, 3].numpy(),
                "theta2": q_cpu[:, 4].numpy(),
            }
        )
        fem = resample_fem_surface_to_pinn(fem_df, pinn, case.case_id)
        pred_xyz = torch.tensor(pinn[["X", "Y", "Z"]].to_numpy(), dtype=torch.float64)
        fem_xyz = torch.tensor(fem[["X", "Y", "Z"]].to_numpy(), dtype=torch.float64)
        pred_u = torch.tensor(pinn[["U1", "U2", "U3"]].to_numpy(), dtype=torch.float64)
        fem_u = torch.tensor(fem[["U1", "U2", "U3"]].to_numpy(), dtype=torch.float64)
        component = {}
        for col in ["U1", "U2", "U3"]:
            a = torch.tensor(pinn[col].to_numpy(), dtype=torch.float64)
            b = torch.tensor(fem[col].to_numpy(), dtype=torch.float64)
            component[f"validation_{col.lower()}_l2"] = float(torch.linalg.norm(a - b) / torch.clamp(torch.linalg.norm(b), min=1.0e-12))
        out = {
            "step": step,
            "form": form,
            "load_factor": load_factor,
            "validation_shape_error": float(torch.sqrt(torch.mean(torch.sum((pred_xyz - fem_xyz) ** 2, dim=1))) / max(float(case.length), 1.0e-12)),
            "validation_displacement_l2": float(torch.linalg.norm(pred_u - fem_u) / torch.clamp(torch.linalg.norm(fem_u), min=1.0e-12)),
            "validation_theta1_l2": float("nan"),
            "validation_theta2_l2": float("nan"),
        }
        out.update(component)
        return out
    except Exception as exc:
        return {"step": step, "form": form, "load_factor": load_factor, "validation_error": str(exc)}
    finally:
        if was_training:
            model.train()


def export_surface_results(case: CaseDefinition, model: ShellSurfacePINN, run_dir: Path, device: torch.device) -> None:
    xi = surface_eval_grid(case, device)
    fields = shell_fields_for_case(case, model, xi)
    r = fields["r"].detach().cpu()
    r0 = fields["r0"].detach().cpu()
    mem = fields["membrane"].detach().cpu()
    bend = fields["bending"].detach().cpu()
    shear = fields["shear"].detach().cpu()
    stress = fields["stress"].detach().cpu()
    nres = fields["membrane_resultant"].detach().cpu()
    qres = fields["shear_resultant"].detach().cpu()
    moment = fields["moment"].detach().cpu()
    density = fields["density"].detach().cpu()
    q = fields["q"].detach().cpu()
    xi_cpu = xi.detach().cpu()

    surface = pd.DataFrame(
        {
            "s": xi_cpu[:, 0].numpy(),
            "eta": xi_cpu[:, 1].numpy(),
            "X0": r0[:, 0].numpy(),
            "Y0": r0[:, 1].numpy(),
            "Z0": r0[:, 2].numpy(),
            "X": r[:, 0].numpy(),
            "Y": r[:, 1].numpy(),
            "Z": r[:, 2].numpy(),
            "U1": (r[:, 0] - r0[:, 0]).numpy(),
            "U2": (r[:, 1] - r0[:, 1]).numpy(),
            "U3": (r[:, 2] - r0[:, 2]).numpy(),
            "theta1": q[:, 3].numpy(),
            "theta2": q[:, 4].numpy(),
            "LE11": mem[:, 0].numpy(),
            "LE22": mem[:, 1].numpy(),
            "LE12": mem[:, 2].numpy(),
            "K11": bend[:, 0].numpy(),
            "K22": bend[:, 1].numpy(),
            "K12": bend[:, 2].numpy(),
            "G1": shear[:, 0].numpy(),
            "G2": shear[:, 1].numpy(),
            "S11": stress[:, 0].numpy(),
            "S22": stress[:, 1].numpy(),
            "S12": stress[:, 2].numpy(),
            "N11": nres[:, 0].numpy(),
            "N22": nres[:, 1].numpy(),
            "N12": nres[:, 2].numpy(),
            "Q1": qres[:, 0].numpy(),
            "Q2": qres[:, 1].numpy(),
            "M11": moment[:, 0].numpy(),
            "M22": moment[:, 1].numpy(),
            "M12": moment[:, 2].numpy(),
            "energy_density": density.reshape(-1).numpy(),
        }
    )
    surface.to_csv(run_dir / "predicted_surface.csv", index=False)

    fem = surface.copy()
    fem.insert(0, "NodeLabel", range(1, len(fem) + 1))
    fem.insert(0, "Instance", "PINN_SHELL2D")
    for col in ["RF1", "RF2", "RF3", "RM1", "RM2", "RM3", "LE33", "LE13", "LE23", "S33", "S13", "S23"]:
        fem[col] = 0.0
    fem.to_csv(run_dir / "predicted_fem_format.csv", index=False)

    center = surface.loc[surface["eta"].abs() == surface["eta"].abs().min()].copy()
    if case.geometry.get("fem_plane", "xy") == "xz":
        x, y, x0, y0 = center["X"], center["Z"], center["X0"], center["Z0"]
    else:
        x, y, x0, y0 = center["X"], center["Y"], center["X0"], center["Y0"]
    pd.DataFrame(
        {
            "s": center["s"].to_numpy(),
            "x": x.to_numpy(),
            "y": y.to_numpy(),
            "theta": 0.0,
            "x0": x0.to_numpy(),
            "y0": y0.to_numpy(),
            "theta0": 0.0,
        }
    ).to_csv(run_dir / "predicted_shape.csv", index=False)
    surface[
        [
            "s",
            "eta",
            "LE11",
            "LE22",
            "LE12",
            "K11",
            "K22",
            "K12",
            "G1",
            "G2",
            "S11",
            "S22",
            "S12",
            "N11",
            "N22",
            "N12",
            "Q1",
            "Q2",
            "M11",
            "M22",
            "M12",
            "energy_density",
        ]
    ].to_csv(run_dir / "predicted_strain.csv", index=False)


def train_case(case: CaseDefinition, form: str, seed: int, device: torch.device, resume: bool = False) -> Path:
    run_dir = Path(case.output_dir) / f"{form}_seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    is_shell2d = getattr(case, "analysis_dim", "centerline1d") in {"shell2d", "plate5d"}
    model = (ShellSurfacePINN(case) if is_shell2d else StripPINN(case)).to(device)
    checkpoint_path = run_dir / "model.pt"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        print(f"resumed model: {checkpoint_path}", flush=True)
    n = int(case.training["collocation_points"])
    if is_shell2d:
        s = sample_surface_points(n, case, device)
        pd.DataFrame(
            {"s": s.detach().cpu()[:, 0].numpy(), "eta": s.detach().cpu()[:, 1].numpy()}
        ).to_csv(run_dir / "collocation_points.csv", index=False)
    else:
        s = sample_arc_length(n, case.length, case.training.get("sampling", "uniform"), device)
        pd.DataFrame({"s": s.detach().cpu().reshape(-1).numpy()}).to_csv(run_dir / "collocation_points.csv", index=False)

    print(
        f"start training: case={case.case_id}, form={form}, analysis_dim={case.analysis_dim}, "
        f"collocation_points={n}, device={device}",
        flush=True,
    )
    criterion = loss_fn_for(form)
    history: list[dict] = []
    validation_history: list[dict] = []
    mechanics_history: list[dict] = []
    validation_interval = int(case.training.get("validation_interval", 0))
    start = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=float(case.training.get("adam_lr", 1e-3)))
    epochs = int(case.training.get("adam_epochs", 1000))
    continuation = case.training.get("continuation", [1.0])
    if not isinstance(continuation, list) or not continuation:
        continuation = [1.0]
    continuation = [float(v) for v in continuation]
    stage_epochs = max(1, epochs // len(continuation))
    global_step = 0
    for stage_id, load_factor in enumerate(continuation):
        case.current_load_factor = load_factor
        print(
            f"[{case.case_id} {form} seed={seed}] continuation stage "
            f"{stage_id + 1}/{len(continuation)}: load_factor={load_factor:.3f}",
            flush=True,
        )
        for local_step in trange(stage_epochs, desc=f"{case.case_id} {form} lf={load_factor:.2f}", leave=False):
            opt.zero_grad(set_to_none=True)
            loss, logs = criterion(model, s, case, case.loss_weights)
            loss.backward()
            opt.step()
            if global_step % max(1, epochs // 200) == 0 or (stage_id == len(continuation) - 1 and local_step == stage_epochs - 1):
                logs["step"] = global_step
                logs["stage"] = "adam"
                logs["load_factor"] = load_factor
                history.append(logs)
                if validation_interval > 0 and (global_step % validation_interval == 0 or (stage_id == len(continuation) - 1 and local_step == stage_epochs - 1)):
                    validation_history.append(validation_snapshot(case, model, device, global_step, form, load_factor))
                if global_step % max(1, epochs // 20) == 0 or (stage_id == len(continuation) - 1 and local_step == stage_epochs - 1):
                    terms = ", ".join(f"{k}={v:.3e}" for k, v in logs.items() if isinstance(v, float))
                    print(f"[{case.case_id} {form} seed={seed}] adam step {global_step:05d}: {terms}", flush=True)
            global_step += 1
        if is_shell2d:
            snap = mechanics_snapshot(model, case, device, load_factor)
            snap.update({"step": global_step - 1, "stage": "adam", "form": form})
            mechanics_history.append(snap)
    case.current_load_factor = 1.0

    lbfgs_iter = int(case.training.get("lbfgs_max_iter", 0))
    if lbfgs_iter > 0:
        opt2 = torch.optim.LBFGS(
            model.parameters(),
            max_iter=lbfgs_iter,
            line_search_fn="strong_wolfe",
            tolerance_grad=1e-9,
            tolerance_change=1e-11,
        )
        calls = 0

        def closure():
            nonlocal calls
            opt2.zero_grad(set_to_none=True)
            loss, logs = criterion(model, s, case, case.loss_weights)
            loss.backward()
            if calls % 10 == 0:
                logs["step"] = global_step + calls
                logs["stage"] = "lbfgs"
                logs["load_factor"] = 1.0
                history.append(logs)
                if validation_interval > 0:
                    validation_history.append(validation_snapshot(case, model, device, global_step + calls, form, 1.0))
                terms = ", ".join(f"{k}={v:.3e}" for k, v in logs.items() if isinstance(v, float))
                print(f"[{case.case_id} {form} seed={seed}] lbfgs call {calls:04d}: {terms}", flush=True)
            calls += 1
            return loss

        opt2.step(closure)
        if is_shell2d:
            snap = mechanics_snapshot(model, case, device, 1.0)
            snap.update({"step": global_step + calls, "stage": "lbfgs", "form": form})
            mechanics_history.append(snap)

    runtime = time.time() - start
    if is_shell2d:
        export_surface_results(case, model, run_dir, device)
        xi_eval = surface_eval_grid(case, device)
        fields = shell_fields_for_case(case, model, xi_eval)
        metrics = {
            "energy": float((fields["density"].mean() * float(case.length) * float(case.geometry.get("width", 1.0))).detach().cpu()),
            "membrane_rms": float(fields["membrane"].pow(2).mean().sqrt().detach().cpu()),
            "bending_rms": float(fields["bending"].pow(2).mean().sqrt().detach().cpu()),
            "shear_rms": float(fields["shear"].pow(2).mean().sqrt().detach().cpu()),
            "s11_rms": float(fields["stress"][:, 0].pow(2).mean().sqrt().detach().cpu()),
            "s22_rms": float(fields["stress"][:, 1].pow(2).mean().sqrt().detach().cpu()),
            "s12_rms": float(fields["stress"][:, 2].pow(2).mean().sqrt().detach().cpu()),
        }
        arrays = None
        eval_s = None
    else:
        eval_s = fixed_grid(int(case.training.get("eval_points", 401)), case.length, device)
        metrics, arrays = evaluate_model(model, case, eval_s)
    metrics["runtime_sec"] = runtime
    metrics["form"] = form
    metrics["seed"] = seed
    metrics["case_id"] = case.case_id
    try:
        if is_shell2d:
            raise RuntimeError("centerline reactions are not defined for shell2d")
        metrics["left_Rx"] = end_reaction(arrays["q"], eval_s, arrays["kappa0"], case.material, "left")["Rx"]
        metrics["left_Ry"] = end_reaction(arrays["q"], eval_s, arrays["kappa0"], case.material, "left")["Ry"]
        metrics["right_Rx"] = end_reaction(arrays["q"], eval_s, arrays["kappa0"], case.material, "right")["Rx"]
        metrics["right_Ry"] = end_reaction(arrays["q"], eval_s, arrays["kappa0"], case.material, "right")["Ry"]
    except RuntimeError:
        pass

    torch.save({"model": model.state_dict(), "case_id": case.case_id, "form": form, "seed": seed}, run_dir / "model.pt")
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
    if validation_history:
        pd.DataFrame(validation_history).to_csv(run_dir / "validation_history.csv", index=False)
    if mechanics_history:
        pd.DataFrame(mechanics_history).to_csv(run_dir / "mechanics_history.csv", index=False)
    if is_shell2d:
        save_json(metrics, run_dir / "metrics.json")
        save_yaml(
            {
                "case_id": case.case_id,
                "name": case.name,
                "analysis_dim": case.analysis_dim,
                "material": case.material,
                "constraints": case.constraints,
                "loads": case.loads,
                "training": case.training,
                "loss_weights": case.loss_weights,
                "model": case.model_config,
                "geometry": case.geometry,
            },
            run_dir / "config_used.yaml",
        )
        plot_shape(run_dir / "predicted_shape.csv", run_dir / "shape_comparison.png", f"{case.name} ({form})")
        plot_loss(run_dir / "loss_history.csv", run_dir / "loss_curve.png", f"{case.case_id} {form}")
        return run_dir

    shape = pd.DataFrame(
        {
            "s": eval_s.detach().cpu().reshape(-1).numpy(),
            "x": arrays["q"][:, 0].detach().cpu().numpy(),
            "y": arrays["q"][:, 1].detach().cpu().numpy(),
            "theta": arrays["q"][:, 2].detach().cpu().numpy(),
            "x0": arrays["q0"][:, 0].detach().cpu().numpy(),
            "y0": arrays["q0"][:, 1].detach().cpu().numpy(),
            "theta0": arrays["q0"][:, 2].detach().cpu().numpy(),
        }
    )
    shape.to_csv(run_dir / "predicted_shape.csv", index=False)
    export_pinn_fem_format(case, shape, run_dir)
    pd.DataFrame(
        {
            "s": eval_s.detach().cpu().reshape(-1).numpy(),
            "epsilon": arrays["epsilon"].detach().cpu().reshape(-1).numpy(),
            "gamma": arrays["gamma"].detach().cpu().reshape(-1).numpy(),
            "kappa": arrays["kappa"].detach().cpu().reshape(-1).numpy(),
            "kappa0": arrays["kappa0"].detach().cpu().reshape(-1).numpy(),
            "bend": arrays["bend"].detach().cpu().reshape(-1).numpy(),
            "N": arrays["N"].detach().cpu().reshape(-1).numpy(),
            "Q": arrays["Q"].detach().cpu().reshape(-1).numpy(),
            "M": arrays["moment"].detach().cpu().reshape(-1).numpy(),
            "Fx": arrays["force"][:, 0].detach().cpu().numpy(),
            "Fy": arrays["force"][:, 1].detach().cpu().numpy(),
            "residual_norm": arrays["residual"].pow(2).sum(dim=1).sqrt().detach().cpu().numpy(),
            "Wb": (0.5 * float(case.material["EI"]) * arrays["bend"].pow(2)).detach().cpu().reshape(-1).numpy(),
        }
    ).to_csv(run_dir / "predicted_strain.csv", index=False)
    save_json(metrics, run_dir / "metrics.json")
    save_yaml(
        {
            "case_id": case.case_id,
            "name": case.name,
            "analysis_dim": case.analysis_dim,
            "material": case.material,
            "constraints": case.constraints,
            "loads": case.loads,
            "training": case.training,
            "loss_weights": case.loss_weights,
            "model": case.model_config,
            "geometry": case.geometry,
        },
        run_dir / "config_used.yaml",
    )
    plot_shape(run_dir / "predicted_shape.csv", run_dir / "shape_comparison.png", f"{case.name} ({form})")
    plot_loss(run_dir / "loss_history.csv", run_dir / "loss_curve.png", f"{case.case_id} {form}")
    return run_dir
