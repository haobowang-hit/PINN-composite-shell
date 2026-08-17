# Data-Free Physics-Informed Neural Modeling of Elastic Flattening in Curved Deployable Strips

## 1. Project Overview

This project develops a fully AI-assisted workflow for a small research demo on **data-free physics-informed neural modeling of elastic deformation and flattening in curved deployable strips**.

The goal is to reproduce the general logic of strong-form/weak-form PINN benchmarking, but with a simpler mechanical model and application-oriented deployable structures. The work does **not** use finite-element data for training. The neural model is trained only by physical losses, including equilibrium residuals, elastic energy, geometric constraints, and boundary conditions. FEM is used only as an independent reference for validation.

The project covers six forward problems:

1. Flat strip bending under tip force.
2. Flat strip bending under prescribed end displacement.
3. Symmetric flattening of a C-shaped strip.
4. Asymmetric flattening of a C-shaped strip.
5. Vertical flattening of an Ω-shaped strip.
6. Constrained full flattening of an Ω-shaped strip.

For every case, both **strong-form PINN** and **weak-form PINN** will be implemented, trained, evaluated, and compared.

The final deliverables include:

* A working PyTorch codebase.
* Six reproducible benchmark cases.
* Strong-form vs weak-form comparison.
* FEM validation results.
* Publication-ready figures.
* A short manuscript draft.
* Supplementary material and reproducibility scripts.

---

## 2. Scientific Positioning

### 2.1 Core Idea

The central scientific question is:

> Can a data-free physics-informed neural model solve elastic deformation and flattening problems of curved deployable strips without using FEM data for training?

The model is intentionally simple:

* No viscoelasticity.
* No thermal effect.
* No shape-memory constitutive model.
* No inverse identification.
* No contact.
* No UMAT.
* No FEM training data.

Instead, the work focuses on a clean and controllable mechanics problem:

> linear elastic material + geometrically nonlinear strip deformation + data-free PINN solver.

### 2.2 Intended Contribution

The expected contributions are:

1. A data-free physics-informed neural solver for elastic deformation of flat and curved strips.
2. A unified elastica/Reissner-beam formulation for flat bending, C-shaped flattening, and Ω-shaped flattening.
3. A systematic strong-form vs weak-form comparison across six representative forward problems.
4. Independent FEM validation showing where weak-form training is more robust than strong-form training.
5. A fully AI-assisted research pipeline covering task design, model implementation, evaluation, visualization, and manuscript writing.

### 2.3 Paper Title Options

Recommended title:

**Data-Free Physics-Informed Neural Modeling of Elastic Flattening in Curved Deployable Strips**

Alternative titles:

* **Strong and Weak Form Physics-Informed Neural Solvers for Elastic Flattening of Curved Strips**
* **Physics-Informed Neural Modeling of Curved Strip Flattening without Training Data**
* **A Data-Free PINN Benchmark for Elastic Bending and Flattening of Deployable Strips**

---

## 3. Mechanical Model

## 3.1 Geometry

Each structure is represented as a planar strip centerline parameterized by arc length:

[
s \in [0,L]
]

The deformed centerline is described by:

[
\mathbf{r}(s) = [x(s), y(s)]
]

and the section rotation angle is:

[
\theta(s)
]

The neural network predicts:

[
[x(s), y(s), \theta(s)] = \mathcal{N}_{\tau}(s)
]

where (\tau) denotes trainable neural-network parameters.

---

## 3.2 Strain Measures

Use a geometrically nonlinear planar Reissner-beam model.

The tangent vector is:

[
\mathbf{r}'(s) = [x'(s), y'(s)]
]

The local director is:

[
\mathbf{d}_1 = [\cos \theta, \sin \theta]
]

and the transverse director is:

[
\mathbf{d}_2 = [-\sin \theta, \cos \theta]
]

Define axial strain:

[
\varepsilon(s) = \mathbf{r}'(s) \cdot \mathbf{d}_1 - 1
]

Define shear strain:

[
\gamma(s) = \mathbf{r}'(s) \cdot \mathbf{d}_2
]

Define curvature:

[
\kappa(s) = \theta'(s)
]

The initial curvature is:

[
\kappa_0(s)
]

Therefore the bending strain is:

[
\kappa(s)-\kappa_0(s)
]

---

## 3.3 Constitutive Model

Use the simplest elastic model:

[
N = EA \varepsilon
]

[
Q = kGA \gamma
]

[
M = EI(\kappa-\kappa_0)
]

where:

* (N) is axial force.
* (Q) is shear force.
* (M) is bending moment.
* (E) is Young’s modulus.
* (A) is cross-sectional area.
* (I) is second moment of area.
* (G) is shear modulus.
* (k) is shear correction factor.

For nondimensional tests, use:

[
L=1,\quad EI=1,\quad EA=100,\quad kGA=100
]

This avoids excessive numerical stiffness while still approximating a nearly inextensible strip.

---

## 4. Strong Form and Weak Form

## 4.1 Weak-Form PINN

The weak-form model minimizes the total potential energy:

[
\Pi =
\int_0^L
\left[
\frac{1}{2}EA\varepsilon^2
+
\frac{1}{2}kGA\gamma^2
+
\frac{1}{2}EI(\kappa-\kappa_0)^2
\right] ds
----------

W_{\mathrm{ext}}
]

The weak-form loss is:

[
\mathcal{L}_{\mathrm{weak}}
===========================

\Pi
+
\lambda_{\mathrm{bc}}\mathcal{L}*{\mathrm{bc}}
+
\lambda*{\mathrm{sym}}\mathcal{L}*{\mathrm{sym}}
+
\lambda*{\mathrm{reg}}\mathcal{L}_{\mathrm{reg}}
]

where:

* (\Pi) is total potential energy.
* (\mathcal{L}_{\mathrm{bc}}) enforces displacement and rotation boundary conditions.
* (\mathcal{L}_{\mathrm{sym}}) enforces symmetry when applicable.
* (\mathcal{L}_{\mathrm{reg}}) prevents nonphysical oscillations.

For displacement-controlled problems, (W_{\mathrm{ext}}) can be omitted because the external work is replaced by prescribed boundary constraints.

---

## 4.2 Strong-Form PINN

The strong-form model minimizes equilibrium residuals.

Instead of hand-deriving every equation, the recommended implementation is to define the energy density:

[
\mathcal{W}(q,q',s)
===================

\frac{1}{2}EA\varepsilon^2
+
\frac{1}{2}kGA\gamma^2
+
\frac{1}{2}EI(\theta'-\kappa_0)^2
]

where:

[
q(s) = [x(s),y(s),\theta(s)]
]

The Euler–Lagrange residual is:

[
\mathbf{R}_{\mathrm{EL}}
========================

\frac{d}{ds}
\left(
\frac{\partial \mathcal{W}}{\partial q'}
\right)
-------

\frac{\partial \mathcal{W}}{\partial q}
]

The strong-form loss is:

[
\mathcal{L}_{\mathrm{strong}}
=============================

\frac{1}{N_c}
\sum_{i=1}^{N_c}
\left|
\mathbf{R}*{\mathrm{EL}}(s_i)
\right|^2
+
\lambda*{\mathrm{bc}}\mathcal{L}*{\mathrm{bc}}
+
\lambda*{\mathrm{nbc}}\mathcal{L}_{\mathrm{nbc}}
]

where:

* (N_c) is the number of collocation points.
* (\mathcal{L}_{\mathrm{nbc}}) enforces natural boundary conditions when needed.

This implementation allows strong-form residuals to be generated automatically using PyTorch autograd.

---

## 4.3 Comparison Principle

For every benchmark case:

* Use the same geometry.
* Use the same material parameters.
* Use the same neural architecture.
* Use the same collocation points.
* Use the same optimizer settings.
* Train strong-form and weak-form models independently.
* Compare accuracy, convergence, training time, robustness, and residual distribution.

Expected trend:

* Strong form may work for simple bending cases.
* Weak form should be more stable for C-shaped and Ω-shaped flattening.
* Strong form may become sensitive to natural boundary conditions and high-order derivatives.
* Weak form should converge faster and be less sensitive to collocation density.

---

# 5. Six Benchmark Cases

## Case 1: Flat Strip Bending under Tip Force

### Purpose

A simple benchmark to verify that the PINN can solve classical elastic bending.

### Geometry

Initial centerline:

[
x_0(s)=s,\quad y_0(s)=0
]

Initial curvature:

[
\kappa_0(s)=0
]

### Boundary Conditions

At (s=0):

[
x=0,\quad y=0,\quad \theta=0
]

At (s=L):

* tip force (P_y) applied in the negative (y)-direction;
* no prescribed tip displacement.

### Expected Output

* Deformed shape.
* Tip displacement.
* Rotation field.
* Moment distribution.
* Strong vs weak convergence curves.
* Comparison with analytical small-deflection solution and FEM.

### Difficulty Level

Low.

---

## Case 2: Flat Strip Bending under Prescribed End Displacement

### Purpose

Test geometrically nonlinear bending under displacement control.

### Geometry

Initial centerline:

[
x_0(s)=s,\quad y_0(s)=0
]

Initial curvature:

[
\kappa_0(s)=0
]

### Boundary Conditions

At (s=0):

[
x=0,\quad y=0,\quad \theta=0
]

At (s=L):

[
x=L-\Delta_x,\quad y=-\Delta_y
]

Tip rotation can be left free or weakly constrained depending on stability.

### Recommended Loading Levels

Use three displacement levels:

[
\Delta_y/L = 0.1,\quad 0.2,\quad 0.3
]

### Expected Output

* Large-deflection shape.
* End reaction force.
* Elastic energy.
* Strong vs weak error comparison.
* FEM validation with nonlinear geometry enabled.

### Difficulty Level

Low to medium.

---

## Case 3: Symmetric Flattening of a C-Shaped Strip

### Purpose

Main deployable-structure benchmark.

### Geometry

The initial shape is a circular arc.

Total angle:

[
\alpha_C = 220^\circ
]

Radius:

[
R_C = \frac{L}{\alpha_C}
]

Initial curvature:

[
\kappa_0(s)=\frac{1}{R_C}
]

### Boundary Conditions

Use symmetric end flattening.

Initial endpoints are gradually pulled apart until the C-shape becomes flatter.

At (s=0) and (s=L), prescribe final endpoint positions:

[
\mathbf{r}(0)=\mathbf{r}_0^{*},\quad
\mathbf{r}(L)=\mathbf{r}_L^{*}
]

End rotations may be:

* free for a natural flattening process;
* or weakly constrained for numerical stability.

### Expected Output

* Initial C-shape.
* Flattened shape.
* Curvature evolution.
* Bending energy distribution.
* Boundary reaction forces.
* Strong vs weak performance.

### Difficulty Level

Medium.

---

## Case 4: Asymmetric Flattening of a C-Shaped Strip

### Purpose

Test robustness under asymmetric boundary conditions and free-boundary effects.

### Geometry

Same as Case 3:

[
\kappa_0(s)=\frac{1}{R_C}
]

### Boundary Conditions

At (s=0):

[
x=x_0,\quad y=y_0,\quad \theta=\theta_0
]

At (s=L):

[
x=x_L^{*},\quad y=y_L^{*}
]

The right end is displaced to flatten the C-shape asymmetrically.

### Expected Output

* Deformed shape.
* Boundary reaction.
* Strong-form residual map.
* Weak-form energy convergence.
* FEM validation.

### Key Scientific Use

This case should expose the difficulty of strong-form training more clearly than Case 3 because boundary conditions are less symmetric and the deformation is less uniform.

### Difficulty Level

Medium to high.

---

## Case 5: Vertical Flattening of an Ω-Shaped Strip

### Purpose

Introduce a more complex initial curvature distribution.

### Geometry

The initial Ω shape is generated from control points, then resampled by arc length.

Recommended control points:

[
(-0.5L,0),
(-0.35L,0.35H),
(-0.15L,H),
(0,H),
(0.15L,H),
(0.35L,0.35H),
(0.5L,0)
]

Use cubic spline interpolation, then compute:

[
\theta_0(s),\quad \kappa_0(s)=\theta_0'(s)
]

### Boundary Conditions

The two bottom endpoints are fixed horizontally.

A vertical displacement is applied to the top region using one or several control points.

No real contact is included in the first version.

Instead, flattening is enforced through prescribed geometric constraints:

[
y(s_{\mathrm{top}})=y_{\mathrm{target}}
]

### Expected Output

* Ω-shape before and after flattening.
* Local curvature concentration.
* Energy density distribution.
* Strong vs weak convergence.
* FEM validation.

### Difficulty Level

High.

---

## Case 6: Constrained Full Flattening of an Ω-Shaped Strip

### Purpose

Most difficult benchmark, used to demonstrate the advantage of weak-form PINN.

### Geometry

Same Ω initial geometry as Case 5.

### Boundary Conditions

Prescribe multiple displacement constraints so that the Ω-shape approaches a nearly flat configuration.

Suggested constraints:

[
y(0)=0,\quad y(L)=0
]

[
y(s_1)=0,\quad y(s_2)=0,\quad y(s_3)=0
]

where (s_1,s_2,s_3) are representative control locations near the upper arch.

Rotations are not fully prescribed unless training becomes unstable.

### Expected Output

* Full flattening shape.
* Reaction forces at constrained points.
* High-curvature zones.
* Strong-form failure or slow convergence.
* Weak-form stable convergence.
* FEM validation.

### Difficulty Level

High.

---

# 6. FEM Validation

## 6.1 FEM Is Not Used for Training

Important rule:

> FEM data must never be used in the PINN loss.

FEM is used only for:

* validation;
* visualization;
* error calculation;
* reaction-force comparison.

## 6.2 Recommended FEM Model

Use Abaqus beam models for simplicity.

Recommended element:

```text
B21
```

or for 3D visualization:

```text
B31
```

Use nonlinear geometry:

```text
NLGEOM = ON
```

Material:

```text
Elastic, isotropic
```

Cross-section:

```text
A, I
```

The FEM model should use the same nondimensional parameters as the PINN model:

```text
L = 1
EI = 1
EA = 100
kGA = 100
```

## 6.3 FEM Outputs

Extract:

* node coordinates;
* displacement field;
* rotation field;
* reaction forces;
* reaction moments;
* strain energy;
* curvature or moment distribution if available.

---

# 7. Neural Network Design

## 7.1 Input and Output

Input:

[
\bar{s}=2s/L-1
]

Output:

[
[x(s),y(s),\theta(s)]
]

Optional conditional input for future extension:

[
[\bar{s}, \lambda]
]

where (\lambda) is a loading parameter.

For the first version, train one model per loading case.

---

## 7.2 Recommended Architecture

Use a simple MLP:

```text
Input dimension: 1
Output dimension: 3
Hidden layers: 4
Neurons per hidden layer: 64
Activation: tanh
```

Optional alternatives:

```text
SIREN
Fourier-feature MLP
```

For the first version, use tanh MLP for all cases.

---

## 7.3 Boundary Condition Enforcement

Use a hybrid strategy.

For essential boundary conditions at (s=0), impose them hard:

[
q(s)=q(0)+s\mathcal{N}_{\tau}(s)
]

For complex endpoint or multi-point constraints, use penalty terms:

[
\mathcal{L}_{\mathrm{bc}}
=========================

\sum_j
\left|
q(s_j)-q_j^{*}
\right|^2
]

Recommended penalty weights:

```text
lambda_bc = 1e3 to 1e5
lambda_sym = 1e2
lambda_reg = 1e-6
```

Use adaptive penalty scaling if boundary errors remain high.

---

# 8. Training Protocol

## 8.1 Collocation Points

Use Sobol or uniform sampling.

Recommended values:

```text
Nc = 512
Nc = 2048
Nc = 8192
```

The default setting is:

```text
Nc = 2048
```

For final paper figures, use:

```text
Nc = 8192
```

## 8.2 Optimizer

Two-stage optimization:

Stage 1:

```text
Adam
learning rate = 1e-3
epochs = 5000
```

Stage 2:

```text
L-BFGS
max_iter = 500
line_search = strong_wolfe
```

## 8.3 Random Seeds

Use at least five seeds:

```text
0, 1, 2, 3, 4
```

Report:

* best seed;
* mean error;
* standard deviation;
* failure rate.

## 8.4 Training Runs

For each case:

```text
strong form × 5 seeds
weak form × 5 seeds
```

Total training jobs:

```text
6 cases × 2 forms × 5 seeds = 60 runs
```

For fast debugging:

```text
6 cases × 2 forms × 1 seed = 12 runs
```

---

# 9. Evaluation Metrics

## 9.1 Shape Error

Compare PINN and FEM centerlines.

Relative (L_2) error:

[
e_{\mathbf{r}}
==============

\frac{
\left|
\mathbf{r}*{\mathrm{PINN}}-\mathbf{r}*{\mathrm{FEM}}
\right|*2
}{
\left|
\mathbf{r}*{\mathrm{FEM}}
\right|_2
}
]

## 9.2 Rotation Error

[
e_{\theta}
==========

\frac{
\left|
\theta_{\mathrm{PINN}}-\theta_{\mathrm{FEM}}
\right|*2
}{
\left|
\theta*{\mathrm{FEM}}
\right|_2
}
]

## 9.3 Reaction Error

[
e_R
===

\frac{
|R_{\mathrm{PINN}}-R_{\mathrm{FEM}}|
}{
|R_{\mathrm{FEM}}|
}
]

## 9.4 Energy Error

[
e_{\Pi}
=======

\frac{
|\Pi_{\mathrm{PINN}}-\Pi_{\mathrm{FEM}}|
}{
|\Pi_{\mathrm{FEM}}|
}
]

## 9.5 Strong Residual Norm

For both strong and weak models, evaluate the strong residual after training:

[
\bar{R}
=======

\frac{1}{N_c}
\sum_{i=1}^{N_c}
\left|
\mathbf{R}_{\mathrm{EL}}(s_i)
\right|_2
]

This allows fair comparison of physical consistency.

## 9.6 Boundary Error

[
e_{\mathrm{bc}}
===============

\sum_j
\left|
q(s_j)-q_j^{*}
\right|_2
]

## 9.7 Convergence Metrics

Record:

* final loss;
* best loss;
* epochs to convergence;
* wall-clock time;
* gradient norm;
* seed-to-seed variance.

---

# 10. Expected Results

## 10.1 Main Expected Trend

Weak-form PINN should generally outperform strong-form PINN in:

* convergence speed;
* final shape accuracy;
* robustness across random seeds;
* sensitivity to collocation density;
* complex-boundary cases.

Strong-form PINN may work well in:

* Case 1;
* Case 2;
* possibly Case 3.

Strong-form PINN is expected to struggle in:

* Case 4;
* Case 5;
* Case 6.

## 10.2 Minimum Success Criteria

The demo is considered successful if:

1. Weak-form PINN solves all six cases.
2. Strong-form PINN solves at least the first three cases.
3. FEM is used only for validation.
4. Weak-form accuracy is better than strong-form in at least four of six cases.
5. Final centerline error is below 5% for weak-form PINN in simple cases.
6. Final centerline error is below 10% for weak-form PINN in Ω-shape cases.
7. The paper can clearly explain why weak form is more robust.

---

# 11. Code Structure

Recommended repository structure:

```text
project_root/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── configs/
│   ├── case01_flat_tip_force.yaml
│   ├── case02_flat_tip_displacement.yaml
│   ├── case03_c_symmetric_flattening.yaml
│   ├── case04_c_asymmetric_flattening.yaml
│   ├── case05_omega_vertical_flattening.yaml
│   └── case06_omega_full_flattening.yaml
│
├── src/
│   ├── geometry/
│   │   ├── flat.py
│   │   ├── c_shape.py
│   │   ├── omega_shape.py
│   │   └── spline_utils.py
│   │
│   ├── models/
│   │   ├── mlp.py
│   │   ├── siren.py
│   │   └── fourier_mlp.py
│   │
│   ├── physics/
│   │   ├── strain.py
│   │   ├── energy.py
│   │   ├── strong_form.py
│   │   ├── weak_form.py
│   │   └── reactions.py
│   │
│   ├── losses/
│   │   ├── boundary_loss.py
│   │   ├── strong_loss.py
│   │   ├── weak_loss.py
│   │   └── regularization.py
│   │
│   ├── training/
│   │   ├── trainer.py
│   │   ├── optimizers.py
│   │   └── scheduler.py
│   │
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── compare_fem.py
│   │   └── summary_table.py
│   │
│   ├── visualization/
│   │   ├── plot_shape.py
│   │   ├── plot_loss.py
│   │   ├── plot_residual.py
│   │   └── plot_summary.py
│   │
│   └── utils/
│       ├── sampling.py
│       ├── io.py
│       ├── seed.py
│       └── logger.py
│
├── scripts/
│   ├── run_case.py
│   ├── run_all_cases.py
│   ├── evaluate_case.py
│   ├── make_all_figures.py
│   └── export_tables.py
│
├── fem/
│   ├── abaqus_scripts/
│   │   ├── build_case01.py
│   │   ├── build_case02.py
│   │   ├── build_case03.py
│   │   ├── build_case04.py
│   │   ├── build_case05.py
│   │   └── build_case06.py
│   │
│   └── results/
│
├── results/
│   ├── raw/
│   ├── processed/
│   ├── figures/
│   └── tables/
│
├── paper/
│   ├── manuscript.md
│   ├── figures/
│   ├── tables/
│   └── references.bib
│
└── agents/
    ├── task_design_agent.md
    ├── coding_agent.md
    ├── fem_agent.md
    ├── evaluation_agent.md
    ├── figure_agent.md
    └── writing_agent.md
```

---

# 12. AI Agent Workflow

This project is intended to be completed by AI agents with human supervision.

## 12.1 Task Design Agent

Responsibilities:

* Finalize six benchmark definitions.
* Check whether each case has clear geometry, boundary conditions, and validation targets.
* Write all YAML config files.
* Ensure no FEM data is used in training.
* Define expected outputs for each case.

Acceptance criteria:

* Six configs are complete.
* Each config includes geometry, material, BCs, optimizer, loss type, collocation settings, and output directory.

---

## 12.2 Coding Agent

Responsibilities:

* Implement geometry generators.
* Implement MLP model.
* Implement strain calculation.
* Implement energy density.
* Implement weak-form loss.
* Implement strong-form Euler–Lagrange residual.
* Implement boundary-condition penalties.
* Implement Adam + L-BFGS training.
* Implement checkpoint saving.

Acceptance criteria:

* `python scripts/run_case.py --config configs/case01_flat_tip_force.yaml --form weak` runs without error.
* Strong and weak forms both run for Case 1.
* Results are saved automatically.
* Loss curves and final shapes are generated.

---

## 12.3 FEM Agent

Responsibilities:

* Generate Abaqus models for six cases.
* Use beam elements for validation.
* Enable nonlinear geometry.
* Export centerline coordinates, rotations, reactions, and energy.
* Save FEM results in a standard CSV format.

Acceptance criteria:

* Each FEM case runs.
* Output files are placed in `fem/results/caseXX/`.
* FEM output format matches evaluation script.

---

## 12.4 Evaluation Agent

Responsibilities:

* Load PINN and FEM results.
* Compute all metrics.
* Generate comparison tables.
* Generate seed statistics.
* Identify failed training runs.
* Summarize strong vs weak trends.

Acceptance criteria:

* `results/tables/table_case_metrics.csv` is generated.
* `results/tables/table_strong_vs_weak.csv` is generated.
* Each case has shape error, rotation error, reaction error, energy error, boundary error, and runtime.

---

## 12.5 Figure Agent

Responsibilities:

Generate publication-ready figures:

1. Framework diagram.
2. Six benchmark cases.
3. Strong vs weak convergence.
4. Shape comparison for flat cases.
5. Shape comparison for C-shape cases.
6. Shape comparison for Ω-shape cases.
7. Summary bar chart of errors and runtime.

Acceptance criteria:

* Figures are saved as `.png`, `.pdf`, and `.svg`.
* All figures use consistent fonts.
* Figure captions are drafted.

---

## 12.6 Writing Agent

Responsibilities:

* Draft manuscript.
* Write abstract.
* Write introduction.
* Write method.
* Write results.
* Write conclusion.
* Write figure captions.
* Write supplementary material.
* Ensure claims are consistent with results.

Acceptance criteria:

* `paper/manuscript.md` is complete.
* All figures and tables are referenced.
* No unsupported claim is included.
* Manuscript clearly states that the model is data-free and FEM is only for validation.

---

# 13. Running the Project

## 13.1 Install Environment

```bash
conda create -n elastic_pinn python=3.10
conda activate elastic_pinn
pip install -r requirements.txt
```

Recommended packages:

```text
torch
numpy
scipy
matplotlib
pandas
pyyaml
tqdm
seaborn
```

For publication figures, use:

```text
matplotlib
scienceplots
```

---

## 13.2 Run One Case

Weak form:

```bash
python scripts/run_case.py \
  --config configs/case03_c_symmetric_flattening.yaml \
  --form weak \
  --seed 0
```

Strong form:

```bash
python scripts/run_case.py \
  --config configs/case03_c_symmetric_flattening.yaml \
  --form strong \
  --seed 0
```

---

## 13.3 Run All Cases

```bash
python scripts/run_all_cases.py \
  --forms strong weak \
  --seeds 0 1 2 3 4
```

---

## 13.4 Evaluate

```bash
python scripts/evaluate_case.py --case case03
python scripts/export_tables.py
```

---

## 13.5 Generate Figures

```bash
python scripts/make_all_figures.py
```

---

# 14. Output Files

Each training run should save:

```text
results/raw/caseXX/form_seed/
│
├── config_used.yaml
├── model.pt
├── loss_history.csv
├── collocation_points.csv
├── predicted_shape.csv
├── predicted_strain.csv
├── predicted_reaction.csv
├── metrics.json
├── shape_comparison.png
├── loss_curve.png
└── residual_distribution.png
```

The final summary should save:

```text
results/tables/
├── table_case_metrics.csv
├── table_strong_vs_weak.csv
├── table_seed_statistics.csv
└── table_runtime.csv
```

---

# 15. Paper Structure

## Abstract

The abstract should state:

* Data-free PINN.
* Elastic curved strips.
* Strong vs weak form.
* Six benchmark problems.
* FEM only for validation.
* Weak form generally improves robustness.

## 1. Introduction

Recommended logic:

1. Curved strips are common in deployable and morphing structures.
2. Elastic flattening involves geometric nonlinearity and boundary-condition sensitivity.
3. FEM is reliable but requires meshing and case-specific setup.
4. PINNs offer a mesh-free, data-free alternative.
5. Existing PINN studies show strong/weak formulation matters.
6. This work builds a compact benchmark for flat, C-shaped, and Ω-shaped elastic strip deformation.

## 2. Mechanics and PINN Formulation

Sections:

* 2.1 Geometrically nonlinear strip model.
* 2.2 Strong-form residual.
* 2.3 Weak-form energy minimization.
* 2.4 Training and validation protocol.

## 3. Benchmark Problems

Sections:

* 3.1 Flat strip bending.
* 3.2 C-shaped strip flattening.
* 3.3 Ω-shaped strip flattening.

Each section includes two cases.

## 4. Results

Sections:

* 4.1 Convergence of strong and weak forms.
* 4.2 Shape prediction accuracy.
* 4.3 Reaction and energy comparison.
* 4.4 Robustness across cases and seeds.

## 5. Discussion

Discuss:

* Why weak form is more stable.
* When strong form still works.
* Why Ω-shape flattening is harder.
* Limitations of the simplified model.
* Future extension to viscoelasticity, contact, shell strips, and SMPCs.

## 6. Conclusion

State:

* Six forward problems solved.
* No training data used.
* Weak form shows better robustness.
* Framework provides a lightweight benchmark for deployable strip modeling.

---

# 16. Figure Plan

## Figure 1: Framework

Content:

* Initial geometry.
* Neural network.
* Strong-form residual.
* Weak-form energy.
* FEM validation.

## Figure 2: Six Benchmark Cases

Content:

* Case 1 flat tip force.
* Case 2 flat tip displacement.
* Case 3 C symmetric.
* Case 4 C asymmetric.
* Case 5 Ω vertical flattening.
* Case 6 Ω full flattening.

## Figure 3: Flat Strip Results

Content:

* FEM vs strong PINN vs weak PINN.
* Loss curves.
* Tip displacement or reaction.

## Figure 4: C-Shaped Strip Results

Content:

* Initial and flattened C-shape.
* Curvature distribution.
* Reaction comparison.
* Strong vs weak convergence.

## Figure 5: Ω-Shaped Strip Results

Content:

* Initial and flattened Ω-shape.
* Local high-curvature zones.
* Strong-form residual map.
* Weak-form energy result.

## Figure 6: Strong vs Weak Summary

Content:

* Relative shape error across six cases.
* Runtime.
* Seed variance.
* Failure rate.

---

# 17. Tables

## Table 1: Benchmark Definitions

Columns:

* Case ID.
* Geometry.
* Boundary condition.
* Loading type.
* Difficulty.
* Validation target.

## Table 2: Training Settings

Columns:

* Network.
* Collocation points.
* Optimizer.
* Epochs.
* Loss weights.
* Seeds.

## Table 3: Strong vs Weak Accuracy

Columns:

* Case.
* Strong shape error.
* Weak shape error.
* Strong reaction error.
* Weak reaction error.
* Strong runtime.
* Weak runtime.

## Table 4: Seed Robustness

Columns:

* Case.
* Form.
* Mean error.
* Standard deviation.
* Best error.
* Worst error.
* Failure rate.

---

# 18. Rules and Constraints

The following rules must be strictly followed:

1. Do not use FEM data for PINN training.
2. Do not add viscoelasticity in the first version.
3. Do not add thermal loading.
4. Do not add inverse identification.
5. Do not add contact.
6. Do not use UMAT.
7. Do not overcomplicate the material model.
8. Use the same material and geometry parameters in PINN and FEM.
9. Compare strong and weak forms fairly.
10. Save every result automatically.
11. Every figure must be reproducible by script.
12. Every claim in the paper must be supported by a figure, table, or metric.

---

# 19. Development Milestones

## Milestone 1: Minimal Working Case

Target:

* Case 1 weak form runs.
* Case 1 strong form runs.
* Shape plot generated.

Deliverables:

* `case01` config.
* Training script.
* Loss curves.
* Predicted shape.

## Milestone 2: Full Six-Case PINN

Target:

* All six cases run using weak form.
* All six cases run using strong form.

Deliverables:

* 12 baseline runs.
* Debug plots.
* Initial comparison table.

## Milestone 3: FEM Validation

Target:

* Abaqus validation completed for all six cases.

Deliverables:

* FEM output CSV files.
* FEM vs PINN shape plots.
* Error metrics.

## Milestone 4: Multi-Seed Study

Target:

* Five seeds per case per form.

Deliverables:

* 60 training runs.
* Seed statistics.
* Failure-rate analysis.

## Milestone 5: Paper Figures

Target:

* All main figures generated.

Deliverables:

* Figure 1–6.
* Table 1–4.
* Supplementary plots.

## Milestone 6: Manuscript Draft

Target:

* Complete manuscript draft.

Deliverables:

* `paper/manuscript.md`
* figure captions;
* abstract;
* discussion;
* conclusion.

---

# 20. Minimal Abstract Draft

**Abstract.**
Physics-informed neural networks provide a mesh-free route for solving mechanical boundary-value problems without relying on labeled simulation data. In this work, we develop a data-free physics-informed neural framework for elastic deformation and flattening of curved deployable strips. A geometrically nonlinear planar strip model is adopted, in which the neural network predicts the deformed centerline and rotation field from the arc-length coordinate. Two formulations are implemented and compared: a strong-form formulation based on Euler–Lagrange residuals and a weak-form formulation based on total potential energy minimization. Six benchmark problems are considered, including flat strip bending, symmetric and asymmetric flattening of C-shaped strips, and vertical and constrained flattening of Ω-shaped strips. The model is trained solely from physical losses, while finite-element simulations are used only as independent validation references. Results are expected to show that the weak-form PINN provides more stable convergence and better robustness for complex curved-strip flattening problems, whereas the strong-form PINN remains effective mainly for simpler bending cases. This compact benchmark provides a reproducible basis for extending physics-informed neural solvers toward deployable structures, viscoelastic materials, and shape-memory composite systems.

---

# 21. One-Sentence Project Summary

> This project builds a fully data-free PINN benchmark for elastic bending and flattening of flat, C-shaped, and Ω-shaped strips, with systematic strong-form and weak-form comparisons across six forward problems.

