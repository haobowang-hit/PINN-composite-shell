# Data-Free Physics-Informed Neural Modeling of Elastic Flattening in Curved Deployable Strips

##  Project Overview

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


