This directory contains four independent Abaqus/CAE case scripts and one shared
data export script:

- `build_case01_flat_bending.py`
- `build_case02_c_flattening.py`
- `build_case03_omega_flattening.py`
- `build_case04_c_curvature_study.py`
- `datasavebystep.py`

Run a script manually in Abaqus/CAE, or from this directory with:

```text
abaqus cae noGUI=build_case01_flat_bending.py
```

Case 4 runs the 120, 170 and 260 degree C-shells sequentially by default.  The
angles and mesh can be overridden after `--`, for example:

```text
abaqus cae noGUI=build_case04_c_curvature_study.py -- --angles 120,170,260 --nx 160 --nz 24
```

Use `--no-submit` to build the `.cae` and `.inp` files without solving them.
The script is compatible with the Python 2.7 interpreter embedded in the
project's Abaqus release.  A fully exported case is skipped automatically,
which also supports direct execution through the CAE `execfile` interface.  A
partial result directory or conflicting Abaqus job artifact still stops the
run; no previous result is deleted or overwritten automatically.

Each case script now builds the model, writes the `.cae` and `.inp`, submits the
job, waits for completion, opens the ODB, and calls `datasavebystep.py`
automatically.

FEM output must not be used in PINN training. Validation CSV files should be
placed under `fem/results/<case_id>/`.
