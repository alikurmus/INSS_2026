# INSS 2026 Statistics Exercises

This repository contains a worked Jupyter notebook with detailed solutions to the INSS 2026 statistics exercises.

Recent refactor: the low-level fit helpers have been moved into an `app` package and a notebook-friendly `FitManager` was added to simplify running fits and profile-likelihood scans from the notebook.

## Contents

- `INSS_Exercise1_Detailed.ipynb` — main solution notebook (now uses `app.FitManager`).
- `app/` — package containing `helper.py` (low-level fit/NLL code) and `fit_manager.py` (high-level `FitManager`).
- `env.yaml` — Conda environment file.
- `data/exercise_1/` — data for Exercise 1.
- `data/exercise_2/` — data for Exercise 2.
- `data/exercise_3/` — data for Exercise 3.
- `reference_material/` — original exercise statement and lecture/reference PDFs.

## Setup

Create the Conda environment:

```bash
conda env create -f env.yaml
```

Activate it:

```bash
conda activate inss-stats-2026
```

Register the Jupyter kernel:

```bash
python -m ipykernel install --user --name inss-stats-2026 --display-name "Python (INSS Stats 2026)"
```

Start JupyterLab from the repository root:

```bash
cd ~/Desktop/project_summer_school_UCSB/INSS_2026
jupyter lab
```

Open `INSS_Exercise1_Detailed.ipynb`, choose the kernel `Python (INSS Stats 2026)`, and run the cells from the top.

## Using `FitManager`

The notebook uses a high-level `FitManager` (in `app/fit_manager.py`) to run fits and profile-likelihood scans.
From the notebook or a Python session you can create a manager and run the full workflow:

```python
from app.fit_manager import FitManager
fm = FitManager()
# Run both fits (no-shape and with profiled shape) and store results on `fm`:
fm.run_both()
# Run profile scans and compute intervals (plotting is handled in the notebook):
# fm.run_full_profile(S_values, plot=False)
```

See `INSS_Exercise1_Detailed.ipynb` for full examples and plots.
