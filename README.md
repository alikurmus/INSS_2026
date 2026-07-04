# INSS 2026 Statistics Exercises

This repository contains a worked Jupyter notebook with detailed solutions to the INSS 2026 statistics exercises.

## Contents

- `INSS_Stats_Detailed_Solutions.ipynb` — main solution notebook.
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
cd ~/Repos/INSS_2026
jupyter lab
```

Open a notebook you would like to use your solutions, choose the kernel `Python (INSS Stats 2026)`, and run all cells.
