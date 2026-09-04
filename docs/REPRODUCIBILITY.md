# Reproducibility

## Environment

Recommended platform:

- Python 3.10
- Windows PowerShell or a POSIX shell with equivalent commands
- Optional CUDA-capable GPU for XGBoost acceleration

Install dependencies:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Pipeline

Run from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\run_experiments.py --targets Nuavg DelP_Pa
.\.venv\Scripts\python.exe scripts\regenerate_figures.py
.\.venv\Scripts\python.exe scripts\write_article.py
```

The pipeline writes processed data, metrics, selected equations, predictions, models, figures, and run reports under `data/` and `results/`.

## Main Experiments

- Random 80/20 interpolation split.
- High-Reynolds-number holdout.
- Grouped geometry holdout.
- Sparse-data training fractions of 20, 40, 60, and 80 percent.
- Training-target noise levels of 1, 3, and 5 percent.

## Metrics and Diagnostics

Predictive metrics:

- MAE
- RMSE
- MAPE
- R2

Reliability diagnostics:

- Symbolic complexity
- Physics-violation rate
- Local sensitivity
- Extrapolation-stability index
- Applicability-domain score
- High-error detection AUROC

## Determinism

The base random seed is 42. Split-specific and diagnostic-specific offsets are recorded in `results/reports/run_manifest.json`.

Small numerical differences may occur across operating systems, BLAS backends, CPU/GPU execution, and library versions.
