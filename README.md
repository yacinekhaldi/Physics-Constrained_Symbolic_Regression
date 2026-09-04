# Reliability-Aware Symbolic Regression

Code, data pointers, and result artifacts for:

**Reliability-Aware Physics-Constrained Symbolic Regression for Generalizable Heat-Transfer Correlation Discovery**

The repository trains symbolic and black-box regressors for thermo-hydraulic prediction in partially porous wavy channels, evaluates interpolation and holdout behavior, computes reliability diagnostics, and exports tables, figures, equations, predictions, and model artifacts.

## Repository Contents

- `src/rcsr/` - reusable Python package for data preparation, model training, metrics, diagnostics, and plotting.
- `scripts/` - command-line entry points for downloading data, running experiments, regenerating figures, and writing article artifacts.
- `data/raw/` - source dataset archive and Mendeley metadata. The extracted dataset directory is ignored because it is recoverable from the archive.
- `data/processed/` - canonical processed CSV and column mapping used by the experiments.
- `results/tables/` - metrics, feature ranges, model summaries, and certification summaries.
- `results/equations/` - selected symbolic equations in JSON and LaTeX formats.
- `results/predictions/` - consolidated predictions for all reported experiments.
- `results/figures/` and `results/figures_reliability_aware/` - exported PNG/PDF figures.
- `results/models/` - fitted model artifacts. These are configured for Git LFS because several files exceed normal GitHub file limits.
- `docs/` - dataset notes, artifact manifest, and reproducibility instructions.

## Dataset

Primary dataset:

- Prince Kumar and K. M. Pandey, *CFD-informed machine learning surrogate dataset for thermo-hydraulic prediction in partially porous wavy channels*, Mendeley Data, Version 2, DOI: `10.17632/5b5n3cg32n.2`.

The repository keeps the downloaded source archive at:

- `data/raw/ML-CFD-Wavy-Channel-Surrogate.zip`

The extracted source directory is not intended to be committed. It can be recreated with:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
```

See `docs/DATASET.md` for checksums, retained variables, and data provenance.

## Setup

Python 3.10 is recommended.

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Optional:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

## Reproduce Results

From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\download_dataset.py
.\.venv\Scripts\python.exe scripts\run_experiments.py --targets Nuavg DelP_Pa
.\.venv\Scripts\python.exe scripts\regenerate_figures.py
.\.venv\Scripts\python.exe scripts\write_article.py
```

The experiment runner uses CPU cores for symbolic regression and requests CUDA acceleration for XGBoost when a compatible GPU is available. CPU fallback is supported.

## Main Outputs

Key outputs are written under `results/`:

- `tables/metrics_by_experiment.csv`
- `tables/model_summary.csv`
- `tables/certification_summary.csv`
- `tables/feature_ranges.csv`
- `equations/selected_equations.json`
- `equations/selected_equations.tex`
- `predictions/all_predictions.csv`
- `figures_reliability_aware/*.png`
- `figures_reliability_aware/*.pdf`
- `reports/run_manifest.json`
- `reports/dataset_profile.json`

The manuscript/artifact writer creates an `Elsevier/` folder one level above this repository when run in the original workspace layout. In a standalone clone, create a sibling `Elsevier/` folder first if you want the article files written there.

## GitHub Upload Notes

Git LFS is required before adding or pushing this repository because the dataset archive and several model artifacts are larger than 100 MB.

```powershell
git lfs install
git add .gitattributes .gitignore README.md CITATION.cff LICENSE docs src scripts data results requirements.txt requirements-optional.txt pyproject.toml
git commit -m "Initial reproducibility release"
git branch -M main
git remote add origin https://github.com/<owner>/<repo>.git
git push -u origin main
```

Run `git lfs ls-files` before pushing to confirm large files are tracked by LFS.
