# Results

This repository includes the completed result artifacts used for reporting and reproducibility.

## Tables

Located in `results/tables/`:

- `metrics_by_experiment.csv` - predictive metrics by target, experiment, and model.
- `model_summary.csv` - trained model summary and diagnostic metrics.
- `certification_summary.csv` - symbolic reliability diagnostics.
- `feature_ranges.csv` - feature and target ranges used by the workflow.
- `symbolic_selection_scores.csv` - reliability-aware selection scores.

## Equations

Located in `results/equations/`:

- `selected_equations.json` - machine-readable selected equations and metadata.
- `selected_equations.tex` - LaTeX table of selected equations.

## Figures

Located in:

- `results/figures/`
- `results/figures_reliability_aware/`

The `figures_reliability_aware/` directory is the preferred figure set for article use.

## Predictions

Located in `results/predictions/`:

- `all_predictions.csv` - consolidated measured/predicted values and applicability scores.

## Models

Located in `results/models/`.

These files are fitted model artifacts. They are tracked through Git LFS because several random-forest models are larger than GitHub's normal file limit.

## Reports

Located in `results/reports/`:

- `run_manifest.json` - run configuration, software/hardware metadata, and output paths.
- `dataset_profile.json` - dataset profile and ranges.

## Reported Headline Values

- Selected symbolic `Nuavg` MAPE: 2.923 percent on random 80/20 split, 5.314 percent on high-Re holdout, 2.982 percent on geometry holdout.
- Selected symbolic `DelP_Pa` MAPE: 9.521 percent on random 80/20 split, 11.786 percent on high-Re holdout, 10.277 percent on geometry holdout.
- The selected symbolic model is the sparse symbolic lasso candidate chosen by the reliability-aware ranking layer.
