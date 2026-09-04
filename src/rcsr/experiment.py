from __future__ import annotations

import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from .certification import ApplicabilityScorer, physics_report
from .data import FEATURE_ALIASES, TARGET_ALIASES, dataset_profile, load_prepared_dataset
from .metrics import high_error_detection_auc, regression_metrics
from .models import FittedModel, fit_black_box_models, symbolic_factories
from .paths import (
    EQUATIONS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_dirs,
)
from .plotting import (
    plot_applicability,
    plot_metric_bars,
    plot_noise,
    plot_pareto,
    plot_predictions,
    plot_re_sweep,
)


FEATURES = list(FEATURE_ALIASES)
TARGETS = list(TARGET_ALIASES)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        if not np.isfinite(value):
            return None
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _safe_name(text: str) -> str:
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("(", "")
        .replace(")", "")
    )


def _nvidia_smi() -> str:
    try:
        completed = subprocess.run(
            ["nvidia-smi"],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return completed.stdout.strip() or completed.stderr.strip()
    except Exception as exc:
        return f"nvidia-smi unavailable: {exc}"


def make_splits(df: pd.DataFrame, random_state: int) -> Dict[str, Dict[str, object]]:
    indices = np.arange(len(df))
    train, test = train_test_split(indices, test_size=0.20, random_state=random_state)
    splits: Dict[str, Dict[str, object]] = {
        "random80_20": {
            "train_idx": train,
            "test_idx": test,
            "include_gp": True,
            "description": "Random 80/20 interpolation split.",
        }
    }

    re_threshold = float(df["Re"].quantile(0.80))
    train = indices[df["Re"].to_numpy() < re_threshold]
    test = indices[df["Re"].to_numpy() >= re_threshold]
    if len(train) > 20 and len(test) > 20:
        splits["range_re_high_holdout"] = {
            "train_idx": train,
            "test_idx": test,
            "include_gp": True,
            "description": f"High-Re range holdout with Re >= {re_threshold:.6g}.",
        }

    geom_cols = ["thickness", "amplitude", "wavelength"]
    groups = (
        df[geom_cols]
        .round(10)
        .astype(str)
        .agg("|".join, axis=1)
        .astype("category")
        .cat.codes
        .to_numpy()
    )
    if len(np.unique(groups)) > 1:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=0.20, random_state=random_state + 1
        )
        train, test = next(splitter.split(indices, groups=groups))
        splits["geometry_holdout"] = {
            "train_idx": train,
            "test_idx": test,
            "include_gp": True,
            "description": "Grouped holdout over thickness/amplitude/wavelength combinations.",
        }

    return splits


def _selection_score(rows: pd.DataFrame) -> pd.Series:
    score_parts = []
    for column, weight, log_transform in [
        ("selection_RMSE", 1.0, False),
        ("complexity", 0.04, False),
        ("physics_violation_rate", 2.0, False),
        ("sensitivity_p95", 0.10, True),
        ("extrapolation_stability_index", 0.10, True),
    ]:
        values = rows[column].astype(float).to_numpy()
        if log_transform:
            values = np.log1p(values)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            scaled = np.ones_like(values)
        else:
            lo = np.nanmin(values[finite])
            hi = np.nanmax(values[finite])
            if hi - lo < 1e-12:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - lo) / (hi - lo)
                scaled[~finite] = 1.0
        score_parts.append(weight * scaled)
    return pd.Series(np.sum(score_parts, axis=0), index=rows.index)


def fit_symbolic_candidates(
    X_train: np.ndarray,
    y_train: np.ndarray,
    target: str,
    feature_names: List[str],
    random_state: int,
    include_gp: bool,
    gp_population: int,
    gp_generations: int,
) -> Tuple[List[FittedModel], pd.DataFrame, FittedModel]:
    train_idx, val_idx = train_test_split(
        np.arange(len(X_train)), test_size=0.25, random_state=random_state
    )
    X_fit, y_fit = X_train[train_idx], y_train[train_idx]
    X_val, y_val = X_train[val_idx], y_train[val_idx]

    fitted: List[FittedModel] = []
    rows: List[Dict[str, object]] = []
    for factory in symbolic_factories(
        feature_names,
        random_state=random_state,
        gp_population=gp_population,
        gp_generations=gp_generations,
        include_gp=include_gp,
    ):
        start = time.perf_counter()
        try:
            model = factory(X_fit, y_fit, target)
            fit_seconds = time.perf_counter() - start
            y_val_pred = model.estimator.predict(X_val)
            metrics = regression_metrics(y_val, y_val_pred)
            cert = physics_report(
                model.estimator,
                X_fit,
                feature_names,
                target,
                random_state=random_state + 101,
                n_samples=1024,
            )
            row = {
                "target": target,
                "model": model.name,
                "family": model.family,
                "selection_MAE": metrics["MAE"],
                "selection_RMSE": metrics["RMSE"],
                "selection_MAPE": metrics["MAPE"],
                "selection_R2": metrics["R2"],
                "complexity": model.complexity,
                "expression": model.expression,
                "notes": model.notes,
                "fit_seconds": fit_seconds,
                **cert,
            }
            fitted.append(model)
            rows.append(row)
        except Exception as exc:
            rows.append(
                {
                    "target": target,
                    "model": getattr(factory, "__name__", "symbolic_factory"),
                    "family": "symbolic",
                    "selection_MAE": np.nan,
                    "selection_RMSE": np.nan,
                    "selection_MAPE": np.nan,
                    "selection_R2": np.nan,
                    "complexity": np.nan,
                    "expression": "",
                    "notes": f"Fit failed: {exc}",
                    "fit_seconds": time.perf_counter() - start,
                    "finite_violation_rate": np.nan,
                    "positive_violation_rate": np.nan,
                    "extra_finite_violation_rate": np.nan,
                    "extra_positive_violation_rate": np.nan,
                    "derivative_sign_violation_rate": np.nan,
                    "sensitivity_median": np.nan,
                    "sensitivity_p95": np.nan,
                    "extrapolation_stability_index": np.nan,
                    "physics_violation_rate": np.nan,
                }
            )

    selection = pd.DataFrame(rows)
    valid = selection[np.isfinite(selection["selection_RMSE"].astype(float))].copy()
    if valid.empty or not fitted:
        raise RuntimeError(f"No symbolic candidate fitted successfully for target {target}.")
    valid["selection_score"] = _selection_score(valid)
    selection = selection.merge(
        valid[["model", "selection_score"]], on="model", how="left"
    )
    winner_name = str(valid.sort_values("selection_score").iloc[0]["model"])
    winner = next(model for model in fitted if model.name == winner_name)
    return fitted, selection, winner


def _evaluate_model(
    model: FittedModel,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_idx: np.ndarray,
    df: pd.DataFrame,
    experiment: str,
    target: str,
    feature_names: List[str],
    random_state: int,
) -> Tuple[Dict[str, object], pd.DataFrame, Dict[str, float]]:
    if model.estimator is None:
        row = {
            "experiment": experiment,
            "target": target,
            "model": model.name,
            "family": model.family,
            "MAE": np.nan,
            "RMSE": np.nan,
            "MAPE": np.nan,
            "R2": np.nan,
            "complexity": model.complexity,
            "expression": model.expression,
            "notes": model.notes,
            "gpu_requested": model.gpu_requested,
            "gpu_used": model.gpu_used,
            "high_error_detection_AUROC": np.nan,
        }
        return row, pd.DataFrame(), {}

    y_pred = np.asarray(model.estimator.predict(X_test), dtype=float)
    metrics = regression_metrics(y_test, y_pred)
    scorer = ApplicabilityScorer(k=10).fit(X_train)
    support_score = scorer.score(X_test)
    valid_prediction = np.isfinite(y_pred) & (y_pred > 0)
    applicability = support_score * valid_prediction.astype(float)
    auc = high_error_detection_auc(y_test, y_pred, applicability)
    cert = physics_report(
        model.estimator,
        X_train,
        feature_names,
        target,
        random_state=random_state + 303,
        n_samples=1024,
    )
    row = {
        "experiment": experiment,
        "target": target,
        "model": model.name,
        "family": model.family,
        **metrics,
        "complexity": model.complexity,
        "expression": model.expression,
        "notes": model.notes,
        "gpu_requested": model.gpu_requested,
        "gpu_used": model.gpu_used,
        "high_error_detection_AUROC": auc,
        **cert,
    }
    pred = df.iloc[test_idx][feature_names].copy()
    pred.insert(0, "row_id", test_idx)
    pred.insert(1, "experiment", experiment)
    pred.insert(2, "target", target)
    pred.insert(3, "model", model.name)
    pred["y_true"] = y_test
    pred["y_pred"] = y_pred
    pred["absolute_error"] = np.abs(y_test - y_pred)
    pred["relative_error"] = pred["absolute_error"] / np.maximum(
        np.abs(y_test), np.finfo(float).eps
    )
    pred["applicability_score"] = applicability
    return row, pred, cert


def _save_model(model: FittedModel, experiment: str, target: str) -> None:
    if model.estimator is None:
        return
    path = MODELS_DIR / f"{_safe_name(experiment)}__{_safe_name(target)}__{_safe_name(model.name)}.joblib"
    joblib.dump(model.estimator, path)


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def save_equation_exports(selected: List[Dict[str, object]]) -> None:
    EQUATIONS_DIR.mkdir(parents=True, exist_ok=True)
    (EQUATIONS_DIR / "selected_equations.json").write_text(
        json.dumps(selected, indent=2, default=_json_default), encoding="utf-8"
    )
    lines = [
        "% Auto-generated selected symbolic equations.",
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Selected reliability-aware symbolic correlations.}",
        "\\begin{tabular}{lll}",
        "\\hline",
        "Target & Experiment & Equation \\\\",
        "\\hline",
    ]
    for item in selected:
        expr = _latex_escape(str(item["expression"]))
        lines.append(
            f"{_latex_escape(str(item['target']))} & "
            f"{_latex_escape(str(item['experiment']))} & "
            f"\\scriptsize{{{expr}}} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}", ""])
    (EQUATIONS_DIR / "selected_equations.tex").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def run_experiments(
    targets: List[str],
    random_state: int = 42,
    gp_population: int = 250,
    gp_generations: int = 6,
    full_gp: bool = False,
    xgb_estimators: int = 700,
) -> Dict[str, object]:
    ensure_dirs()
    df, mapping, csv_path = load_prepared_dataset()
    profile = dataset_profile(df)
    profile["source_csv"] = str(csv_path)
    profile["column_mapping"] = mapping
    (REPORTS_DIR / "dataset_profile.json").write_text(
        json.dumps(profile, indent=2, default=_json_default), encoding="utf-8"
    )

    ranges = []
    for col in FEATURES + [target for target in TARGETS if target in df.columns]:
        ranges.append(
            {
                "variable": col,
                "min": df[col].min(),
                "max": df[col].max(),
                "mean": df[col].mean(),
                "std": df[col].std(ddof=0),
            }
        )
    pd.DataFrame(ranges).to_csv(TABLES_DIR / "feature_ranges.csv", index=False)

    X = df[FEATURES].to_numpy(dtype=float)
    splits = make_splits(df, random_state=random_state)
    metrics_rows: List[Dict[str, object]] = []
    selection_rows: List[pd.DataFrame] = []
    predictions: List[pd.DataFrame] = []
    selected_equations: List[Dict[str, object]] = []
    selected_models_for_plots: Dict[str, Tuple[object, np.ndarray]] = {}

    base_random_train = splits["random80_20"]["train_idx"]
    base_random_test = splits["random80_20"]["test_idx"]

    experiment_items = list(splits.items())
    for frac in [0.20, 0.40, 0.60, 0.80]:
        rng = np.random.default_rng(random_state + int(frac * 1000))
        n_sub = max(20, int(len(base_random_train) * frac))
        train_sub = rng.choice(base_random_train, size=n_sub, replace=False)
        experiment_items.append(
            (
                f"sparse_{int(frac * 100)}",
                {
                    "train_idx": train_sub,
                    "test_idx": base_random_test,
                    "include_gp": False,
                    "description": f"Sparse-data run using {int(frac * 100)}% of the random training pool.",
                },
            )
        )
    for noise_pct in [1, 3, 5]:
        experiment_items.append(
            (
                f"noise_{noise_pct}",
                {
                    "train_idx": base_random_train,
                    "test_idx": base_random_test,
                    "include_gp": False,
                    "noise_pct": noise_pct,
                    "description": f"Training targets perturbed with {noise_pct}% Gaussian noise.",
                },
            )
        )

    for target in targets:
        if target not in df.columns:
            print(f"Skipping unknown target: {target}")
            continue
        y_full = df[target].to_numpy(dtype=float)
        for experiment, spec in experiment_items:
            print(f"\n=== {target} :: {experiment} ===")
            train_idx = np.asarray(spec["train_idx"], dtype=int)
            test_idx = np.asarray(spec["test_idx"], dtype=int)
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y_full[train_idx].copy()
            y_test = y_full[test_idx]
            if "noise_pct" in spec:
                rng = np.random.default_rng(random_state + int(spec["noise_pct"]) * 17)
                sigma = float(spec["noise_pct"]) / 100.0
                y_train = y_train * (1.0 + rng.normal(0.0, sigma, size=len(y_train)))
                y_train = np.maximum(y_train, np.finfo(float).eps)

            include_gp = bool(spec.get("include_gp", False)) and (
                full_gp or experiment in {"random80_20", "range_re_high_holdout", "geometry_holdout"}
            )
            symbolic_models, selection, winner = fit_symbolic_candidates(
                X_train,
                y_train,
                target,
                FEATURES,
                random_state=random_state,
                include_gp=include_gp,
                gp_population=gp_population,
                gp_generations=gp_generations,
            )
            selection.insert(0, "experiment", experiment)
            selection_rows.append(selection)

            certified = FittedModel(
                name="certified_symbolic",
                family="symbolic",
                estimator=winner.estimator,
                expression=winner.expression,
                complexity=winner.complexity,
                notes=f"Reliability-aware selection of {winner.name}: {winner.notes}",
            )
            selected_equations.append(
                {
                    "experiment": experiment,
                    "target": target,
                    "selected_model": winner.name,
                    "expression": winner.expression,
                    "complexity": winner.complexity,
                    "selection_record": selection[
                        selection["model"] == winner.name
                    ].to_dict(orient="records")[0],
                }
            )

            models = symbolic_models + [certified]
            if not experiment.startswith("noise_"):
                models.extend(
                    fit_black_box_models(
                        X_train,
                        y_train,
                        FEATURES,
                        target,
                        random_state=random_state,
                        xgb_estimators=xgb_estimators,
                    )
                )
            else:
                models.extend(
                    fit_black_box_models(
                        X_train,
                        y_train,
                        FEATURES,
                        target,
                        random_state=random_state,
                        xgb_estimators=max(250, xgb_estimators // 2),
                    )
                )

            for model in models:
                row, pred, _ = _evaluate_model(
                    model,
                    X_train,
                    X_test,
                    y_test,
                    test_idx,
                    df,
                    experiment,
                    target,
                    FEATURES,
                    random_state=random_state,
                )
                row["train_n"] = len(train_idx)
                row["test_n"] = len(test_idx)
                row["split_description"] = str(spec.get("description", ""))
                metrics_rows.append(row)
                if not pred.empty:
                    predictions.append(pred)
                _save_model(model, experiment, target)

            if experiment == "random80_20":
                selected_models_for_plots[target] = (winner.estimator, X_train)

    metrics = pd.DataFrame(metrics_rows)
    selections = pd.concat(selection_rows, ignore_index=True)
    prediction_table = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()

    metrics.to_csv(TABLES_DIR / "metrics_by_experiment.csv", index=False)
    selections.to_csv(TABLES_DIR / "symbolic_selection_scores.csv", index=False)
    prediction_table.to_csv(PREDICTIONS_DIR / "all_predictions.csv", index=False)

    model_summary = metrics[
        [
            "experiment",
            "target",
            "model",
            "family",
            "MAE",
            "RMSE",
            "MAPE",
            "R2",
            "complexity",
            "physics_violation_rate",
            "sensitivity_p95",
            "extrapolation_stability_index",
            "high_error_detection_AUROC",
            "gpu_requested",
            "gpu_used",
            "expression",
            "notes",
        ]
    ].copy()
    model_summary.to_csv(TABLES_DIR / "model_summary.csv", index=False)
    certification_cols = [
        col
        for col in metrics.columns
        if "violation" in col
        or col.startswith("sensitivity")
        or col == "extrapolation_stability_index"
        or col == "high_error_detection_AUROC"
    ]
    metrics[
        ["experiment", "target", "model", "family"] + certification_cols
    ].to_csv(TABLES_DIR / "certification_summary.csv", index=False)
    save_equation_exports(selected_equations)

    for target in targets:
        plot_pareto(model_summary, target, FIGURES_DIR)
        plot_predictions(prediction_table, target, FIGURES_DIR)
        plot_applicability(prediction_table, target, FIGURES_DIR)
        plot_noise(metrics, target, FIGURES_DIR)
        plot_metric_bars(metrics, target, FIGURES_DIR)
        if target in selected_models_for_plots:
            model, X_train_for_plot = selected_models_for_plots[target]
            plot_re_sweep(model, X_train_for_plot, FEATURES, target, FIGURES_DIR)

    manifest = {
        "timestamp_utc": pd.Timestamp.utcnow().isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "nvidia_smi": _nvidia_smi(),
        "dataset_csv": str(csv_path),
        "targets": targets,
        "random_state": random_state,
        "gp_population": gp_population,
        "gp_generations": gp_generations,
        "full_gp": full_gp,
        "xgb_estimators": xgb_estimators,
        "outputs": {
            "metrics": str(TABLES_DIR / "metrics_by_experiment.csv"),
            "model_summary": str(TABLES_DIR / "model_summary.csv"),
            "selection_scores": str(TABLES_DIR / "symbolic_selection_scores.csv"),
            "predictions": str(PREDICTIONS_DIR / "all_predictions.csv"),
            "selected_equations": str(EQUATIONS_DIR / "selected_equations.json"),
        },
    }
    (REPORTS_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8"
    )
    return manifest
