from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
ELSEVIER_DIR = WORKSPACE_ROOT / "Elsevier"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"
EQUATIONS_DIR = RESULTS_DIR / "equations"


def _fmt(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except Exception:
        return "--"
    if not pd.notna(value):
        return "--"
    if abs(value) >= 1000 or (0 < abs(value) < 0.01):
        return f"{value:.{digits}e}"
    return f"{value:.{digits}f}"


def _latex_escape(text: str) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _target_label(target: str) -> str:
    if target == "Nuavg":
        return "$Nu_{avg}$"
    if target == "DelP_Pa":
        return "$\\Delta p$"
    return _latex_escape(target)


MODEL_LABELS = {
    "ordinary_power_law": "Ordinary power law",
    "complexity_penalized_symbolic_lasso": "Sparse symbolic lasso",
    "unconstrained_gp_sr": "Unconstrained GP-SR",
    "positive_complexity_gp_sr": "Positive GP-SR",
    "certified_symbolic": "Selected symbolic",
    "random_forest": "Random forest",
    "xgboost_hist": "XGBoost",
}

EXPERIMENT_LABELS = {
    "random80_20": "Random 80/20 split",
    "range_re_high_holdout": "High-Reynolds-number holdout",
    "geometry_holdout": "Geometry holdout",
    "sparse_20": "20% sparse-data training",
    "sparse_40": "40% sparse-data training",
    "sparse_60": "60% sparse-data training",
    "sparse_80": "80% sparse-data training",
    "noise_1": "1% training-noise test",
    "noise_3": "3% training-noise test",
    "noise_5": "5% training-noise test",
}


def _model_label(model: str) -> str:
    return MODEL_LABELS.get(str(model), str(model).replace("_", " "))


def _experiment_label(experiment: str) -> str:
    return EXPERIMENT_LABELS.get(str(experiment), str(experiment).replace("_", " "))


def _display_expression(expression: str) -> str:
    escaped = _latex_escape(expression)
    return (
        escaped
        .replace("Nuavg", "$Nu_{avg}$")
        .replace("DelP\\_Pa", "$\\Delta p$")
        .replace("porosity", "$\\epsilon$")
        .replace("thickness", "$H_p$")
        .replace("amplitude", "$a$")
        .replace("wavelength", "$L_w$")
    )


def copy_artifacts() -> tuple[Path, Path]:
    out_fig = ELSEVIER_DIR / "article_artifacts" / "figures"
    out_tab = ELSEVIER_DIR / "article_artifacts" / "tables"
    out_fig.mkdir(parents=True, exist_ok=True)
    out_tab.mkdir(parents=True, exist_ok=True)
    for path in FIGURES_DIR.glob("*"):
        if path.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(path, out_fig / path.name)
    for path in TABLES_DIR.glob("*.csv"):
        shutil.copy2(path, out_tab / path.name)
    if (EQUATIONS_DIR / "selected_equations.tex").exists():
        shutil.copy2(EQUATIONS_DIR / "selected_equations.tex", out_tab / "selected_equations.tex")
    return out_fig, out_tab


def make_metric_table(metrics: pd.DataFrame, target: str, out_path: Path) -> None:
    keep_experiments = ["random80_20", "range_re_high_holdout", "geometry_holdout"]
    keep_models = [
        "ordinary_power_law",
        "complexity_penalized_symbolic_lasso",
        "unconstrained_gp_sr",
        "positive_complexity_gp_sr",
        "certified_symbolic",
        "random_forest",
        "xgboost_hist",
    ]
    df = metrics[
        (metrics["target"] == target)
        & (metrics["experiment"].isin(keep_experiments))
        & (metrics["model"].isin(keep_models))
    ].copy()
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{Predictive performance for {_target_label(target)}.}}",
        "\\label{tab:metrics-" + target.lower().replace("_", "-") + "}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "Experiment & Model & MAE & RMSE & MAPE (\\%) & $R^2$ \\\\",
        "\\midrule",
    ]
    for _, row in df.sort_values(["experiment", "MAPE", "model"]).iterrows():
        lines.append(
            f"{_latex_escape(_experiment_label(row['experiment']))} & {_latex_escape(_model_label(row['model']))} & "
            f"{_fmt(row['MAE'])} & {_fmt(row['RMSE'])} & {_fmt(row['MAPE'])} & {_fmt(row['R2'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_cert_table(metrics: pd.DataFrame, target: str, out_path: Path) -> None:
    df = metrics[
        (metrics["target"] == target)
        & (metrics["experiment"] == "random80_20")
        & (metrics["family"] == "symbolic")
    ].copy()
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        f"\\caption{{{_target_label(target)} reliability diagnostics.}}",
        "\\label{tab:cert-" + target.lower().replace("_", "-") + "}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Model & PVR & Sensitivity p95 & ESI & AUROC \\\\",
        "\\midrule",
    ]
    for _, row in df.sort_values(["physics_violation_rate", "MAPE", "model"]).iterrows():
        lines.append(
            f"{_latex_escape(_model_label(row['model']))} & {_fmt(row['physics_violation_rate'])} & "
            f"{_fmt(row['sensitivity_p95'])} & {_fmt(row['extrapolation_stability_index'])} & "
            f"{_fmt(row['high_error_detection_AUROC'])} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_data_dictionary_table(profile: dict, out_path: Path) -> None:
    ranges = profile.get("feature_ranges", {})
    target_ranges = profile.get("target_ranges", {})
    rows = [
        ("$Re$", "Reynolds number", "1", ranges.get("Re", {}), "Retained input."),
        ("$Pr$", "Prandtl number", "1", ranges.get("Pr", {}), "Retained input."),
        ("$Da$", "Darcy number", "1", ranges.get("Da", {}), "Retained input."),
        ("$\\epsilon$", "Porosity", "1", ranges.get("porosity", {}), "Retained input."),
        ("$H_p$", "Porous-slab thickness", "mm", ranges.get("thickness", {}), "Retained input."),
        ("$a$", "Wave amplitude", "mm", ranges.get("amplitude", {}), "Retained input."),
        ("$L_w$", "Wavelength", "mm", ranges.get("wavelength", {}), "Retained input; zero denotes the straight-channel case."),
        ("$Nu_{avg}$", "Average Nusselt number", "1", target_ranges.get("Nuavg", {}), "Target output."),
        ("$\\Delta p$", "Pressure drop", "Pa", target_ranges.get("DelP_Pa", {}), "Target output."),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Variable ranges.}",
        "\\label{tab:data-dictionary}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llllp{0.36\\linewidth}}",
        "\\toprule",
        "Symbol & Quantity & Unit & Range & Role and note \\\\",
        "\\midrule",
    ]
    for symbol, quantity, unit, stats, note in rows:
        lo = _fmt(stats.get("min"))
        hi = _fmt(stats.get("max"))
        lines.append(
            f"{symbol} & {_latex_escape(quantity)} & {unit} & {lo}--{hi} & {_latex_escape(note)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_nomenclature_table(profile: dict, out_path: Path) -> None:
    ranges = profile.get("feature_ranges", {})
    target_ranges = profile.get("target_ranges", {})

    def range_text(name: str, source: dict) -> str:
        stats = source.get(name, {})
        if not stats:
            return "--"
        return f"{_fmt(stats.get('min'))}--{_fmt(stats.get('max'))}"

    rows = [
        ("$Re$", "Reynolds number", "1", range_text("Re", ranges), "Flow inertial-to-viscous ratio."),
        ("$Pr$", "Prandtl number", "1", range_text("Pr", ranges), "Momentum-to-thermal diffusivity ratio."),
        ("$Da$", "Darcy number", "1", range_text("Da", ranges), "Permeability scale relative to channel scale."),
        ("$\\epsilon$", "Porosity", "1", range_text("porosity", ranges), "Porous-zone void fraction."),
        ("$H_p$", "Porous-slab thickness", "mm", range_text("thickness", ranges), "Thickness of the porous insert."),
        ("$a$", "Wave amplitude", "mm", range_text("amplitude", ranges), "Wavy-wall amplitude."),
        ("$L_w$", "Wavelength", "mm", range_text("wavelength", ranges), "Wavy-wall wavelength; zero denotes the straight-channel case."),
        ("$Nu_{avg}$", "Average Nusselt number", "1", range_text("Nuavg", target_ranges), "Heat-transfer target."),
        ("$\\Delta p$", "Pressure drop", "Pa", range_text("DelP_Pa", target_ranges), "Pressure-loss target."),
        ("PVR", "PVR", "1", "0--1", "Physics-violation rate for sampled finite, positive and monotonic checks."),
        ("ESI", "ESI", "1", "$\\ge 0$", "Extrapolation-stability index; lower values indicate less near-domain response growth."),
        ("AUROC", "AUROC", "1", "0--1", "Area under the ROC curve for ranking high relative-error cases using applicability risk."),
        ("MAPE", "MAPE", "\\%", "$\\ge 0$", "Mean absolute percentage error on the stated test split."),
        ("$A(x)$", "applicability score", "1", "0--1", "kNN feature-space support score; larger values indicate stronger local data support."),
    ]
    lines = [
        "\\begingroup",
        "\\setlength{\\tabcolsep}{2pt}",
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Nomenclature.}",
        "\\label{tab:nomenclature}",
        "\\scriptsize",
        "\\begin{tabular}{@{}p{0.10\\linewidth}p{0.18\\linewidth}p{0.12\\linewidth}p{0.15\\linewidth}p{0.35\\linewidth}@{}}",
        "\\toprule",
        "Symbol & Quantity & Unit & Range & Meaning \\\\",
        "\\midrule",
    ]
    for symbol, name, unit, valid_range, meaning in rows:
        lines.append(
            f"{symbol} & {_latex_escape(name)} & {unit} & {valid_range} & {_latex_escape(meaning)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", "\\endgroup", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_reproducibility_table(
    metrics: pd.DataFrame, manifest: dict, out_path: Path
) -> None:
    random_state = manifest.get("random_state", 42)
    gp_population = manifest.get("gp_population", "--")
    gp_generations = manifest.get("gp_generations", "--")
    xgb_estimators = manifest.get("xgb_estimators", "--")

    split_rows = (
        metrics[["experiment", "train_n", "test_n", "split_description"]]
        .drop_duplicates()
        .sort_values("experiment")
    )
    split_sizes = []
    for _, row in split_rows.iterrows():
        split_sizes.append(
            f"{_experiment_label(row['experiment'])}: {int(row['train_n'])}/{int(row['test_n'])}"
        )
    split_text = "; ".join(split_sizes)

    rows = [
        (
            "Random seeds",
            f"Base seed {random_state}; grouped geometry split seed {random_state + 1}; positive-target GP seed {random_state + 11}; selection diagnostic seed {random_state + 101}; test diagnostic seed {random_state + 303}; sparse/noise studies use deterministic offsets.",
        ),
        (
            "Outer train/test splits",
            split_text,
        ),
        (
            "Internal validation split",
            "Each symbolic-selection run holds out 25% of the outer training set for validation RMSE in J(f); the remaining 75% fits candidate equations.",
        ),
        (
            "Ordinary power law",
            "Ordinary least-squares regression on the logarithm of the target; strictly positive inputs enter through logarithmic factors; nonpositive-range geometry terms enter linearly inside the exponent.",
        ),
        (
            "Sparse symbolic lasso",
            "Sparse regression on the logarithm of the target with standardized basis terms; fivefold cross-validation over 80 regularization values; maximum 30000 iterations.",
        ),
        (
            "GP-SR",
            f"Population {gp_population}; generations {gp_generations}; tournament size 20; RMSE fitness; initial tree depth 2--4; constants in -5 to 5; 90% sample fraction; crossover, subtree mutation, hoist mutation and point mutation probabilities 0.70, 0.10, 0.05 and 0.10; all CPU cores; function set: addition, subtraction, multiplication, protected division, protected square root, protected logarithm, absolute value, negation and inverse; parsimony coefficients 0.0005 and 0.01.",
        ),
        (
            "Random Forest",
            "400 trees; minimum leaf size 1; base seed; all CPU cores; remaining settings are defaults.",
        ),
        (
            "XGBoost",
            f"Squared-error histogram boosting; {xgb_estimators} trees for non-noise runs and at least 250 for noise runs; depth 4; learning rate 0.035; row and feature subsampling 0.92; L2 regularization 1.0; base seed; CUDA requested with CPU fallback.",
        ),
        (
            "GPU configuration",
            "NVIDIA GeForce RTX 4060-class GPU, 8188 MiB memory; CUDA requested for XGBoost and CPU fallback allowed.",
        ),
    ]
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Reproducibility settings.}",
        "\\label{tab:reproducibility-settings}",
        "\\scriptsize",
        "\\begin{tabular}{p{0.24\\linewidth}p{0.70\\linewidth}}",
        "\\toprule",
        "Component & Setting \\\\",
        "\\midrule",
    ]
    for component, setting in rows:
        lines.append(f"{_latex_escape(component)} & {_latex_escape(setting)} \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _selection_score_for_weights(rows: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    columns = [
        ("selection_RMSE", False),
        ("complexity", False),
        ("physics_violation_rate", False),
        ("sensitivity_p95", True),
        ("extrapolation_stability_index", True),
    ]
    score_parts = []
    for column, log_transform in columns:
        weight = weights.get(column, 0.0)
        if weight == 0.0:
            continue
        values = rows[column].astype(float).to_numpy()
        if log_transform:
            values = np.log1p(values)
        finite = np.isfinite(values)
        if finite.sum() == 0:
            scaled = np.ones_like(values)
        else:
            finite_values = values[finite]
            lo = float(np.nanmin(finite_values))
            hi = float(np.nanmax(finite_values))
            if hi - lo < 1e-12:
                scaled = np.zeros_like(values)
            else:
                scaled = (values - lo) / (hi - lo)
                scaled[~finite] = 1.0
        score_parts.append(weight * scaled)
    if not score_parts:
        return pd.Series([0.0] * len(rows), index=rows.index)
    return pd.Series(sum(score_parts), index=rows.index)


def make_selection_ablation_table(
    selection: pd.DataFrame, metrics: pd.DataFrame, out_path: Path
) -> None:
    variants = [
        ("Error only", {"selection_RMSE": 1.0}),
        ("Error + complexity", {"selection_RMSE": 1.0, "complexity": 0.04}),
        (
            "Error + PVR",
            {"selection_RMSE": 1.0, "physics_violation_rate": 2.0},
        ),
        (
            "Full reliability-aware",
            {
                "selection_RMSE": 1.0,
                "complexity": 0.04,
                "physics_violation_rate": 2.0,
                "sensitivity_p95": 0.10,
                "extrapolation_stability_index": 0.10,
            },
        ),
    ]
    df = selection[selection["experiment"] == "random80_20"].copy()
    rows = []
    for target in df["target"].drop_duplicates():
        target_rows = df[df["target"] == target].copy()
        for variant_name, weights in variants:
            scores = _selection_score_for_weights(target_rows, weights)
            selected = target_rows.loc[scores.idxmin()]
            selected_model = str(selected["model"])
            test_row = _row(metrics, str(target), "random80_20", selected_model)
            source = test_row if test_row is not None else selected
            rows.append(
                (
                    _target_label(str(target)),
                    variant_name,
                    _model_label(selected_model),
                    source.get("MAPE", selected.get("selection_MAPE")),
                    source.get("complexity", selected.get("complexity")),
                    source.get(
                        "physics_violation_rate",
                        selected.get("physics_violation_rate"),
                    ),
                    source.get(
                        "extrapolation_stability_index",
                        selected.get("extrapolation_stability_index"),
                    ),
                    source.get("high_error_detection_AUROC", np.nan),
                )
            )
    lines = [
        "\\begin{table}[!htbp]",
        "\\centering",
        "\\caption{Selection-objective ablation.}",
        "\\label{tab:selection-ablation}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lllrrrrr}",
        "\\toprule",
        "Target & Objective & Selected candidate & MAPE (\\%) & Complexity & PVR & ESI & AUROC \\\\",
        "\\midrule",
    ]
    for target_label, variant_name, model, mape, complexity, pvr, esi, auroc in rows:
        lines.append(
            f"{target_label} & {_latex_escape(variant_name)} & {_latex_escape(model)} & "
            f"{_fmt(mape)} & {_fmt(complexity, 0)} & {_fmt(pvr, 4)} & "
            f"{_fmt(esi)} & {_fmt(auroc)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _row(metrics: pd.DataFrame, target: str, experiment: str, model: str) -> pd.Series | None:
    df = metrics[
        (metrics["target"] == target)
        & (metrics["experiment"] == experiment)
        & (metrics["model"] == model)
    ]
    if df.empty:
        return None
    return df.iloc[0]


def _best_blackbox(metrics: pd.DataFrame, target: str, experiment: str) -> pd.Series | None:
    df = metrics[
        (metrics["target"] == target)
        & (metrics["experiment"] == experiment)
        & (metrics["model"].isin(["random_forest", "xgboost_hist"]))
    ].copy()
    df = df[pd.notna(df["MAPE"])]
    if df.empty:
        return None
    return df.sort_values("MAPE").iloc[0]


def _metric(
    metrics: pd.DataFrame,
    target: str,
    experiment: str,
    model: str,
    column: str,
    digits: int = 3,
) -> str:
    row = _row(metrics, target, experiment, model)
    if row is None:
        return "--"
    return _fmt(row[column], digits=digits)


def write_bib() -> None:
    bib = r"""
@article{PachecoVega2006,
  title = {Heat transfer correlations by symbolic regression},
  author = {Pacheco-Vega, Arturo and Sen, Mihir and Yang, K. T. and McClain, R. L.},
  journal = {International Journal of Heat and Mass Transfer},
  volume = {49},
  pages = {4352--4359},
  year = {2006},
  doi = {10.1016/j.ijheatmasstransfer.2006.04.029}
}

@article{Zhu2026,
  title = {Genetic programming-symbolic regression: enabling interpretable and high-accuracy heat transfer correlations for supercritical fluids},
  author = {Zhu, Bingguo and Zhang, Qing and Wei, Mingtong and Shi, Houdong and Zhou, Zihao},
  journal = {Applied Thermal Engineering},
  volume = {283},
  pages = {128862},
  year = {2026},
  doi = {10.1016/j.applthermaleng.2025.128862}
}

@article{PINNSR2026,
  title = {Construction of heat transfer correlation by integrating physics-informed neural network and symbolic regression: Application to pyrolytic n-decane},
  author = {Wang, Ziwen and Zhan, Taotao and Jiang, Tao and Li, Yihang and Yang, Kai and Chen, Jian and Wang, Ning and Pan, Yu},
  journal = {International Journal of Heat and Mass Transfer},
  volume = {271},
  pages = {129149},
  year = {2026},
  doi = {10.1016/j.ijheatmasstransfer.2026.129149}
}

@article{Li2026,
  title = {Physics-Constrained Symbolic Regression of Collision Integrals for Dilute-Gas Viscosity},
  author = {Li, Z. and Duan, Y. and Yang, X.},
  journal = {International Journal of Thermophysics},
  volume = {47},
  pages = {136},
  year = {2026},
  doi = {10.1007/s10765-026-03809-4}
}

@misc{KumarPandey2026,
  title = {CFD-informed machine learning surrogate dataset for thermo-hydraulic prediction in partially porous wavy channels},
  author = {Kumar, Prince and Pandey, K. M.},
  howpublished = {Mendeley Data},
  note = {Version 2, published 16 June 2026},
  year = {2026},
  url = {https://data.mendeley.com/datasets/5b5n3cg32n/2},
  doi = {10.17632/5b5n3cg32n.2}
}

@inproceedings{ChenGuestrin2016,
  title = {XGBoost: A scalable tree boosting system},
  author = {Chen, Tianqi and Guestrin, Carlos},
  booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages = {785--794},
  year = {2016},
  doi = {10.1145/2939672.2939785}
}

@article{Cranmer2023,
  title = {Interpretable machine learning for science with PySR and SymbolicRegression.jl},
  author = {Cranmer, Miles},
  journal = {arXiv preprint arXiv:2305.01582},
  year = {2023}
}

@article{XiangChen2025,
  title = {Discovery of classical gas-solid flow correlations using a reinforcement learning-based symbolic regression framework},
  author = {Xiang, Zhong and Chen, Xi},
  journal = {Chemical Engineering Science},
  volume = {314},
  pages = {121767},
  year = {2025},
  doi = {10.1016/j.ces.2025.121767}
}

@article{Wu2025,
  title = {A framework for learning symbolic turbulence models from indirect observation data via neural networks and feature importance analysis},
  author = {Wu, Chutian and Zhang, Xin-Lei and Xu, Duo and He, Guowei},
  journal = {Journal of Computational Physics},
  volume = {537},
  pages = {114068},
  year = {2025},
  doi = {10.1016/j.jcp.2025.114068}
}

@article{Tang2024,
  title = {Enhancing the SST turbulence model for predicting heat flux in hypersonic flows through symbolic regression},
  author = {Tang, Denggao and Zeng, Fanzhi and Yi, Chen and Zhang, Tianxin and Yan, Chao},
  journal = {Acta Astronautica},
  volume = {222},
  pages = {244--271},
  year = {2024},
  doi = {10.1016/j.actaastro.2024.06.018}
}

@article{Shi2025,
  title = {Hybrid symbolic regression for data-driven discovery: Governing dimensionless numbers in supercritical heat transfer},
  author = {Shi, Yunzhi and Song, Meiqi and Bi, Hongtao and Xu, Wei and Liu, Xiaojing},
  journal = {Energy},
  volume = {338},
  pages = {138752},
  year = {2025},
  doi = {10.1016/j.energy.2025.138752}
}

@article{Chowdhury2026,
  title = {Modeling and prediction of Carreau fluid flow with generalized thermal conductivity using hybrid artificial neural network and symbolic regression},
  author = {Chowdhury, Rajkumar Saha and Bhaumik, Bivas and Sarkar, Golam Mortuja and Sahoo, Bikash},
  journal = {Engineering Applications of Artificial Intelligence},
  volume = {167},
  pages = {113781},
  year = {2026},
  doi = {10.1016/j.engappai.2026.113781}
}

@article{Ji2026,
  title = {A few-shot and physically restorable symbolic regression turbulence model based on normalized general effective-viscosity hypothesis},
  author = {Ji, Ziqi and Duan, Penghao and Du, Gang},
  journal = {Aerospace Science and Technology},
  volume = {179},
  pages = {113475},
  year = {2026},
  doi = {10.1016/j.ast.2026.113475}
}

@article{Panczyk2025,
  title = {Opening the AI black-box: Symbolic regression with Kolmogorov-Arnold Networks for advanced energy applications},
  author = {Panczyk, Nataly R. and Erdem, Omer F. and Radaideh, Majdi I.},
  journal = {Energy and AI},
  volume = {22},
  pages = {100595},
  year = {2025},
  doi = {10.1016/j.egyai.2025.100595}
}

@article{ZhaoZhao2025,
  title = {Uncertainty quantification based on symbolic regression and probabilistic programming and its application},
  author = {Zhao, Yuyang and Zhao, Hongbo},
  journal = {Machine Learning with Applications},
  volume = {20},
  pages = {100632},
  year = {2025},
  doi = {10.1016/j.mlwa.2025.100632}
}

@misc{Darooneh2026,
  title = {Enhancing Symbolic Regression and Universal Physics-Informed Neural Networks with Dimensional Analysis},
  author = {Darooneh, Diba and Podina, Lena and Grewal, Joshveer and Kohandel, Mohammad},
  year = {2026},
  eprint = {2411.15919},
  archivePrefix = {arXiv}
}

@article{ZhuChenLi2025,
  title = {A physics-guided symbolic regression framework for efficient and interpretable sand constitutive modeling},
  author = {Zhu, Yi and Chen, Su and Li, Xiaojun},
  journal = {Canadian Geotechnical Journal},
  volume = {62},
  pages = {1--14},
  year = {2025},
  doi = {10.1139/cgj-2025-0201}
}

@inproceedings{FongMotani2025,
  title = {Pareto-Optimal Fronts for Benchmarking Symbolic Regression Algorithms},
  author = {Fong, Kei Sen and Motani, Mehul},
  booktitle = {Proceedings of the 42nd International Conference on Machine Learning},
  pages = {17392--17410},
  series = {Proceedings of Machine Learning Research},
  volume = {267},
  publisher = {PMLR},
  year = {2025},
  url = {https://proceedings.mlr.press/v267/fong25b.html}
}

@article{CoverHart1967,
  title = {Nearest Neighbor Pattern Classification},
  author = {Cover, Thomas M. and Hart, Peter E.},
  journal = {IEEE Transactions on Information Theory},
  volume = {13},
  number = {1},
  pages = {21--27},
  year = {1967},
  doi = {10.1109/TIT.1967.1053964}
}

@article{Fawcett2006,
  title = {An Introduction to ROC Analysis},
  author = {Fawcett, Tom},
  journal = {Pattern Recognition Letters},
  volume = {27},
  number = {8},
  pages = {861--874},
  year = {2006},
  doi = {10.1016/j.patrec.2005.10.010}
}

@book{Saltelli2008,
  title = {Global Sensitivity Analysis: The Primer},
  author = {Saltelli, Andrea and Ratto, Marco and Andres, Terry and Campolongo, Francesca and Cariboni, Jessica and Gatelli, Debora and Saisana, Michaela and Tarantola, Stefano},
  publisher = {John Wiley \& Sons},
  year = {2008},
  doi = {10.1002/9780470725184}
}

@article{Schultz2025,
  title = {A general approach for determining applicability domain of machine learning models},
  author = {Schultz, Lane E. and Wang, Yiqi and Jacobs, Ryan and Morgan, Dane},
  journal = {npj Computational Materials},
  year = {2025},
  doi = {10.1038/s41524-025-01573-x}
}

@article{Taskin2026,
  title = {Knowledge integration for physics-informed symbolic regression using pre-trained large language models},
  author = {Taskin, Bilge and Xie, Wenxiong and Lazebnik, Teddy},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-35327-6}
}

@article{LazebnikLiberzon2026,
  title = {Moving from table to graph in physics-informed spatio-temporal symbolic regression},
  author = {Lazebnik, Teddy and Liberzon, Alex},
  journal = {Scientific Reports},
  year = {2026},
  doi = {10.1038/s41598-026-53882-w}
}

@article{Anselment2025,
  title = {Systematic tree search for symbolic regression: deterministically searching the space of dimensionally homogeneous models},
  author = {Anselment, Marcel and Neumaier, Moritz and Rudolph, Stephan},
  journal = {CEAS Aeronautical Journal},
  volume = {17},
  pages = {793--808},
  year = {2026},
  doi = {10.1007/s13272-025-00886-3}
}
"""
    (ELSEVIER_DIR / "rcsr_refs.bib").write_text(bib.strip() + "\n", encoding="utf-8")


def make_sota_table(metrics: pd.DataFrame, primary: str, out_path: Path) -> None:
    cert_random = _row(metrics, primary, "random80_20", "certified_symbolic")
    range_cert = _row(metrics, primary, "range_re_high_holdout", "certified_symbolic")
    geom_cert = _row(metrics, primary, "geometry_holdout", "certified_symbolic")
    delp_random = _row(metrics, "DelP_Pa", "random80_20", "certified_symbolic")
    delp_range = _row(metrics, "DelP_Pa", "range_re_high_holdout", "certified_symbolic")
    pvr = _fmt(cert_random["physics_violation_rate"], 4) if cert_random is not None else "--"
    random_mape = _fmt(cert_random["MAPE"]) if cert_random is not None else "--"
    range_mape = _fmt(range_cert["MAPE"]) if range_cert is not None else "--"
    geom_mape = _fmt(geom_cert["MAPE"]) if geom_cert is not None else "--"
    delp_random_mape = _fmt(delp_random["MAPE"]) if delp_random is not None else "--"
    delp_range_mape = _fmt(delp_range["MAPE"]) if delp_range is not None else "--"
    target_label = _target_label(primary)

    rows = [
        (
            "\\cite{PachecoVega2006}",
            "Heat-transfer correlations",
            "Tabular engineering data",
            "Genetic-programming SR",
            "Variable choice and equation form",
            "Not reported as a separate applicability-domain score",
            "Correlation recovery and comparison",
            "Demonstrated compact symbolic heat-transfer equations.",
        ),
        (
            "\\cite{Zhu2026}",
            "Supercritical-fluid heat transfer",
            "Supercritical-fluid data",
            "GP-SR",
            "Problem-specific heat-transfer variables",
            "No explicit kNN applicability score reported",
            "Comparison with traditional formulas",
            "Reported interpretable high-accuracy correlations.",
        ),
        (
            "\\cite{PINNSR2026}",
            "Pyrolytic n-decane regenerative cooling",
            "CFD or simulation-derived field data",
            "Feature screening plus SR",
            "Physics-informed broad neural-network stage",
            "Validity restricted by pyrolysis conversion limit",
            "Independent-test field validation",
            "Field errors below 5\\%; final Nu correlation limited to conversion below 35\\%.",
        ),
        (
            "\\cite{Shi2025}",
            "Supercritical heat-transfer scaling",
            "1492 supercritical-water points",
            "Hybrid SR neural network",
            "Dimensional invariance and active subspaces",
            "No separate applicability-domain score reported",
            "Data-fit and scaling-factor evaluation",
            "Identified compact dimensionless heat-transfer factors.",
        ),
        (
            "\\cite{Tang2024}",
            "Hypersonic wall heat flux",
            "RANS/field-inversion data",
            "SR for closure correction",
            "Embedded turbulent-Prandtl correction",
            "Assessed through flow-case transfer",
            "Wall-cooling, Mach, Reynolds and 3D cases",
            "Improved heat-flux prediction across tested cases.",
        ),
        (
            "\\cite{Wu2025}",
            "Turbulence-model discovery from indirect data",
            "Indirect observation data",
            "Feature-screened SR",
            "Neural inference of latent quantities",
            "Not framed as applicability-domain scoring",
            "Case-study computational comparison",
            "Feature screening reduced SR cost from 1237 min to 176 min.",
        ),
        (
            "\\cite{Chowdhury2026}",
            "Carreau fluid flow and heat transfer",
            "Numerical solver or ANN response data",
            "ANN-to-symbolic distillation",
            "Governing non-Newtonian flow setup",
            "No explicit applicability-domain score reported",
            "Comparison with numerical/ANN solutions",
            "Reported ANN errors below 1\\% for skin-friction and Nu quantities.",
        ),
        (
            "\\cite{Ji2026}",
            "Few-shot turbulence modeling",
            "Few-shot turbulence data",
            "Physically restorable SR",
            "Normalized effective-viscosity hypothesis",
            "Reliability treated through restoration tests",
            "Few-shot recovery benchmarks",
            "Reported improved recovery of turbulence-model forms.",
        ),
        (
            "\\cite{XiangChen2025}",
            "Gas-solid minimum-fluidization velocity",
            "Gas-solid flow correlation data",
            "PhySO reinforcement-learning SR",
            "Dimensional homogeneity and dimensionless groups",
            "Noise robustness reported",
            "Regression accuracy and noise tests",
            "Reduced Ergun-form complexity from 24 to 14 with $R^2=0.9930$.",
        ),
        (
            "\\cite{Darooneh2026}",
            "Dimensional-analysis-enhanced SR and PINNs",
            "Benchmark or physics-system data",
            "SR/PINN with Buckingham-Pi preprocessing",
            "Dimensionless-group transformation",
            "Not centered on applicability-domain scoring",
            "Benchmark tests",
            "Showed benefits from enforcing dimensionless structure before learning.",
        ),
        (
            "\\cite{Anselment2025}",
            "Dimensionally homogeneous model search",
            "Symbolic-regression benchmarks",
            "Deterministic tree search",
            "Dimensional homogeneity enforced during search",
            "No data-support applicability score reported",
            "Benchmark model search",
            "Found physically admissible expressions by pruning invalid units.",
        ),
        (
            "\\cite{ZhuChenLi2025}",
            "Sand constitutive modeling",
            "Geotechnical constitutive data",
            "Physics-guided SR",
            "Classical constitutive forms and boundary constraints",
            "Robustness and physical consistency reported",
            "Constitutive-equation benchmarks",
            "Reported improved convergence and physical consistency.",
        ),
        (
            "\\cite{Li2026}",
            "Dilute-gas viscosity collision integrals",
            "87 fluids; 11263 property points",
            "Physics-constrained SR",
            "Transformed collision-integral formulation",
            "No kNN applicability score reported",
            "Property-correlation validation",
            "Reported overall AARD of 2.00\\%.",
        ),
        (
            "\\cite{Panczyk2025}",
            "KAN symbolic equations for energy systems",
            "Low-output-dimensional energy cases",
            "KAN symbolic extraction",
            "SHAP-based physical interpretation",
            "No explicit applicability-domain score reported",
            "KAN/FNN accuracy comparison",
            "Found KAN and FNN accuracy comparable in selected cases.",
        ),
        (
            "\\cite{ZhaoZhao2025}",
            "Uncertainty-aware empirical equations",
            "Empirical roughness-correlation data",
            "SR with probabilistic programming",
            "Problem-specific empirical-equation structure",
            "Uncertainty quantification included",
            "Generalization comparison",
            "Reported improved generalization over deterministic empirical equations.",
        ),
        (
            "\\cite{Schultz2025}",
            "Applicability domain for ML models",
            "Materials-ML datasets",
            "No SR engine",
            "Feature-space density and dissimilarity",
            "Applicability-domain method",
            "Residual and uncertainty reliability tests",
            "High dissimilarity associated with larger residuals and unreliable uncertainty.",
        ),
        (
            "\\cite{Taskin2026}",
            "LLM-guided physics-informed SR",
            "Physics-equation discovery benchmarks",
            "LLM-scored physics-informed SR",
            "Pre-trained LLM supplies physics-informed score",
            "No kNN applicability score reported",
            "Engine and prompt comparisons",
            "Reported gains depend on engine, language model and prompt.",
        ),
        (
            "\\cite{LazebnikLiberzon2026}",
            "Spatio-temporal physics-informed SR",
            "Functional, ODE, PDE, integral and delay systems",
            "Graph and tabular physics-informed SR",
            "Representation-level physical structure",
            "Noise cases included",
            "Spatio-temporal recovery benchmarks",
            "Reported improved recovery in several noisy settings.",
        ),
        (
            "\\cite{FongMotani2025}",
            "SR benchmarking",
            "34 SRBench datasets",
            "Multiple SR algorithms",
            "Accuracy--length Pareto framing",
            "No physics-specific applicability score",
            "Exhaustive APO-front benchmarking",
            "Constructed Pareto-optimal accuracy--length fronts using large compute budgets.",
        ),
        (
            "Present work",
            f"Partially porous wavy-channel {target_label} and $\\Delta p$",
            "CFD-derived full-factorial surrogate dataset",
            "Power law, sparse symbolic lasso and GP-SR candidates",
            "Post-fit finite, positive and monotonic checks; dimensional homogeneity not enforced",
            "PVR, sensitivity, ESI, kNN applicability score and AUROC",
            "Random split, high-Re holdout, geometry holdout, sparse-data and noise tests",
            f"{target_label}: random MAPE {random_mape}\\%, high-Re {range_mape}\\%, geometry {geom_mape}\\%, PVR {pvr}; $\\Delta p$: random MAPE {delp_random_mape}\\%, high-Re {delp_range_mape}\\%.",
        ),
    ]

    lines = [
        "\\begingroup",
        "\\setlength{\\tabcolsep}{1pt}",
        "\\begin{center}",
        "\\tiny",
        "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.08\\linewidth}>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.13\\linewidth}>{\\raggedright\\arraybackslash}p{0.13\\linewidth}>{\\raggedright\\arraybackslash}p{0.12\\linewidth}>{\\raggedright\\arraybackslash}p{0.13\\linewidth}@{}}",
        "\\caption{Methodological comparison.}",
        "\\label{tab:sota-comparison}\\\\",
        "\\toprule",
        "Study & Problem & Data type & Symbolic engine & Physics constraint & Uncertainty or AD & Validation & Reported result \\\\",
        "\\midrule",
        "\\endfirsthead",
        "\\caption[]{Methodological comparison (continued).}\\\\",
        "\\toprule",
        "Study & Problem & Data type & Symbolic engine & Physics constraint & Uncertainty or AD & Validation & Reported result \\\\",
        "\\midrule",
        "\\endhead",
        "\\midrule",
        "\\multicolumn{8}{r}{Continued on next page}\\\\",
        "\\endfoot",
        "\\bottomrule",
        "\\endlastfoot",
    ]
    for row in rows:
        lines.append(" & ".join(row) + " \\\\")
    lines.extend(["\\end{longtable}", "\\end{center}", "\\endgroup", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_supplementary_figure_table(
    rows: list[tuple[str, str, str, str]], out_path: Path
) -> None:
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        "\\caption{Supplementary figure index.}",
        "\\label{tab:supplementary-figures}",
        "\\scriptsize",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llll}",
        "\\toprule",
        "Artifact file & Target & Split/model context & Main conclusion \\\\",
        "\\midrule",
    ]
    for filename, target_label, context, conclusion in rows:
        lines.append(
            f"{_latex_escape(filename)} & {target_label} & {_latex_escape(context)} & "
            f"{_latex_escape(conclusion)} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}", ""])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    metrics_path = TABLES_DIR / "metrics_by_experiment.csv"
    selection_path = TABLES_DIR / "symbolic_selection_scores.csv"
    profile_path = RESULTS_DIR / "reports" / "dataset_profile.json"
    manifest_path = RESULTS_DIR / "reports" / "run_manifest.json"
    equations_path = EQUATIONS_DIR / "selected_equations.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing {metrics_path}. Run scripts/run_experiments.py first.")

    metrics = pd.read_csv(metrics_path)
    selection = pd.read_csv(selection_path)
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = json.loads(equations_path.read_text(encoding="utf-8"))
    _, article_tables = copy_artifacts()
    write_bib()

    make_data_dictionary_table(profile, article_tables / "data_dictionary.tex")
    make_nomenclature_table(profile, article_tables / "nomenclature.tex")
    make_reproducibility_table(metrics, manifest, article_tables / "reproducibility_settings.tex")
    make_selection_ablation_table(selection, metrics, article_tables / "selection_ablation.tex")

    targets = list(metrics["target"].drop_duplicates())
    for target in targets:
        make_metric_table(metrics, target, article_tables / f"metrics_{target}.tex")
        make_cert_table(metrics, target, article_tables / f"certification_{target}.tex")

    primary = "Nuavg" if "Nuavg" in targets else targets[0]
    make_sota_table(metrics, primary, article_tables / "sota_comparison.tex")
    certified_random = _row(metrics, primary, "random80_20", "certified_symbolic")
    power_random = _row(metrics, primary, "random80_20", "ordinary_power_law")
    black_random = _best_blackbox(metrics, primary, "random80_20")
    range_cert = _row(metrics, primary, "range_re_high_holdout", "certified_symbolic")
    geom_cert = _row(metrics, primary, "geometry_holdout", "certified_symbolic")
    selected_primary = next(
        item
        for item in selected
        if item["target"] == primary and item["experiment"] == "random80_20"
    )

    certified_mape = _fmt(certified_random["MAPE"]) if certified_random is not None else "--"
    power_mape = _fmt(power_random["MAPE"]) if power_random is not None else "--"
    black_model = str(black_random["model"]) if black_random is not None else "black-box baseline"
    black_mape = _fmt(black_random["MAPE"]) if black_random is not None else "--"
    range_mape = _fmt(range_cert["MAPE"]) if range_cert is not None else "--"
    geom_mape = _fmt(geom_cert["MAPE"]) if geom_cert is not None else "--"
    selected_primary_model = str(selected_primary.get("selected_model", "--"))

    nu_target = "Nuavg" if "Nuavg" in targets else primary
    delp_target = "DelP_Pa" if "DelP_Pa" in targets else primary
    selected_delp = next(
        (
            item
            for item in selected
            if item["target"] == delp_target and item["experiment"] == "random80_20"
        ),
        {},
    )
    selected_delp_model = str(selected_delp.get("selected_model", "--"))
    nu_cert_random_mape = _metric(metrics, nu_target, "random80_20", "certified_symbolic", "MAPE")
    nu_rf_random_mape = _metric(metrics, nu_target, "random80_20", "random_forest", "MAPE")
    nu_xgb_random_mape = _metric(metrics, nu_target, "random80_20", "xgboost_hist", "MAPE")
    nu_cert_range_mape = _metric(metrics, nu_target, "range_re_high_holdout", "certified_symbolic", "MAPE")
    nu_rf_range_mape = _metric(metrics, nu_target, "range_re_high_holdout", "random_forest", "MAPE")
    nu_xgb_range_mape = _metric(metrics, nu_target, "range_re_high_holdout", "xgboost_hist", "MAPE")
    nu_cert_geom_mape = _metric(metrics, nu_target, "geometry_holdout", "certified_symbolic", "MAPE")
    nu_rf_geom_mape = _metric(metrics, nu_target, "geometry_holdout", "random_forest", "MAPE")
    nu_xgb_geom_mape = _metric(metrics, nu_target, "geometry_holdout", "xgboost_hist", "MAPE")
    delp_cert_random_mape = _metric(metrics, delp_target, "random80_20", "certified_symbolic", "MAPE")
    delp_rf_random_mape = _metric(metrics, delp_target, "random80_20", "random_forest", "MAPE")
    delp_xgb_random_mape = _metric(metrics, delp_target, "random80_20", "xgboost_hist", "MAPE")
    delp_cert_range_mape = _metric(metrics, delp_target, "range_re_high_holdout", "certified_symbolic", "MAPE")
    delp_rf_range_mape = _metric(metrics, delp_target, "range_re_high_holdout", "random_forest", "MAPE")
    delp_xgb_range_mape = _metric(metrics, delp_target, "range_re_high_holdout", "xgboost_hist", "MAPE")
    delp_cert_geom_mape = _metric(metrics, delp_target, "geometry_holdout", "certified_symbolic", "MAPE")
    delp_rf_geom_mape = _metric(metrics, delp_target, "geometry_holdout", "random_forest", "MAPE")
    delp_xgb_geom_mape = _metric(metrics, delp_target, "geometry_holdout", "xgboost_hist", "MAPE")
    delp_cert_random_auroc = _metric(
        metrics,
        delp_target,
        "random80_20",
        "certified_symbolic",
        "high_error_detection_AUROC",
    )
    delp_cert_geom_auroc = _metric(
        metrics,
        delp_target,
        "geometry_holdout",
        "certified_symbolic",
        "high_error_detection_AUROC",
    )
    symbolic_pvr_max = _fmt(
        metrics[metrics["family"] == "symbolic"]["physics_violation_rate"].max(),
        digits=4,
    )
    ranges = profile["feature_ranges"]
    validity_ranges = (
        f"$Re$={_fmt(ranges['Re']['min'])}--{_fmt(ranges['Re']['max'])}, "
        f"$Pr$={_fmt(ranges['Pr']['min'])}--{_fmt(ranges['Pr']['max'])}, "
        f"$Da$={_fmt(ranges['Da']['min'])}--{_fmt(ranges['Da']['max'])}, "
        f"$\\epsilon$={_fmt(ranges['porosity']['min'])}--{_fmt(ranges['porosity']['max'])}, "
        f"$H_p$={_fmt(ranges['thickness']['min'])}--{_fmt(ranges['thickness']['max'])} mm, "
        f"$a$={_fmt(ranges['amplitude']['min'])}--{_fmt(ranges['amplitude']['max'])} mm, "
        f"and $L_w$={_fmt(ranges['wavelength']['min'])}--{_fmt(ranges['wavelength']['max'])} mm"
    )

    figure_dir_name = "figures"
    if (ELSEVIER_DIR / "article_artifacts" / "figures_reliability_aware").exists():
        figure_dir_name = "figures_reliability_aware"

    method_pipeline_intro = (
        "Figure~\\ref{fig:method-pipeline} summarizes the proposed reliability-aware "
        "symbolic-regression pipeline and clarifies where data splitting, model fitting, "
        "soft diagnostic ranking, applicability gating, and artifact export enter the workflow."
    )
    method_pipeline_figure = ""
    method_pipeline_path = None
    for suffix in ("pdf", "png"):
        candidate = (
            ELSEVIER_DIR
            / "article_artifacts"
            / figure_dir_name
            / f"method_pipeline.{suffix}"
        )
        if candidate.exists():
            method_pipeline_path = candidate
            break
    if method_pipeline_path is not None:
        method_pipeline_figure = (
            "\\begin{figure}[!htbp]\n"
            "\\centering\n"
            f"\\includegraphics[width=\\linewidth]{{article_artifacts/{figure_dir_name}/{method_pipeline_path.name}}}\n"
            "\\caption{Reliability-aware workflow.}\n"
            "\\label{fig:method-pipeline}\n"
            "\\end{figure}\n"
        )

    main_figure_lines = []
    supplementary_figure_rows = []
    for target in targets:
        target_label = _target_label(target)
        safe_target = target.lower().replace("_", "-")
        main_specs = [
            (
                f"pareto_front_{target}.pdf",
                "Pareto front",
            ),
            (
                f"predicted_vs_measured_{target}.pdf",
                "Predicted versus measured",
            ),
            (
                f"model_comparison_mape_{target}.pdf",
                "Split-wise MAPE",
            ),
        ]
        supplementary_specs = [
            (
                f"applicability_error_{target}.pdf",
                "all exported test splits; selected symbolic model",
                f"Applicability-risk ranking is summarized by random-split AUROC {_metric(metrics, target, 'random80_20', 'certified_symbolic', 'high_error_detection_AUROC')}.",
            ),
            (
                f"reynolds_sweep_{target}.pdf",
                "random-split selected symbolic model; median non-Re inputs",
                "The selected correlation varies smoothly across the sampled Reynolds-number range.",
            ),
            (
                f"noise_robustness_{target}.pdf",
                "noise-perturbed training runs; selected symbolic, power-law and XGBoost models",
                "The exported plot reports clean-test RMSE sensitivity to injected target noise.",
            ),
        ]
        panel_lines = []
        for panel_number, (filename, panel_title) in enumerate(main_specs):
            if (ELSEVIER_DIR / "article_artifacts" / figure_dir_name / filename).exists():
                panel_letter = chr(ord("a") + panel_number)
                if panel_lines:
                    panel_lines.append("\\vspace{0.45em}")
                panel_lines.extend(
                    [
                        f"\\includegraphics[width=0.78\\linewidth,height=0.25\\textheight,keepaspectratio]{{article_artifacts/{figure_dir_name}/{filename}}}",
                        f"\\par\\small\\textbf{{({panel_letter})}} {panel_title}.\\par",
                    ]
                )
        if panel_lines:
            main_figure_lines.append(
                "\\begin{figure}[!htbp]\n"
                "\\centering\n"
                + "\n".join(panel_lines)
                + "\n"
                f"\\caption{{{target_label} diagnostics.}}\n"
                f"\\label{{fig:{safe_target}-diagnostics}}\n"
                "\\end{figure}\n"
                )
        for filename, context, conclusion in supplementary_specs:
            if (ELSEVIER_DIR / "article_artifacts" / figure_dir_name / filename).exists():
                supplementary_figure_rows.append((filename, target_label, context, conclusion))
    make_supplementary_figure_table(
        supplementary_figure_rows, article_tables / "supplementary_figure_index.tex"
    )

    table_inputs = []
    for target in targets:
        table_inputs.append(f"\\input{{article_artifacts/tables/metrics_{target}.tex}}")
        table_inputs.append(f"\\input{{article_artifacts/tables/certification_{target}.tex}}")

    manuscript = rf"""
\documentclass[preprint,12pt]{{elsarticle}}

\usepackage{{amssymb}}
\usepackage{{amsmath}}
\usepackage{{booktabs}}
\usepackage{{graphicx}}
\usepackage{{longtable}}
\usepackage{{array}}
\usepackage{{placeins}}
\usepackage{{hyperref}}
\pdfstringdefDisableCommands{{%
  \def\corref#1{{}}%
  \def\cnotenum#1{{}}%
}}
\urlstyle{{same}}
\emergencystretch=2em

\journal{{AI Thermal Fluids}}

\begin{{document}}
\hypersetup{{pageanchor=false}}

\begin{{frontmatter}}

\title{{Reliability-Aware Physics-Constrained Symbolic Regression for Generalizable Heat-Transfer Correlation Discovery}}

\author[inst1]{{Yacine Khaldi}}
\ead{{khaldi.yacine@ens-ouargla.dz}}

\author[inst2]{{Hanane Azzaoui\corref{{cor1}}}}
\ead{{hanane.azzaoui@univ-ouargla.dz}}

\author[inst2]{{Mohamed Ben Bezziane}}
\ead{{benbezziane.mohamed@univ-ouargla.dz}}

\author[inst3]{{Amir Benzaoui}}
\ead{{a.benzaoui@univ-skikda.dz}}

\cortext[cor1]{{Corresponding author}}

\address[inst1]{{Mathematics Department, \'{{E}}cole Normale Sup\'{{e}}rieure de Ouargla, Ouargla, Algeria}}
\address[inst2]{{LINATI Laboratory, Kasdi Merbah University, Ouargla, Algeria}}
\address[inst3]{{Electrical Engineering Department, University 20 August 1955, Skikda, Algeria}}

\begin{{abstract}}
Symbolic regression is attractive for heat-transfer modeling because it produces explicit engineering correlations rather than only numerical predictors. However, symbolic heat-transfer correlation discovery is already established, so the central contribution of this work is a reliability-aware screening layer around symbolic regression. The proposed pipeline qualifies correlations using predictive error, algebraic complexity, physics-violation rate, extrapolation stability, local sensitivity, and an applicability-domain score. These diagnostics are heuristic, sampled checks; they do not constitute a formal guarantee of global equation validity. On the CFD-derived partially porous wavy-channel dataset of {profile['n_rows']} samples and {profile['n_geometry_combinations']} geometry combinations, the reliability-aware selected symbolic correlation for {_target_label(primary)} reached a random-split MAPE of {certified_mape}\%, compared with {power_mape}\% for an ordinary power law and {black_mape}\% for the best black-box reference ({_latex_escape(black_model)}). Under high-Reynolds-number and geometry holdout tests, the selected symbolic MAPE was {range_mape}\% and {geom_mape}\%, respectively. The results support treating symbolic regression as an equation discovery and qualification workflow, not only as a curve-fitting tool.
\end{{abstract}}

\begin{{highlights}}
\item Reliability-aware SR screens explicit heat-transfer correlations.
\item PVR, sensitivity, ESI and applicability score qualify equations.
\item The selected $Nu_{{avg}}$ equation gives {certified_mape}\% random-split MAPE.
\item Holdout tests assess reliability beyond interpolation.
\end{{highlights}}

\begin{{keyword}}
Symbolic regression \sep Heat transfer \sep Nusselt number \sep Applicability domain \sep Physics constraints \sep Out-of-domain generalization
\end{{keyword}}

\end{{frontmatter}}
\hypersetup{{pageanchor=true}}

\section{{Introduction}}

Compact heat-transfer correlations remain useful because they can be inspected, implemented in design calculations, and compared against limiting physical behavior. Symbolic regression (SR) is therefore a natural tool for thermal-fluid modeling. In this work, SR denotes data-driven search over mathematical expression structures and their constants, with the goal of producing an explicit equation rather than an opaque predictor \cite{{PachecoVega2006,Cranmer2023,FongMotani2025}}. The idea is not new: Pacheco-Vega et al. used SR for heat-transfer correlations in 2006 \cite{{PachecoVega2006}}, and recent work has applied GP-SR, neural-symbolic, and physics-informed SR variants to supercritical heat transfer, pyrolytic fuels, thermophysical-property correlations, turbulence closures and energy systems \cite{{Zhu2026,PINNSR2026,Shi2025,Li2026,Tang2024,Wu2025,Panczyk2025}}.

The practical limitation is that a low-error expression can still be unusable if it becomes negative, singular, oversensitive, or plausible only inside a dense interpolation cloud. This concern is consistent with two themes in recent literature. First, SR benchmarking is naturally Pareto based because an equation should be judged by accuracy and expression length rather than error alone \cite{{FongMotani2025}}. Second, machine-learning predictions should be interpreted with respect to an applicability domain, here understood as the feature-space region where the training data support reliable use of the model \cite{{Schultz2025}}. Physics-guided SR adds another layer by enforcing or encouraging dimensional homogeneity, dimensionless structure, governing-variable selection or physically meaningful search spaces \cite{{XiangChen2025,Darooneh2026,Anselment2025,Shi2025,ZhuChenLi2025,Taskin2026}}.

This paper therefore evaluates a reliability-aware SR workflow. The intended output is not merely a low-MAPE equation, but an equation accompanied by physical-consistency diagnostics, out-of-domain behavior, and a declared applicability score. The study uses the public partially porous wavy-channel CFD surrogate dataset of Kumar and Pandey \cite{{KumarPandey2026}}. The contribution is an executable workflow that qualifies candidate equations under stated data-domain tests and exports symbolic equations, trained models, predictions, split-wise diagnostics, large-label figures, article tables and a manuscript-ready comparison against state-of-the-art SR approaches.

\section{{Related Work}}

The related literature is grouped into four strands needed for the present workflow: explicit symbolic heat-transfer correlations, physics-guided equation search, flow-model discovery beyond scalar correlations, and reliability concepts such as applicability domains, uncertainty and accuracy--complexity benchmarking.

\subsection{{Symbolic heat-transfer and thermophysical correlations}}

SR has a long connection with explicit heat-transfer correlation discovery. Pacheco-Vega et al. showed that symbolic search could recover compact heat-transfer equations from data, establishing the core attraction of SR for engineering correlations: the final model can be read and implemented as an equation \cite{{PachecoVega2006}}. Recent work has extended this idea to more difficult thermal-fluid regimes. Zhu et al. used GP-SR to discover interpretable high-accuracy correlations for supercritical-fluid heat transfer, reporting improvements over traditional correlations while preserving equation-level interpretability \cite{{Zhu2026}}. Wang et al. combined a physics-informed broad neural network with symbolic regression for pyrolytic n-decane heat transfer; the neural stage reconstructed physical fields with independent-test errors below 5\%, and the final Nusselt correlation was restricted to pyrolysis conversion below 35\% \cite{{PINNSR2026}}.

Other recent thermal-property and energy studies show the same trend toward hybrid pipelines. Shi et al. proposed a hybrid symbolic-regression neural network for supercritical heat transfer that uses dimensional invariance and active subspaces to identify governing dimensionless factors from 1492 supercritical-water data points \cite{{Shi2025}}. Li et al. constrained SR for dilute-gas viscosity by modeling collision-integral terms; using 87 fluids and 11263 data points, they reported an overall AARD of 2.00\% \cite{{Li2026}}. Chowdhury et al. used ANN predictions for Carreau fluid flow with generalized thermal conductivity and then extracted symbolic expressions, reporting ANN errors below 1\% for skin-friction and Nusselt-number quantities \cite{{Chowdhury2026}}. These studies support the relevance of SR in heat transfer, but most emphasize accuracy or physical formulation more than split-wise reliability diagnostics.

\subsection{{Physics-guided and dimensionally constrained SR}}

Physics-guided SR methods reduce the candidate-equation space by embedding known structure. Xiang and Chen used a reinforcement-learning SR framework with dimensional homogeneity and dimensionless groups for gas-solid flow correlations; for an Ergun-form case, they reduced expression complexity from 24 to 14 while retaining $R^2=0.9930$ and demonstrated robustness up to 10\% noise \cite{{XiangChen2025}}. Darooneh et al. used dimensional analysis to enhance SR and universal physics-informed neural networks by transforming variables into Buckingham-Pi groups before learning \cite{{Darooneh2026}}. Anselment et al. pursued deterministic tree search over dimensionally homogeneous model structures, showing how unit consistency can prune invalid expressions before numerical fitting \cite{{Anselment2025}}.

Physics-guided symbolic discovery also appears beyond thermal-fluid correlations. Zhu, Chen and Li proposed a physics-guided SR framework for sand constitutive modeling, beginning with classical constitutive equations and refining Pareto candidate expressions under dimensional balance and boundary constraints \cite{{ZhuChenLi2025}}. Taskin et al. incorporated scores from pre-trained large language models into physics-informed SR losses; their results show that knowledge-integration benefits depend on the SR engine, language model and prompt \cite{{Taskin2026}}. These works motivate physics constraints in the search stage. The present work instead keeps the candidate set deliberately lightweight and adds an explicit diagnostic-screening layer after fitting.

\subsection{{Flow-field, turbulence and spatio-temporal symbolic discovery}}

Several related studies use SR to discover flow models rather than scalar engineering correlations. Tang et al. coupled field inversion with SR to improve the SST turbulence model for hypersonic heat-flux prediction through a variable turbulent-Prandtl correction; the resulting closure improved heat-flux prediction across wall-cooling, Mach-number, Reynolds-number and three-dimensional cases \cite{{Tang2024}}. Wu et al. learned symbolic turbulence models from indirect observation data by using neural networks and feature-importance analysis to narrow the SR feature space; the reported case study reduced SR cost from 1237 min to 176 min after feature screening \cite{{Wu2025}}. Ji et al. proposed a few-shot physically restorable SR turbulence model based on a normalized effective-viscosity hypothesis \cite{{Ji2026}}. Lazebnik and Liberzon moved physics-informed spatio-temporal SR from tabular to graph representations for functional equations, ODEs, PDEs, integral equations and delay equations, with improved recovery in several noisy settings \cite{{LazebnikLiberzon2026}}. These studies show that SR can be embedded into physics solvers and dynamical systems, whereas this paper focuses on qualifying explicit steady thermo-hydraulic correlations.

\subsection{{Applicability, uncertainty and benchmarking}}

Reliability assessment is a distinct issue from equation discovery. Schultz et al. define a general applicability-domain approach using feature-space density and dissimilarity; high dissimilarity is associated with high residuals and unreliable uncertainty estimates \cite{{Schultz2025}}. Zhao and Zhao coupled SR with probabilistic programming for uncertainty quantification and reported better generalization than deterministic empirical equations in their application \cite{{ZhaoZhao2025}}. Fong and Motani argued for Pareto-optimal fronts as a benchmark for SR algorithms, constructing exhaustive accuracy--length fronts for 34 SRBench datasets \cite{{FongMotani2025}}. Panczyk et al. explored Kolmogorov-Arnold Networks as symbolic equations for energy applications and found comparable KAN/FNN accuracy in selected low-output-dimensional settings while using SHAP to inspect physical relationships \cite{{Panczyk2025}}. The gap addressed here is the coupling of explicit thermal-fluid SR with physical-violation checks, extrapolation-stability diagnostics, data-support scoring and complete artifact export for article use.

\section{{Data and Problem Formulation}}

The study uses the Kumar and Pandey Mendeley Data archive, version 2, DOI 10.17632/5b5n3cg32n.2 \cite{{KumarPandey2026}}, distributed as \texttt{{ML-CFD-Wavy-\allowbreak Channel-\allowbreak Surrogate.zip}}. The experiment runner consumes the published cleaned long-form file \path{{02_processed_data/ML_dataset_longform.csv}}. The dataset metadata identifies the original CFD spreadsheet outputs as \texttt{{Nuavg.xlsx}} and \texttt{{DelP.xlsx}}; rows 1--768 and the geometry-specific output columns are represented in the long-form file used here.

The dataset contains {profile['n_rows']} CFD-derived samples from {profile['n_geometry_combinations']} geometry combinations: four Reynolds numbers, four Prandtl numbers, four Darcy numbers, four porosity values, three porous-slab thicknesses, and six straight or wavy geometry settings. The retained inputs are $Re$, $Pr$, $Da$, $\epsilon$, porous slab thickness $H_p$, wave amplitude $a$, and wavelength $L_w$. The modeled targets are the average Nusselt number $Nu_{{avg}}$ and pressure drop $\Delta p$. Preprocessing in this study resolves the source columns to canonical names, coerces the retained columns to numeric values, removes nonfinite rows and duplicate rows, and exports \texttt{{canonical\_wavy\_channel\_dataset.csv}} and the column-mapping JSON. No additional rows, features, external data, centering, scaling, or normalization are introduced before split construction; scaling is applied only inside the specific estimators and applicability-domain scorer described below.

Table~\ref{{tab:data-dictionary}} lists the retained variables, source-column names, units and validity ranges. Table~\ref{{tab:nomenclature}} defines the symbols, target names and diagnostic abbreviations used in the equations and tables. The geometry variables $H_p$, $a$ and $L_w$ are dimensional millimetre values from the source columns \texttt{{Hp\_mm}}, \texttt{{a\_mm}} and \texttt{{Lw\_mm}}, not normalized geometry variables. The value $L_w=0$ is a straight-channel code rather than a physical zero wavelength. Consequently, the exported symbolic equations must be evaluated with geometry supplied in millimetres and within the ranges in Table~\ref{{tab:data-dictionary}}.

\input{{article_artifacts/tables/data_dictionary.tex}}

\input{{article_artifacts/tables/nomenclature.tex}}

\FloatBarrier

\section{{Methods}}

The method has three stages. First, symbolic and black-box candidates are trained on each split. Second, symbolic candidates are ranked with a reliability-aware objective that combines error with diagnostic penalties. Third, the selected equation is evaluated with the same predictive metrics, physical checks and applicability-domain score used to qualify its intended use.

{method_pipeline_intro}

{method_pipeline_figure}

\FloatBarrier

\subsection{{Candidate Models}}

The symbolic candidate set contains an ordinary log-linear power-law correlation, a sparse log-target symbolic basis model, an unconstrained GP-SR model, and a positive complexity-penalized GP-SR model. The GP-SR implementation uses protected arithmetic, square-root, logarithm, inverse, absolute-value, and negation operators. These candidate families are intentionally close to the explicit-correlation tradition in heat-transfer SR \cite{{PachecoVega2006,Zhu2026,PINNSR2026}}. Random forest and XGBoost are included as black-box accuracy references, with XGBoost configured to request CUDA acceleration \cite{{ChenGuestrin2016}}. PySR remains compatible with the study design \cite{{Cranmer2023}}, but this run uses the lighter Python-native GP-SR engine to keep the pipeline reproducible without a Julia runtime dependency. Table~\ref{{tab:reproducibility-settings}} gives the random seed, split sizes, validation split, GP settings, lasso settings, random-forest settings, XGBoost settings and GPU configuration used in the completed run.

\input{{article_artifacts/tables/reproducibility_settings.tex}}

\FloatBarrier

\subsection{{Reliability-Aware Selection and Equation Qualification}}

The phrase reliability-aware is used here in a limited diagnostic sense: the workflow ranks and reports candidate equations under sampled checks, and it does not provide a formal certificate of global validity. This framing follows the distinction between model discovery and validity-domain assessment in applicability-domain work \cite{{Schultz2025}}. Candidate equations are selected with an internal validation split using
\begin{{equation}}
J(f)=\widetilde E_{{val}}(f)+0.04\widetilde C(f)+2.0\widetilde{{PVR}}(f)+0.10\widetilde S(f)+0.10\widetilde{{ESI}}(f),
\end{{equation}}
where $E_{{val}}$ is validation RMSE, $C$ is symbolic complexity, $PVR$ is the physics-violation rate, $S$ is local sensitivity, and $ESI$ is an extrapolation-stability index. Tildes denote split-wise min--max normalized quantities; $S$ and $ESI$ are transformed with $\log(1+\cdot)$ before normalization. The accuracy--complexity term follows the Pareto view of SR benchmarking \cite{{FongMotani2025}}, while the physical checks follow the heat-transfer requirement that explicit correlations remain finite and sign-consistent over their intended parameter range. The weights are fixed before test evaluation and are not learned from the test data: validation RMSE has unit weight, $PVR$ is given the largest diagnostic weight because nonfinite, nonpositive, or derivative-inconsistent behavior can make an equation unusable, and complexity, sensitivity and $ESI$ receive smaller weights so that they act mainly as tie-breakers among similarly accurate symbolic candidates.

Table~\ref{{tab:selection-ablation}} reports a compact ablation of the random-split symbolic-selection objective. Recomputing the symbolic winner using validation error only, error plus complexity, error plus $PVR$, and the full reliability-aware objective selects the same sparse symbolic lasso candidate for both targets. The table reports the selected candidate's random-split test MAPE, complexity, $PVR$, $ESI$ and high-error AUROC. This does not prove invariance to every possible weighting scheme; it documents that the reported random-split symbolic choice is not caused by a narrow tuning of the stated accuracy-dominant weights.

\input{{article_artifacts/tables/selection_ablation.tex}}

\FloatBarrier

For a fitted equation $f$, $PVR$ is the average of five violation rates:
\begin{{equation}}
PVR(f)=\frac{{1}}{{5}}\left(V_{{fin}}+V_{{pos}}+V^+_{{fin}}+V^+_{{pos}}+V_{{mono}}\right).
\end{{equation}}
$V_{{fin}}$ is the fraction of uniformly sampled points in the training-feature hyperrectangle for which $f$ is nonfinite. $V_{{pos}}$ is the fraction of those points with nonfinite or nonpositive predictions. $V^+_{{fin}}$ and $V^+_{{pos}}$ repeat the same checks on a near-domain extrapolation sample formed by expanding each feature range by 15\%, with positive variables lower-bounded at half of their minimum training value. $V_{{mono}}$ is the mean rate of failed finite-difference derivative-sign checks: negative response with respect to $Re$ is penalized for both targets, and negative response with respect to $Pr$ is also penalized for $Nu_{{avg}}$. This $PVR$ definition is an operational diagnostic introduced in this work, motivated by the broader use of physical admissibility and dimensional restrictions in physics-guided SR \cite{{XiangChen2025,Darooneh2026,Anselment2025}}.

The sensitivity term used in selection is the 95th percentile of
\begin{{equation}}
\frac{{|f(x+\delta)-f(x)|}}{{\max(|f(x)|,10^{{-12}})}},
\end{{equation}}
where each perturbation component has zero mean and standard deviation equal to 1\% of the corresponding training-feature standard deviation. The median sensitivity is exported as an additional diagnostic but is not used in $J$. Sensitivity screening follows the standard idea that model response should be inspected under controlled input perturbations \cite{{Saltelli2008}}. The extrapolation-stability index is the 95th percentile of $|f(x)|$ on the near-domain extrapolation sample divided by the median absolute prediction on the training points, with a small denominator floor. Lower sensitivity and lower $ESI$ indicate less local amplification and less near-domain growth; the $ESI$ itself is a local diagnostic defined here rather than a standard statistical confidence bound.

The selection objective treats validation error, complexity, $PVR$, sensitivity, and $ESI$ as soft ranking criteria. A candidate is excluded from selection only if fitting fails or the validation metric cannot be computed. No reported equation receives a mathematical proof of global validity, and no fixed pass/fail threshold is applied to $PVR$, $ESI$, sensitivity, or AUROC.

\subsection{{Applicability Domain}}

For each trained split, standardized feature-space support is measured by the distance to the $k$th nearest training point, using $k=10$, standard-score feature scaling, Euclidean distance, and $\alpha=1$. Nearest-neighbor distance provides a simple local support measure \cite{{CoverHart1967}}, and the reliability interpretation follows the applicability-domain view that reliability should decrease as a query point becomes dissimilar from the training data \cite{{Schultz2025}}. The applicability score is
\begin{{equation}}
A(x)=\exp\left(-\alpha d_k(x)/s_k\right) I[\mathrm{{finite}}(f(x)) \wedge f(x)>0],
\end{{equation}}
where $d_k(x)$ is the distance from $x$ to its $k$th nearest training point in standardized feature space, $s_k$ is the median training-set $k$th-neighbor distance, and $I[\cdot]$ sets the score to zero for nonphysical nonpositive or nonfinite predictions. Thus $A(x)=1$ for a point coincident with the training support, $A(x)=e^{{-1}}$ when the $k$th-neighbor distance equals the median training $k$th-neighbor distance, and $A(x)$ approaches zero as feature-space dissimilarity increases. This pointwise validity gate is the only hard reliability rule applied after fitting. No fixed acceptance threshold is used in the reported metrics; the score is an engineering reliability indicator and ranking variable, not a calibrated probabilistic guarantee.

High-error detection AUROC evaluates whether low applicability identifies large prediction errors. For each test split, the absolute relative error is thresholded at its 80th percentile; points at or above this threshold are labeled high-error cases. The AUROC is then computed using $1-A(x)$ as the risk score, following the standard ROC interpretation for ranking binary outcomes by a continuous score \cite{{Fawcett2006}}. Values above 0.5 indicate that lower applicability tends to flag the largest relative errors, whereas values near 0.5 indicate chance-level ranking.

\subsection{{Experimental Design}}

The pipeline evaluates a random 80/20 interpolation split, a high-Reynolds-number range holdout, a grouped geometry holdout, sparse-data training fractions of 20--80\%, and training-target noise levels of 1--5\%. Metrics include MAE, RMSE, MAPE, $R^2$, symbolic complexity, PVR, sensitivity, ESI, applicability score, and high-error detection AUROC.

\section{{Results}}

For the primary {_target_label(primary)} target, the selected random-split reliability-aware equation was:
\begin{{quote}}
\small\raggedright
{_display_expression(selected_primary['expression'])}
\end{{quote}}
The reliability-aware selection layer selected the sparse symbolic lasso candidate, \texttt{{{_latex_escape(selected_primary_model)}}}, for the random-split {_target_label(primary)} equation and \texttt{{{_latex_escape(selected_delp_model)}}} for the random-split $\Delta p$ equation. Consequently, the \texttt{{certified\_symbolic}} rows and the corresponding \texttt{{complexity\_penalized\_symbolic\_lasso}} rows have identical predictive metrics. The \texttt{{certified\_symbolic}} label denotes the candidate chosen by the reliability-aware ranking layer; it is not a separate regression estimator.

In this printed expression, \texttt{{thickness}}, \texttt{{amplitude}} and \texttt{{wavelength}} denote the numerical millimetre-valued geometry columns listed in Table~\ref{{tab:data-dictionary}}. Coefficients attached to dimensional numerical inputs absorb the corresponding reciprocal units, so expressions containing exponentials or logarithms of geometry terms should be read as dataset-unit empirical transformations rather than dimensionally homogeneous Buckingham-Pi correlations. The reported final equations should be used only over the data-supported range {validity_ranges}. Use outside these limits is an extrapolation and should require recomputing the applicability score, inspecting extrapolation-stability and sensitivity diagnostics, and preferably adding external CFD or experimental validation. The full selected-equation list is exported with the article artifacts, and the machine-readable equation file is stored with the project results.

Tables~\ref{{tab:metrics-nuavg}} and \ref{{tab:cert-nuavg}} report predictive performance and reliability diagnostics for {_target_label(primary)}. Tables~\ref{{tab:metrics-delp-pa}} and \ref{{tab:cert-delp-pa}} report the corresponding results for $\Delta p$.

{chr(10).join(table_inputs)}

\FloatBarrier

Figure~\ref{{fig:nuavg-diagnostics}} summarizes the main $Nu_{{avg}}$ diagnostics in stacked panels: (a) symbolic-candidate selection, (b) random-split prediction agreement and (c) split-wise MAPE transfer. Figure~\ref{{fig:delp-pa-diagnostics}} gives the corresponding diagnostics for $\Delta p$. To reduce repetition in the paper, the applicability-error scatter plots, Reynolds sweeps and noise-robustness plots are retained as exported supplementary figure artifacts rather than embedded here; Table~\ref{{tab:supplementary-figures}} lists their target, split/model context and main conclusion.

{chr(10).join(main_figure_lines)}

\FloatBarrier

\input{{article_artifacts/tables/supplementary_figure_index.tex}}

\FloatBarrier

\section{{Discussion}}

The experiment separates equation accuracy from equation qualification. The black-box baselines provide useful accuracy references, but they do not produce compact correlations. The ordinary power law is compact and stable but may underfit nonlinear interactions. The reliability-aware selected symbolic model is chosen as a compromise among validation error, compactness, positivity, finite behavior, monotonic checks and extrapolation stability. This design is complementary to search-stage physics constraints in recent SR work \cite{{XiangChen2025,Darooneh2026,Anselment2025,Taskin2026}} and to Pareto-front benchmarking \cite{{FongMotani2025}}. The applicability-domain score is especially useful for article presentation because it converts local data support into a plotted validity envelope, aligning the final correlation with broader ML reliability practice \cite{{Schultz2025}}.

For $Nu_{{avg}}$, the black-box models are clearly more accurate in the random interpolation split: the selected symbolic equation gives {nu_cert_random_mape}\% MAPE, while random forest and XGBoost give {nu_rf_random_mape}\% and {nu_xgb_random_mape}\%, respectively. This is expected because the random split preserves nearby full-factorial operating points and geometry combinations in the training set, allowing tree ensembles to interpolate local response surfaces very closely. In the high-Reynolds-number holdout, however, the selected symbolic equation gives {nu_cert_range_mape}\% MAPE whereas random forest and XGBoost increase to {nu_rf_range_mape}\% and {nu_xgb_range_mape}\%. The high-Re test removes the $Re=500$ level from training, and tree ensembles have limited ability to extrapolate a monotonic Reynolds-number trend beyond their trained partitions. The symbolic equation sacrifices some interpolation accuracy but retains a smooth algebraic dependence on $Re$, $Pr$ and $Da$, which is more useful when the goal is an inspectable engineering correlation rather than only in-domain numerical prediction.

The pressure-drop target shows the same pattern, with a larger gap between interpolation and range extrapolation. Random forest reaches {delp_rf_random_mape}\% MAPE on the random split because the train and test points share the same factorial grid structure and the pressure-drop surface can be locally memorized with high fidelity. In the high-Re holdout, random forest and XGBoost rise to {delp_rf_range_mape}\% and {delp_xgb_range_mape}\% MAPE, while the selected symbolic equation gives {delp_cert_range_mape}\%. Pressure drop varies steeply with Reynolds number and Darcy number, so removing the largest Reynolds-number level exposes the weakness of tree-based extrapolation. The sparse symbolic lasso is still imperfect, but its log-target form and explicit $Re$/$Da$ terms preserve a scalable trend that transfers better to the withheld high-Re cases.

The diagnostic results also show limitations of the current reliability layer. The maximum symbolic-candidate PVR reported in the completed experiments is {symbolic_pvr_max}, so PVR confirms that the sampled equations remained finite, positive and compliant with the simple derivative-sign checks, but it does not distinguish among viable symbolic candidates in this dataset. For pressure drop, the random-split AUROC of the selected symbolic equation is {delp_cert_random_auroc}, and the geometry-holdout AUROC is {delp_cert_geom_auroc}; these near- or below-chance values mean the kNN applicability score is not reliably ranking the largest pressure-drop errors. The likely reason is that pressure-drop errors are driven by target-scale nonlinearities and systematic residual structure, not only by feature-space sparsity. Future versions should add stricter dimensional constraints, stronger target-specific monotonic/asymptotic tests, and calibrated uncertainty or conformal risk estimates rather than relying on kNN distance alone.

Table~\ref{{tab:sota-comparison}} gives a neutral methodological comparison with representative symbolic-regression, physics-constrained modeling and applicability-domain studies. The table is intended to locate the present workflow by problem setting, data type, symbolic engine, physics constraint, uncertainty/applicability treatment, validation type and reported result, rather than to rank the studies.

\input{{article_artifacts/tables/sota_comparison.tex}}

\FloatBarrier

This study should be interpreted as computational evidence for a workflow. It does not prove uniqueness of the discovered equation, does not claim that SR itself is novel, does not enforce dimensional homogeneity of every discovered term, and does not mathematically prove validity outside the sampled checks and stated applicability domain. A dimensionally constrained rerun should replace the dimensional millimetre geometry variables with specified nondimensional groups or normalized geometry ratios before allowing logarithmic or exponential transformations. External experimental measurements or independent CFD reruns remain needed before the reported equations are used as design-standard correlations.

\section{{Conclusions}}

A reliability-aware symbolic-regression workflow was implemented for thermo-hydraulic correlation discovery. The pipeline produces explicit symbolic equations, physical diagnostics, out-of-domain tests, applicability scores, article-ready plots, tables, model files, and prediction files. For $Nu_{{avg}}$, the selected symbolic equation achieved {nu_cert_random_mape}\% MAPE on the random 80/20 split, compared with {power_mape}\% for the ordinary power law, {nu_rf_random_mape}\% for random forest and {nu_xgb_random_mape}\% for XGBoost. Under the high-Reynolds-number holdout, the selected symbolic equation gave {nu_cert_range_mape}\% MAPE, while random forest and XGBoost increased to {nu_rf_range_mape}\% and {nu_xgb_range_mape}\%; under the geometry holdout, the selected symbolic equation gave {nu_cert_geom_mape}\% MAPE, compared with {nu_rf_geom_mape}\% and {nu_xgb_geom_mape}\% for the two black-box baselines.

For $\Delta p$, the random split strongly favored local interpolation: random forest reached {delp_rf_random_mape}\% MAPE, XGBoost gave {delp_xgb_random_mape}\%, and the selected symbolic equation gave {delp_cert_random_mape}\%. The high-Reynolds-number holdout reversed the practical interpretation, with the selected symbolic equation at {delp_cert_range_mape}\% MAPE compared with {delp_rf_range_mape}\% and {delp_xgb_range_mape}\% for random forest and XGBoost. The geometry-holdout pressure-drop MAPE was {delp_cert_geom_mape}\% for the selected symbolic equation, {delp_rf_geom_mape}\% for random forest and {delp_xgb_geom_mape}\% for XGBoost. These results indicate that the symbolic equations are weaker local interpolants than tree ensembles on dense random splits, but their smooth algebraic structure can be more useful for range-transfer behavior and engineering-correlation interpretation.

The main limitation is that the reliability layer is diagnostic and sampled. It qualifies candidate equations with validation error, complexity, PVR, sensitivity, ESI and applicability scoring, but it does not prove global validity, uniqueness, dimensional correctness of every discovered term, or safe extrapolation outside the tested ranges. Because the geometry variables remain dimensional millimetre inputs, exponentials and logarithms involving $H_p$, $a$ or $L_w$ are dataset-unit transformations rather than dimensionally homogeneous physical groups. The fixed selection weights should also be treated as design choices rather than universal constants. In the current experiments, zero PVR values show that the physical-violation screen does not always distinguish viable symbolic candidates, and weak pressure-drop AUROC values show that kNN applicability is not a calibrated uncertainty model for every target.

Future work should add external CFD or experimental validation, stricter dimensional and asymptotic constraints during equation search, calibrated uncertainty or conformal risk estimates, and target-specific applicability thresholds. These additions would move the workflow from reliability-aware equation qualification toward a stronger validation protocol for deployable heat-transfer and pressure-drop correlations.

\section*{{Ethics Statement}}

This study uses previously published CFD-derived numerical data and does not involve human participants, animals, or new hazardous experiments.

\section*{{CRediT Authorship Contribution Statement}}

Yacine Khaldi: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing - original draft, Writing - review and editing. Hanane Azzaoui: Methodology, Supervision, Validation, Project administration, Writing - review and editing. Mohamed Ben Bezziane: Supervision, Validation, Writing - review and editing. Amir Benzaoui: Validation, Formal analysis, Writing - review and editing.

\section*{{Funding}}

No specific grant funding was declared for this work.

\section*{{Data and Code Availability}}

The source data are available from the \href{{https://doi.org/10.17632/5b5n3cg32n.2}}{{Mendeley Data record}}, DOI 10.17632/5b5n3cg32n.2. The analysis code, generated tables, figures, selected equations, prediction files and reproducibility artifacts will be released upon acceptance.

\section*{{Declaration of Competing Interest}}

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section*{{Declaration of Generative AI and AI-assisted technologies in the writing process}}

During the preparation of this work, the authors used ChatGPT (OpenAI) solely for proofreading and improving the language and readability of the manuscript. The tool was not used to generate scientific results, conduct data analysis, develop the methodology, or draw scientific conclusions. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the publication.

\bibliographystyle{{elsarticle-num}}
\bibliography{{rcsr_refs}}

\end{{document}}
"""

    output = ELSEVIER_DIR / "elsarticle-template-num.tex"
    output.write_text(manuscript.strip() + "\n", encoding="utf-8")
    print(f"Wrote manuscript: {output}")


if __name__ == "__main__":
    main()
