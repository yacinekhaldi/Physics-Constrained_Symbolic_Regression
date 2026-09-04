from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

TITLE_SIZE = 20
LABEL_SIZE = 18
TICK_SIZE = 15
LEGEND_SIZE = 14
ANNOTATION_SIZE = 12

MODEL_LABELS = {
    "ordinary_power_law": "Power law",
    "complexity_penalized_symbolic_lasso": "Sparse symbolic",
    "unconstrained_gp_sr": "GP-SR",
    "positive_complexity_gp_sr": "Positive GP-SR",
    "certified_symbolic": "Reliability-aware symbolic",
    "random_forest": "Random forest",
    "xgboost_hist": "XGBoost",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": TICK_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
            "figure.titlesize": TITLE_SIZE,
            "axes.linewidth": 1.25,
            "lines.linewidth": 2.4,
            "lines.markersize": 7,
        }
    )


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _draw_pipeline_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    title: str,
    body: str,
    facecolor: str,
    edgecolor: str = "#2f4254",
) -> None:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.4,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + width / 2,
        y + height * 0.69,
        title,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#1d2b38",
    )
    ax.text(
        x + width / 2,
        y + height * 0.34,
        body,
        ha="center",
        va="center",
        fontsize=9.4,
        linespacing=1.18,
        color="#263746",
    )


def _draw_arrow(ax, start: tuple[float, float], end: tuple[float, float], dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=1.5,
            color="#314554",
            linestyle="--" if dashed else "-",
            shrinkA=2,
            shrinkB=2,
        )
    )


def plot_method_pipeline(out_dir: Path) -> None:
    configure_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.8, 6.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    main_y = 0.58
    box_w = 0.15
    box_h = 0.25
    main_boxes = [
        (
            0.03,
            "1. Data",
            "$Re$, $Pr$, $Da$, $\\epsilon$\ngeometry variables\nTargets: $Nu_{avg}$, $\\Delta p$",
            "#eaf2f8",
        ),
        (
            0.235,
            "2. Splits",
            "Random 80/20\nhigh-$Re$ holdout\ngeometry holdout",
            "#f7efd9",
        ),
        (
            0.44,
            "3. Candidates",
            "Power law\nsparse symbolic\nGP-SR variants\nRF/XGBoost references",
            "#eaf5ea",
        ),
        (
            0.645,
            "4. Ranking",
            "Validation error\ncomplexity\nPVR, sensitivity, ESI",
            "#f0ecf7",
        ),
        (
            0.82,
            "5. Outputs",
            "Selected equation\nAD score and AUROC\nfigures, tables, models",
            "#e8f4f2",
        ),
    ]
    for x, title, body, color in main_boxes:
        _draw_pipeline_box(ax, x, main_y, box_w, box_h, title, body, color)

    for i in range(len(main_boxes) - 1):
        x0 = main_boxes[i][0] + box_w
        x1 = main_boxes[i + 1][0]
        _draw_arrow(ax, (x0 + 0.012, main_y + box_h / 2), (x1 - 0.012, main_y + box_h / 2))

    lower_boxes = [
        (
            0.20,
            0.22,
            0.22,
            0.20,
            "Stress tests",
            "Sparse-data fractions\ntraining-target noise\nsplit-wise metrics",
            "#f6f6f6",
        ),
        (
            0.48,
            0.18,
            0.22,
            0.24,
            "Soft selection",
            "$J(f)=\\widetilde E_{val}+0.04\\widetilde C$\n$+2\\widetilde{PVR}+0.10\\widetilde S+0.10\\widetilde{ESI}$\nRanks symbolic candidates",
            "#fbf7ed",
        ),
        (
            0.74,
            0.18,
            0.22,
            0.24,
            "Hard prediction gate",
            "$A(x)=0$ for nonfinite\nor nonpositive predictions\notherwise kNN data support",
            "#edf7f6",
        ),
    ]
    for x, y, width, height, title, body, color in lower_boxes:
        _draw_pipeline_box(ax, x, y, width, height, title, body, color, edgecolor="#5a6470")

    _draw_arrow(ax, (0.31, main_y - 0.01), (0.31, 0.43), dashed=True)
    _draw_arrow(ax, (0.72, main_y - 0.01), (0.60, 0.43), dashed=True)
    _draw_arrow(ax, (0.895, main_y - 0.01), (0.85, 0.43), dashed=True)

    ax.text(
        0.5,
        0.08,
        "PVR: physics-violation rate; ESI: extrapolation-stability index; "
        "AD: applicability domain; AUROC: high-error detection ranking.",
        ha="center",
        va="center",
        fontsize=10.5,
        color="#33424f",
    )

    png_path = out_dir / "method_pipeline.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_pareto(model_summary: pd.DataFrame, target: str, out_dir: Path) -> None:
    configure_style()
    df = model_summary[
        (model_summary["target"] == target)
        & (model_summary["experiment"] == "random80_20")
        & (model_summary["family"] == "symbolic")
    ].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    scatter = ax.scatter(
        df["complexity"],
        df["RMSE"],
        c=df["physics_violation_rate"],
        s=140,
        cmap="viridis_r",
        edgecolor="black",
    )
    for _, row in df.iterrows():
        ax.annotate(
            MODEL_LABELS.get(row["model"], row["model"]),
            (row["complexity"], row["RMSE"]),
            fontsize=ANNOTATION_SIZE,
            xytext=(5, 5),
            textcoords="offset points",
        )
    ax.set_xlabel("Symbolic complexity")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Pareto view for {target}")
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Physics violation rate")
    _save(fig, out_dir / f"pareto_front_{target}.png")


def plot_predictions(predictions: pd.DataFrame, target: str, out_dir: Path) -> None:
    configure_style()
    df = predictions[
        (predictions["target"] == target)
        & (predictions["experiment"] == "random80_20")
        & (predictions["model"].isin(["certified_symbolic", "xgboost_hist", "random_forest"]))
    ].copy()
    if df.empty:
        return
    model_order = ["certified_symbolic", "random_forest", "xgboost_hist"]
    models = [model for model in model_order if model in set(df["model"])]
    fig, axes = plt.subplots(
        len(models),
        1,
        figsize=(7.2, 4.1 * len(models)),
        squeeze=False,
    )
    y_min = min(df["y_true"].min(), df["y_pred"].min())
    y_max = max(df["y_true"].max(), df["y_pred"].max())
    for i, (ax, model) in enumerate(zip(axes[:, 0], models)):
        sub = df[df["model"] == model]
        panel = chr(ord("a") + i)
        ax.scatter(sub["y_true"], sub["y_pred"], s=28, alpha=0.65)
        ax.plot([y_min, y_max], [y_min, y_max], color="black", linewidth=1.8)
        ax.set_xlabel("Measured CFD")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{panel}) {MODEL_LABELS.get(model, model)}", loc="left")
    _save(fig, out_dir / f"predicted_vs_measured_{target}.png")


def plot_applicability(predictions: pd.DataFrame, target: str, out_dir: Path) -> None:
    configure_style()
    df = predictions[
        (predictions["target"] == target) & (predictions["model"] == "certified_symbolic")
    ].copy()
    if df.empty or "applicability_score" not in df:
        return
    df["relative_error"] = np.abs(df["y_true"] - df["y_pred"]) / np.maximum(
        np.abs(df["y_true"]), np.finfo(float).eps
    )
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.scatter(df["applicability_score"], df["relative_error"] * 100.0, s=26, alpha=0.45)
    bins = np.linspace(0, 1, 8)
    grouped = df.groupby(pd.cut(df["applicability_score"], bins), observed=True)
    centers = [interval.mid for interval in grouped.groups]
    means = grouped["relative_error"].mean().to_numpy() * 100.0
    if len(centers):
        ax.plot(centers, means, color="black", marker="o", linewidth=2.6)
    ax.set_xlabel("Applicability-domain score")
    ax.set_ylabel("Absolute relative error (%)")
    ax.set_title(f"Applicability diagnostic for {target}")
    _save(fig, out_dir / f"applicability_error_{target}.png")


def plot_re_sweep(
    model,
    X_train: np.ndarray,
    feature_names: List[str],
    target: str,
    out_dir: Path,
) -> None:
    configure_style()
    if "Re" not in feature_names:
        return
    j = feature_names.index("Re")
    median = np.median(X_train, axis=0)
    re_grid = np.linspace(np.min(X_train[:, j]), np.max(X_train[:, j]), 200)
    X = np.tile(median, (len(re_grid), 1))
    X[:, j] = re_grid
    y = model.predict(X)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    ax.plot(re_grid, y, linewidth=3)
    ax.set_xlabel("Re")
    ax.set_ylabel(f"Predicted {target}")
    ax.set_title(f"Selected-correlation Reynolds sweep for {target}")
    _save(fig, out_dir / f"reynolds_sweep_{target}.png")


def plot_noise(metrics: pd.DataFrame, target: str, out_dir: Path) -> None:
    configure_style()
    df = metrics[
        (metrics["target"] == target)
        & (metrics["experiment"].str.startswith("noise_"))
        & (metrics["model"].isin(["certified_symbolic", "ordinary_power_law", "xgboost_hist"]))
    ].copy()
    if df.empty:
        return
    df["noise_percent"] = df["experiment"].str.extract(r"noise_(\d+)").astype(float)
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for model, sub in df.groupby("model"):
        sub = sub.sort_values("noise_percent")
        ax.plot(
            sub["noise_percent"],
            sub["RMSE"],
            marker="o",
            linewidth=2.8,
            label=MODEL_LABELS.get(model, model),
        )
    ax.set_xlabel("Injected training noise (%)")
    ax.set_ylabel("RMSE on clean random test split")
    ax.set_title(f"Noise robustness for {target}")
    ax.legend()
    _save(fig, out_dir / f"noise_robustness_{target}.png")


def plot_metric_bars(metrics: pd.DataFrame, target: str, out_dir: Path) -> None:
    configure_style()
    df = metrics[
        (metrics["target"] == target)
        & (metrics["experiment"].isin(["random80_20", "range_re_high_holdout", "geometry_holdout"]))
    ].copy()
    if df.empty:
        return
    df["model_label"] = df["model"].map(MODEL_LABELS).fillna(df["model"])
    pivot = df.pivot_table(index="model", columns="experiment", values="MAPE", aggfunc="mean")
    pivot.index = [MODEL_LABELS.get(idx, idx) for idx in pivot.index]
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("MAPE (%)")
    ax.set_title(f"Model comparison for {target}")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=28)
    ax.legend(title="Experiment", title_fontsize=LEGEND_SIZE)
    _save(fig, out_dir / f"model_comparison_mape_{target}.png")
