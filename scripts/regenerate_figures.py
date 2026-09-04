from __future__ import annotations

import shutil
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rcsr.experiment import FEATURES, make_splits  # noqa: E402
from rcsr.paths import (  # noqa: E402
    MODELS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    TABLES_DIR,
)
from rcsr.plotting import (  # noqa: E402
    plot_applicability,
    plot_method_pipeline,
    plot_metric_bars,
    plot_noise,
    plot_pareto,
    plot_predictions,
    plot_re_sweep,
)


def main() -> None:
    figure_out_dir = PROJECT_ROOT / "results" / "figures_reliability_aware"
    metrics = pd.read_csv(TABLES_DIR / "metrics_by_experiment.csv")
    model_summary = pd.read_csv(TABLES_DIR / "model_summary.csv")
    predictions = pd.read_csv(PREDICTIONS_DIR / "all_predictions.csv")
    df = pd.read_csv(PROCESSED_DATA_DIR / "canonical_wavy_channel_dataset.csv")
    splits = make_splits(df, random_state=42)
    train_idx = splits["random80_20"]["train_idx"]
    X_train = df[FEATURES].to_numpy(dtype=float)[train_idx]

    plot_method_pipeline(figure_out_dir)

    for target in metrics["target"].drop_duplicates():
        plot_pareto(model_summary, target, figure_out_dir)
        plot_predictions(predictions, target, figure_out_dir)
        plot_applicability(predictions, target, figure_out_dir)
        plot_noise(metrics, target, figure_out_dir)
        plot_metric_bars(metrics, target, figure_out_dir)

        model_path = MODELS_DIR / f"random80_20__{target}__certified_symbolic.joblib"
        if model_path.exists():
            plot_re_sweep(joblib.load(model_path), X_train, FEATURES, target, figure_out_dir)

    article_figures = (
        WORKSPACE_ROOT / "Elsevier" / "article_artifacts" / "figures_reliability_aware"
    )
    article_figures.mkdir(parents=True, exist_ok=True)
    for path in figure_out_dir.glob("*"):
        if path.suffix.lower() in {".png", ".pdf"}:
            shutil.copy2(path, article_figures / path.name)


if __name__ == "__main__":
    main()
