from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "R2": np.nan}
    yt = y_true[mask]
    yp = y_pred[mask]
    denom = np.maximum(np.abs(yt), np.finfo(float).eps)
    return {
        "MAE": float(mean_absolute_error(yt, yp)),
        "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
        "MAPE": float(np.mean(np.abs((yt - yp) / denom)) * 100.0),
        "R2": float(r2_score(yt, yp)),
    }


def high_error_detection_auc(y_true, y_pred, scores, quantile: float = 0.8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    scores = np.asarray(scores, dtype=float)
    rel_error = np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), np.finfo(float).eps)
    finite = np.isfinite(rel_error) & np.isfinite(scores)
    if finite.sum() < 10:
        return np.nan
    rel_error = rel_error[finite]
    scores = scores[finite]
    threshold = np.quantile(rel_error, quantile)
    labels = (rel_error >= threshold).astype(int)
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(roc_auc_score(labels, 1.0 - scores))
