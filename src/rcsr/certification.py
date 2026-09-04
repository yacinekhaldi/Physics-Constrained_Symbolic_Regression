from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def sample_domain(
    X_train: np.ndarray,
    n_samples: int = 2048,
    random_state: int = 42,
    extrapolate: bool = False,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    X_train = np.asarray(X_train, dtype=float)
    lo = np.nanmin(X_train, axis=0)
    hi = np.nanmax(X_train, axis=0)
    width = np.maximum(hi - lo, np.finfo(float).eps)
    if extrapolate:
        lo = lo - 0.15 * width
        hi = hi + 0.15 * width
        positive = np.nanmin(X_train, axis=0) > 0
        lo[positive] = np.maximum(lo[positive], np.nanmin(X_train[:, positive], axis=0) * 0.5)
    return rng.uniform(lo, hi, size=(n_samples, X_train.shape[1]))


def _predict_safe(model, X: np.ndarray) -> np.ndarray:
    try:
        return np.asarray(model.predict(X), dtype=float)
    except Exception:
        return np.full(X.shape[0], np.nan)


def physics_report(
    model,
    X_train: np.ndarray,
    feature_names: List[str],
    target: str,
    random_state: int = 42,
    n_samples: int = 2048,
) -> Dict[str, float]:
    X_train = np.asarray(X_train, dtype=float)
    domain = sample_domain(X_train, n_samples=n_samples, random_state=random_state)
    extra = sample_domain(
        X_train,
        n_samples=max(512, n_samples // 2),
        random_state=random_state + 1,
        extrapolate=True,
    )
    y_domain = _predict_safe(model, domain)
    y_extra = _predict_safe(model, extra)

    finite_violation = float(np.mean(~np.isfinite(y_domain)))
    positive_violation = float(np.mean((~np.isfinite(y_domain)) | (y_domain <= 0)))
    extra_finite_violation = float(np.mean(~np.isfinite(y_extra)))
    extra_positive_violation = float(np.mean((~np.isfinite(y_extra)) | (y_extra <= 0)))

    monotonic_checks = []
    check_columns = ["Re"]
    if target == "Nuavg":
        check_columns.append("Pr")
    for col in check_columns:
        if col not in feature_names:
            continue
        j = feature_names.index(col)
        width = np.nanmax(X_train[:, j]) - np.nanmin(X_train[:, j])
        if not np.isfinite(width) or width <= 0:
            continue
        step = max(1e-8, 0.005 * width)
        x_minus = domain.copy()
        x_plus = domain.copy()
        x_minus[:, j] = np.maximum(x_minus[:, j] - step, np.nanmin(X_train[:, j]))
        x_plus[:, j] = np.minimum(x_plus[:, j] + step, np.nanmax(X_train[:, j]))
        ym = _predict_safe(model, x_minus)
        yp = _predict_safe(model, x_plus)
        denom = x_plus[:, j] - x_minus[:, j]
        ok = denom > 0
        derivative = np.full(len(domain), np.nan)
        derivative[ok] = (yp[ok] - ym[ok]) / denom[ok]
        monotonic_checks.append(float(np.mean(~np.isfinite(derivative) | (derivative < -1e-8))))
    derivative_violation = float(np.mean(monotonic_checks)) if monotonic_checks else 0.0

    rng = np.random.default_rng(random_state + 2)
    scale = np.nanstd(X_train, axis=0)
    perturb = rng.normal(0.0, 0.01, size=domain.shape) * np.maximum(scale, 1e-12)
    y_perturbed = _predict_safe(model, domain + perturb)
    sensitivity = np.abs(y_perturbed - y_domain) / np.maximum(np.abs(y_domain), 1e-12)
    sensitivity_p95 = float(np.nanpercentile(sensitivity, 95))
    sensitivity_median = float(np.nanmedian(sensitivity))

    train_pred = _predict_safe(model, X_train)
    extra_scale = np.nanmedian(np.abs(train_pred))
    if not np.isfinite(extra_scale) or extra_scale <= 0:
        extra_scale = np.nanmedian(np.abs(y_domain))
    extra_growth = np.abs(y_extra) / max(float(extra_scale), 1e-12)
    extrapolation_stability_index = float(np.nanpercentile(extra_growth, 95))

    pvr = float(
        np.clip(
            np.mean(
                [
                    positive_violation,
                    extra_positive_violation,
                    finite_violation,
                    extra_finite_violation,
                    derivative_violation,
                ]
            ),
            0.0,
            1.0,
        )
    )

    return {
        "finite_violation_rate": finite_violation,
        "positive_violation_rate": positive_violation,
        "extra_finite_violation_rate": extra_finite_violation,
        "extra_positive_violation_rate": extra_positive_violation,
        "derivative_sign_violation_rate": derivative_violation,
        "sensitivity_median": sensitivity_median,
        "sensitivity_p95": sensitivity_p95,
        "extrapolation_stability_index": extrapolation_stability_index,
        "physics_violation_rate": pvr,
    }


@dataclass
class ApplicabilityScorer:
    k: int = 10
    alpha: float = 1.0

    def fit(self, X_train: np.ndarray):
        X_train = np.asarray(X_train, dtype=float)
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X_train)
        self.k_ = min(self.k, max(2, Xs.shape[0]))
        self.nn_ = NearestNeighbors(n_neighbors=self.k_)
        self.nn_.fit(Xs)
        train_dist, _ = self.nn_.kneighbors(Xs)
        kth = train_dist[:, -1]
        scale = float(np.median(kth[kth > 0])) if np.any(kth > 0) else 1.0
        self.distance_scale_ = max(scale, 1e-12)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler_.transform(np.asarray(X, dtype=float))
        dist, _ = self.nn_.kneighbors(Xs)
        kth = dist[:, -1]
        return np.exp(-self.alpha * kth / self.distance_scale_)
