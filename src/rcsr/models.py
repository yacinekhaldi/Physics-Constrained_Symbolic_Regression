from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _safe_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.maximum(arr, np.finfo(float).eps)


class PowerLawRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, feature_names: Optional[List[str]] = None):
        self.feature_names = feature_names

    def _transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        cols = []
        for j in range(X.shape[1]):
            if self.modes_[j] == "log":
                cols.append(np.log(_safe_positive(X[:, j])))
            else:
                cols.append(X[:, j])
        return np.column_stack(cols)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() < 5:
            raise ValueError("PowerLawRegressor requires positive target values.")
        X_fit = X[mask]
        y_fit = y[mask]
        self.modes_ = ["log" if np.nanmin(X_fit[:, j]) > 0 else "linear" for j in range(X.shape[1])]
        self.model_ = LinearRegression()
        self.model_.fit(self._transform(X_fit), np.log(y_fit))
        self.feature_names_ = self.feature_names or [f"x{j}" for j in range(X.shape[1])]
        self.complexity_ = 1 + int(np.sum(np.abs(self.model_.coef_) > 1e-12))
        return self

    def predict(self, X):
        log_y = self.model_.predict(self._transform(np.asarray(X, dtype=float)))
        log_y = np.clip(log_y, -700, 700)
        return np.exp(log_y)

    def equation(self, target_name: str = "y") -> str:
        constant = math.exp(float(self.model_.intercept_))
        parts = [f"{constant:.8g}"]
        for name, coef, mode in zip(self.feature_names_, self.model_.coef_, self.modes_):
            if abs(coef) < 1e-10:
                continue
            if mode == "log":
                parts.append(f"{name}^{{{coef:.6g}}}")
            else:
                parts.append(f"exp({coef:.6g}{name})")
        return f"{target_name} = " + " ".join(parts)


class BasisLogLassoRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42,
        cv: int = 5,
    ):
        self.feature_names = feature_names
        self.random_state = random_state
        self.cv = cv

    def _make_basis(self, X: np.ndarray, store_names: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        basis = []
        names = []
        for j, name in enumerate(self.feature_names_):
            basis.append(X[:, j])
            names.append(name)
            if self.positive_cols_[j]:
                basis.append(np.log(_safe_positive(X[:, j])))
                names.append(f"log({name})")
                basis.append(np.sqrt(_safe_positive(X[:, j])))
                names.append(f"sqrt({name})")
        positive_indices = [j for j, ok in enumerate(self.positive_cols_) if ok]
        for pos, j in enumerate(positive_indices):
            for k in positive_indices[pos + 1 :]:
                basis.append(
                    np.log(_safe_positive(X[:, j])) * np.log(_safe_positive(X[:, k]))
                )
                names.append(f"log({self.feature_names_[j]})log({self.feature_names_[k]})")
        if not basis:
            basis = [np.ones(X.shape[0])]
            names = ["1"]
        if store_names:
            self.basis_names_ = names
        return np.column_stack(basis)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mask = np.isfinite(y) & (y > 0)
        if mask.sum() < 10:
            raise ValueError("BasisLogLassoRegressor requires positive target values.")
        X_fit = X[mask]
        y_fit = y[mask]
        self.feature_names_ = self.feature_names or [f"x{j}" for j in range(X.shape[1])]
        self.positive_cols_ = [np.nanmin(X_fit[:, j]) > 0 for j in range(X.shape[1])]
        Phi = self._make_basis(X_fit, store_names=True)
        cv = min(self.cv, max(2, mask.sum() // 20))
        self.model_ = make_pipeline(
            StandardScaler(),
            LassoCV(
                cv=cv,
                random_state=self.random_state,
                max_iter=30000,
                n_alphas=80,
                n_jobs=None,
            ),
        )
        self.model_.fit(Phi, np.log(y_fit))
        lasso = self.model_.named_steps["lassocv"]
        self.complexity_ = 1 + int(np.sum(np.abs(lasso.coef_) > 1e-9))
        return self

    def predict(self, X):
        Phi = self._make_basis(np.asarray(X, dtype=float), store_names=False)
        log_y = self.model_.predict(Phi)
        log_y = np.clip(log_y, -700, 700)
        return np.exp(log_y)

    def equation(self, target_name: str = "y") -> str:
        lasso = self.model_.named_steps["lassocv"]
        scaler = self.model_.named_steps["standardscaler"]
        terms = []
        intercept = float(lasso.intercept_)
        for coef, mean, scale, name in zip(
            lasso.coef_, scaler.mean_, scaler.scale_, self.basis_names_
        ):
            if abs(coef) < 1e-9:
                continue
            if scale == 0:
                continue
            intercept -= float(coef) * float(mean) / float(scale)
            terms.append((float(coef) / float(scale), name))
        chunks = [f"{intercept:.6g}"]
        for coef, name in terms:
            sign = "+" if coef >= 0 else "-"
            chunks.append(f" {sign} {abs(coef):.6g} {name}")
        return f"{target_name} = exp(" + "".join(chunks) + ")"


class GplearnRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        population_size: int = 250,
        generations: int = 6,
        parsimony_coefficient: float = 0.001,
        positive_target: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.feature_names = feature_names
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self.positive_target = positive_target
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        from gplearn.genetic import SymbolicRegressor

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.feature_names_ = self.feature_names or [f"x{j}" for j in range(X.shape[1])]
        if self.positive_target:
            mask = np.isfinite(y) & (y > 0)
            fit_y = np.log(y[mask])
            fit_X = X[mask]
        else:
            fit_X = X
            fit_y = y
        self.estimator_ = SymbolicRegressor(
            population_size=self.population_size,
            generations=self.generations,
            tournament_size=20,
            stopping_criteria=0.0,
            const_range=(-5.0, 5.0),
            init_depth=(2, 4),
            init_method="half and half",
            function_set=("add", "sub", "mul", "div", "sqrt", "log", "abs", "neg", "inv"),
            metric="rmse",
            parsimony_coefficient=self.parsimony_coefficient,
            p_crossover=0.70,
            p_subtree_mutation=0.10,
            p_hoist_mutation=0.05,
            p_point_mutation=0.10,
            max_samples=0.90,
            verbose=0,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            low_memory=True,
            feature_names=self.feature_names_,
        )
        self.estimator_.fit(fit_X, fit_y)
        self.complexity_ = int(getattr(self.estimator_._program, "length_", 0) or 0)
        return self

    def predict(self, X):
        pred = self.estimator_.predict(np.asarray(X, dtype=float))
        if self.positive_target:
            pred = np.exp(np.clip(pred, -700, 700))
        return pred

    def equation(self, target_name: str = "y") -> str:
        expr = str(self.estimator_._program)
        if self.positive_target:
            return f"{target_name} = exp({expr})"
        return f"{target_name} = {expr}"


@dataclass
class FittedModel:
    name: str
    family: str
    estimator: object
    expression: str
    complexity: float
    notes: str = ""
    gpu_requested: bool = False
    gpu_used: bool = False


def symbolic_factories(
    feature_names: List[str],
    random_state: int,
    gp_population: int,
    gp_generations: int,
    include_gp: bool = True,
) -> List[Callable[[np.ndarray, np.ndarray, str], FittedModel]]:
    factories: List[Callable[[np.ndarray, np.ndarray, str], FittedModel]] = []

    def power_law(X, y, target):
        model = PowerLawRegressor(feature_names=feature_names).fit(X, y)
        return FittedModel(
            name="ordinary_power_law",
            family="symbolic",
            estimator=model,
            expression=model.equation(target),
            complexity=model.complexity_,
            notes="Log-linear ordinary least-squares power-law correlation.",
        )

    def basis_lasso(X, y, target):
        model = BasisLogLassoRegressor(
            feature_names=feature_names, random_state=random_state
        ).fit(X, y)
        return FittedModel(
            name="complexity_penalized_symbolic_lasso",
            family="symbolic",
            estimator=model,
            expression=model.equation(target),
            complexity=model.complexity_,
            notes="Sparse log-target model over raw, log, sqrt and log-product basis terms.",
        )

    factories.extend([power_law, basis_lasso])

    if include_gp:

        def gp_raw(X, y, target):
            model = GplearnRegressor(
                feature_names=feature_names,
                population_size=gp_population,
                generations=gp_generations,
                parsimony_coefficient=0.0005,
                positive_target=False,
                random_state=random_state,
            ).fit(X, y)
            return FittedModel(
                name="unconstrained_gp_sr",
                family="symbolic",
                estimator=model,
                expression=model.equation(target),
                complexity=model.complexity_,
                notes="Unconstrained genetic-programming symbolic regression.",
            )

        def gp_positive(X, y, target):
            model = GplearnRegressor(
                feature_names=feature_names,
                population_size=gp_population,
                generations=gp_generations,
                parsimony_coefficient=0.01,
                positive_target=True,
                random_state=random_state + 11,
            ).fit(X, y)
            return FittedModel(
                name="positive_complexity_gp_sr",
                family="symbolic",
                estimator=model,
                expression=model.equation(target),
                complexity=model.complexity_,
                notes="Complexity-penalized GP-SR on log target, enforcing positive predictions.",
            )

        factories.extend([gp_raw, gp_positive])

    return factories


def fit_black_box_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    target: str,
    random_state: int,
    xgb_estimators: int = 700,
) -> List[FittedModel]:
    models: List[FittedModel] = []
    rf = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    models.append(
        FittedModel(
            name="random_forest",
            family="black_box",
            estimator=rf,
            expression=f"{target} = RandomForestRegressor(...)",
            complexity=float(sum(tree.tree_.node_count for tree in rf.estimators_)),
            notes="Accuracy reference; not a publishable explicit correlation.",
        )
    )

    try:
        from xgboost import XGBRegressor

        xgb_params = dict(
            objective="reg:squarederror",
            n_estimators=xgb_estimators,
            max_depth=4,
            learning_rate=0.035,
            subsample=0.92,
            colsample_bytree=0.92,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=0,
            tree_method="hist",
            device="cuda",
        )
        xgb = XGBRegressor(**xgb_params)
        try:
            xgb.fit(X_train, y_train, verbose=False)
            gpu_used = True
            notes = "XGBoost baseline requested CUDA device on the RTX GPU."
        except Exception as exc:
            xgb_params["device"] = "cpu"
            xgb = XGBRegressor(**xgb_params)
            xgb.fit(X_train, y_train, verbose=False)
            gpu_used = False
            notes = f"CUDA training failed; fell back to CPU. CUDA error: {exc}"
        models.append(
            FittedModel(
                name="xgboost_hist",
                family="black_box",
                estimator=xgb,
                expression=f"{target} = XGBRegressor(...)",
                complexity=float(xgb_estimators),
                notes=notes,
                gpu_requested=True,
                gpu_used=gpu_used,
            )
        )
    except Exception as exc:
        models.append(
            FittedModel(
                name="xgboost_hist",
                family="black_box",
                estimator=None,
                expression=f"{target} = XGBRegressor(unavailable)",
                complexity=np.nan,
                notes=f"XGBoost unavailable: {exc}",
                gpu_requested=True,
                gpu_used=False,
            )
        )
    return models
