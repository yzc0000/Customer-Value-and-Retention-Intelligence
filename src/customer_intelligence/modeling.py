"""Leakage-aware tabular candidates for inactivity and revenue."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, LogisticRegression, TweedieRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

from .config import SEED
from .features import NUMERIC_FEATURES

CATEGORY_FEATURES = ["country"]


def _preprocessor(numeric: list[str], categorical: list[str], scale_numeric: bool = True) -> ColumnTransformer:
    numeric_steps = [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric),
        ("categorical", categorical_pipeline, categorical),
    ], remainder="drop")


def classifier_candidates() -> dict[str, object]:
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORY_FEATURES)
    baseline = DummyPrevalence()
    recency = Pipeline([
        ("transform", _preprocessor(["recency_days"], [], scale_numeric=True)),
        ("model", LogisticRegression(C=0.2, max_iter=600, random_state=SEED)),
    ])
    logistic = Pipeline([
        ("transform", _preprocessor(numeric, categorical, scale_numeric=True)),
        ("model", LogisticRegression(C=0.25, max_iter=600, random_state=SEED)),
    ])

    spline_features = [
        "recency_days", "tenure_days", "purchase_day_count", "purchase_count",
        "orders_90d", "orders_365d", "spend_90d_gbp", "lifetime_revenue_gbp",
        "mean_days_between_purchase_days", "spend_recent90_over_prior90",
    ]
    spline_numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("spline", SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")),
        ("scale", StandardScaler()),
    ])
    spline_country = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    spline_transform = ColumnTransformer([
        ("splines", spline_numeric, spline_features),
        ("country", spline_country, categorical),
    ], remainder="drop")
    gam = Pipeline([
        ("transform", spline_transform),
        ("model", LogisticRegression(C=0.2, max_iter=700, random_state=SEED)),
    ])
    random_forest = Pipeline([
        ("transform", _preprocessor(numeric, categorical, scale_numeric=False)),
        ("model", RandomForestClassifier(
            n_estimators=240, max_depth=10, min_samples_leaf=18,
            max_features=0.8, n_jobs=1, random_state=SEED,
        )),
    ])
    candidates: dict[str, object] = {
        "prevalence_baseline": baseline,
        "recency_logistic": recency,
        "logistic_regression": logistic,
        "logistic_gam": gam,
        "random_forest": random_forest,
    }
    try:
        from xgboost import XGBClassifier
        candidates["xgboost"] = Pipeline([
            ("transform", _preprocessor(numeric, categorical, scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=220, max_depth=3, learning_rate=0.04,
                min_child_weight=18, subsample=0.82, colsample_bytree=0.82,
                reg_lambda=8.0, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=2, random_state=SEED,
            )),
        ])
    except ImportError:
        pass
    try:
        from catboost import CatBoostClassifier
        candidates["catboost"] = Pipeline([
            ("transform", _preprocessor(numeric, categorical, scale_numeric=False)),
            ("model", CatBoostClassifier(
                iterations=400, depth=5, learning_rate=0.04,
                loss_function="Logloss", verbose=False, thread_count=2,
                random_seed=SEED,
            )),
        ])
    except ImportError:
        pass
    return candidates


class DummyPrevalence:
    """Fixed prevalence predictor with the classifier interface."""

    def fit(self, x, y):
        self.probability_ = float(np.mean(y))
        return self

    def predict_proba(self, x):
        p = np.full(len(x), self.probability_, dtype=float)
        return np.column_stack([1 - p, p])


@dataclass
class HurdleRevenueModel:
    """Return propensity times a positive conditional GAM-like GLM."""

    return_model: object
    spend_model: object

    @classmethod
    def build(cls) -> "HurdleRevenueModel":
        numeric = list(NUMERIC_FEATURES)
        categorical = list(CATEGORY_FEATURES)
        return cls(
            return_model=Pipeline([
                ("transform", _preprocessor(numeric, categorical, scale_numeric=True)),
                ("model", LogisticRegression(C=0.3, max_iter=600, random_state=SEED)),
            ]),
            spend_model=Pipeline([
                ("transform", _preprocessor(numeric, categorical, scale_numeric=True)),
                ("model", GammaRegressor(alpha=2.0, max_iter=700)),
            ]),
        )

    def fit(self, x: pd.DataFrame, y) -> "HurdleRevenueModel":
        y_values = np.maximum(np.asarray(y, dtype=float), 0)
        self.return_model.fit(x, (y_values > 0).astype(int))
        positive = y_values > 0
        self.spend_model.fit(x.loc[positive], y_values[positive])
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        p_return = self.return_model.predict_proba(x)[:, 1]
        conditional_spend = self.spend_model.predict(x)
        return np.maximum(p_return * conditional_spend, 0)


def revenue_candidates() -> dict[str, object]:
    numeric = list(NUMERIC_FEATURES)
    categorical = list(CATEGORY_FEATURES)
    candidates: dict[str, object] = {
        "tweedie_glm": Pipeline([
            ("transform", _preprocessor(numeric, categorical, scale_numeric=True)),
            ("model", TweedieRegressor(power=1.5, alpha=4.0, link="log", max_iter=800, tol=1e-7)),
        ]),
        "hurdle_gamma_glm": HurdleRevenueModel.build(),
        "random_forest_regressor": Pipeline([
            ("transform", _preprocessor(numeric, categorical, scale_numeric=False)),
            ("model", RandomForestRegressor(
                n_estimators=220, max_depth=10, min_samples_leaf=18,
                max_features=0.8, n_jobs=1, random_state=SEED,
            )),
        ]),
    }
    try:
        from xgboost import XGBRegressor
        candidates["xgboost_tweedie"] = Pipeline([
            ("transform", _preprocessor(numeric, categorical, scale_numeric=False)),
            ("model", XGBRegressor(
                n_estimators=240, max_depth=3, learning_rate=0.035,
                min_child_weight=20, subsample=0.82, colsample_bytree=0.82,
                reg_lambda=10.0, objective="reg:tweedie", tweedie_variance_power=1.5,
                eval_metric="mae", tree_method="hist", n_jobs=2, random_state=SEED,
            )),
        ])
    except ImportError:
        pass
    try:
        from catboost import CatBoostRegressor
        candidates["catboost_tweedie"] = Pipeline([
            ("transform", _preprocessor(numeric, categorical, scale_numeric=False)),
            ("model", CatBoostRegressor(
                iterations=400, depth=5, learning_rate=0.04,
                loss_function="Tweedie:variance_power=1.5", verbose=False,
                thread_count=2, random_seed=SEED,
            )),
        ])
    except ImportError:
        pass
    return candidates

