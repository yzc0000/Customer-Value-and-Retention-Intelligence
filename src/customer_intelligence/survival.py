"""Pooled discrete-time hazard model for time until the next purchase."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from .features import NUMERIC_FEATURES
from .modeling import CATEGORY_FEATURES, _preprocessor

PERIODS = 13
HORIZON_DAYS = 90


class PooledPurchaseHazard:
    """Estimate weekly conditional return hazards and combine to 90-day risk."""

    def __init__(self):
        self.transform = _preprocessor(list(NUMERIC_FEATURES), list(CATEGORY_FEATURES), scale_numeric=True)
        self.period_encoder = OneHotEncoder(categories=[np.arange(PERIODS)], handle_unknown="ignore", sparse_output=True)
        self.model = LogisticRegression(C=0.2, max_iter=400, solver="lbfgs")

    @staticmethod
    def _exposure(returned, first_purchase, cutoffs):
        person_rows = []
        periods = []
        targets = []
        for index, (event, event_date, cutoff) in enumerate(zip(returned, first_purchase, cutoffs, strict=False)):
            if bool(event) and pd.notna(event_date):
                elapsed = max(0.0, (pd.Timestamp(event_date) - pd.Timestamp(cutoff)).total_seconds() / 86400)
                event_period = min(int(elapsed // 7), PERIODS - 1)
                count = event_period + 1
            else:
                event_period = -1
                count = PERIODS
            person_rows.extend([index] * count)
            periods.extend(range(count))
            targets.extend([0] * max(0, count - 1) + ([1] if event_period >= 0 else [0]))
        return np.asarray(person_rows), np.asarray(periods), np.asarray(targets, dtype=np.int8)

    def fit(self, x: pd.DataFrame, returned, first_purchase, cutoffs) -> "PooledPurchaseHazard":
        encoded = self.transform.fit_transform(x)
        people, periods, y = self._exposure(returned, first_purchase, cutoffs)
        self.period_encoder.fit(periods.reshape(-1, 1))
        design = sparse.hstack([
            encoded[people], self.period_encoder.transform(periods.reshape(-1, 1)),
        ], format="csr")
        self.model.fit(design, y)
        return self

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        encoded = self.transform.transform(x)
        row = np.repeat(np.arange(len(x)), PERIODS)
        periods = np.tile(np.arange(PERIODS), len(x))
        design = sparse.hstack([
            encoded[row], self.period_encoder.transform(periods.reshape(-1, 1)),
        ], format="csr")
        hazard = self.model.predict_proba(design)[:, 1].reshape(len(x), PERIODS)
        survival = np.prod(1.0 - hazard, axis=1)
        p_return = np.clip(1.0 - survival, 0, 1)
        return np.column_stack([1 - p_return, p_return])
