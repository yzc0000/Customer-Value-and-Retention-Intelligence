"""Comparable metrics used to rank frozen forecast candidates."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, f1_score, log_loss,
    mean_absolute_error, mean_squared_error, precision_score,
    recall_score, roc_auc_score,
)


def classification_metrics(y_true, probability, capacity: float = 0.10) -> dict:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    n = len(y)
    k = max(1, int(np.ceil(n * capacity)))
    chosen = np.argpartition(-p, k - 1)[:k]
    predicted = p >= 0.5
    result = {
        "n_customers": int(n),
        "positive_rate": float(y.mean()) if n else np.nan,
        "precision_at_capacity": float(y[chosen].mean()) if n else np.nan,
        "recall_at_capacity": float(y[chosen].sum() / max(1, y.sum())) if n else np.nan,
        "contact_capacity_fraction": float(k / n) if n else np.nan,
        "precision_threshold_0_5": float(precision_score(y, predicted, zero_division=0)),
        "recall_threshold_0_5": float(recall_score(y, predicted, zero_division=0)),
        "f1_threshold_0_5": float(f1_score(y, predicted, zero_division=0)),
        "average_precision": float(average_precision_score(y, p)) if y.sum() else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }
    return result


def revenue_metrics(y_true, prediction, capacity: float = 0.10) -> dict:
    y = np.maximum(np.asarray(y_true, dtype=float), 0)
    p = np.maximum(np.asarray(prediction, dtype=float), 0)
    n = len(y)
    k = max(1, int(np.ceil(n * capacity)))
    chosen = np.argpartition(-p, k - 1)[:k]
    total_actual = float(y.sum())
    total_predicted = float(p.sum())
    return {
        "n_customers": int(n),
        "actual_revenue_gbp": total_actual,
        "predicted_revenue_gbp": total_predicted,
        "aggregate_bias_gbp": total_predicted - total_actual,
        "aggregate_bias_pct": (total_predicted - total_actual) / total_actual if total_actual else np.nan,
        "mae_gbp": float(mean_absolute_error(y, p)),
        "rmse_gbp": float(np.sqrt(mean_squared_error(y, p))),
        "revenue_capture_at_capacity": float(y[chosen].sum() / total_actual) if total_actual else np.nan,
        "actual_at_capacity_gbp": float(y[chosen].sum()),
    }


def metrics_table(records: list[dict], task: str) -> pd.DataFrame:
    table = pd.DataFrame(records)
    if not table.empty:
        table.insert(0, "task", task)
    return table
