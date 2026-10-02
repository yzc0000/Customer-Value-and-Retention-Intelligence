"""Independent future-outcome labels with explicit end-exclusive windows."""

from __future__ import annotations

import calendar
import pandas as pd


def _six_calendar_month_end(cutoff: pd.Timestamp) -> pd.Timestamp:
    month = cutoff.month + 6
    year = cutoff.year + (month - 1) // 12
    month = (month - 1) % 12 + 1
    day = min(cutoff.day, calendar.monthrange(year, month)[1])
    return pd.Timestamp(year=year, month=month, day=day)


def build_targets(
    sales: pd.DataFrame,
    features: pd.DataFrame,
    cutoffs: list[str],
    complete_end_exclusive: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create matured 90-day and exact six-calendar-month targets.

    Event rows are filtered on their own timestamps before invoice and
    customer aggregation. This prevents multi-timestamp invoices from moving
    future lines into a historical interval.
    """
    target_sets = {"90d": [], "6m": []}
    for cutoff_value in cutoffs:
        cutoff = pd.Timestamp(cutoff_value)
        cohort = features.loc[features["cutoff"].eq(cutoff), ["customer_id", "cutoff"]].copy()
        for name, end in (("90d", cutoff + pd.Timedelta(days=90)), ("6m", _six_calendar_month_end(cutoff))):
            mature = end <= complete_end_exclusive
            future = sales.loc[sales["invoice_date"].ge(cutoff) & sales["invoice_date"].lt(end)]
            by_customer = future.groupby("customer_id", sort=False).agg(
                future_invoice_count=("invoice_id", "nunique"),
                future_revenue_gbp=("revenue", "sum"),
                first_future_purchase=("invoice_date", "min"),
                future_purchase_days=("purchase_day", "nunique"),
            )
            labels = cohort.join(by_customer, on="customer_id")
            labels["future_invoice_count"] = labels["future_invoice_count"].fillna(0).astype("int32")
            labels["future_revenue_gbp"] = labels["future_revenue_gbp"].fillna(0.0)
            labels["future_purchase_days"] = labels["future_purchase_days"].fillna(0).astype("int16")
            labels["returned"] = labels["future_invoice_count"].gt(0).astype("int8")
            labels["inactive"] = (1 - labels["returned"]).astype("int8")
            labels["horizon"] = name
            labels["horizon_end_exclusive"] = end
            labels["label_mature"] = bool(mature)
            if not mature:
                labels[["future_invoice_count", "future_revenue_gbp", "future_purchase_days", "returned", "inactive"]] = pd.NA
                labels["first_future_purchase"] = pd.NaT
            target_sets[name].append(labels.reset_index(drop=True))
    # Nullable dtypes keep the immature final six-month horizon consistent
    # with mature folds, and avoid concat dtype inference from all-NA blocks.
    target_dtypes = {
        "future_invoice_count": "Int32",
        "future_revenue_gbp": "Float64",
        "future_purchase_days": "Int16",
        "returned": "Int8",
        "inactive": "Int8",
    }
    for frames in target_sets.values():
        for labels in frames:
            for column, dtype in target_dtypes.items():
                labels[column] = labels[column].astype(dtype)
    result90 = pd.concat(target_sets["90d"], ignore_index=True)
    result6m = pd.concat(target_sets["6m"], ignore_index=True)
    return result90, result6m
