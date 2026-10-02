"""Cutoff-safe customer history features and weekly transaction sequences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC_FEATURES = [
    "recency_days", "tenure_days", "history_days", "purchase_count",
    "purchase_day_count", "repeat_purchase_day_count", "lifetime_revenue_gbp",
    "avg_order_value_gbp", "median_order_value_gbp", "max_order_value_gbp",
    "avg_basket_units", "avg_basket_lines", "product_diversity",
    "mean_days_between_purchase_days", "purchase_days_per_30d",
    "orders_30d", "orders_90d", "orders_180d", "orders_365d",
    "spend_30d_gbp", "spend_90d_gbp", "spend_180d_gbp", "spend_365d_gbp",
    "spend_recent90_over_prior90", "return_invoice_rate", "credit_amount_365d_gbp",
    "revenue_per_history_day_gbp", "zero_purchase_weeks_26w",
]
SEQUENCE_CHANNELS = ["orders", "revenue_gbp", "units", "product_codes", "has_purchase", "after_first_observed_purchase"]


def _customer_window(lines: pd.DataFrame, cutoff: pd.Timestamp, days: int) -> tuple[pd.Series, pd.Series]:
    window = lines.loc[lines["invoice_date"].ge(cutoff - pd.Timedelta(days=days))]
    orders = window.groupby("customer_id", sort=False)["invoice_id"].nunique()
    spend = window.groupby("customer_id", sort=False)["revenue"].sum()
    return orders, spend


def build_cutoff_features(
    sales: pd.DataFrame,
    credits: pd.DataFrame,
    cutoff_value: str | pd.Timestamp,
    horizon_days: int = 90,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Return one pre-cutoff feature row and 26 weekly bins per known customer."""
    cutoff = pd.Timestamp(cutoff_value)
    history = sales.loc[sales["invoice_date"].lt(cutoff)]
    if history.empty:
        raise ValueError(f"No eligible purchase history before {cutoff.date()}.")

    invoices = history.groupby(["customer_id", "invoice_id"], sort=False).agg(
        invoice_date=("invoice_date", "min"),
        revenue=("revenue", "sum"),
        units=("quantity", "sum"),
        line_count=("stock_code", "size"),
        product_count=("stock_code", "nunique"),
        country=("country", "last"),
    ).reset_index()
    invoices = invoices.sort_values("invoice_date", kind="stable")
    by_customer = invoices.groupby("customer_id", sort=False)
    features = by_customer.agg(
        first_purchase=("invoice_date", "min"),
        last_purchase=("invoice_date", "max"),
        purchase_count=("invoice_id", "nunique"),
        lifetime_revenue_gbp=("revenue", "sum"),
        avg_order_value_gbp=("revenue", "mean"),
        median_order_value_gbp=("revenue", "median"),
        max_order_value_gbp=("revenue", "max"),
        avg_basket_units=("units", "mean"),
        avg_basket_lines=("line_count", "mean"),
        country=("country", "last"),
    )
    features["product_diversity"] = history.groupby("customer_id", sort=False)["stock_code"].nunique().reindex(features.index, fill_value=0)
    date_events = invoices.assign(purchase_day=invoices["invoice_date"].dt.normalize()).groupby("customer_id", sort=False).agg(
        purchase_day_count=("purchase_day", "nunique"),
        _purchase_days=("purchase_day", lambda values: list(pd.DatetimeIndex(values.drop_duplicates().sort_values()))),
    )
    features = features.join(date_events)
    features["cutoff"] = cutoff
    features["recency_days"] = (cutoff - features["last_purchase"]).dt.total_seconds() / 86400
    features["tenure_days"] = (cutoff - features["first_purchase"]).dt.total_seconds() / 86400
    features["history_days"] = features["tenure_days"].clip(lower=1)
    features["repeat_purchase_day_count"] = (features["purchase_day_count"] - 1).clip(lower=0)
    features["mean_days_between_purchase_days"] = date_events["_purchase_days"].map(
        lambda values: float(np.diff(pd.DatetimeIndex(values).asi8 / 86_400_000_000_000).mean()) if len(values) > 1 else np.nan
    )
    features = features.drop(columns=["_purchase_days"])
    features["invoice_count"] = features["purchase_count"]
    features["purchase_days_per_30d"] = features["purchase_day_count"] / (features["history_days"] / 30).clip(lower=1)

    for window in (30, 90, 180, 365):
        counts, spend = _customer_window(history, cutoff, window)
        features[f"orders_{window}d"] = counts.reindex(features.index, fill_value=0).astype(float)
        features[f"spend_{window}d_gbp"] = spend.reindex(features.index, fill_value=0.0).astype(float)
    features["spend_recent90_over_prior90"] = (
        features["spend_90d_gbp"] + 1
    ) / (features["spend_180d_gbp"] - features["spend_90d_gbp"] + 1)
    features["revenue_per_history_day_gbp"] = features["lifetime_revenue_gbp"] / features["history_days"]

    customer_credit = credits.loc[credits["invoice_date"].lt(cutoff)]
    recent_credits = customer_credit.loc[customer_credit["invoice_date"].ge(cutoff - pd.Timedelta(days=365))]
    credits365 = recent_credits.groupby("customer_id", sort=False)["credit_amount_abs"].sum()
    credit_orders = customer_credit.groupby("customer_id", sort=False)["invoice_id"].nunique()
    features["credit_amount_365d_gbp"] = credits365.reindex(features.index, fill_value=0.0)
    features["return_invoice_rate"] = (
        credit_orders.reindex(features.index, fill_value=0.0) / features["purchase_count"].clip(lower=1)
    )

    weeks = np.zeros((len(features), 26, len(SEQUENCE_CHANNELS)), dtype=np.float32)
    recent = history.loc[history["invoice_date"].ge(cutoff - pd.Timedelta(days=182))].copy()
    recent["week_index"] = ((recent["invoice_date"] - (cutoff - pd.Timedelta(days=182))).dt.total_seconds() // (7 * 86400)).astype(int)
    recent["week_index"] = recent["week_index"].clip(0, 25)
    weekly = recent.groupby(["customer_id", "week_index"], sort=False).agg(
        orders=("invoice_id", "nunique"),
        revenue_gbp=("revenue", "sum"),
        units=("quantity", "sum"),
        product_codes=("stock_code", "nunique"),
    )
    row_by_customer = pd.Series(np.arange(len(features)), index=features.index)
    if len(weekly):
        row_indices = weekly.index.get_level_values("customer_id").map(row_by_customer).to_numpy()
        week_indices = weekly.index.get_level_values("week_index").to_numpy()
        channel_values = weekly[["orders", "revenue_gbp", "units", "product_codes"]].to_numpy(dtype=np.float32)
        weeks[row_indices, week_indices, :4] = np.maximum(channel_values, 0)
        weeks[row_indices, week_indices, 4] = 1.0
    first_days = by_customer["invoice_date"].min().dt.normalize()
    oldest_date = (cutoff - pd.Timedelta(days=182)).normalize()
    first_week = ((first_days - oldest_date).dt.days // 7).clip(0, 25)
    for customer_id, position in row_by_customer.items():
        start_index = int(first_week.loc[customer_id])
        weeks[position, start_index:, 5] = 1.0
    features["zero_purchase_weeks_26w"] = 26 - (weeks[:, :, 4] > 0).sum(axis=1)
    features = features.reset_index(names="customer_id")
    features = features.drop(columns=["invoice_count"])
    features["cutoff"] = cutoff
    features["horizon_days"] = int(horizon_days)
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, weeks, invoices


def build_panel(
    sales: pd.DataFrame,
    credits: pd.DataFrame,
    cutoffs: list[str],
    root: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    processed = root / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    panels = []
    sequences: dict[str, np.ndarray] = {}
    for cutoff in cutoffs:
        print(f"Building customer features for cutoff {cutoff}...", flush=True)
        panel, weeks, _ = build_cutoff_features(sales, credits, cutoff)
        panels.append(panel)
        sequences[cutoff] = weeks
        print(f"  {len(panel):,} eligible customers; tensor shape {weeks.shape}", flush=True)
    features = pd.concat(panels, ignore_index=True)
    features.to_parquet(processed / "customer_features.parquet", index=False)
    np.savez_compressed(processed / "weekly_sequences.npz", **sequences)
    return features, sequences

