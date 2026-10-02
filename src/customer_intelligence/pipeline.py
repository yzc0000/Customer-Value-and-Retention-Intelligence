"""Dataset preparation commands."""

from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from .config import FORECAST_CUTOFFS, SOURCE_FILE
from .features import NUMERIC_FEATURES, build_panel
from .ingestion import load_workbook, sha256, write_dataset
from .targets import build_targets


def prepare_dataset(root: Path) -> None:
    (root / "reports").mkdir(parents=True, exist_ok=True)
    source = root / SOURCE_FILE
    print(f"Preparing source workbook: {source.name}", flush=True)
    sales, credits, quality = load_workbook(source)
    write_dataset(sales, credits, quality, root)
    safe_end = sales["invoice_date"].max().normalize()
    features, sequences = build_panel(sales, credits, FORECAST_CUTOFFS, root)
    target90, target6m = build_targets(sales, features, FORECAST_CUTOFFS, safe_end)

    processed = root / "data" / "processed"
    target90.to_parquet(processed / "targets_90d.parquet", index=False)
    target6m.to_parquet(processed / "targets_6m.parquet", index=False)
    cohort = target90.groupby("cutoff", sort=True).agg(
        customers=("customer_id", "nunique"),
        inactive_rate=("inactive", "mean"),
        returned=("returned", "sum"),
        future_revenue_gbp=("future_revenue_gbp", "sum"),
    ).reset_index()
    recent = features.loc[features["orders_365d"].gt(0)].groupby("cutoff")["customer_id"].nunique()
    cohort["active_365d_customers"] = cohort["cutoff"].map(recent).fillna(0).astype(int)
    cohort.to_csv(root / "reports" / "cohort_summary.csv", index=False)

    installed = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }
    manifest = {
        **quality,
        "source_sha256": sha256(source),
        "complete_end_exclusive": str(safe_end.date()),
        "forecast_cutoffs": FORECAST_CUTOFFS,
        "feature_numeric_columns": NUMERIC_FEATURES,
        "feature_rows": int(len(features)),
        "sequence_shapes": {cutoff: list(value.shape) for cutoff, value in sequences.items()},
        "target_90d_rows": int(len(target90)),
        "target_6m_rows": int(len(target6m)),
        "target_maturity": {
            "90d_mature_rows": int(target90["label_mature"].sum()),
            "six_month_mature_rows": int(target6m["label_mature"].sum()),
        },
        "environment": installed,
        "excluded_model_families": {},
    }
    (processed / "prepare_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print("\n90-day cohort labels (future data only):", flush=True)
    print(cohort.to_string(index=False), flush=True)
    print("\nSaved line marts, customer snapshots, weekly tensors and targets under data/", flush=True)


def load_prepared(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    processed = root / "data" / "processed"
    return (
        pd.read_parquet(processed / "customer_features.parquet"),
        pd.read_parquet(processed / "targets_90d.parquet"),
        pd.read_parquet(processed / "targets_6m.parquet"),
        json.loads((processed / "prepare_manifest.json").read_text(encoding="utf-8")),
    )
