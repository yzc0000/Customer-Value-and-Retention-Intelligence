"""Read-only workbook audit for the project design; writes aggregate evidence only.

Run from the repository root: python scripts/audit_dataset.py
Purchase filters below are provisional sensitivity scenarios, not final cleaning rules.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "online_retail_II.xlsx"
OUTPUT = ROOT / "research" / "evidence"
ALIASES = {
    "Invoice": "invoice_id", "InvoiceNo": "invoice_id",
    "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date",
    "Price": "unit_price", "UnitPrice": "unit_price",
    "Customer ID": "customer_id", "CustomerID": "customer_id",
    "Country": "country",
}


def audit() -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("Loading every worksheet in the supplied workbook...", flush=True)
    sheets = pd.read_excel(SOURCE, sheet_name=None, engine="openpyxl")
    frames = []
    sheet_profiles = []
    for name, source_frame in sheets.items():
        frame = source_frame.rename(columns=ALIASES).copy()
        frame["invoice_date"] = pd.to_datetime(frame["invoice_date"], errors="coerce")
        sheet_profiles.append({
            "sheet": name, "rows": len(frame), "source_columns": list(source_frame.columns),
            "date_min": str(frame["invoice_date"].min()),
            "date_max": str(frame["invoice_date"].max()),
            "missing_customer_rows": int(frame["customer_id"].isna().sum()),
        })
        frame["source_sheet"] = name
        frames.append(frame)
        print(f"Loaded {name}: {len(frame):,} invoice lines", flush=True)
    data = pd.concat(frames, ignore_index=True)
    raw_columns = [c for c in data.columns if c != "source_sheet"]
    data["invoice_id"] = data["invoice_id"].astype("string").str.strip()
    data["stock_code"] = data["stock_code"].astype("string").str.strip()
    cancel = data["invoice_id"].str.upper().str.startswith("C", na=False)
    negative = data["quantity"].lt(0)
    valid_sale = data["quantity"].gt(0) & data["unit_price"].gt(0) & ~cancel
    identified_sale = valid_sale & data["customer_id"].notna()
    product_pattern = data["stock_code"].str.fullmatch(r"\d{5}[A-Za-z]*", na=False)
    duplicates = data.duplicated(subset=raw_columns, keep="first")
    duplicate_group_rows = data.duplicated(subset=raw_columns, keep=False)
    row_hashes = pd.util.hash_pandas_object(data[raw_columns], index=False)
    cross_sheet_hashes = (pd.DataFrame({"hash": row_hashes, "sheet": data["source_sheet"]})
                          .groupby("hash")["sheet"].nunique())
    sales = data.loc[identified_sale & ~duplicates & product_pattern].copy()
    sales["revenue"] = sales["quantity"] * sales["unit_price"]
    orders = sales.groupby(["customer_id", "invoice_id"], as_index=False).agg(
        invoice_date=("invoice_date", "min"), revenue=("revenue", "sum"),
        units=("quantity", "sum"), line_count=("stock_code", "size"),
    )
    customer_counts = orders.groupby("customer_id").size()
    occasion_counts = sales.assign(purchase_day=sales["invoice_date"].dt.normalize()).groupby("customer_id")["purchase_day"].nunique()
    customer_revenue = orders.groupby("customer_id")["revenue"].sum().sort_values(ascending=False)
    known = data.loc[data["customer_id"].notna()]
    invoice_customers = known.groupby("invoice_id")["customer_id"].nunique()
    invoice_timestamps = data.groupby("invoice_id")["invoice_date"].nunique()
    invoice_countries = data.groupby("invoice_id")["country"].nunique()
    customer_countries = known.groupby("customer_id")["country"].nunique()
    monthly = data.assign(month=data["invoice_date"].dt.to_period("M").astype(str)).groupby("month").agg(
        rows=("invoice_id", "size"), missing_customer_rows=("customer_id", lambda s: s.isna().sum()),
        known_customers=("customer_id", "nunique"),
    ).reset_index()
    monthly["missing_customer_pct"] = monthly["missing_customer_rows"] / monthly["rows"] * 100
    monthly.to_csv(OUTPUT / "monthly_profile.csv", index=False)
    special = (data.loc[~product_pattern].groupby("stock_code").agg(
        rows=("invoice_id", "size"), description_example=("description", "first"))
        .sort_values("rows", ascending=False).head(25).reset_index())
    special.to_csv(OUTPUT / "special_stock_codes.csv", index=False)
    duplicate_months = data.loc[duplicates, "invoice_date"].dt.to_period("M").astype(str).value_counts()
    safe_end = data["invoice_date"].max().normalize()
    snapshots = []
    for cutoff_text in ["2010-06-01", "2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01", "2011-09-01"]:
        cutoff = pd.Timestamp(cutoff_text)
        history = sales.loc[sales["invoice_date"] < cutoff]
        customers = history["customer_id"].unique()
        eligible_recent = history.loc[history["invoice_date"] >= cutoff - pd.Timedelta(days=365), "customer_id"].unique()
        for horizon in [90, 180]:
            end = cutoff + pd.Timedelta(days=horizon)
            matured = end <= safe_end
            future = sales.loc[(sales["invoice_date"] >= cutoff) & (sales["invoice_date"] < end)]
            future_counts = future.groupby("customer_id")["invoice_id"].nunique().reindex(customers, fill_value=0)
            future_revenue = future.groupby("customer_id")["revenue"].sum().reindex(customers, fill_value=0)
            recent_counts = future_counts.reindex(eligible_recent, fill_value=0)
            snapshots.append({
                "cutoff": cutoff_text, "horizon_days": horizon, "label_end_exclusive": str(end.date()),
                "fully_observed": bool(matured), "historical_customers": len(customers),
                "active_within_365d_customers": len(eligible_recent),
                "returned_customers": int(future_counts.gt(0).sum()) if matured else None,
                "inactivity_pct_all_history": float(future_counts.eq(0).mean() * 100) if matured else None,
                "inactivity_pct_active_365d": float(recent_counts.eq(0).mean() * 100) if matured else None,
                "future_positive_sales_revenue_gbp": float(future_revenue.sum()) if matured else None,
            })
    pd.DataFrame(snapshots).to_csv(OUTPUT / "snapshot_feasibility.csv", index=False)
    numeric_quantiles = {
        col: {str(q): float(value) for q, value in data[col].quantile([0, .01, .5, .99, 1]).items()}
        for col in ["quantity", "unit_price"]
    }
    daily = data.groupby(data["invoice_date"].dt.normalize()).size()
    all_dates = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    absent_dates = all_dates.difference(daily.index)
    nulls = {col: {"rows": int(data[col].isna().sum()), "pct": float(data[col].isna().mean() * 100)} for col in raw_columns}
    with SOURCE.open("rb") as source_handle:
        source_sha256 = hashlib.file_digest(source_handle, "sha256").hexdigest()
    result = {
        "source": SOURCE.name,
        "sha256": source_sha256,
        "file_bytes": SOURCE.stat().st_size,
        "profile_definition": "All sheets; raw counts unless explicitly marked provisional scenario. Dates are timezone-naive source dates.",
        "raw_rows": len(data), "raw_columns": raw_columns, "sheets": sheet_profiles,
        "date_min": str(data["invoice_date"].min()), "date_max": str(data["invoice_date"].max()),
        "conservative_complete_end_exclusive": str(safe_end),
        "distinct_known_customers": int(data["customer_id"].nunique()),
        "distinct_invoice_ids": int(data["invoice_id"].nunique()),
        "distinct_stock_codes": int(data["stock_code"].nunique()),
        "distinct_countries": int(data["country"].nunique()), "nulls": nulls,
        "exact_duplicate_excess_rows": int(duplicates.sum()),
        "rows_in_exact_duplicate_groups": int(duplicate_group_rows.sum()),
        "cross_sheet_duplicate_hashes": int(cross_sheet_hashes.gt(1).sum()),
        "duplicate_excess_rows_by_month": {str(k): int(v) for k, v in duplicate_months.items()},
        "cancel_prefix_rows": int(cancel.sum()), "negative_quantity_rows": int(negative.sum()),
        "cancel_prefix_nonnegative_quantity_rows": int((cancel & ~negative).sum()),
        "negative_quantity_without_cancel_prefix_rows": int((negative & ~cancel).sum()),
        "zero_quantity_rows": int(data["quantity"].eq(0).sum()),
        "nonpositive_price_rows": int(data["unit_price"].le(0).sum()),
        "negative_price_rows": int(data["unit_price"].lt(0).sum()),
        "invoice_ids_with_multiple_known_customers": int(invoice_customers.gt(1).sum()),
        "invoice_ids_with_multiple_timestamps": int(invoice_timestamps.gt(1).sum()),
        "invoice_ids_with_multiple_countries": int(invoice_countries.gt(1).sum()),
        "customer_ids_with_multiple_countries": int(customer_countries.gt(1).sum()),
        "nonnumeric_5digit_product_pattern_rows": int((~product_pattern).sum()),
        "numeric_quantiles": numeric_quantiles,
        "dates_without_rows": len(absent_dates),
        "absent_dates_by_weekday": pd.Series(absent_dates.day_name()).value_counts().to_dict(),
        "provisional_purchase_scenario": {
            "definition": "Known customer, positive quantity/price, no C invoice prefix, remove excess exact duplicate rows, stock code matches five digits plus optional letters.",
            "positive_noncancel_rows_all_ids": int(valid_sale.sum()),
            "identified_positive_noncancel_rows": int(identified_sale.sum()),
            "identified_positive_noncancel_customers": int(data.loc[identified_sale, "customer_id"].nunique()),
            "identified_positive_noncancel_deduplicated_rows": int((identified_sale & ~duplicates).sum()),
            "candidate_product_rows": len(sales), "orders": len(orders),
            "customers": int(sales["customer_id"].nunique()),
            "one_order_customers": int(customer_counts.eq(1).sum()),
            "repeat_order_customers": int(customer_counts.gt(1).sum()),
            "repeat_purchase_day_customers": int(occasion_counts.gt(1).sum()),
            "one_purchase_day_customers": int(occasion_counts.eq(1).sum()),
            "positive_sales_revenue_gbp": float(sales["revenue"].sum()),
            "top_1pct_customer_revenue_share_pct": float(customer_revenue.head(max(1, int(np.ceil(len(customer_revenue) * .01)))).sum() / customer_revenue.sum() * 100),
            "top_10pct_customer_revenue_share_pct": float(customer_revenue.head(max(1, int(np.ceil(len(customer_revenue) * .1)))).sum() / customer_revenue.sum() * 100),
            "order_value_quantiles_gbp": {str(q): float(v) for q, v in orders["revenue"].quantile([.5, .9, .99, 1]).items()},
        },
        "snapshots": snapshots,
        "limitations": [
            "No ingestion timestamp or event log is available; source recording completeness cannot be proven.",
            "Missing dates can reflect trading schedules rather than ingestion gaps.",
            "A repeated invoice line may be legitimate; duplicate removal remains a sensitivity scenario.",
            "The product-code regex is provisional; accounting/service/gift codes need explicit review.",
            "The final date is treated as incomplete because the last transaction occurs during that day.",
        ],
    }
    (OUTPUT / "dataset_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ["raw_rows", "date_min", "date_max", "distinct_known_customers", "exact_duplicate_excess_rows"]}, indent=2), flush=True)
    print("Aggregate evidence saved under research/evidence", flush=True)
    return result


if __name__ == "__main__":
    audit()
