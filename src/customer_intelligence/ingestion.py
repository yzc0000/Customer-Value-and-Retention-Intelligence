"""Workbook ingestion with conservative overlap reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "Invoice": "invoice_id", "InvoiceNo": "invoice_id",
    "StockCode": "stock_code", "Description": "description",
    "Quantity": "quantity", "InvoiceDate": "invoice_date",
    "Price": "unit_price", "UnitPrice": "unit_price",
    "Customer ID": "customer_id", "CustomerID": "customer_id",
    "Country": "country",
}
SHEET_OWNER = {"Year 2009-2010": "legacy", "Year 2010-2011": "current"}
PRODUCT_PATTERN = r"\d{5}[A-Za-z]*"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_workbook(path: Path) -> tuple[pd.DataFrame, dict]:
    """Read both sheets and keep the maximum observed exact-row multiplicity.

    This reconciles the overlapping date ranges without globally deduplicating
    legitimate repeated rows within either source sheet. A 64-bit row
    fingerprint identifies source row signatures across sheets.
    """
    if not path.exists():
        raise FileNotFoundError(f"Workbook not found: {path}")

    book = pd.ExcelFile(path, engine="openpyxl")
    parts: list[pd.DataFrame] = []
    sheet_summary = []
    for sheet_name in book.sheet_names:
        source = pd.read_excel(book, sheet_name=sheet_name)
        unknown = sorted(set(source.columns) - set(ALIASES))
        missing = sorted(set(ALIASES).intersection({"Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "Customer ID", "Country"}) - set(source.columns))
        if missing:
            raise ValueError(f"Unexpected workbook schema in {sheet_name}: missing {missing}")
        source = source.rename(columns=ALIASES)
        source["source_sheet"] = sheet_name
        source["source_row"] = np.arange(2, len(source) + 2, dtype=np.int64)
        source["source_sheet_role"] = SHEET_OWNER.get(sheet_name, "unmapped")
        sheet_summary.append({
            "sheet": sheet_name,
            "rows": int(len(source)),
            "date_min": str(pd.to_datetime(source["invoice_date"], errors="coerce").min()),
            "date_max": str(pd.to_datetime(source["invoice_date"], errors="coerce").max()),
            "missing_customer_rows": int(source["customer_id"].isna().sum()),
            "unrecognized_source_columns": unknown,
        })
        parts.append(source)
        print(f"Read worksheet {sheet_name}: {len(source):,} lines", flush=True)

    frame = pd.concat(parts, ignore_index=True, sort=False)
    raw_columns = list(dict.fromkeys(ALIASES.values()))
    raw_hash = pd.util.hash_pandas_object(frame[raw_columns], index=False).astype("uint64")
    source_counts = pd.DataFrame({"hash": raw_hash, "sheet": frame["source_sheet"]}).groupby(["hash", "sheet"], sort=False).size()
    multiplicity = source_counts.groupby(level="hash").max()
    row_rank = pd.Series(raw_hash.to_numpy()).groupby(raw_hash.to_numpy(), sort=False).cumcount()
    allowed = multiplicity.reindex(raw_hash.to_numpy()).to_numpy()
    keep = row_rank.to_numpy() < allowed

    reconciled = frame.loc[keep].copy()
    reconciled["invoice_id"] = reconciled["invoice_id"].astype("string").str.strip()
    reconciled["stock_code"] = reconciled["stock_code"].astype("string").str.strip()
    reconciled["description"] = reconciled["description"].astype("string")
    reconciled["country"] = reconciled["country"].astype("string")
    reconciled["invoice_date"] = pd.to_datetime(reconciled["invoice_date"], errors="coerce")
    reconciled["quantity"] = pd.to_numeric(reconciled["quantity"], errors="coerce")
    reconciled["unit_price"] = pd.to_numeric(reconciled["unit_price"], errors="coerce")
    reconciled["customer_id"] = pd.to_numeric(reconciled["customer_id"], errors="coerce").astype("Int64")

    invoice_text = reconciled["invoice_id"].str.upper()
    is_credit = invoice_text.str.startswith("C", na=False)
    stock_product = reconciled["stock_code"].str.fullmatch(PRODUCT_PATTERN, na=False)
    valid_positive = reconciled["quantity"].gt(0) & reconciled["unit_price"].gt(0) & ~is_credit
    identifiable = reconciled["customer_id"].notna()
    is_sale = valid_positive & identifiable & stock_product & reconciled["invoice_date"].notna()

    sales = reconciled.loc[is_sale, [
        "invoice_id", "stock_code", "invoice_date", "quantity", "unit_price",
        "customer_id", "country", "source_sheet", "source_row",
    ]].copy()
    sales["customer_id"] = sales["customer_id"].astype("int64")
    sales["revenue"] = sales["quantity"] * sales["unit_price"]
    sales["purchase_day"] = sales["invoice_date"].dt.normalize()

    credits = reconciled.loc[
        is_credit & reconciled["quantity"].lt(0) & identifiable & reconciled["invoice_date"].notna(),
        ["invoice_id", "stock_code", "invoice_date", "quantity", "unit_price", "customer_id", "country", "source_sheet", "source_row"],
    ].copy()
    credits["customer_id"] = credits["customer_id"].astype("int64")
    credits["credit_amount_abs"] = (credits["quantity"] * credits["unit_price"]).abs()

    rejected_nonproduct = valid_positive & identifiable & ~stock_product
    unscored_known_sale = valid_positive & ~identifiable & stock_product
    result = {
        "source": path.name,
        "source_sha256": sha256(path),
        "row_counts": {
            "raw_lines": int(len(frame)),
            "overlap_excess_lines_removed": int((~keep).sum()),
            "reconciled_lines": int(len(reconciled)),
            "identified_purchase_lines": int(len(sales)),
            "identified_credit_lines": int(len(credits)),
            "anonymous_candidate_purchase_lines": int(unscored_known_sale.sum()),
            "known_customer_nonproduct_sale_lines_excluded": int(rejected_nonproduct.sum()),
            "missing_customer_ids": int(reconciled["customer_id"].isna().sum()),
            "cancel_prefix_lines": int(is_credit.sum()),
            "negative_quantity_lines_without_cancel_prefix": int((reconciled["quantity"].lt(0) & ~is_credit).sum()),
            "nonpositive_price_lines": int(reconciled["unit_price"].le(0).sum()),
        },
        "sheets": sheet_summary,
        "policy": {
            "overlap": "For exact matching rows, retain the maximum multiplicity observed in either worksheet. Preserve repeated identical rows within each sheet.",
            "purchase": "Known customer; positive quantity and unit price; invoice is not C-prefixed; stock code is five digits with optional letters.",
            "revenue": "Gross positive merchandise sales in source currency (GBP); credits are separately recorded, not allocated to original orders.",
            "time": "Source local naive timestamps; windows are end-exclusive; retain row dates when filtering before aggregation.",
        },
    }
    return sales, credits, result


def write_dataset(sales: pd.DataFrame, credits: pd.DataFrame, summary: dict, root: Path) -> None:
    interim = root / "data" / "interim"
    interim.mkdir(parents=True, exist_ok=True)
    sales.to_parquet(interim / "purchase_lines.parquet", index=False)
    credits.to_parquet(interim / "credit_lines.parquet", index=False)
    (interim / "data_quality.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    invoice_summary = sales.groupby(["customer_id", "invoice_id"], sort=False).agg(
        invoice_date=("invoice_date", "min"),
        revenue=("revenue", "sum"),
        units=("quantity", "sum"),
        line_count=("stock_code", "size"),
        product_count=("stock_code", "nunique"),
        country=("country", "last"),
    ).reset_index()
    invoice_summary.to_parquet(interim / "purchase_invoices.parquet", index=False)
