"""Join customer predictions and segments into a historical retention review.

Priority is a declared heuristic based only on predictions and prior history.
Future outcomes are deliberately absent from the joined customer artifact.
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["customer_id", "cutoff"]
HISTORY = ["country", "recency_days", "tenure_days", "purchase_count",
           "avg_order_value_gbp", "lifetime_revenue_gbp", "spend_90d_gbp", "spend_365d_gbp"]
POLICIES = {
    "risk_value": "Risk × historical average order value",
    "risk": "Inactivity risk only",
    "forecast_value": "Expected 90-day revenue only",
    "historical_value": "Historical 365-day spend only",
}
POLICY_COLUMNS = dict(risk_value="review_priority_score", risk="inactive_probability_90d",
                      forecast_value="predicted_revenue_90d_gbp", historical_value="spend_365d_gbp")
GROUPS = ["Higher value, higher risk", "Higher value, lower risk",
          "Lower value, higher risk", "Lower value, lower risk"]


def _cohort(frame, cutoff, name, columns):
    missing = set(KEYS + columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name}: missing required columns {sorted(missing)}")
    rows = frame[KEYS + columns].copy()
    rows["cutoff"] = pd.to_datetime(rows.cutoff, errors="raise")
    rows = rows.loc[rows.cutoff.eq(pd.Timestamp(cutoff))].copy()
    if rows.empty:
        raise ValueError(f"{name}: no rows at the shared forecast cutoff {cutoff}")
    if rows[KEYS].isna().any().any() or rows.duplicated(KEYS).any():
        raise ValueError(f"{name}: customer/cutoff keys must be non-null and unique")
    return rows


def combine_customer_review(features, risk, revenue, segments, cutoff, risk_model, revenue_model):
    """Require complete one-to-one coverage at the same forecast origin."""
    rows = _cohort(features, cutoff, "Customer history", HISTORY)
    risk_column = f"{risk_model}_risk"
    sources = [
        ("Inactivity predictions", _cohort(risk, cutoff, "Inactivity predictions", [risk_column]).rename(columns={risk_column: "inactive_probability_90d"})),
        ("Revenue predictions", _cohort(revenue, cutoff, "Revenue predictions", ["predicted_revenue_gbp"]).rename(columns={"predicted_revenue_gbp": "predicted_revenue_90d_gbp"})),
        ("Segment assignments", _cohort(segments, cutoff, "Segment assignments", ["kmeans_segment", "gmm_segment", "gmm_max_membership"])),
    ]
    expected = pd.MultiIndex.from_frame(rows[KEYS])
    for name, source in sources:
        observed = pd.MultiIndex.from_frame(source[KEYS])
        absent, extra = expected.difference(observed), observed.difference(expected)
        if len(absent) or len(extra):
            raise ValueError(f"{name}: cohort coverage mismatch ({len(absent)} missing, {len(extra)} extra customer/cutoff keys)")
        rows = rows.merge(source, on=KEYS, how="left", validate="one_to_one")
    numeric = [c for c in HISTORY if c != "country"] + ["inactive_probability_90d", "predicted_revenue_90d_gbp", "gmm_max_membership"]
    if not np.isfinite(rows[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Review inputs contain missing or non-finite numeric values")
    if not rows.inactive_probability_90d.between(0, 1).all() or not rows.gmm_max_membership.between(0, 1).all():
        raise ValueError("Probabilities must be between zero and one")
    if (rows[[c for c in HISTORY if c != "country"] + ["predicted_revenue_90d_gbp"]] < 0).any().any():
        raise ValueError("Historical values and expected gross revenue must be nonnegative")
    if (rows.avg_order_value_gbp <= 0).any():
        raise ValueError("Customers with qualifying positive sales must have positive historical average order values")
    if rows.country.isna().any() or rows[["kmeans_segment", "gmm_segment"]].isna().any().any():
        raise ValueError("Country and segment assignments must be present")
    rows["review_priority_score"] = rows.inactive_probability_90d * rows.avg_order_value_gbp
    rows["forecast_end_exclusive"] = rows.cutoff + pd.Timedelta(days=90)
    rows["inactivity_model"], rows["revenue_model"] = risk_model, revenue_model
    return rows


def assign_review_groups(rows, risk_threshold, value_threshold):
    if not 0 <= risk_threshold <= 1 or not np.isfinite(value_threshold) or value_threshold < 0:
        raise ValueError("Invalid risk/value thresholds")
    result = rows.copy()
    high_risk = result.inactive_probability_90d >= risk_threshold
    high_value = result.avg_order_value_gbp >= value_threshold
    result["review_group"] = np.select(
        [high_value & high_risk, high_value & ~high_risk, ~high_value & high_risk],
        GROUPS[:3], default=GROUPS[3],
    )
    result["risk_threshold"] = risk_threshold
    result["historical_order_value_threshold_gbp"] = value_threshold
    return result


def filter_review(rows, segment_ids=None, country=None, minimum_risk=0.0, customer_search="", groups=None):
    if not 0 <= minimum_risk <= 1:
        raise ValueError("Minimum risk must be a probability")
    result = rows.loc[rows.inactive_probability_90d >= minimum_risk].copy()
    if segment_ids is not None:
        result = result.loc[result.kmeans_segment.isin(segment_ids)]
    if country is not None:
        result = result.loc[result.country.eq(country)]
    if groups is not None:
        result = result.loc[result.review_group.isin(groups)]
    if customer_search.strip():
        result = result.loc[result.customer_id.astype(str).str.contains(customer_search.strip(), regex=False)]
    return result


def rank_review(rows, policy="risk_value", capacity=0.10):
    if policy not in POLICIES or not np.isfinite(capacity) or not 0 <= capacity <= 1:
        raise ValueError("Unknown policy or invalid review capacity")
    sort_columns = list(dict.fromkeys([POLICY_COLUMNS[policy], "inactive_probability_90d", "predicted_revenue_90d_gbp", "customer_id"]))
    result = rows.sort_values(sort_columns, ascending=[c == "customer_id" for c in sort_columns], kind="stable").copy()
    result["review_rank"] = np.arange(1, len(result) + 1)
    result["in_review_queue"] = result.review_rank <= math.ceil(len(result) * capacity)
    result["ranking_policy"] = POLICIES[policy]
    result["review_capacity_fraction"] = capacity
    result["filtered_population_customers"] = len(result)
    return result.reset_index(drop=True)


def compare_policies(rows, capacity=0.10):
    records = []
    for policy, title in POLICIES.items():
        ranked = rank_review(rows, policy, capacity)
        queue = ranked.loc[ranked.in_review_queue]
        records.append(dict(policy=policy, ranking=title, customers=len(queue),
            mean_inactivity_probability=float(queue.inactive_probability_90d.mean()) if len(queue) else np.nan,
            median_historical_order_value_gbp=float(queue.avg_order_value_gbp.median()) if len(queue) else np.nan,
            predicted_revenue_90d_gbp=float(queue.predicted_revenue_90d_gbp.sum())))
    return pd.DataFrame(records)


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_retention_review(root: Path):
    processed = root / "data" / "processed"
    champions = _json(root / "artifacts" / "champions.json")
    prepared = _json(processed / "prepare_manifest.json")
    risk_model = champions["inactivity_90d"]["selected_model"]
    cutoff = pd.Timestamp(champions["inactivity_90d"]["test_cutoff"])
    paths = {
        "data_preparation": processed / "prepare_manifest.json",
        "customer_history": processed / "customer_features.parquet",
        "inactivity_predictions": processed / "final_customer_scores_90d.parquet",
        "segment_assignments": processed / "customer_segments.parquet",
        "model_registry": root / "artifacts" / "champions.json",
        "segment_profiles": root / "reports" / "segment_profiles.csv",
        "segment_fit_metadata": root / "artifacts" / "runs" / "segments" / "kmeans_champion.json",
    }
    choice_path = root / "artifacts" / "runs" / "revenue_challengers" / "90d" / "selection.json"
    expanded_path = processed / "revenue_challenger_scores_90d.parquet"
    if choice_path.exists() and expanded_path.exists():
        protocol = _json(choice_path.parent.parent / "protocol.json")
        if protocol["source_sha256"] != prepared["source_sha256"]:
            raise ValueError("Expanded revenue scores use a different workbook. Rerun the revenue challengers first.")
        revenue_model = _json(choice_path)["selected_model"]
        paths["revenue_predictions"] = expanded_path
        paths["revenue_selection"] = choice_path
        paths["revenue_protocol"] = choice_path.parent.parent / "protocol.json"
        revenue = pd.read_parquet(expanded_path)
        if not revenue.selected_model.eq(revenue_model).all():
            raise ValueError("Selected revenue model and saved score identities disagree")
        revenue_source = "expanded_expected_revenue_selection"
    else:
        if champions["future_revenue_90d"]["source_sha256"] != prepared["source_sha256"]:
            raise ValueError("Revenue scores use a different workbook. Rerun revenue first.")
        revenue_model = champions["future_revenue_90d"]["selected_model"]
        paths["revenue_predictions"] = processed / "final_customer_scores_90d_revenue.parquet"
        revenue = pd.read_parquet(paths["revenue_predictions"])
        revenue = revenue.rename(columns={revenue_model: "predicted_revenue_gbp"})
        revenue_source = "original_revenue_portfolio"
    if champions["inactivity_90d"]["source_sha256"] != prepared["source_sha256"]:
        raise ValueError("Inactivity scores use a different workbook. Rerun inactivity first.")
    features = pd.read_parquet(paths["customer_history"])
    rows = combine_customer_review(features, pd.read_parquet(paths["inactivity_predictions"]), revenue,
        pd.read_parquet(paths["segment_assignments"]), cutoff, risk_model, revenue_model)
    segment_fit = _json(root / "artifacts" / "runs" / "segments" / "kmeans_champion.json")
    if segment_fit["source_sha256"] != prepared["source_sha256"]:
        raise ValueError("Segment assignments use a different workbook. Rerun segmentation first.")
    profiles = pd.read_csv(root / "reports" / "segment_profiles.csv")
    labels = {int(s): f"Segment {int(s)}" for s in rows.kmeans_segment.unique()}
    if len(profiles) == 2 and profiles.purchase_day_count_median.nunique() == 2:
        repeat = int(profiles.loc[profiles.purchase_day_count_median.idxmax(), "segment"])
        sparse = int(profiles.loc[profiles.purchase_day_count_median.idxmin(), "segment"])
        labels[repeat] = f"Segment {repeat}: repeat purchase history"
        labels[sparse] = f"Segment {sparse}: sparse purchase history"
    rows["segment_label"] = rows.kmeans_segment.map(labels)
    reference_cutoff = pd.Timestamp("2011-06-01")
    reference = features.loc[features.cutoff.eq(reference_cutoff), "avg_order_value_gbp"]
    if reference.empty or reference_cutoff >= cutoff:
        raise ValueError("An earlier customer cohort is required for the value threshold")
    value_threshold = float(reference.quantile(0.75))
    rows = assign_review_groups(rows, 0.5, value_threshold)
    ranked = rank_review(rows)
    output = processed / "customer_retention_review_90d.parquet"
    ranked.to_parquet(output, index=False)
    queue = ranked.loc[ranked.in_review_queue].copy()
    queue.to_csv(processed / "retention_review_priority_90d.csv", index=False)
    compare_policies(rows).to_csv(root / "reports" / "retention_policy_comparison.csv", index=False)
    summary = ranked.groupby(["segment_label", "review_group"], as_index=False).agg(
        customers=("customer_id", "size"), queue_customers=("in_review_queue", "sum"),
        mean_inactivity_probability=("inactive_probability_90d", "mean"),
        predicted_revenue_90d_gbp=("predicted_revenue_90d_gbp", "sum"),
        median_historical_order_value_gbp=("avg_order_value_gbp", "median"),
    )
    summary.to_csv(root / "reports" / "retention_review_summary.csv", index=False)
    manifest = dict(schema_version=1, created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=prepared["source_sha256"], cutoff=str(cutoff.date()), horizon_days=90,
        forecast_end_exclusive=str((cutoff + pd.Timedelta(days=90)).date()),
        population="Identified customers with qualifying merchandise history before cutoff; one row per customer and cutoff.",
        customers=len(rows), default_queue_customers=len(queue), join="Complete one-to-one customer_id + cutoff across all four inputs.",
        classification="historical_retrospective_model_predictions", inactivity_model=risk_model, revenue_model=revenue_model,
        revenue_selection_source=revenue_source, segment_fit=segment_fit,
        source_files={name: dict(path=p.relative_to(root).as_posix(), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name, p in paths.items()},
        review_output=dict(path=output.relative_to(root).as_posix(), sha256=hashlib.sha256(output.read_bytes()).hexdigest()),
        defaults=dict(policy="risk_value", capacity_fraction=0.10, risk_threshold=0.5, value_threshold_gbp=value_threshold,
            value_reference_cutoff=str(reference_cutoff.date()), value_reference_quantile=0.75),
        metric_definitions=dict(inactive_probability_90d="Calibrated probability of no qualifying purchase over the next 90 days.",
            predicted_revenue_90d_gbp="Unconditional expected gross merchandise sales over the same 90-day horizon, in GBP.",
            avg_order_value_gbp="Mean observed qualifying invoice value strictly before cutoff, in GBP.",
            review_priority_score="inactive_probability_90d * avg_order_value_gbp; a review heuristic using a historical order-value proxy.",
            in_review_queue="Top ceil(filtered customer count * capacity fraction) under the selected ranking; ties use risk, expected revenue, then customer ID.",
            review_group="Risk compared with 0.5 and historical average order value compared with the earlier June cohort's 75th percentile by default."),
        limitations=["Historical source ending December 2011; forecasts are retrospective.",
            "Review priority does not estimate treatment response, recovered revenue or profit.",
            "September 90-day and June six-month forecasts have different origins and are not combined.",
            "Unconditional predicted revenue is not divided by a separately modeled return probability to invent conditional value.",
            "Observed future outcomes do not enter review records, filtering, ranking or policy comparisons.",
            "Revenue forecasts have material retrospective bias; model comparisons remain in the Revenue tab."])
    path = root / "artifacts" / "runs" / "retention" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Combined {len(rows):,} customers at {cutoff.date()}: {risk_model} + {revenue_model} + customer segments.", flush=True)
    print(f"Default 10% review queue: {len(queue):,} customers. Historical order-value threshold: GBP {value_threshold:,.2f}.", flush=True)
    return ranked
