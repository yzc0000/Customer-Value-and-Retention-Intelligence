"""Streamlit presentation of the combined, saved customer review dataset."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from .retention import GROUPS, POLICIES, assign_review_groups, compare_policies, filter_review, rank_review


FILTER_KEYS = ["retention_segments", "retention_country", "retention_group", "retention_search",
               "retention_minimum_risk", "retention_capacity", "retention_policy",
               "retention_risk_threshold", "retention_value_threshold", "retention_customer"]


def reset_filters():
    for key in FILTER_KEYS:
        st.session_state.pop(key, None)


def render_retention_review(root: Path):
    path = root / "data" / "processed" / "customer_retention_review_90d.parquet"
    manifest_path = root / "artifacts" / "runs" / "retention" / "manifest.json"
    st.subheader("Customer retention review")
    if not path.exists() or not manifest_path.exists():
        st.info("Run `python scripts/project.py retention` after the model and segmentation stages to create the combined customer view.")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("review_output") and hashlib.sha256(path.read_bytes()).hexdigest() != manifest["review_output"]["sha256"]:
        st.info("The saved review data changed. Run `python scripts/project.py retention` to rebuild it from model outputs.")
        return
    for source in manifest["source_files"].values():
        p = root / source["path"]
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != source["sha256"]:
            st.info("An input to this review changed. Run `python scripts/project.py retention` to refresh the combined view.")
            return
    rows = pd.read_parquet(path)
    defaults = manifest["defaults"]
    cutoff = pd.Timestamp(manifest["cutoff"])
    final_day = pd.Timestamp(manifest["forecast_end_exclusive"]) - pd.Timedelta(days=1)
    st.caption(f"Historical snapshot: {cutoff:%d %B %Y} · 90-day forecast through {final_day:%d %B %Y}. Saved model predictions for the retrospective cohort.")
    st.write("Review priority = inactivity probability × historical average order value. Expected revenue is shown separately and describes gross sales without an intervention. Campaign recovery is not estimated.")
    labels = rows.drop_duplicates("kmeans_segment").set_index("kmeans_segment").segment_label.to_dict()
    left, middle, right = st.columns(3)
    chosen_segments = left.multiselect("Customer segments", sorted(labels), default=sorted(labels),
        format_func=lambda value: labels[value], key="retention_segments")
    chosen_country = middle.selectbox("Country", ["All countries"] + sorted(rows.country.unique()), key="retention_country")
    chosen_group = right.selectbox("Risk/value group", ["All groups"] + GROUPS, key="retention_group")
    left, middle, right = st.columns(3)
    minimum_risk = left.slider("Minimum inactivity probability (%)", 0, 100, 0, step=5, key="retention_minimum_risk") / 100
    capacity = middle.slider("Review capacity (% of filtered customers)", 0, 100, int(defaults["capacity_fraction"] * 100), step=1, key="retention_capacity") / 100
    policy = right.selectbox("Rank customers by", list(POLICIES), format_func=lambda value: POLICIES[value], key="retention_policy")
    left, right = st.columns([3, 1])
    customer_search = left.text_input("Customer ID contains", key="retention_search")
    right.button("Reset review filters", on_click=reset_filters, key="retention_reset")
    with st.expander("Risk/value group thresholds"):
        a, b = st.columns(2)
        risk_threshold = a.slider("Higher-risk threshold (%)", 0, 100, int(defaults["risk_threshold"] * 100), key="retention_risk_threshold") / 100
        value_threshold = b.number_input("Higher-value threshold: historical average order (GBP)", min_value=0.0,
            value=float(defaults["value_threshold_gbp"]), step=25.0, key="retention_value_threshold")
        st.caption("The default value threshold is the June 2011 cohort's 75th percentile of historical average order value. It stays fixed when filtering; you can change it for exploration.")
    grouped = assign_review_groups(rows, risk_threshold, value_threshold)
    filtered = filter_review(grouped, chosen_segments, None if chosen_country == "All countries" else chosen_country,
        minimum_risk, customer_search, None if chosen_group == "All groups" else [chosen_group])
    ranked = rank_review(filtered, policy, capacity)
    queue = ranked.loc[ranked.in_review_queue].copy()
    a, b, c, d = st.columns(4)
    a.metric("Filtered customers", f"{len(ranked):,}")
    b.metric("Customers in review queue", f"{len(queue):,}")
    c.metric("Queue mean inactivity probability", f"{queue.inactive_probability_90d.mean():.1%}" if len(queue) else "—")
    d.metric("Queue expected 90-day revenue", f"£{queue.predicted_revenue_90d_gbp.sum():,.0f}" if len(queue) else "£0",
        help="Sum of unconditional expected gross merchandise revenue for these customers; no campaign effect is assumed.")
    if ranked.empty:
        st.info("No customers match these filters. Reset the filters or broaden the selection.")
        return

    st.subheader("Customer review queue")
    st.caption(f"Ranking: {POLICIES[policy]}. Showing up to 100 of {len(queue):,} selected customers. Capacity is rounded up to a whole customer.")
    columns = ["review_rank", "customer_id", "segment_label", "review_group", "country",
        "inactive_probability_90d", "predicted_revenue_90d_gbp", "avg_order_value_gbp", "review_priority_score", "recency_days"]
    if queue.empty:
        st.info("Review capacity is zero. Increase it to create a customer queue.")
    else:
        display = queue[columns].head(100).copy()
        display["inactive_probability_90d"] *= 100
        st.dataframe(display, width="stretch", hide_index=True, height=400, column_config={
            "review_rank": st.column_config.NumberColumn("Rank", format="%d"),
            "customer_id": st.column_config.NumberColumn("Customer ID", format="%d"),
            "segment_label": "Segment", "review_group": "Risk/value group", "country": "Country",
            "inactive_probability_90d": st.column_config.NumberColumn("Inactivity probability (%)", format="%.1f"),
            "predicted_revenue_90d_gbp": st.column_config.NumberColumn("Expected 90-day revenue (£)", format="£%.2f"),
            "avg_order_value_gbp": st.column_config.NumberColumn("Historical average order (£)", format="£%.2f"),
            "review_priority_score": st.column_config.NumberColumn("Review priority score", format="%.2f"),
            "recency_days": st.column_config.NumberColumn("Days since last purchase", format="%.1f"),
        })
    st.download_button("Download this review queue", queue.to_csv(index=False), "customer_retention_review_90d.csv", "text/csv", key="retention_download_queue")
    st.download_button("Download all filtered customers", ranked.to_csv(index=False), "customer_retention_filtered_90d.csv", "text/csv", key="retention_download_filtered")

    st.subheader("Inactivity risk and historical order value")
    plotted = ranked.copy()
    plotted["inactivity_percent"] = plotted.inactive_probability_90d * 100
    plotted["queue_status"] = plotted.in_review_queue.map({True: "In review queue", False: "Outside queue"})
    st.vega_lite_chart(plotted, {
        "height": 400,
        "layer": [
            {"mark": {"type": "point", "filled": True, "opacity": 0.55}, "encoding": {
                "x": {"field": "avg_order_value_gbp", "type": "quantitative", "scale": {"type": "log"}, "axis": {"tickCount": 6}, "title": "Historical average order (£, log scale)"},
                "y": {"field": "inactivity_percent", "type": "quantitative", "scale": {"domain": [0, 100]}, "title": "Inactivity probability (%)"},
                "color": {"field": "segment_label", "type": "nominal", "title": "Segment", "legend": {"orient": "bottom", "columns": 1, "direction": "vertical", "labelLimit": 290}},
                "size": {"field": "queue_status", "type": "nominal", "scale": {"domain": ["Outside queue", "In review queue"], "range": [15, 65]}, "title": "Review selection", "legend": None},
                "tooltip": [{"field": "customer_id", "title": "Customer ID", "format": "d"},
                    {"field": "segment_label", "title": "Segment"},
                    {"field": "inactivity_percent", "title": "Inactivity (%)", "format": ".1f"},
                    {"field": "avg_order_value_gbp", "title": "Historical average order (£)", "format": ".2f"},
                    {"field": "predicted_revenue_90d_gbp", "title": "Expected 90-day revenue (£)", "format": ".2f"},
                    {"field": "review_group", "title": "Risk/value group"}, {"field": "queue_status", "title": "Review selection"}],
            }},
            {"data": {"values": [{"threshold": risk_threshold * 100}]}, "mark": {"type": "rule", "strokeDash": [5, 5], "color": "#666"},
                "encoding": {"y": {"field": "threshold", "type": "quantitative"}}},
            *([{ "data": {"values": [{"threshold": value_threshold}]}, "mark": {"type": "rule", "strokeDash": [5, 5], "color": "#666"},
                "encoding": {"x": {"field": "threshold", "type": "quantitative"}}}] if value_threshold > 0 else []),
        ],
    }, width="stretch")
    st.caption("Each point is one filtered customer; larger points are in the review queue. Dashed lines mark the risk/value thresholds. The value axis uses a log scale.")
    counts = ranked.groupby("review_group").agg(customers=("customer_id", "size"), selected=("in_review_queue", "sum")).reindex(GROUPS, fill_value=0)
    counts["In review queue"] = counts.selected
    counts["Outside queue"] = counts.customers - counts.selected
    st.subheader("Review selection by risk/value group")
    chart_rows = counts[["In review queue", "Outside queue"]].reset_index().melt("review_group", var_name="Selection", value_name="Customers")
    st.vega_lite_chart(chart_rows, {"mark": "bar", "height": 320, "encoding": {
        "y": {"field": "review_group", "type": "nominal", "sort": GROUPS, "title": None, "axis": {"labelLimit": 160}},
        "x": {"field": "Customers", "type": "quantitative", "title": "Filtered customers", "axis": {"tickCount": 4}},
        "color": {"field": "Selection", "type": "nominal", "legend": {"orient": "bottom", "columns": 1, "direction": "vertical"}},
        "tooltip": [{"field": "review_group", "title": "Risk/value group"}, {"field": "Selection"}, {"field": "Customers"}],
    }}, width="stretch")

    st.subheader("Ranking tradeoffs at the same capacity")
    comparison = compare_policies(filtered, capacity).drop(columns="policy").rename(columns={
        "ranking": "Ranking", "customers": "Customers", "mean_inactivity_probability": "Mean inactivity probability",
        "median_historical_order_value_gbp": "Median historical order (£)", "predicted_revenue_90d_gbp": "Expected 90-day revenue (£)",
    })
    st.dataframe(comparison.style.format({"Mean inactivity probability": "{:.1%}",
        "Median historical order (£)": "£{:,.2f}", "Expected 90-day revenue (£)": "£{:,.2f}"}, na_rep="—"), width="stretch", hide_index=True)
    st.caption("Every ranking uses the same filtered population and queue size. The columns summarize saved predictions and observed history; campaign response and future outcomes do not choose a ranking.")

    if len(queue):
        with st.expander("Inspect a customer in the review queue"):
            customer = st.selectbox("Customer ID", queue.customer_id.tolist(), key="retention_customer")
            row = queue.loc[queue.customer_id.eq(customer)].iloc[0]
            st.write(f"{row.segment_label} · {row.country} · {row.review_group}")
            st.write(f"Inactivity probability {row.inactive_probability_90d:.1%}; historical average order £{row.avg_order_value_gbp:,.2f}; review priority score {row.review_priority_score:,.2f}.")
            st.write(f"Expected 90-day revenue £{row.predicted_revenue_90d_gbp:,.2f}; {row.purchase_count:,.0f} prior orders; last purchase {row.recency_days:,.1f} days before the snapshot.")
            st.write(f"Historical spend: £{row.spend_90d_gbp:,.2f} over the prior 90 days and £{row.spend_365d_gbp:,.2f} over the prior 365 days.")
            st.caption(f"Alternative GMM assignment: {int(row.gmm_segment)}; maximum membership {row.gmm_max_membership:.1%}. The segment description comes from cluster profiles; the review score comes from the declared ranking rule.")
    with st.expander("Data sources and definitions"):
        st.write(f"Inactivity model: {manifest['inactivity_model']}. Revenue model: {manifest['revenue_model']}. All records join on customer ID and the same forecast date.")
        st.json(manifest["metric_definitions"])
        st.dataframe(pd.DataFrame([dict(Input=name.replace("_", " ").title(), File=source["path"]) for name, source in manifest["source_files"].items()]), width="stretch", hide_index=True)
        st.caption("The separate six-month forecasts are from June 2011 and remain in the Revenue tab. September review records use the aligned 90-day forecasts.")
