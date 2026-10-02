"""Local Streamlit reader for the saved, versioned project artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from customer_intelligence.retention_view import render_retention_review
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

st.set_page_config(page_title="Customer Value & Retention Intelligence", layout="wide")
st.title("Customer Value & Retention Intelligence")
st.caption("Online Retail II · local historical model comparison")


def load_json(path: Path, fallback=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else (fallback or {})


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


manifest = load_json(PROCESSED / "prepare_manifest.json")
champions = load_json(ROOT / "artifacts" / "champions.json")
if not manifest:
    st.info("Run `python scripts/project.py all` to prepare the data and save model runs.")
    st.stop()

cohort = load_csv(REPORTS / "cohort_summary.csv")
inactivity = load_csv(REPORTS / "inactivity_model_comparison.csv")
revenue90 = load_csv(REPORTS / "revenue_90d_model_comparison.csv")
revenue6m = load_csv(REPORTS / "revenue_6m_model_comparison.csv")
segments = load_csv(REPORTS / "segment_profiles.csv")
segment_profile_index = load_csv(REPORTS / "segment_profile_index.csv")
segment_outcomes = load_csv(REPORTS / "segment_holdout_outcomes.csv")
gmm_candidates = load_csv(REPORTS / "gmm_candidates.csv")

tab_retention, tab_overview, tab_risk, tab_value, tab_segments, tab_data = st.tabs(
    ["Retention review", "Overview", "Inactivity", "Revenue", "Segments", "Data policy"]
)

with tab_retention:
    render_retention_review(ROOT)

with tab_overview:
    left, middle, right = st.columns(3)
    left.metric("Source lines", f"{manifest['row_counts']['raw_lines']:,}")
    middle.metric("Purchase lines modeled", f"{manifest['row_counts']['identified_purchase_lines']:,}")
    right.metric("Identified buyers at Sep 2011", f"{cohort.iloc[-1]['customers']:,}" if len(cohort) else "—")
    st.subheader("Observed 90-day outcomes")
    if not cohort.empty:
        display_cohort = cohort.copy()
        display_cohort["inactive_rate"] = display_cohort["inactive_rate"].map(lambda x: f"{x:.1%}")
        display_cohort["future_revenue_gbp"] = display_cohort["future_revenue_gbp"].map(lambda x: f"£{x:,.0f}")
        st.dataframe(display_cohort, use_container_width=True, hide_index=True)
    st.write("The label is no qualifying merchandise purchase in the end-exclusive 90-day window. It does not mean permanent churn.")
    st.write("Development rows select model families; June probabilities calibrate; September 2011 is the frozen out-of-time test.")
    st.caption(f"Workbook SHA-256: `{manifest.get('source_sha256', 'unknown')}`")

with tab_risk:
    champion = champions.get("inactivity_90d", {})
    st.subheader("90-day inactivity model comparison")
    if champion:
        st.metric("Project lead model", champion.get("selected_model", "not selected"), help=champion.get("selection_rationale", ""))
        metric_winner = champion.get("development_metric_winner")
        if metric_winner:
            st.caption(f"Development metric winner: {metric_winner}. LSTM is the project lead for explicit temporal modeling.")
    if not inactivity.empty:
        view = inactivity.loc[inactivity["split"].isin(["development", "development_ensemble", "final_test_platt"])].copy()
        columns = [column for column in ["model", "split", "n_customers", "precision_at_capacity", "recall_at_capacity", "average_precision", "roc_auc", "brier_score", "log_loss"] if column in view]
        st.dataframe(view[columns].sort_values(["split", "precision_at_capacity"], ascending=[True, False]), use_container_width=True, hide_index=True)
        plot = view.pivot(index="model", columns="split", values="average_precision").dropna(how="all")
        if not plot.empty:
            st.bar_chart(plot)
    st.subheader("September 2011 test cohort score view")
    score_path = PROCESSED / "final_customer_scores_90d.parquet"
    if score_path.exists() and champion.get("selected_model"):
        frame = pd.read_parquet(score_path)
        risk_column = f"{champion['selected_model']}_risk"
        if risk_column in frame:
            # Show model outputs only. Actual held-out outcomes remain in report metrics.
            shown = frame[["customer_id", "cutoff", risk_column]].rename(columns={risk_column: "inactive_90d_probability"})
            shown = shown.sort_values("inactive_90d_probability", ascending=False).head(500)
            st.caption("Retrospective frozen-test scores for model comparison, not a live campaign list.")
            st.dataframe(shown.head(50), use_container_width=True, hide_index=True)
            st.download_button("Download top 500 test predictions", shown.to_csv(index=False), "september_test_risk_predictions.csv", "text/csv")
        else:
            st.info("Selected run has no score file yet.")
    else:
        st.info("Run the inactivity model stage to create this comparison.")

with tab_value:
    st.subheader("Expanded customer revenue experiments")
    experiment_horizon = st.selectbox(
        "Forecast horizon", ["90d", "6m"],
        format_func=lambda h: "90 days" if h == "90d" else "Six calendar months",
        key="revenue_experiment_horizon",
    )
    robustness_summary = load_csv(REPORTS / f"revenue_challengers_{experiment_horizon}_summary.csv")
    experiment_summary = load_csv(REPORTS / f"revenue_challengers_{experiment_horizon}_selection_summary.csv")
    if experiment_summary.empty:
        experiment_summary = robustness_summary
    experiment_selection = load_json(
        ROOT / "artifacts" / "runs" / "revenue_challengers" / experiment_horizon / "selection.json"
    )
    if not experiment_summary.empty:
        if experiment_selection:
            st.metric("Selected expected-revenue candidate", experiment_selection["selected_model"])
            st.caption("Selection uses mean Tweedie deviance (power 1.5) on outcomes complete by the final forecast origin. Lower is better.")
            st.caption("Selection origins: " + ", ".join(experiment_selection["development_origins"]))
        else:
            st.caption("Experiment in progress. A partial run does not select a model.")
        display = experiment_summary.rename(columns={
            "mean_mae_gbp": "Mean MAE (GBP)", "mean_rmse_gbp": "Mean RMSE (GBP)",
            "mean_absolute_bias_pct": "Mean absolute total bias", "mean_tweedie_deviance": "Mean Tweedie deviance",
            "mean_revenue_capture_top10": "Top 10% revenue capture", "n_origins": "Development origins",
        })
        st.dataframe(display.style.format({
            "Mean MAE (GBP)": "{:.2f}", "Mean RMSE (GBP)": "{:.2f}",
            "Mean absolute total bias": "{:.1%}", "Mean Tweedie deviance": "{:.2f}",
            "Top 10% revenue capture": "{:.1%}",
        }), width="stretch", hide_index=True)
        eligible = experiment_summary.loc[experiment_summary.eligible].set_index("model")
        # Zero recent-spend forecasts incur extremely large deviance on future buyers.
        # Keep those numbers in the table; use a log axis to keep the chart readable.
        if not eligible.empty:
            st.vega_lite_chart(eligible.reset_index(), {
                "mark": "bar",
                "encoding": {
                    "y": {"field": "model", "type": "nominal", "sort": "-x", "title": "Model"},
                    "x": {"field": "mean_tweedie_deviance", "type": "quantitative", "scale": {"type": "log"}, "title": "Mean Tweedie deviance (log scale, lower is better)"},
                    "tooltip": [{"field": "model"}, {"field": "mean_tweedie_deviance", "format": ".2f"}, {"field": "mean_mae_gbp", "format": ".2f"}],
                },
            }, width="stretch")
        st.caption("The same customers can appear at multiple origins; these are dependent temporal checks.")
        if experiment_selection.get("retrospective_robustness_origins") and not robustness_summary.empty:
            with st.expander("Additional retrospective six-month rolling checks"):
                st.caption("December-May outcome windows overlap the final June cohort. Only December's outcomes were available by June; the combined averages below cannot select an as-of June model.")
                st.dataframe(robustness_summary, width="stretch", hide_index=True)
        retrospective = load_csv(REPORTS / f"revenue_challengers_{experiment_horizon}_retrospective.csv")
        if not retrospective.empty:
            with st.expander("Retrospective comparison and predictive intervals"):
                st.caption("These final cohorts were inspected in earlier work. They are retrospective checks, not fresh untouched tests, and did not select the new candidate.")
                columns = [c for c in ["model", "cutoff", "status", "mae_gbp", "rmse_gbp", "aggregate_bias_pct", "tweedie_deviance_p1_5", "interval_90_coverage", "interval_90_mean_width_gbp", "error"] if c in retrospective]
                st.dataframe(retrospective[columns], width="stretch", hide_index=True)
                st.caption("The LSTM and NGBoost intervals target 90% coverage. Actual coverage is measured; the intervals have not been conformally calibrated.")
        forecast_path = PROCESSED / f"revenue_challenger_scores_{experiment_horizon}.parquet"
        if forecast_path.exists() and experiment_selection:
            with st.expander("Selected candidate: historical customer forecasts"):
                forecasts = pd.read_parquet(forecast_path)
                columns = [c for c in ["customer_id", "cutoff", "selected_model", "predicted_revenue_gbp", "lower_90_gbp", "upper_90_gbp"] if c in forecasts]
                forecasts = forecasts[columns].sort_values("predicted_revenue_gbp", ascending=False)
                st.caption("Expected gross merchandise revenue in GBP for the historical cohort.")
                st.dataframe(forecasts.head(100), width="stretch", hide_index=True)
                st.download_button("Download customer revenue forecasts", forecasts.to_csv(index=False), f"customer_revenue_{experiment_horizon}.csv", "text/csv")
    else:
        st.info("Run `python scripts/revenue_challengers.py` to compare GAMs, Pareto/NBD, NGBoost and the probabilistic LSTM.")

    st.subheader("Original model portfolio")
    for title, frame, champion_key in (
        ("90-day revenue", revenue90, "future_revenue_90d"),
        ("Six-calendar-month revenue", revenue6m, "future_revenue_6m"),
    ):
        st.subheader(title)
        current = champions.get(champion_key, {})
        if current:
            st.metric("Development champion", current.get("selected_model", "not selected"))
        if frame.empty:
            st.info("Run the revenue model stage to create this comparison.")
        else:
            shown = frame.loc[frame["split"].isin(["development", "final_test"])].copy()
            columns = [column for column in ["model", "split", "n_customers", "mae_gbp", "rmse_gbp", "aggregate_bias_pct", "revenue_capture_at_capacity"] if column in shown]
            st.dataframe(shown[columns].sort_values(["split", "mae_gbp"]), use_container_width=True, hide_index=True)
    clv_path = REPORTS / "behavioral_clv_comparison.csv"
    if clv_path.exists():
        st.subheader("BG/NBD × Gamma-Gamma forecast-origin fits")
        clv = load_csv(clv_path)
        cols = [col for col in ["model", "split", "cutoff", "mae_gbp", "aggregate_bias_pct", "bg_nbd_a", "gamma_gamma_q", "gamma_gamma_finite_mean", "status", "error"] if col in clv]
        st.dataframe(clv[cols], use_container_width=True, hide_index=True)

with tab_segments:
    st.subheader("Segment profiles")
    if not segments.empty:
        st.dataframe(segments, use_container_width=True, hide_index=True)
    if not segment_profile_index.empty:
        profile_index = segment_profile_index.set_index("segment").T
        profile_index.index = [
            {
                "recency_days": "Recency (days)",
                "purchase_day_count": "Purchase days",
                "purchase_count": "Repeat purchase occasions",
                "lifetime_revenue_gbp": "Historical revenue (£)",
                "avg_order_value_gbp": "Average order value (£)",
                "avg_basket_units": "Average basket units",
                "product_diversity": "Product diversity",
                "spend_365d_gbp": "Spend in prior 365 days (£)",
            }.get(str(feature), str(feature))
            for feature in profile_index.index
        ]
        st.caption("Segment median divided by the June 2011 population median. Values above 1 are higher than the cohort median.")
        st.dataframe(
            profile_index.style.format("{:.2f}×").background_gradient(cmap="RdYlGn", axis=None, vmin=0.25, vmax=2.5),
            use_container_width=True,
        )

    projection_path = PROCESSED / "segment_projection.parquet"
    if projection_path.exists():
        projection = pd.read_parquet(projection_path)
        projection_roles = projection["snapshot_role"].drop_duplicates().tolist()
        selected_role = st.selectbox("Customer snapshot", projection_roles, key="segment_projection_role")
        coloring = st.selectbox("Color points by", ["K-Means segment", "GMM segment"], key="segment_projection_method")
        label_column = "kmeans_segment" if coloring == "K-Means segment" else "gmm_segment"
        plotted = projection.loc[projection["snapshot_role"].eq(selected_role), ["pc1", "pc2", label_column]].copy()
        plotted["segment_label"] = "Segment " + plotted[label_column].astype(str)
        info = load_json(ROOT / "artifacts" / "runs" / "segments" / "kmeans_champion.json")
        variance = info.get("projection_variance_ratio") or []
        axis_labels = ["PC 1", "PC 2"]
        if len(variance) >= 2:
            axis_labels = [f"PC 1 ({variance[0]:.1%} variance)", f"PC 2 ({variance[1]:.1%} variance)"]
        st.subheader("Customer map in the fitted PCA space")
        st.scatter_chart(plotted, x="pc1", y="pc2", color="segment_label", x_label=axis_labels[0], y_label=axis_labels[1])
        st.caption("Each point is one customer. This two-dimensional projection helps inspect the clusters; it does not prove that customer groups are naturally separated.")

    if not segment_outcomes.empty:
        st.subheader("September outcomes by assigned segment")
        st.caption("Out-of-time descriptive check only: June history formed the segments; these September outcomes were not used to fit or select them.")
        kmeans_outcomes = segment_outcomes.loc[segment_outcomes["method"].eq("kmeans")].copy()
        if not kmeans_outcomes.empty:
            kmeans_outcomes["segment_label"] = "Segment " + kmeans_outcomes["segment"].astype(str)
            left, right = st.columns(2)
            left.bar_chart(kmeans_outcomes, x="segment_label", y="inactivity_rate", y_label="90-day inactivity rate")
            right.bar_chart(kmeans_outcomes, x="segment_label", y="mean_future_revenue_gbp", y_label="Mean 90-day revenue (£)")
            show = kmeans_outcomes[["segment_label", "customers", "inactivity_rate", "mean_future_revenue_gbp", "median_future_revenue_gbp", "total_future_revenue_gbp"]].copy()
            show = show.rename(columns={
                "segment_label": "segment", "customers": "customers",
                "inactivity_rate": "inactivity_rate", "mean_future_revenue_gbp": "mean_future_revenue_gbp",
                "median_future_revenue_gbp": "median_future_revenue_gbp", "total_future_revenue_gbp": "total_future_revenue_gbp",
            })
            st.dataframe(show, use_container_width=True, hide_index=True)
        if not gmm_candidates.empty:
            with st.expander("Gaussian Mixture component selection"):
                st.dataframe(gmm_candidates, use_container_width=True, hide_index=True)

    score_path = PROCESSED / "customer_segments.parquet"
    if score_path.exists():
        cluster_scores = pd.read_parquet(score_path)
        st.subheader("September customer membership")
        st.dataframe(cluster_scores.head(100), use_container_width=True, hide_index=True)
        st.download_button("Download segment assignments", cluster_scores.to_csv(index=False), "customer_segments.csv", "text/csv")
    if not segments.empty:
        st.caption("Segments describe customer history and join with inactivity risk and expected revenue in the Retention review tab. Segment labels are not inputs to the predictive models.")

with tab_data:
    st.subheader("Reconciliation and purchase policy")
    st.json(manifest.get("row_counts", {}))
    st.write(manifest.get("policy", {}))
    st.write("Known-customer purchases require positive quantity and price, a non-cancel invoice, and a five-digit product stock code. The code screen is conservative and excludes service/accounting-like codes for review.")
    st.write("Credits are kept in a separate ledger. They are not subtracted from merchandise sales because this workbook does not reliably link each credit to an original purchase.")
