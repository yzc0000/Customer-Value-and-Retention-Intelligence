"""Replay retained revenue models and quantify paired development differences.

Bootstrap intervals condition on the observed dates and do not adjust for model
selection. Repeated appearances of a customer are resampled together.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if (ROOT / ".local-deps").exists():
    sys.path.insert(0, str(ROOT / ".local-deps"))

from customer_intelligence.config import SEED
from customer_intelligence.features import NUMERIC_FEATURES
from customer_intelligence.revenue_challengers import DEVELOPMENT, MODELS, TEST, weeks_for


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def point_deviance(y, prediction):
    p = np.maximum(prediction, 1e-6)
    return 4 * (y / np.sqrt(p) + np.sqrt(p) - 2 * np.sqrt(y))


def bootstrap_difference(parts, draws=2000):
    frame = pd.concat(parts)
    pivot = frame.pivot(index="customer_id", columns="cutoff", values="difference")
    values = pivot.fillna(0).to_numpy()
    present = pivot.notna().to_numpy(dtype=float)
    rng = np.random.default_rng(SEED)
    sampled = []
    for start in range(0, draws, 100):
        weights = rng.multinomial(len(pivot), np.full(len(pivot), 1 / len(pivot)), size=min(100, draws - start))
        sampled.extend(((weights @ values) / np.maximum(weights @ present, 1)).mean(axis=1))
    return float(pivot.mean().mean()), *map(float, np.quantile(sampled, [0.025, 0.975]))


def main(uncertainty_only=False):
    base = ROOT / "artifacts" / "runs" / "revenue_challengers"
    protocol = read_json(base / "protocol.json")
    original_features = pd.read_parquet(ROOT / "data" / "processed" / "customer_features.parquet")
    sequences = np.load(ROOT / "data" / "processed" / "revenue_challengers" / "weeks.npz")
    replays, numerical, comparisons, intervals = [], [], [], []
    for horizon, dates in DEVELOPMENT.items():
        panel = pd.read_parquet(ROOT / "data" / "processed" / "revenue_challengers" / f"panel_{horizon}.parquet")
        if horizon == "90d":
            features = ["customer_id", "cutoff"] + list(NUMERIC_FEATURES) + ["country"]
            actual = panel.loc[panel.cutoff.isin(original_features.cutoff.unique()), features].sort_values(["cutoff", "customer_id"]).reset_index(drop=True)
            expected = original_features[features].sort_values(["cutoff", "customer_id"]).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
        selection = read_json(base / horizon / "selection.json")
        selected = selection["selected_model"]
        selection_dates = selection["development_origins"]
        for origin in ([] if uncertainty_only else dates + [TEST[horizon]]):
            valid = panel.loc[panel.cutoff.eq(pd.Timestamp(origin))].copy()
            train = panel.loc[(panel.cutoff < pd.Timestamp(origin)) & panel.label_mature & (panel.horizon_end_exclusive <= pd.Timestamp(origin))]
            assert train.horizon_end_exclusive.max() <= pd.Timestamp(origin)
            assert valid.label_mature.all() and valid.customer_id.is_unique
            days = (valid.horizon_end_exclusive.iloc[0] - valid.cutoff.iloc[0]).days
            for name in MODELS:
                directory = base / horizon / name / origin
                receipt = read_json(directory / "receipt.json")
                assert receipt["status"] == "completed", (horizon, name, origin, receipt)
                assert receipt["key"]["signature"] == protocol["signature"]
                saved = pd.read_parquet(directory / "predictions.parquet")
                np.testing.assert_array_equal(saved.customer_id, valid.customer_id)
                np.testing.assert_allclose(saved.actual_revenue_gbp, valid.future_revenue_gbp, rtol=0, atol=0)
                model = joblib.load(directory / "model.joblib")
                with threadpool_limits(limits=2):
                    if name == "recent_spend":
                        prediction = valid[f"spend_{model['window_days']}d_gbp"].to_numpy() * model["horizon_days"] / model["window_days"]
                    elif name == "lstm_hurdle_lognormal":
                        weekly = weeks_for(valid, sequences)
                        prediction = model.predict(valid, weekly)
                        bounds = model.interval(valid, weekly)
                    elif name.startswith("pareto_nbd") or name == "bg_nbd_gamma_gamma":
                        prediction = model.predict(valid, days)
                    else:
                        prediction = model.predict(valid)
                        if name == "hurdle_ngboost_lognormal":
                            bounds = model.interval(valid)
                np.testing.assert_allclose(prediction, saved.predicted_revenue_gbp, rtol=1e-7, atol=1e-5)
                if name in ["hurdle_ngboost_lognormal", "lstm_hurdle_lognormal"]:
                    for column, value in zip(["lower_90_gbp", "upper_90_gbp"], bounds):
                        np.testing.assert_allclose(saved[column], value, rtol=1e-7, atol=1e-5)
                    for group, mask in [("all", np.ones(len(saved), dtype=bool)), ("positive", saved.actual_revenue_gbp.gt(0).to_numpy()), ("zero", saved.actual_revenue_gbp.eq(0).to_numpy())]:
                        y, lo, hi = saved.loc[mask, "actual_revenue_gbp"], saved.loc[mask, "lower_90_gbp"], saved.loc[mask, "upper_90_gbp"]
                        intervals.append(dict(horizon=horizon, model=name, origin=origin, group=group, customers=int(mask.sum()), coverage=float(((y >= lo) & (y <= hi)).mean()), mean_width_gbp=float((hi - lo).mean())))
                if name.startswith("pareto_nbd"):
                    pnbd = model.purchase_
                    summary = model.summary_.reindex(valid.customer_id)
                    original_nodes = pnbd.quadrature_nodes
                    count24 = pnbd.expected_purchases(summary, days)
                    likelihood24 = pnbd.likelihood_parts(pnbd.parameters_, pnbd._arrays(summary))[0]
                    pnbd.quadrature_nodes = 64
                    count64 = pnbd.expected_purchases(summary, days)
                    likelihood64 = pnbd.likelihood_parts(pnbd.parameters_, pnbd._arrays(summary))[0]
                    pnbd.quadrature_nodes = original_nodes
                    relative = float(np.max(np.abs(count24 - count64) / np.maximum(count64, 1e-10)))
                    log_difference = float(np.max(np.abs(likelihood24 - likelihood64)))
                    assert relative < 1e-4 and log_difference < 1e-4, (horizon, name, origin, relative, log_difference)
                    numerical.append(dict(horizon=horizon, model=name, origin=origin, max_relative_purchase_difference_24_vs_64=relative, max_loglikelihood_difference_24_vs_64=log_difference))
                effective = train.loc[train.cutoff.dt.month.isin([3, 6, 9, 12])] if name == "xgboost_tweedie" else train
                history_model = name.startswith("pareto_nbd") or name == "bg_nbd_gamma_gamma"
                replays.append(dict(horizon=horizon, model=name, origin=origin, customers=len(saved), max_absolute_replay_difference_gbp=float(np.max(np.abs(prediction - saved.predicted_revenue_gbp))), supervised_training_rows=0 if history_model or name == "recent_spend" else len(effective), fit_source="as-of customer transaction histories" if history_model else "baseline formula" if name == "recent_spend" else "mature supervised snapshots"))
                known_development = origin in ["2011-03-01", "2011-06-01"] if horizon == "90d" else origin == "2010-12-01"
                if name == "xgboost_tweedie" and (known_development or origin == TEST[horizon]):
                    old_dir = ROOT / "artifacts" / "runs" / ("90d" if horizon == "90d" else "six_calendar_months") / "revenue" / name
                    old_name = "final_test_predictions.parquet" if origin == TEST[horizon] else f"development_{origin}_predictions.parquet" if horizon == "90d" else "development_predictions.parquet"
                    old = pd.read_parquet(old_dir / old_name)
                    assert old.customer_id.tolist() == saved.customer_id.tolist()
                    forecast_column = "prediction" if "prediction" in old else "predicted_revenue_gbp"
                    np.testing.assert_allclose(old[forecast_column], saved.predicted_revenue_gbp, rtol=1e-7, atol=1e-5)
            print(f"Replayed all models: {horizon} / {origin}", flush=True)
        reference = "xgboost_tweedie" if horizon == "90d" else "recent_spend"
        for metric in ["mae_gbp", "tweedie_deviance_p1_5"]:
            parts = []
            for origin in selection_dates:
                ref = pd.read_parquet(base / horizon / reference / origin / "predictions.parquet")
                cand = pd.read_parquet(base / horizon / selected / origin / "predictions.parquet")
                assert ref.customer_id.tolist() == cand.customer_id.tolist()
                y = ref.actual_revenue_gbp.to_numpy()
                fn = (lambda y, p: np.abs(y - p)) if metric == "mae_gbp" else point_deviance
                difference = fn(y, ref.predicted_revenue_gbp.to_numpy()) - fn(y, cand.predicted_revenue_gbp.to_numpy())
                parts.append(pd.DataFrame(dict(customer_id=ref.customer_id, cutoff=origin, difference=difference)))
            mean, lower, upper = bootstrap_difference(parts)
            comparisons.append(dict(horizon=horizon, selected_model=selected, reference_model=reference, metric=metric, mean_improvement=mean, conditional_customer_bootstrap_lower_95=lower, conditional_customer_bootstrap_upper_95=upper, positive_origin_improvements=sum(p.difference.mean() > 0 for p in parts), origins=len(parts), note="Conditional on observed origins; repeated customers resampled together; no correction for model selection or future seasonal uncertainty."))
    if not uncertainty_only:
        pd.DataFrame(replays).to_csv(ROOT / "reports" / "revenue_challengers_replay_audit.csv", index=False)
        pd.DataFrame(numerical).to_csv(ROOT / "reports" / "revenue_challengers_quadrature_audit.csv", index=False)
        pd.DataFrame(intervals).to_csv(ROOT / "reports" / "revenue_challengers_interval_diagnostics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(ROOT / "reports" / "revenue_challengers_uncertainty.csv", index=False)
    if uncertainty_only:
        print("Updated paired uncertainty for the temporally eligible selections; prediction replay reports were preserved.", flush=True)
    else:
        print(f"Verified {len(replays)} retained fits and {len(numerical)} Pareto/NBD numerical checks.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uncertainty-only", action="store_true", help="Update paired differences after reselection, without repeating prediction replays.")
    main(parser.parse_args().uncertainty_only)
