"""Expanded temporal revenue benchmarks, preserving the original portfolio.

Recipes and mean-forecast selection are fixed before retrospective test scoring.
Quarterly 90-day origins span all four seasons. Six-month targets use six
overlapping monthly origins; these are dependent robustness checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import mean_tweedie_deviance
from threadpoolctl import threadpool_limits

from .bgnbd import BetaGeoNBD, GammaGamma, MONTH_DAYS, customer_day_summary
from .config import SEED
from .evaluation import revenue_metrics
from .features import NUMERIC_FEATURES, build_cutoff_features
from .modeling import revenue_candidates
from .pareto_nbd import ParetoRevenue
from .revenue_challenger_models import HurdleNGBoost, HurdleRevenueLSTM, RevenueGAM, VALUE_FEATURES
from .revenue_research import CALENDAR, EXTRA, TwoPart, enriched_features, candidate
from .targets import build_targets


ROOT = Path(__file__).resolve().parents[2]
MODELS = ["recent_spend", "xgboost_tweedie", "xgboost_enriched_monthly", "hurdle_gamma_hist",
          "tweedie_gam", "hurdle_gamma_gam", "hurdle_ngboost_lognormal", "lstm_hurdle_lognormal",
          "bg_nbd_gamma_gamma", "pareto_nbd_gamma_gamma", "pareto_nbd_seasonal_gamma_gamma"]
DEVELOPMENT = {
    "90d": ["2010-09-01", "2010-12-01", "2011-03-01", "2011-06-01"],
    "6m": ["2010-12-01", "2011-01-01", "2011-02-01", "2011-03-01", "2011-04-01", "2011-05-01"],
}
TEST = {"90d": "2011-09-01", "6m": "2011-06-01"}


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def build_panels(root):
    cache = root / "data" / "processed" / "revenue_challengers"
    cache.mkdir(parents=True, exist_ok=True)
    prepare = json.loads((root / "data" / "processed" / "prepare_manifest.json").read_text(encoding="utf-8"))
    cache_signature = dict(source_sha256=prepare["source_sha256"], version=1, start="2010-06-01", end="2011-09-01")
    receipt = cache / "manifest.json"
    if receipt.exists() and json.loads(receipt.read_text(encoding="utf-8")) == cache_signature:
        return {h: pd.read_parquet(cache / f"panel_{h}.parquet") for h in DEVELOPMENT}, np.load(cache / "weeks.npz")
    sales = pd.read_parquet(root / "data" / "interim" / "purchase_lines.parquet")
    credits = pd.read_parquet(root / "data" / "interim" / "credit_lines.parquet")
    dates = pd.date_range("2010-06-01", "2011-09-01", freq="MS")
    parts, sequences = [], {}
    for cutoff in dates:
        print(f"Building challenger snapshot {cutoff.date()}", flush=True)
        f, w, _ = build_cutoff_features(sales, credits, cutoff)
        f["sequence_row"] = np.arange(len(f))
        parts.append(f)
        sequences[str(cutoff.date())] = w
    base = enriched_features(pd.concat(parts, ignore_index=True), sales)
    target90, target6m = build_targets(sales, base, [str(d.date()) for d in dates], pd.Timestamp(prepare["complete_end_exclusive"]))
    panels = {}
    source_start = sales.invoice_date.min().normalize()
    for horizon, targets in [("90d", target90), ("6m", target6m)]:
        panel = base.merge(targets, on=["customer_id", "cutoff"], validate="one_to_one")
        if horizon == "6m":
            for cutoff, indices in panel.groupby("cutoff").groups.items():
                end = cutoff + pd.DateOffset(months=6)
                dates_future = pd.date_range(cutoff, end - pd.Timedelta(days=1), freq="D")
                panel.loc[indices, "forecast_q4_share"] = np.mean(dates_future.month >= 10)
                panel.loc[indices, "horizon_days"] = (end - cutoff).days
                prior_start, prior_end = cutoff - pd.DateOffset(years=1), end - pd.DateOffset(years=1)
                assert prior_end <= cutoff
                if prior_start >= source_start:
                    prior = sales.loc[(sales.invoice_date >= prior_start) & (sales.invoice_date < prior_end)]
                    spend = prior.groupby("customer_id").revenue.sum()
                    count = prior.groupby("customer_id").invoice_id.nunique()
                    panel.loc[indices, EXTRA[-2]] = panel.loc[indices, "customer_id"].map(spend).fillna(0)
                    panel.loc[indices, EXTRA[-1]] = panel.loc[indices, "customer_id"].map(count).fillna(0)
                else:
                    panel.loc[indices, EXTRA[-2:]] = np.nan
        panel.to_parquet(cache / f"panel_{horizon}.parquet", index=False)
        panels[horizon] = panel
    np.savez_compressed(cache / "weeks.npz", **sequences)
    save_json(receipt, cache_signature)
    return panels, np.load(cache / "weeks.npz")


def weeks_for(rows, sequences):
    result = np.empty((len(rows), 26, 6), dtype=np.float32)
    for cutoff, positions in rows.groupby("cutoff", sort=False).indices.items():
        indices = rows.iloc[positions].sequence_row.to_numpy(dtype=int)
        result[positions] = sequences[str(cutoff.date())][indices]
    return result


def summarize_metrics(frame, expected_origins):
    complete = frame.loc[frame.status == "completed"]
    summary = complete.groupby("model", as_index=False).agg(
        mean_mae_gbp=("mae_gbp", "mean"), mean_rmse_gbp=("rmse_gbp", "mean"),
        mean_absolute_bias_pct=("aggregate_bias_pct", lambda v: v.abs().mean()),
        mean_tweedie_deviance=("tweedie_deviance_p1_5", "mean"),
        mean_revenue_capture_top10=("revenue_capture_at_capacity", "mean"), n_origins=("cutoff", "nunique"),
    ).sort_values(["mean_tweedie_deviance", "mean_absolute_bias_pct", "mean_rmse_gbp"])
    summary["eligible"] = summary.n_origins == expected_origins
    return summary


def select_from_available_outcomes(root, horizon, frame, panel, signature):
    """Outcome windows used for selection must finish before the final origin."""
    ends = panel[["cutoff", "horizon_end_exclusive"]].drop_duplicates().set_index("cutoff").horizon_end_exclusive
    available = [d for d in DEVELOPMENT[horizon] if ends.loc[pd.Timestamp(d)] <= pd.Timestamp(TEST[horizon])]
    selected_rows = frame.loc[frame.cutoff.isin(available)].copy()
    summary = summarize_metrics(selected_rows, len(available))
    summary.to_csv(root / "reports" / f"revenue_challengers_{horizon}_selection_summary.csv", index=False)
    eligible = summary.loc[summary.eligible]
    if eligible.empty:
        raise RuntimeError(f"No complete {horizon} candidate with outcomes available at the final origin.")
    selected = str(eligible.iloc[0].model)
    save_json(root / "artifacts" / "runs" / "revenue_challengers" / horizon / "selection.json",
        dict(selected_model=selected,
             selection_rule="Lowest mean fixed-power Tweedie deviance (p=1.5), then absolute bias, then RMSE; complete origins and outcome windows ending no later than the final forecast origin.",
             development_origins=available, retrospective_robustness_origins=[d for d in DEVELOPMENT[horizon] if d not in available],
             development_summary=eligible.iloc[0].to_dict(), signature=signature,
             note="Final cohorts were already inspected. Later overlapping six-month comparison windows cannot select an as-of final-origin model. Original champion pointers are retained."))
    return selected


class BGRevenue:
    def fit(self, sales, origin):
        self.summary = customer_day_summary(sales, origin)
        s = self.summary
        self.purchase = BetaGeoNBD().fit(s.frequency, s.recency_months, s.age_months)
        repeat = (s.frequency > 0) & s.repeat_mean_daily_spend.notna()
        self.spend = GammaGamma().fit(s.loc[repeat, "frequency"], s.loc[repeat, "repeat_mean_daily_spend"])
        if not self.spend.finite_population_mean_:
            raise RuntimeError("Gamma-Gamma population mean not finite.")
        self.fit_info_ = dict(bg_a=self.purchase.a_, gamma_q=self.spend.q_, optimizer_success=self.purchase.optimizer_success_)
        return self

    def predict(self, x, horizon_days):
        s = self.summary.reindex(x.customer_id)
        count = self.purchase.expected_purchases(horizon_days / MONTH_DAYS, s.frequency, s.recency_months, s.age_months)
        amount = self.spend.expected_average_spend(s.frequency, s.repeat_mean_daily_spend.fillna(0))
        return count * amount


def fit_and_score(name, train, valid, sales, sequences):
    y = train.future_revenue_gbp.to_numpy(dtype=float)
    horizon_days = int((valid.horizon_end_exclusive.iloc[0] - valid.cutoff.iloc[0]).days)
    if name == "recent_spend":
        window = 90 if valid.horizon.iloc[0] == "90d" else 180
        model = dict(kind="recent_spend", window_days=window, horizon_days=horizon_days)
        return model, valid[f"spend_{window}d_gbp"].to_numpy() * horizon_days / window, None
    if name.startswith("pareto_nbd"):
        model = ParetoRevenue(seasonal="seasonal" in name).fit(sales, valid.cutoff.iloc[0])
        return model, model.predict(valid, horizon_days), None
    if name == "bg_nbd_gamma_gamma":
        model = BGRevenue().fit(sales, valid.cutoff.iloc[0])
        return model, model.predict(valid, horizon_days), None
    if name == "xgboost_tweedie":
        # Match the original quarterly supervised training portfolio.
        train = train.loc[train.cutoff.dt.month.isin([3, 6, 9, 12])].copy()
        y = train.future_revenue_gbp.to_numpy(dtype=float)
        model = revenue_candidates()["xgboost_tweedie"]
    elif name == "xgboost_enriched_monthly":
        model = candidate("xgboost_reference", VALUE_FEATURES)
    elif name == "hurdle_gamma_hist":
        model = TwoPart("hurdle_hist", VALUE_FEATURES)
    elif name == "tweedie_gam":
        model = RevenueGAM()
    elif name == "hurdle_gamma_gam":
        model = RevenueGAM(hurdle=True)
    elif name == "hurdle_ngboost_lognormal":
        model = HurdleNGBoost()
    elif name == "lstm_hurdle_lognormal":
        model = HurdleRevenueLSTM()
        model.fit(train, y, weeks_for(train, sequences))
        w = weeks_for(valid, sequences)
        return model, model.predict(valid, w), model.interval(valid, w)
    else:
        raise ValueError(name)
    model.fit(train, y)
    interval = model.interval(valid) if name == "hurdle_ngboost_lognormal" else None
    return model, model.predict(valid), interval


def evaluate_origin(root, panel, sequences, sales, horizon, origin_text, names, split, signature):
    origin = pd.Timestamp(origin_text)
    train = panel.loc[(panel.cutoff < origin) & panel.label_mature & (panel.horizon_end_exclusive <= origin)].copy()
    valid = panel.loc[(panel.cutoff == origin) & panel.label_mature].copy()
    assert len(train) and len(valid) and train.horizon_end_exclusive.max() <= origin
    assert valid.customer_id.is_unique
    records = []
    print(f"{horizon} / {split} / {origin_text}: {len(train):,} mature training snapshots, {len(valid):,} scored customers", flush=True)
    for name in names:
        path = root / "artifacts" / "runs" / "revenue_challengers" / horizon / name / origin_text
        receipt = path / "receipt.json"
        key = dict(signature=signature, model=name, horizon=horizon, origin=origin_text)
        if receipt.exists():
            old = json.loads(receipt.read_text(encoding="utf-8"))
            if old.get("key") == key and old.get("status") == "completed" and (path / "predictions.parquet").exists() and (path / "model.joblib").exists():
                saved = old["metrics"]
                records.append(dict(model=name, cutoff=origin_text, horizon=horizon, split=split, status="completed", **saved))
                print(f"  {name}: cached MAE={saved['mae_gbp']:.2f}, deviance={saved['tweedie_deviance_p1_5']:.2f}", flush=True)
                continue
        start = time.monotonic()
        try:
            with threadpool_limits(limits=2):
                model, prediction, interval = fit_and_score(name, train, valid, sales, sequences)
            prediction = np.maximum(np.asarray(prediction, dtype=float), 0)
            if len(prediction) != len(valid) or not np.isfinite(prediction).all():
                raise RuntimeError("Forecast contains missing/non-finite customer predictions.")
            actual = valid.future_revenue_gbp.to_numpy(dtype=float)
            metrics = revenue_metrics(actual, prediction)
            metrics["tweedie_deviance_p1_5"] = float(mean_tweedie_deviance(actual, np.maximum(prediction, 1e-6), power=1.5))
            metrics["training_seconds"] = time.monotonic() - start
            output = valid[["customer_id", "cutoff"]].copy()
            output["actual_revenue_gbp"], output["predicted_revenue_gbp"] = actual, prediction
            if interval is not None:
                lower, upper = map(lambda a: np.asarray(a, dtype=float), interval)
                if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
                    raise RuntimeError("Invalid predictive interval.")
                output["lower_90_gbp"], output["upper_90_gbp"] = lower, upper
                metrics["interval_90_coverage"] = float(np.mean((actual >= lower) & (actual <= upper)))
                metrics["interval_90_mean_width_gbp"] = float(np.mean(upper - lower))
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, path / "model.joblib", compress=3)
            if name == "lstm_hurdle_lognormal":
                torch.save(model.net.state_dict(), path / "state_dict.pt")
                save_json(path / "training_loss.json", model.training_loss_)
            output.to_parquet(path / "predictions.parquet", index=False)
            info = getattr(model, "fit_info_", {})
            save_json(receipt, dict(key=key, status="completed", metrics=metrics, fit_info=info,
                train_rows=len(train), latest_training_label_end=str(train.horizon_end_exclusive.max().date()),
                train_origins=[str(d.date()) for d in sorted(train.cutoff.unique())],
                note="Statistical customer-process models fit histories strictly before the scored origin, without future labels."))
            records.append(dict(model=name, cutoff=origin_text, horizon=horizon, split=split, status="completed", **metrics))
            print(f"  {name}: MAE={metrics['mae_gbp']:.2f}, RMSE={metrics['rmse_gbp']:.2f}, bias={metrics['aggregate_bias_pct']:.1%}, deviance={metrics['tweedie_deviance_p1_5']:.2f} ({metrics['training_seconds']:.1f}s)", flush=True)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            save_json(receipt, dict(key=key, status="failed", error=error))
            records.append(dict(model=name, cutoff=origin_text, horizon=horizon, split=split, status="failed", error=error))
            print(f"  {name}: FAILED {error}", flush=True)
    return records


def run(root, horizons, models, origins=None, retrospective=True):
    local_deps = root / ".local-deps"
    if local_deps.exists():
        sys.path.insert(0, str(local_deps))
    panels, sequences = build_panels(root)
    sales = pd.read_parquet(root / "data" / "interim" / "purchase_lines.parquet")
    prepare = json.loads((root / "data" / "processed" / "prepare_manifest.json").read_text(encoding="utf-8"))
    code_files = [Path(__file__), Path(__file__).with_name("pareto_nbd.py"), Path(__file__).with_name("revenue_challenger_models.py")]
    code_hash = hashlib.sha256(b"".join(p.read_bytes() for p in code_files)).hexdigest()
    signature = hashlib.sha256((prepare["source_sha256"] + code_hash).encode()).hexdigest()
    protocol = dict(signature=signature, source_sha256=prepare["source_sha256"], code_hash=code_hash, seed=SEED,
        models=MODELS, comparison_origins=DEVELOPMENT,
        development_origins={"90d": DEVELOPMENT["90d"], "6m": ["2010-12-01"]}, retrospective_test_origins=TEST,
        selection_rule="Lowest mean fixed-power Tweedie deviance (p=1.5), then absolute bias, then RMSE; complete origins and outcome windows ending no later than the final forecast origin.",
        lstm_epochs=30, ngboost_iterations=250, new_models_training="All fully matured monthly snapshots; original XGBoost reference uses quarterly snapshots.",
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, torch=torch.__version__),
        limitations="Test cohorts were previously inspected. Final rescoring is retrospective; origins and repeated customers are dependent. January-May six-month windows overlap the final cohort and are retrospective robustness checks only.")
    save_json(root / "artifacts" / "runs" / "revenue_challengers" / "protocol.json", protocol)
    for horizon in horizons:
        records = []
        dates = origins or DEVELOPMENT[horizon]
        for origin in dates:
            outcome_end = panels[horizon].loc[panels[horizon].cutoff.eq(pd.Timestamp(origin)), "horizon_end_exclusive"].iloc[0]
            split = "development" if outcome_end <= pd.Timestamp(TEST[horizon]) else "retrospective_robustness"
            records.extend(evaluate_origin(root, panels[horizon], sequences, sales, horizon, origin, models, split, signature))
            pd.DataFrame(records).to_csv(root / "reports" / f"revenue_challengers_{horizon}_development.csv", index=False)
        frame = pd.DataFrame(records)
        complete = frame.loc[frame.status == "completed"]
        summary = complete.groupby("model", as_index=False).agg(
            mean_mae_gbp=("mae_gbp", "mean"), mean_rmse_gbp=("rmse_gbp", "mean"),
            mean_absolute_bias_pct=("aggregate_bias_pct", lambda v: v.abs().mean()),
            mean_tweedie_deviance=("tweedie_deviance_p1_5", "mean"),
            mean_revenue_capture_top10=("revenue_capture_at_capacity", "mean"), n_origins=("cutoff", "nunique"),
        ).sort_values(["mean_tweedie_deviance", "mean_absolute_bias_pct", "mean_rmse_gbp"])
        summary["eligible"] = summary.n_origins == len(dates)
        summary.to_csv(root / "reports" / f"revenue_challengers_{horizon}_summary.csv", index=False)
        if origins is not None:
            print("Smoke run completed; no model selected from partial origins.", flush=True)
            continue
        selected = select_from_available_outcomes(root, horizon, frame, panels[horizon], signature)
        print(f"{horizon} expected-revenue candidate selected on development: {selected}", flush=True)
        if retrospective:
            audit = evaluate_origin(root, panels[horizon], sequences, sales, horizon, TEST[horizon], models, "retrospective_test", signature)
            pd.DataFrame(audit).to_csv(root / "reports" / f"revenue_challengers_{horizon}_retrospective.csv", index=False)
            source = root / "artifacts" / "runs" / "revenue_challengers" / horizon / selected / TEST[horizon] / "predictions.parquet"
            if source.exists():
                score = pd.read_parquet(source)
                score["selected_model"] = selected
                score.to_parquet(root / "data" / "processed" / f"revenue_challenger_scores_{horizon}.parquet", index=False)
        print(summary.to_string(index=False), flush=True)


def summarize_saved(root, horizons):
    """Correct selection eligibility using retained fits; do not refit models."""
    base = root / "artifacts" / "runs" / "revenue_challengers"
    protocol = json.loads((base / "protocol.json").read_text(encoding="utf-8"))
    protocol["selection_outcome_availability_required"] = True
    protocol["comparison_origins"] = DEVELOPMENT
    protocol["development_origins"] = {"90d": DEVELOPMENT["90d"], "6m": ["2010-12-01"]}
    protocol["selection_rule"] = "Lowest mean fixed-power Tweedie deviance (p=1.5), then absolute bias, then RMSE; complete origins and outcome windows ending no later than the final forecast origin."
    protocol["selection_code_hash"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for horizon in horizons:
        panel = pd.read_parquet(root / "data" / "processed" / "revenue_challengers" / f"panel_{horizon}.parquet")
        frame = pd.read_csv(root / "reports" / f"revenue_challengers_{horizon}_development.csv")
        ends = panel[["cutoff", "horizon_end_exclusive"]].drop_duplicates().set_index("cutoff").horizon_end_exclusive
        frame["selection_eligible_origin"] = pd.to_datetime(frame.cutoff).map(ends).le(pd.Timestamp(TEST[horizon]))
        frame["split"] = np.where(frame.selection_eligible_origin, "development", "retrospective_robustness")
        frame.to_csv(root / "reports" / f"revenue_challengers_{horizon}_development.csv", index=False)
        selected = select_from_available_outcomes(root, horizon, frame, panel, protocol["signature"])
        source = base / horizon / selected / TEST[horizon] / "predictions.parquet"
        score = pd.read_parquet(source)
        score["selected_model"] = selected
        score.to_parquet(root / "data" / "processed" / f"revenue_challenger_scores_{horizon}.parquet", index=False)
        print(f"{horizon}: selected {selected} using outcome windows complete before {TEST[horizon]}", flush=True)
    save_json(base / "protocol.json", protocol)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizons", nargs="+", choices=list(DEVELOPMENT), default=list(DEVELOPMENT))
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--origins", nargs="+", help="Optional smoke-run dates; disables model selection.")
    parser.add_argument("--no-retrospective", action="store_true")
    parser.add_argument("--summarize", action="store_true", help="Recompute temporally eligible selections from retained results without refitting.")
    args = parser.parse_args()
    if args.summarize:
        summarize_saved(ROOT, args.horizons)
    else:
        run(ROOT, args.horizons, args.models, args.origins, not args.no_retrospective)


if __name__ == "__main__":
    main()
