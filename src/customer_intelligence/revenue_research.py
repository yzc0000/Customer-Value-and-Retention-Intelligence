"""Isolated, reproducible 90-day revenue experiments; never changes champions.

Run ``python scripts/revenue_research.py`` to compare quarterly and monthly
training, safer GLMs, nonlinear two-part models, and feature ablations.
Only March and June development targets are scored. The already inspected
September test is deliberately not used for candidate selection here.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import GammaRegressor, LogisticRegression, PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_tweedie_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from customer_intelligence.config import REVENUE_90D_DEVELOPMENT_CUTOFFS, SEED
from customer_intelligence.bgnbd import BetaGeoNBD, GammaGamma, MONTH_DAYS, customer_day_summary
from customer_intelligence.evaluation import revenue_metrics
from customer_intelligence.features import NUMERIC_FEATURES, build_cutoff_features
from customer_intelligence.modeling import _preprocessor, revenue_candidates
from customer_intelligence.targets import build_targets


CALENDAR = ["cutoff_month_sin", "cutoff_month_cos", "forecast_q4_share"]
EXTRA = [
    "recent_order_value_30d", "recent_order_value_90d", "prior_90d_spend_gbp",
    "recency_over_mean_gap", "last_order_value_gbp", "order_value_std_gbp",
    "spend_same_future_window_last_year_gbp", "orders_same_future_window_last_year",
]


def enriched_features(features: pd.DataFrame, sales: pd.DataFrame) -> pd.DataFrame:
    """All amounts are computed before each cutoff; future calendar is known."""
    parts = []
    source_start = sales.invoice_date.min().normalize()
    for cutoff, cohort in features.groupby("cutoff", sort=True):
        cohort = cohort.copy()
        angle = 2 * np.pi * (cutoff.month - 1) / 12
        cohort[CALENDAR[0]], cohort[CALENDAR[1]] = np.sin(angle), np.cos(angle)
        dates = pd.date_range(cutoff, periods=90, freq="D")
        cohort[CALENDAR[2]] = np.mean(dates.month >= 10)
        for days in (30, 90):
            cohort[f"recent_order_value_{days}d"] = (
                cohort[f"spend_{days}d_gbp"] / cohort[f"orders_{days}d"].replace(0, np.nan)
            )
        cohort["prior_90d_spend_gbp"] = (cohort.spend_180d_gbp - cohort.spend_90d_gbp).clip(lower=0)
        cohort["recency_over_mean_gap"] = cohort.recency_days / cohort.mean_days_between_purchase_days.clip(lower=1)
        history = sales.loc[sales.invoice_date < cutoff]
        # Filter event timestamps before invoice aggregation, as in production.
        orders = history.groupby(["customer_id", "invoice_id"], sort=False).agg(
            date=("invoice_date", "max"), amount=("revenue", "sum"),
        ).reset_index().sort_values("date", kind="stable")
        last = orders.groupby("customer_id", sort=False).amount.last()
        std = orders.groupby("customer_id", sort=False).amount.std()
        cohort["last_order_value_gbp"] = cohort.customer_id.map(last)
        cohort["order_value_std_gbp"] = cohort.customer_id.map(std)
        prior_start = cutoff - pd.DateOffset(years=1)
        prior_end = prior_start + pd.Timedelta(days=90)
        assert prior_end <= cutoff
        if source_start <= prior_start:
            prior = history.loc[(history.invoice_date >= prior_start) & (history.invoice_date < prior_end)]
            spend = prior.groupby("customer_id").revenue.sum()
            count = prior.groupby("customer_id").invoice_id.nunique()
            cohort[EXTRA[-2]] = cohort.customer_id.map(spend).fillna(0)
            cohort[EXTRA[-1]] = cohort.customer_id.map(count).fillna(0)
        else:
            # A missing source period must not masquerade as observed zero spend.
            cohort[EXTRA[-2]], cohort[EXTRA[-1]] = np.nan, np.nan
        parts.append(cohort)
    return pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def load_panel(root: Path, frequency: str) -> pd.DataFrame:
    cache = root / "data" / "processed" / "revenue_research"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / f"{frequency}_panel.parquet"
    if path.exists():
        return pd.read_parquet(path)
    sales = pd.read_parquet(root / "data" / "interim" / "purchase_lines.parquet")
    if frequency == "quarterly":
        features = pd.read_parquet(root / "data" / "processed" / "customer_features.parquet")
        features = features.loc[features.cutoff <= pd.Timestamp("2011-06-01")].copy()
    else:
        credits = pd.read_parquet(root / "data" / "interim" / "credit_lines.parquet")
        dates = pd.date_range("2010-06-01", "2011-06-01", freq="MS")
        parts = []
        for cutoff in dates:
            print(f"Building research-only monthly snapshot {cutoff.date()}", flush=True)
            f, _, _ = build_cutoff_features(sales, credits, cutoff)
            parts.append(f)
        features = pd.concat(parts, ignore_index=True)
    features = enriched_features(features, sales)
    dates = [str(d.date()) for d in sorted(features.cutoff.unique())]
    targets, _ = build_targets(sales, features, dates, pd.Timestamp("2011-12-09"))
    panel = features.merge(targets, on=["customer_id", "cutoff"], validate="one_to_one")
    panel.to_parquet(path, index=False)
    return panel


def transform(numeric: list[str], logarithmic: bool = True) -> ColumnTransformer:
    nonnegative = [c for c in numeric if c not in CALENDAR]
    steps = [("impute", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True))]
    if logarithmic:
        # Transform inputs, not targets: predictions still estimate money means.
        steps.append(("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("numeric", Pipeline(steps), nonnegative),
        ("calendar", "passthrough", [c for c in CALENDAR if c in numeric]),
        ("country", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country"]),
    ])


class TwoPart:
    """Return propensity x positive revenue, or count x weighted order value."""

    def __init__(self, mode: str, numeric: list[str]):
        self.mode = mode
        self.transformer = transform(numeric)

    def fit(self, x, y, counts=None):
        z = self.transformer.fit_transform(x)
        positive = y > 0
        if self.mode == "hurdle_hist":
            self.first = HistGradientBoostingClassifier(**hist_parameters()).fit(z, positive)
            self.second = HistGradientBoostingRegressor(loss="gamma", **hist_parameters()).fit(z[positive], y[positive])
        elif self.mode == "hurdle_glm":
            self.first = LogisticRegression(C=0.3, max_iter=1000, random_state=SEED).fit(z, positive)
            self.second = GammaRegressor(alpha=1.0, max_iter=1000).fit(z[positive], y[positive])
        else:
            self.first = PoissonRegressor(alpha=1.0, max_iter=1000).fit(z, counts)
            self.second = GammaRegressor(alpha=1.0, max_iter=1000).fit(
                z[positive], y[positive] / counts[positive], sample_weight=counts[positive],
            )
        return self

    def predict(self, x):
        z = self.transformer.transform(x)
        if self.mode.startswith("hurdle"):
            first = self.first.predict_proba(z)[:, 1]
        else:
            first = self.first.predict(z)
        return first * self.second.predict(z)


def hist_parameters() -> dict:
    # Disable random early-stopping splits in this temporal experiment.
    return dict(max_iter=180, learning_rate=0.045, max_leaf_nodes=15,
                min_samples_leaf=40, l2_regularization=10.0,
                early_stopping=False, random_state=SEED)


def candidate(name: str, numeric: list[str]):
    if name == "xgboost_reference":
        model = revenue_candidates()["xgboost_tweedie"].named_steps["model"]
        return Pipeline([("transform", _preprocessor(numeric, ["country"], scale_numeric=False)), ("model", model)])
    if name.startswith("hurdle") or name == "frequency_severity_glm":
        return TwoPart(name, numeric)
    if name == "tweedie_log_glm":
        model = TweedieRegressor(power=1.5, alpha=1.0, max_iter=1200)
    elif name == "poisson_hist":
        model = HistGradientBoostingRegressor(loss="poisson", **hist_parameters())
    elif name == "median_hist":
        model = HistGradientBoostingRegressor(loss="absolute_error", **hist_parameters())
    else:
        raise ValueError(name)
    return Pipeline([("transform", transform(numeric)), ("model", model)])


def run(root: Path, frequencies: list[str], names: list[str]) -> None:
    records, manifest = [], []
    artifact = root / "artifacts" / "runs" / "revenue_research"
    artifact.mkdir(parents=True, exist_ok=True)
    for frequency in frequencies:
        panel = load_panel(root, frequency)
        for origin_text in REVENUE_90D_DEVELOPMENT_CUTOFFS:
            origin = pd.Timestamp(origin_text)
            # Monthly target windows overlap. Date of origin alone is insufficient.
            train = panel.loc[(panel.cutoff < origin) & (panel.horizon_end_exclusive <= origin) & panel.label_mature].copy()
            valid = panel.loc[panel.cutoff == origin].copy()
            assert len(train) and len(valid) and train.horizon_end_exclusive.max() <= origin
            assert not train.cutoff.eq(origin).any()
            assert valid.customer_id.is_unique
            y = train.future_revenue_gbp.to_numpy(dtype=float)
            count = train.future_invoice_count.to_numpy(dtype=float)
            actual = valid.future_revenue_gbp.to_numpy(dtype=float)
            print(f"{frequency} / {origin_text}: train={len(train):,} rows, latest label end={train.horizon_end_exclusive.max().date()}", flush=True)
            for feature_set in ("basic", "calendar", "enriched"):
                numeric = list(NUMERIC_FEATURES)
                if feature_set != "basic":
                    numeric += CALENDAR
                if feature_set == "enriched":
                    numeric += EXTRA
                for name in names:
                    label = f"{frequency}__{feature_set}__{name}"
                    model = candidate(name, numeric)
                    x_train, x_valid = train[numeric + ["country"]], valid[numeric + ["country"]]
                    with threadpool_limits(limits=2):
                        if isinstance(model, TwoPart):
                            model.fit(x_train, y, counts=count)
                        else:
                            model.fit(x_train, y)
                        prediction = np.maximum(model.predict(x_valid), 0)
                    if not np.isfinite(prediction).all():
                        raise RuntimeError(f"Non-finite predictions: {label}")
                    metrics = revenue_metrics(actual, prediction)
                    metrics["tweedie_deviance_p1_5"] = float(mean_tweedie_deviance(actual, np.maximum(prediction, 1e-6), power=1.5))
                    records.append(dict(model=label, cutoff=origin_text, **metrics))
                    run_dir = artifact / label
                    run_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(model, run_dir / f"{origin_text}_model.joblib", compress=3)
                    output = valid[["customer_id", "cutoff"]].copy()
                    output["actual_revenue_gbp"], output["predicted_revenue_gbp"] = actual, prediction
                    output.to_parquet(run_dir / f"{origin_text}_predictions.parquet", index=False)
                    print(f"  {label}: MAE={metrics['mae_gbp']:.2f}, RMSE={metrics['rmse_gbp']:.2f}, bias={metrics['aggregate_bias_pct']:.1%}, deviance={metrics['tweedie_deviance_p1_5']:.2f}", flush=True)
            manifest.append(dict(frequency=frequency, origin=origin_text, train_rows=len(train),
                                 train_origins=[str(d.date()) for d in sorted(train.cutoff.unique())],
                                 latest_available_training_outcome=str(train.horizon_end_exclusive.max().date())))
            metrics_path = root / "reports" / "revenue_research_development.csv"
            previous = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
            pd.concat([previous, pd.DataFrame(records)], ignore_index=True).drop_duplicates(
                ["model", "cutoff"], keep="last",
            ).to_csv(metrics_path, index=False)
    summarize(root)
    (artifact / "manifest.json").write_text(json.dumps(dict(seed=SEED, target="90-day gross customer revenue",
        selection_data="March and June development only; September not scored", runs=manifest,
        calendar=CALENDAR, added_behavioral_features=EXTRA,
        source_sha256=json.loads((root / "data" / "processed" / "prepare_manifest.json").read_text(encoding="utf-8"))["source_sha256"],
        environment=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        model_names=names, feature_sets=["basic", "calendar", "enriched"],
        note="Monthly snapshots and overlapping horizons are dependent observations, not independent customers. This research run does not update champion pointers."), indent=2), encoding="utf-8")


def summarize(root: Path) -> pd.DataFrame:
    """Score pre-specified equal-weight blends and paired customer resampling."""
    artifact = root / "artifacts" / "runs" / "revenue_research"
    metrics = pd.read_csv(root / "reports" / "revenue_research_development.csv")
    pairs = [
        ("xgboost_plus_bgnbd", "quarterly__basic__xgboost_reference", "BG_NBD"),
        ("quarterly_hurdle_plus_bgnbd", "quarterly__enriched__hurdle_hist", "BG_NBD"),
        ("monthly_hurdle_plus_bgnbd", "monthly__enriched__hurdle_hist", "BG_NBD"),
        ("monthly_hurdle_plus_xgboost", "monthly__enriched__hurdle_hist", "quarterly__basic__xgboost_reference"),
        ("monthly_hurdle_plus_frequency_severity", "monthly__enriched__hurdle_hist", "monthly__enriched__frequency_severity_glm"),
    ]
    records = []
    for label, left, right in pairs:
        for origin in REVENUE_90D_DEVELOPMENT_CUTOFFS:
            a_path = artifact / left / f"{origin}_predictions.parquet"
            b_path = (root / "artifacts" / "runs" / "behavioral_clv" / origin / "90d_predictions.parquet"
                      if right == "BG_NBD" else artifact / right / f"{origin}_predictions.parquet")
            if not a_path.exists() or not b_path.exists():
                continue
            a, b = pd.read_parquet(a_path), pd.read_parquet(b_path)
            if right == "BG_NBD":
                b = b.rename(columns={"future_revenue_gbp": "actual_revenue_gbp"})
            joined = a.merge(b[["customer_id", "cutoff", "actual_revenue_gbp", "predicted_revenue_gbp"]],
                             on=["customer_id", "cutoff"], suffixes=("_left", "_right"), validate="one_to_one")
            assert len(joined) == len(a) == len(b)
            assert np.allclose(joined.actual_revenue_gbp_left, joined.actual_revenue_gbp_right)
            prediction = (joined.predicted_revenue_gbp_left + joined.predicted_revenue_gbp_right) / 2
            actual = joined.actual_revenue_gbp_left
            result = revenue_metrics(actual, prediction)
            result["tweedie_deviance_p1_5"] = float(mean_tweedie_deviance(actual, np.maximum(prediction, 1e-6), power=1.5))
            records.append(dict(model="ensemble__" + label, cutoff=origin, **result))
            run_dir = artifact / ("ensemble__" + label)
            run_dir.mkdir(parents=True, exist_ok=True)
            output = joined[["customer_id", "cutoff"]].copy()
            output["actual_revenue_gbp"], output["predicted_revenue_gbp"] = actual, prediction
            output.to_parquet(run_dir / f"{origin}_predictions.parquet", index=False)
            (run_dir / "formula.json").write_text(json.dumps(dict(left=left, right=right, weights=[0.5, 0.5],
                note="Fixed equal weights; not optimized on development or final outcomes."), indent=2), encoding="utf-8")
    pd.DataFrame(records).to_csv(root / "reports" / "revenue_research_ensembles.csv", index=False)
    metrics = pd.concat([metrics, pd.DataFrame(records)], ignore_index=True)
    summary = metrics.groupby("model", as_index=False).agg(
        mean_mae_gbp=("mae_gbp", "mean"), mean_rmse_gbp=("rmse_gbp", "mean"),
        mean_absolute_bias_pct=("aggregate_bias_pct", lambda s: s.abs().mean()),
        mean_tweedie_deviance=("tweedie_deviance_p1_5", "mean"),
        mean_revenue_capture_top10=("revenue_capture_at_capacity", "mean"),
        n_origins=("cutoff", "nunique"),
    ).sort_values("mean_tweedie_deviance")
    summary.to_csv(root / "reports" / "revenue_research_summary.csv", index=False)
    uncertainty = []
    for label in ["quarterly__enriched__hurdle_hist", "monthly__enriched__hurdle_hist", "ensemble__monthly_hurdle_plus_xgboost"]:
        rows = []
        for origin in REVENUE_90D_DEVELOPMENT_CUTOFFS:
            p = artifact / label / f"{origin}_predictions.parquet"
            if not p.exists():
                break
            a = pd.read_parquet(artifact / "quarterly__basic__xgboost_reference" / f"{origin}_predictions.parquet")
            b = pd.read_parquet(p)
            z = a.merge(b, on=["customer_id", "cutoff"], suffixes=("_reference", "_candidate"), validate="one_to_one")
            assert np.allclose(z.actual_revenue_gbp_reference, z.actual_revenue_gbp_candidate)
            z["mae_improvement_gbp"] = (np.abs(z.predicted_revenue_gbp_reference - z.actual_revenue_gbp_reference)
                                        - np.abs(z.predicted_revenue_gbp_candidate - z.actual_revenue_gbp_candidate))
            rows.append(z[["customer_id", "cutoff", "mae_improvement_gbp"]])
        if len(rows) != len(REVENUE_90D_DEVELOPMENT_CUTOFFS):
            continue
        pivot = pd.concat(rows).pivot(index="customer_id", columns="cutoff", values="mae_improvement_gbp")
        observed = pivot.notna().to_numpy(dtype=float)
        delta = pivot.fillna(0).to_numpy()
        # Repeated appearances of a customer are sampled together across origins.
        rng = np.random.default_rng(SEED)
        draws = []
        for _ in range(10):
            weights = rng.multinomial(len(pivot), np.full(len(pivot), 1 / len(pivot)), size=100)
            draws.extend(np.mean((weights @ delta) / (weights @ observed), axis=1))
        lower, upper = np.quantile(draws, [0.025, 0.975])
        uncertainty.append(dict(model=label, mean_mae_improvement_gbp=float(pivot.mean().mean()),
            customer_bootstrap_lower_95_gbp=float(lower), customer_bootstrap_upper_95_gbp=float(upper),
            note="Conditional on these two origins; does not measure uncertainty across future seasonal regimes or correct for candidate selection."))
    pd.DataFrame(uncertainty).to_csv(root / "reports" / "revenue_research_uncertainty.csv", index=False)
    existing_error_diagnostics(root)
    print(summary.to_string(index=False), flush=True)
    return summary


def existing_error_diagnostics(root: Path) -> pd.DataFrame:
    """Diagnose the previously published forecasts, including the old test.

    This does not evaluate research candidates on September or affect selection.
    """
    records = []
    for origin in [*REVENUE_90D_DEVELOPMENT_CUTOFFS, "2011-09-01"]:
        for model in ["xgboost_tweedie", "recent_90d_spend_baseline"]:
            filename = "final_test_predictions.parquet" if origin == "2011-09-01" else f"development_{origin}_predictions.parquet"
            z = pd.read_parquet(root / "artifacts" / "runs" / "90d" / "revenue" / model / filename)
            y, p = z.actual_revenue_gbp.to_numpy(), z.predicted_revenue_gbp.to_numpy()
            top = np.argsort(-y)[:int(np.ceil(0.01 * len(y)))]
            records.append(dict(cutoff=origin, model=model, n=len(y), zero_fraction=float(np.mean(y == 0)),
                top1pct_revenue_fraction=float(y[top].sum() / y.sum()), top1pct_signed_error_gbp=float((p - y)[top].sum()),
                all_signed_error_gbp=float((p - y).sum()), zero_customer_mean_prediction_gbp=float(p[y == 0].mean()),
                positive_customer_mae_gbp=float(np.abs(p[y > 0] - y[y > 0]).mean())))
    result = pd.DataFrame(records)
    result.to_csv(root / "reports" / "revenue_error_diagnostics.csv", index=False)
    return result


def statistical_sensitivity(root: Path) -> pd.DataFrame:
    """Check BG/NBD regularization on development origins, never September.

    Evaluate six months only at the original December development origin.
    Gamma-Gamma is held fixed so the effect of BG/NBD penalties is inspectable.
    """
    sales = pd.read_parquet(root / "data" / "interim" / "purchase_lines.parquet")
    target90 = pd.read_parquet(root / "data" / "processed" / "targets_90d.parquet")
    target6m = pd.read_parquet(root / "data" / "processed" / "targets_6m.parquet")
    records = []
    for origin_text in ["2010-12-01", *REVENUE_90D_DEVELOPMENT_CUTOFFS]:
        origin = pd.Timestamp(origin_text)
        summary = customer_day_summary(sales, origin)
        repeat = (summary.frequency > 0) & summary.repeat_mean_daily_spend.notna()
        gamma = GammaGamma().fit(summary.loc[repeat, "frequency"], summary.loc[repeat, "repeat_mean_daily_spend"])
        assert gamma.finite_population_mean_
        amount = gamma.expected_average_spend(summary.frequency, summary.repeat_mean_daily_spend.fillna(0))
        targets = target6m if origin_text == "2010-12-01" else target90
        horizon = "6m" if origin_text == "2010-12-01" else "90d"
        labels = targets.loc[(targets.cutoff == origin) & targets.label_mature].set_index("customer_id")
        aligned = summary.reindex(labels.index)
        amount_aligned = pd.Series(amount, index=summary.index).reindex(labels.index).to_numpy()
        days = (labels.horizon_end_exclusive.iloc[0] - origin).days
        for penalty in [0.0, 1e-5, 1e-4, 1e-3, 1e-2]:
            model = BetaGeoNBD(penalizer=penalty).fit(summary.frequency, summary.recency_months, summary.age_months)
            prediction = model.expected_purchases(days / MONTH_DAYS, aligned.frequency, aligned.recency_months, aligned.age_months) * amount_aligned
            assert np.isfinite(prediction).all()
            record = dict(model="BG_NBD_x_Gamma_Gamma", cutoff=origin_text, horizon=horizon,
                bg_penalty=penalty, gamma_penalty=gamma.penalizer, bg_a=model.a_, bg_b=model.b_,
                optimizer_success=model.optimizer_success_, a_at_lower_bound=bool(np.isclose(model.a_, np.exp(-7), rtol=1e-4)),
                gamma_q=gamma.q_, **revenue_metrics(labels.future_revenue_gbp, prediction))
            records.append(record)
            path = root / "artifacts" / "runs" / "revenue_research" / "bgnbd_sensitivity" / origin_text / str(penalty)
            path.mkdir(parents=True, exist_ok=True)
            joblib.dump(dict(bg_nbd=model, gamma_gamma=gamma), path / "model.joblib", compress=3)
            out = labels.reset_index()[["customer_id", "cutoff", "future_revenue_gbp"]].copy()
            out["predicted_revenue_gbp"] = prediction
            out.to_parquet(path / f"{horizon}_predictions.parquet", index=False)
            print(f"{horizon} / {origin_text} / penalty={penalty}: a={model.a_:.6f}, MAE={record['mae_gbp']:.2f}, bias={record['aggregate_bias_pct']:.1%}", flush=True)
    result = pd.DataFrame(records)
    result.to_csv(root / "reports" / "revenue_bgnbd_sensitivity.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frequency", choices=["quarterly", "monthly", "both"], default="both")
    parser.add_argument("--models", nargs="+", default=["xgboost_reference", "tweedie_log_glm", "hurdle_glm", "frequency_severity_glm", "hurdle_hist", "poisson_hist", "median_hist"])
    parser.add_argument("--summarize", action="store_true", help="Rescore saved predictions; do not refit any models.")
    parser.add_argument("--statistical-sensitivity", action="store_true", help="Run the development-only BG/NBD regularization check.")
    args = parser.parse_args()
    if args.statistical_sensitivity:
        statistical_sensitivity(ROOT)
    elif args.summarize:
        summarize(ROOT)
    else:
        run(ROOT, ["quarterly", "monthly"] if args.frequency == "both" else [args.frequency], args.models)


if __name__ == "__main__":
    main()
