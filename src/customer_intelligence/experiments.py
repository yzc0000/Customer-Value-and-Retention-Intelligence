"""Historical model fitting, calibration and frozen out-of-time scoring."""

from __future__ import annotations

import json
import math
import platform
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression

from .bgnbd import BetaGeoNBD, GammaGamma, MONTH_DAYS, customer_day_summary
from .config import (
    CALIBRATION_CUTOFF, CLASSIFICATION_DEVELOPMENT_CUTOFF,
    CLASSIFICATION_TRAIN_CUTOFFS, CONTACT_CAPACITY, FINAL_TEST_CUTOFF,
    INACTIVITY_SELECTED_MODEL, REVENUE_90D_DEVELOPMENT_CUTOFFS, SEED,
)
from .evaluation import classification_metrics, metrics_table, revenue_metrics
from .features import NUMERIC_FEATURES
from .modeling import classifier_candidates, revenue_candidates
from .pipeline import load_prepared
from .sequence_models import fit_sequence_model
from .survival import PooledPurchaseHazard

MODEL_INPUTS = list(NUMERIC_FEATURES) + ["country"]
SEQUENCE_MODELS = ["mlp_weekly", "gru_weekly", "lstm_weekly"]


def _merge_targets(features: pd.DataFrame, targets: pd.DataFrame, horizon: str) -> pd.DataFrame:
    labels = targets.loc[targets["horizon"].eq(horizon)]
    merged = features.merge(labels, on=["customer_id", "cutoff"], how="inner", validate="one_to_one")
    return merged.loc[merged["label_mature"].eq(True)].reset_index(drop=True)


def _metric_record(task: str, model: str, split: str, cutoff: str, metrics: dict, **extra) -> dict:
    return {"task": task, "model": model, "split": split, "cutoff": cutoff, **metrics, **extra}


def _save_prediction(path: Path, rows: pd.DataFrame, y, prediction, target: str) -> None:
    output = rows[["customer_id", "cutoff"]].copy()
    output["actual"] = np.asarray(y)
    output["prediction"] = np.asarray(prediction)
    output["target"] = target
    output.to_parquet(path, index=False)


def _sequences_for_rows(all_features: pd.DataFrame, rows: pd.DataFrame, sequence_file) -> np.ndarray:
    result = np.zeros((len(rows), 26, 6), dtype=np.float32)
    positions = rows.groupby("cutoff", sort=False).indices
    for cutoff_value, local_pos in positions.items():
        cutoff = pd.Timestamp(cutoff_value)
        key = cutoff.strftime("%Y-%m-%d")
        seq = sequence_file[key]
        cutoff_features = all_features.loc[all_features["cutoff"].eq(cutoff), "customer_id"].reset_index(drop=True)
        index_for_customer = pd.Series(np.arange(len(cutoff_features)), index=cutoff_features)
        offset = rows.iloc[local_pos]["customer_id"].map(index_for_customer).to_numpy(dtype=int)
        result[local_pos] = seq[offset]
    return result


def _platt_fit(calibration_rows: pd.DataFrame, calibration_probability: np.ndarray) -> LogisticRegression:
    p = np.clip(calibration_probability, 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=500)
    model.fit(logits, calibration_rows["inactive"].to_numpy(dtype=int))
    return model


def _platt_predict(model, probability: np.ndarray) -> np.ndarray:
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p)).reshape(-1, 1)
    return model.predict_proba(logits)[:, 1]


def _revenue_baselines(train_features: pd.DataFrame, y_train, score_features: pd.DataFrame, horizon_days: int, recent_window: int) -> dict[str, np.ndarray]:
    """Training-only constant and transparent run-rate baselines."""
    target = np.asarray(y_train, dtype=float)
    mean_value = float(np.mean(target))
    median_value = float(np.median(target))
    recent_name = f"recent_{recent_window}d_spend_baseline"
    return {
        "training_mean_baseline": np.full(len(score_features), mean_value, dtype=float),
        "training_median_baseline": np.full(len(score_features), median_value, dtype=float),
        "historical_spend_rate_baseline": (
            score_features["lifetime_revenue_gbp"].to_numpy(dtype=float)
            / np.maximum(score_features["history_days"].to_numpy(dtype=float), 1)
            * horizon_days
        ),
        recent_name: (
            score_features[f"spend_{recent_window}d_gbp"].to_numpy(dtype=float)
            * (horizon_days / recent_window)
        ),
    }


def _fit_discrete_survival(x_train, rows_train, x_valid) -> np.ndarray:
    model = PooledPurchaseHazard().fit(
        x_train,
        rows_train["returned"].to_numpy(dtype=int),
        rows_train["first_future_purchase"].to_numpy(),
        rows_train["cutoff"].to_numpy(),
    )
    return model, model.predict_proba(x_valid)[:, 0]


def fit_inactivity(root: Path) -> pd.DataFrame:
    features, targets90, _, manifest = load_prepared(root)
    population = _merge_targets(features, targets90, "90d")
    train = population.loc[population["cutoff"].isin(pd.to_datetime(CLASSIFICATION_TRAIN_CUTOFFS))].copy().reset_index(drop=True)
    development = population.loc[population["cutoff"].eq(pd.Timestamp(CLASSIFICATION_DEVELOPMENT_CUTOFF))].copy().reset_index(drop=True)
    calibration = population.loc[population["cutoff"].eq(pd.Timestamp(CALIBRATION_CUTOFF))].copy().reset_index(drop=True)
    final_test = population.loc[population["cutoff"].eq(pd.Timestamp(FINAL_TEST_CUTOFF))].copy().reset_index(drop=True)
    if any(part.empty for part in [train, development, calibration, final_test]):
        raise ValueError("Expected mature rows are missing from one or more configured forecast splits.")
    x_train, y_train = train[MODEL_INPUTS], train["inactive"].to_numpy(dtype=int)
    x_dev, y_dev = development[MODEL_INPUTS], development["inactive"].to_numpy(dtype=int)
    x_cal, y_cal = calibration[MODEL_INPUTS], calibration["inactive"].to_numpy(dtype=int)
    x_test, y_test = final_test[MODEL_INPUTS], final_test["inactive"].to_numpy(dtype=int)

    artifact_root = root / "artifacts" / "runs" / "90d" / "inactivity"
    artifact_root.mkdir(parents=True, exist_ok=True)
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict] = []
    development_predictions: dict[str, np.ndarray] = {}
    statuses = {}
    candidates = classifier_candidates()
    try:
        import catboost  # noqa: F401
        catboost_status = {"status": "ready"}
    except ImportError:
        catboost_status = {"status": "unavailable", "reason": "CatBoost is not installed in the current modeling environment."}
    if catboost_status["status"] != "ready":
        statuses["catboost"] = catboost_status
    print(f"Fitting {len(candidates)} tabular inactivity candidates on {len(train):,} matured snapshots...", flush=True)

    for name, model in candidates.items():
        run_dir = artifact_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            with warnings.catch_warnings(record=True) as seen:
                warnings.simplefilter("always")
                model.fit(x_train, y_train)
                raw_dev = model.predict_proba(x_dev)[:, 1]
            joblib.dump(model, run_dir / "development_model.joblib", compress=3)
            development_predictions[name] = raw_dev
            metric_rows.append(_metric_record(
                "inactivity", name, "development", CLASSIFICATION_DEVELOPMENT_CUTOFF,
                classification_metrics(y_dev, raw_dev, CONTACT_CAPACITY),
            ))
            statuses[name] = {
                "status": "completed",
                "development_warnings": [str(item.message) for item in seen if issubclass(item.category, Warning)],
            }
            print(f"  {name}: development AP={metric_rows[-1]['average_precision']:.4f}, P@10%={metric_rows[-1]['precision_at_capacity']:.4f}", flush=True)
        except Exception as exc:  # preserve the outcome of an unsupported model run
            statuses[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: failed ({type(exc).__name__}: {exc})", flush=True)

    for name in ("pooled_purchase_hazard",):
        run_dir = artifact_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            model, raw_dev = _fit_discrete_survival(x_train, train, x_dev)
            joblib.dump(model, run_dir / "development_model.joblib", compress=3)
            development_predictions[name] = raw_dev
            metric_rows.append(_metric_record(
                "inactivity", name, "development", CLASSIFICATION_DEVELOPMENT_CUTOFF,
                classification_metrics(y_dev, raw_dev, CONTACT_CAPACITY),
            ))
            statuses[name] = {"status": "completed", "event": "time until first purchase after cutoff"}
            print(f"  {name}: development AP={metric_rows[-1]['average_precision']:.4f}, P@10%={metric_rows[-1]['precision_at_capacity']:.4f}", flush=True)
        except Exception as exc:
            statuses[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: failed ({type(exc).__name__}: {exc})", flush=True)

    for name in SEQUENCE_MODELS:
        run_dir = artifact_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        seed_predictions = []
        epochs = []
        per_seed = []
        cache = np.load(root / "data" / "processed" / "weekly_sequences.npz")
        train_sequences = _sequences_for_rows(features, train, cache)
        dev_sequences = _sequences_for_rows(features, development, cache)
        for seed in (SEED, SEED + 1, SEED + 2):
            try:
                predictor, raw_dev, best_epoch = fit_sequence_model(
                    name, train, train_sequences, y_train,
                    development, dev_sequences, y_dev, seed=seed,
                )
                seed_predictions.append(raw_dev)
                epochs.append(best_epoch)
                per_seed.append(predictor)
                metric_rows.append(_metric_record(
                    "inactivity", name, "development_seed", CLASSIFICATION_DEVELOPMENT_CUTOFF,
                    classification_metrics(y_dev, raw_dev, CONTACT_CAPACITY), seed=seed, best_epoch=best_epoch,
                ))
                print(f"  {name} seed {seed}: best epoch {best_epoch}, AP={metric_rows[-1]['average_precision']:.4f}", flush=True)
            except Exception as exc:
                print(f"  {name} seed {seed}: failed ({type(exc).__name__}: {exc})", flush=True)
        if seed_predictions:
            mean_dev = np.mean(seed_predictions, axis=0)
            development_predictions[name] = mean_dev
            metric_rows.append(_metric_record(
                "inactivity", name, "development_ensemble", CLASSIFICATION_DEVELOPMENT_CUTOFF,
                classification_metrics(y_dev, mean_dev, CONTACT_CAPACITY),
                seeds=len(seed_predictions), median_best_epoch=int(np.median(epochs)),
            ))
            statuses[name] = {"status": "completed", "seeds": len(seed_predictions), "median_best_epoch": int(np.median(epochs))}
            joblib.dump(per_seed, run_dir / "development_ensemble.joblib", compress=3)
            joblib.dump(epochs, run_dir / "development_epochs.joblib")
        else:
            statuses[name] = {"status": "failed", "error": "All recorded seed runs failed."}

    development_table = pd.DataFrame(metric_rows)
    model_development = development_table.loc[
        development_table["split"].isin(["development", "development_ensemble"])
    ].dropna(subset=["precision_at_capacity"]).copy()
    if model_development.empty:
        raise RuntimeError("No valid inactivity candidate completed on the development snapshot.")
    model_development = model_development.sort_values(
        ["precision_at_capacity", "average_precision", "brier_score"],
        ascending=[False, False, True], kind="stable",
    )
    development_metric_winner = str(model_development.iloc[0]["model"])
    if INACTIVITY_SELECTED_MODEL not in statuses or statuses[INACTIVITY_SELECTED_MODEL].get("status") != "completed":
        raise RuntimeError(f"Configured inactivity lead model {INACTIVITY_SELECTED_MODEL!r} did not complete.")
    selected_model = INACTIVITY_SELECTED_MODEL
    print(
        f"Project-selected inactivity lead: {selected_model}; development metric winner: {development_metric_winner}",
        flush=True,
    )

    fit_rows = population.loc[
        population["cutoff"].le(pd.Timestamp(CLASSIFICATION_DEVELOPMENT_CUTOFF))
        & population["cutoff"].isin([*pd.to_datetime(CLASSIFICATION_TRAIN_CUTOFFS), pd.Timestamp(CLASSIFICATION_DEVELOPMENT_CUTOFF)])
    ].copy().reset_index(drop=True)
    x_fit, y_fit = fit_rows[MODEL_INPUTS], fit_rows["inactive"].to_numpy(dtype=int)
    calibrators: dict[str, object] = {}
    final_probabilities: dict[str, np.ndarray] = {}
    fit_sequences = _sequences_for_rows(features, fit_rows, cache)
    cal_sequences = _sequences_for_rows(features, calibration, cache)
    test_sequences = _sequences_for_rows(features, final_test, cache)

    for name, status in list(statuses.items()):
        if status.get("status") != "completed":
            continue
        run_dir = artifact_root / name
        try:
            if name in SEQUENCE_MODELS:
                epoch_file = run_dir / "development_epochs.joblib"
                epochs = joblib.load(epoch_file)
                ensemble = []
                cal_seed_probabilities = []
                test_seed_probabilities = []
                for seed_idx, (seed, epoch_count) in enumerate(zip((SEED, SEED + 1, SEED + 2), epochs, strict=False)):
                    predictor, _, _ = fit_sequence_model(
                        name, fit_rows, fit_sequences, y_fit,
                        fit_rows, fit_sequences, y_fit,
                        seed=seed, fixed_epochs=max(1, int(epoch_count)),
                    )
                    ensemble.append(predictor)
                    cal_seed_probabilities.append(predictor.predict_proba(calibration, cal_sequences)[:, 1])
                    test_seed_probabilities.append(predictor.predict_proba(final_test, test_sequences)[:, 1])
                raw_cal = np.mean(cal_seed_probabilities, axis=0)
                raw_test = np.mean(test_seed_probabilities, axis=0)
                joblib.dump(ensemble, run_dir / "scoring_ensemble.joblib", compress=3)
            elif name == "pooled_purchase_hazard":
                model, raw_cal = _fit_discrete_survival(x_fit, fit_rows, x_cal)
                raw_test = model.predict_proba(x_test)[:, 0]
                joblib.dump(model, run_dir / "scoring_model.joblib", compress=3)
            else:
                model = classifier_candidates()[name]
                model.fit(x_fit, y_fit)
                raw_cal = model.predict_proba(x_cal)[:, 1]
                raw_test = model.predict_proba(x_test)[:, 1]
                joblib.dump(model, run_dir / "scoring_model.joblib", compress=3)
            calibrator = _platt_fit(calibration, raw_cal)
            calibrated_cal = _platt_predict(calibrator, raw_cal)
            calibrated_test = _platt_predict(calibrator, raw_test)
            joblib.dump(calibrator, run_dir / "platt_calibrator.joblib")
            calibrators[name] = calibrator
            final_probabilities[name] = raw_test
            metric_rows.append(_metric_record(
                "inactivity", name, "calibration_raw", CALIBRATION_CUTOFF,
                classification_metrics(y_cal, raw_cal, CONTACT_CAPACITY),
            ))
            metric_rows.append(_metric_record(
                "inactivity", name, "calibration_platt", CALIBRATION_CUTOFF,
                classification_metrics(y_cal, calibrated_cal, CONTACT_CAPACITY),
            ))
            metric_rows.append(_metric_record(
                "inactivity", name, "final_test_raw", FINAL_TEST_CUTOFF,
                classification_metrics(y_test, raw_test, CONTACT_CAPACITY),
            ))
            metric_rows.append(_metric_record(
                "inactivity", name, "final_test_platt", FINAL_TEST_CUTOFF,
                classification_metrics(y_test, calibrated_test, CONTACT_CAPACITY),
            ))
            prediction_dir = run_dir / "predictions"
            prediction_dir.mkdir(exist_ok=True)
            _save_prediction(prediction_dir / "development.parquet", development, y_dev, development_predictions[name], "inactive_90d")
            _save_prediction(prediction_dir / "calibration.parquet", calibration, y_cal, calibrated_cal, "inactive_90d")
            _save_prediction(prediction_dir / "final_test.parquet", final_test, y_test, calibrated_test, "inactive_90d")
            print(f"  {name}: calibration + frozen September evaluation completed", flush=True)
        except Exception as exc:
            statuses[name] = {"status": "refit_failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: scoring refit failed ({type(exc).__name__}: {exc})", flush=True)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(reports / "inactivity_model_comparison.csv", index=False)
    score_output = final_test[["customer_id", "cutoff", "inactive", "future_revenue_gbp"]].copy()
    for name, model in final_probabilities.items():
        calibrator = calibrators[name]
        score_output[f"{name}_risk"] = _platt_predict(calibrator, model)
    score_output.to_parquet(root / "data" / "processed" / "final_customer_scores_90d.parquet", index=False)

    champion_payload = {
        "task": "90-day inactivity probability",
        "metric": "development precision among the top 10% highest-risk customers; ties break by average precision then Brier score",
        "development_cutoff": CLASSIFICATION_DEVELOPMENT_CUTOFF,
        "selected_model": selected_model,
        "development_metric_winner": development_metric_winner,
        "selection_rationale": "Weekly LSTM is the project lead to test explicit temporal modeling; this design choice does not claim that it won the prespecified development metric.",
        "test_cutoff": FINAL_TEST_CUTOFF,
        "model_status": statuses,
        "source_sha256": manifest["source_sha256"],
        "test_used_for_selection": False,
        "all_candidates_retained": True,
    }
    champion_path = root / "artifacts" / "champions.json"
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(champion_path.read_text(encoding="utf-8")) if champion_path.exists() else {}
    existing["inactivity_90d"] = champion_payload
    champion_path.write_text(json.dumps(existing, indent=2, allow_nan=False), encoding="utf-8")
    (artifact_root / "run_manifest.json").write_text(json.dumps({
        "source_sha256": manifest["source_sha256"],
        "task": "inactivity_90d",
        "feature_columns": MODEL_INPUTS,
        "training_cutoffs": CLASSIFICATION_TRAIN_CUTOFFS,
        "development_cutoff": CLASSIFICATION_DEVELOPMENT_CUTOFF,
        "calibration_cutoff": CALIBRATION_CUTOFF,
        "final_test_cutoff": FINAL_TEST_CUTOFF,
        "primary_metric": "precision_at_capacity",
        "capacity": CONTACT_CAPACITY,
        "seed": SEED,
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "status": statuses,
        "selection": f"Project lead fixed to {selected_model} for temporal modeling. Development metric winner is {development_metric_winner}; calibration and final test were not used to select the lead.",
    }, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Inactivity comparison saved. Project lead: {selected_model}; development metric winner: {development_metric_winner}", flush=True)
    print(metrics.loc[metrics["split"].isin(["development", "development_ensemble", "final_test_platt"]), ["model", "split", "precision_at_capacity", "average_precision", "brier_score", "recall_at_capacity"]].to_string(index=False), flush=True)
    return metrics


def fit_revenue(root: Path) -> pd.DataFrame:
    features, _, targets6m, manifest = load_prepared(root)
    population = _merge_targets(features, targets6m, "6m")
    train_cutoff = pd.Timestamp("2010-06-01")
    development_cutoff = pd.Timestamp("2010-12-01")
    calibration_cutoff = pd.Timestamp("2011-06-01")
    train = population.loc[population["cutoff"].eq(train_cutoff)].reset_index(drop=True)
    development = population.loc[population["cutoff"].eq(development_cutoff)].reset_index(drop=True)
    final_test = population.loc[population["cutoff"].eq(calibration_cutoff)].reset_index(drop=True)
    train = train.loc[train["label_mature"].eq(True)].reset_index(drop=True)
    development = development.loc[development["label_mature"].eq(True)].reset_index(drop=True)
    final_test = final_test.loc[final_test["label_mature"].eq(True)].reset_index(drop=True)
    if any(part.empty for part in [train, development, final_test]):
        raise ValueError("The June/December/June six-month holdout schedule is incomplete.")
    x_train = train[MODEL_INPUTS]
    y_train = train["future_revenue_gbp"].to_numpy(dtype=float)
    x_dev = development[MODEL_INPUTS]
    y_dev = development["future_revenue_gbp"].to_numpy(dtype=float)
    x_test = final_test[MODEL_INPUTS]
    y_test = final_test["future_revenue_gbp"].to_numpy(dtype=float)
    artifact_root = root / "artifacts" / "runs" / "six_calendar_months" / "revenue"
    artifact_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    predictions = {}
    statuses = {}
    candidates = revenue_candidates()
    print(f"Fitting {len(candidates)} six-month revenue candidates on {len(train):,} customer examples...", flush=True)
    development_days = (development["horizon_end_exclusive"].iloc[0] - development_cutoff).days
    test_days = (final_test["horizon_end_exclusive"].iloc[0] - calibration_cutoff).days
    recent_baselines = _revenue_baselines(x_train, y_train, x_dev, development_days, 180)
    for name, prediction in recent_baselines.items():
        run_dir = artifact_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions[name] = prediction
        statuses[name] = {"status": "completed", "kind": "training-only arithmetic baseline"}
        dev_output = development[["customer_id", "cutoff"]].copy()
        dev_output["actual_revenue_gbp"] = y_dev
        dev_output["predicted_revenue_gbp"] = prediction
        dev_output.to_parquet(run_dir / "development_predictions.parquet", index=False)
        metric_rows.append(_metric_record("future_revenue_6m", name, "development", str(development_cutoff.date()), revenue_metrics(y_dev, prediction, CONTACT_CAPACITY)))
        print(f"  {name}: development MAE=GBP {metric_rows[-1]['mae_gbp']:,.2f}, bias={metric_rows[-1]['aggregate_bias_pct']:.1%}", flush=True)
    for name, model in candidates.items():
        run_dir = artifact_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        try:
            model.fit(x_train, y_train)
            pred_dev = np.maximum(model.predict(x_dev), 0)
            joblib.dump(model, run_dir / "development_model.joblib", compress=3)
            predictions[name] = pred_dev
            dev_output = development[["customer_id", "cutoff"]].copy()
            dev_output["actual_revenue_gbp"] = y_dev
            dev_output["predicted_revenue_gbp"] = pred_dev
            dev_output.to_parquet(run_dir / "development_predictions.parquet", index=False)
            metric_rows.append(_metric_record("future_revenue_6m", name, "development", str(development_cutoff.date()), revenue_metrics(y_dev, pred_dev, CONTACT_CAPACITY)))
            statuses[name] = {"status": "completed"}
            print(f"  {name}: development MAE=GBP {metric_rows[-1]['mae_gbp']:,.2f}, bias={metric_rows[-1]['aggregate_bias_pct']:.1%}", flush=True)
        except Exception as exc:
            statuses[name] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: failed ({type(exc).__name__}: {exc})", flush=True)
    table = pd.DataFrame(metric_rows)
    table = table.dropna(subset=["mae_gbp"]).copy()
    table["abs_aggregate_bias_pct"] = table["aggregate_bias_pct"].abs()
    table = table.sort_values(["mae_gbp", "abs_aggregate_bias_pct"], kind="stable")
    if table.empty:
        raise RuntimeError("No revenue model completed on the development snapshot.")
    champion = str(table.iloc[0]["model"])
    # Once the development choice is frozen, use every matured historical
    # snapshot available before the June 2011 test origin. In particular, the
    # September 2010 six-month label matured on 2011-03-01; it can strengthen
    # the final refit without changing model selection or touching test labels.
    fit_rows = population.loc[
        population["cutoff"].lt(calibration_cutoff)
        & population["label_mature"].eq(True)
        & population["horizon_end_exclusive"].le(calibration_cutoff)
    ].reset_index(drop=True)
    if fit_rows.empty or fit_rows["horizon_end_exclusive"].max() > calibration_cutoff:
        raise ValueError("Six-month refit labels must be complete at the forecast origin.")
    final_predictions = {}
    for name, status in list(statuses.items()):
        if status.get("status") != "completed":
            continue
        try:
            if name in candidates:
                model = revenue_candidates()[name]
                model.fit(fit_rows[MODEL_INPUTS], fit_rows["future_revenue_gbp"].to_numpy(dtype=float))
                pred = np.maximum(model.predict(x_test), 0)
                joblib.dump(model, artifact_root / name / "scoring_model.joblib", compress=3)
            else:
                values = _revenue_baselines(fit_rows[MODEL_INPUTS], fit_rows["future_revenue_gbp"], x_test, test_days, 180)
                pred = values[name]
                (artifact_root / name / "scoring_baseline.json").write_text(json.dumps({
                    "training_median_gbp": float(np.median(fit_rows["future_revenue_gbp"])),
                    "training_mean_gbp": float(np.mean(fit_rows["future_revenue_gbp"])),
                    "source_sha256": manifest["source_sha256"],
                }, indent=2), encoding="utf-8")
            final_predictions[name] = pred
            metric_rows.append(_metric_record("future_revenue_6m", name, "final_test", str(calibration_cutoff.date()), revenue_metrics(y_test, pred, CONTACT_CAPACITY)))
            out = final_test[["customer_id", "cutoff"]].copy()
            out["actual_revenue_gbp"] = y_test
            out["predicted_revenue_gbp"] = pred
            out.to_parquet(artifact_root / name / "final_test_predictions.parquet", index=False)
            print(f"  {name}: frozen June 2011 test MAE=GBP {metric_rows[-1]['mae_gbp']:,.2f}", flush=True)
        except Exception as exc:
            statuses[name] = {"status": "refit_failed", "error": f"{type(exc).__name__}: {exc}"}
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metric_rows).to_csv(reports / "revenue_6m_model_comparison.csv", index=False)
    score_output = final_test[["customer_id", "cutoff", "future_revenue_gbp"]].copy()
    score_output = score_output.rename(columns={"future_revenue_gbp": "actual_revenue_gbp"})
    for name, prediction in final_predictions.items():
        score_output[name] = prediction
    score_output.to_parquet(root / "data" / "processed" / "final_customer_scores_6m.parquet", index=False)

    champion_path = root / "artifacts" / "champions.json"
    existing = json.loads(champion_path.read_text(encoding="utf-8")) if champion_path.exists() else {}
    existing["future_revenue_6m"] = {
        "task": "six-calendar-month gross merchandise revenue forecast in GBP",
        "metric": "development MAE in GBP; aggregate bias is the supporting diagnostic",
        "development_cutoff": str(development_cutoff.date()),
        "selected_model": champion,
        "test_cutoff": str(calibration_cutoff.date()),
        "test_used_for_selection": False,
        "all_candidates_retained": True,
        "source_sha256": manifest["source_sha256"],
        "model_status": statuses,
        "baseline_comparison_included": True,
        "scoring_train_rows": int(len(fit_rows)),
        "latest_scoring_training_label_end": str(fit_rows["horizon_end_exclusive"].max().date()),
        "scoring_training_labels_complete_at_origin": True,
    }
    champion_path.parent.mkdir(parents=True, exist_ok=True)
    champion_path.write_text(json.dumps(existing, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Six-month revenue champion selected on December development data: {champion}", flush=True)
    return pd.DataFrame(metric_rows)


def fit_revenue_90d(root: Path) -> pd.DataFrame:
    """Select 90-day revenue models across March and June development origins."""
    features, targets90, _, manifest = load_prepared(root)
    population = _merge_targets(features, targets90, "90d")
    development_cutoffs = pd.to_datetime(REVENUE_90D_DEVELOPMENT_CUTOFFS).tolist()
    final_test = population.loc[population["cutoff"].eq(pd.Timestamp(FINAL_TEST_CUTOFF))].copy().reset_index(drop=True)
    x_test, y_test = final_test[MODEL_INPUTS], final_test["future_revenue_gbp"].to_numpy(dtype=float)
    artifact_root = root / "artifacts" / "runs" / "90d" / "revenue"
    artifact_root.mkdir(parents=True, exist_ok=True)
    metric_rows = []
    statuses = {}
    development_errors = {}
    candidates = revenue_candidates()
    baseline_names = [
        "training_mean_baseline", "training_median_baseline",
        "historical_spend_rate_baseline", "recent_90d_spend_baseline",
    ]
    all_names = [*baseline_names, *candidates]
    fold_metrics = []
    print(
        f"Evaluating {len(all_names)} 90-day revenue candidates across "
        f"{len(development_cutoffs)} rolling development origins...",
        flush=True,
    )
    for cutoff in development_cutoffs:
        train = population.loc[
            population["cutoff"].lt(cutoff)
            & population["label_mature"].eq(True)
        ].reset_index(drop=True)
        development = population.loc[population["cutoff"].eq(cutoff)].reset_index(drop=True)
        if train.empty or development.empty:
            raise ValueError(f"No matured training data or evaluation rows at development origin {cutoff.date()}.")
        x_train, y_train = train[MODEL_INPUTS], train["future_revenue_gbp"].to_numpy(dtype=float)
        x_dev, y_dev = development[MODEL_INPUTS], development["future_revenue_gbp"].to_numpy(dtype=float)
        cutoff_text = str(cutoff.date())
        print(f"  Origin {cutoff_text}: {len(train):,} matured training snapshots, {len(development):,} evaluation customers", flush=True)

        for name, prediction in _revenue_baselines(x_train, y_train, x_dev, 90, 90).items():
            run_dir = artifact_root / name
            run_dir.mkdir(parents=True, exist_ok=True)
            statuses[name] = {"status": "completed", "kind": "training-only arithmetic baseline"}
            dev_output = development[["customer_id", "cutoff"]].copy()
            dev_output["actual_revenue_gbp"] = y_dev
            dev_output["predicted_revenue_gbp"] = prediction
            dev_output.to_parquet(run_dir / f"development_{cutoff_text}_predictions.parquet", index=False)
            result = _metric_record("future_revenue_90d", name, "development", cutoff_text, revenue_metrics(y_dev, prediction, CONTACT_CAPACITY))
            metric_rows.append(result)
            fold_metrics.append(result)
            print(f"    {name}: MAE=GBP {result['mae_gbp']:,.2f}, bias={result['aggregate_bias_pct']:.1%}", flush=True)

        for name in candidates:
            run_dir = artifact_root / name
            run_dir.mkdir(parents=True, exist_ok=True)
            try:
                model = revenue_candidates()[name]
                model.fit(x_train, y_train)
                pred_dev = np.maximum(model.predict(x_dev), 0)
                joblib.dump(model, run_dir / f"development_model_{cutoff_text}.joblib", compress=3)
                dev_output = development[["customer_id", "cutoff"]].copy()
                dev_output["actual_revenue_gbp"] = y_dev
                dev_output["predicted_revenue_gbp"] = pred_dev
                dev_output.to_parquet(run_dir / f"development_{cutoff_text}_predictions.parquet", index=False)
                result = _metric_record("future_revenue_90d", name, "development", cutoff_text, revenue_metrics(y_dev, pred_dev, CONTACT_CAPACITY))
                metric_rows.append(result)
                fold_metrics.append(result)
                statuses[name] = {"status": "completed", "development_origins_completed": len([r for r in fold_metrics if r["model"] == name])}
                print(f"    {name}: MAE=GBP {result['mae_gbp']:,.2f}, bias={result['aggregate_bias_pct']:.1%}", flush=True)
            except Exception as exc:
                development_errors.setdefault(name, []).append(f"{cutoff_text}: {type(exc).__name__}: {exc}")
                print(f"    {name}: failed ({type(exc).__name__}: {exc})", flush=True)

    for name in candidates:
        completed_origins = int(pd.DataFrame(fold_metrics).loc[lambda frame: frame["model"].eq(name), "cutoff"].nunique())
        if completed_origins == len(development_cutoffs):
            statuses[name] = {"status": "completed", "development_origins_completed": completed_origins}
        else:
            statuses[name] = {
                "status": "failed_development_validation",
                "development_origins_completed": completed_origins,
                "errors": development_errors.get(name, []),
            }

    development_scores = pd.DataFrame(fold_metrics).groupby("model", as_index=False).agg(
        mean_origin_mae_gbp=("mae_gbp", "mean"),
        mean_absolute_origin_bias_pct=("aggregate_bias_pct", lambda values: values.abs().mean()),
        mean_signed_origin_bias_pct=("aggregate_bias_pct", "mean"),
        mean_origin_rmse_gbp=("rmse_gbp", "mean"),
        mean_origin_revenue_capture=("revenue_capture_at_capacity", "mean"),
        development_origins=("cutoff", "nunique"),
    )
    development_scores = development_scores.loc[
        development_scores["development_origins"].eq(len(development_cutoffs))
    ].sort_values(["mean_origin_mae_gbp", "mean_absolute_origin_bias_pct"], kind="stable")
    if development_scores.empty:
        raise RuntimeError("No 90-day revenue model completed on every development origin.")
    champion = str(development_scores.iloc[0]["model"])
    for row in development_scores.to_dict("records"):
        metric_rows.append({
            "task": "future_revenue_90d", "model": row["model"],
            "split": "development_summary", "cutoff": "|".join(str(c.date()) for c in development_cutoffs),
            "mae_gbp": row["mean_origin_mae_gbp"],
            "rmse_gbp": row["mean_origin_rmse_gbp"],
            "aggregate_bias_pct": row["mean_signed_origin_bias_pct"],
            "revenue_capture_at_capacity": row["mean_origin_revenue_capture"],
            "mean_absolute_origin_bias_pct": row["mean_absolute_origin_bias_pct"],
            "n_origins": int(row["development_origins"]),
        })
    # Refit on all mature snapshots strictly before the frozen September 2011
    # test origin. The June cohort's 90-day target is complete by 2011-09-01.
    refit = population.loc[
        population["cutoff"].lt(pd.Timestamp(FINAL_TEST_CUTOFF))
        & population["label_mature"].eq(True)
        & population["horizon_end_exclusive"].le(pd.Timestamp(FINAL_TEST_CUTOFF))
    ].reset_index(drop=True)
    predictions = {}
    refit_features = refit[MODEL_INPUTS]
    refit_targets = refit["future_revenue_gbp"].to_numpy(dtype=float)
    scoring_baselines = _revenue_baselines(refit_features, refit_targets, x_test, 90, 90)
    for name, prediction in scoring_baselines.items():
        run_dir = artifact_root / name
        (run_dir / "scoring_baseline.json").write_text(json.dumps({
            "training_median_gbp": float(np.median(refit_targets)),
            "training_mean_gbp": float(np.mean(refit_targets)),
            "source_sha256": manifest["source_sha256"],
        }, indent=2), encoding="utf-8")
        scored = final_test[["customer_id", "cutoff"]].copy()
        scored["actual_revenue_gbp"] = y_test
        scored["predicted_revenue_gbp"] = prediction
        scored.to_parquet(run_dir / "final_test_predictions.parquet", index=False)
        predictions[name] = prediction
        metric_rows.append(_metric_record("future_revenue_90d", name, "final_test", FINAL_TEST_CUTOFF, revenue_metrics(y_test, prediction, CONTACT_CAPACITY)))
    for name, status in list(statuses.items()):
        if status.get("status") != "completed":
            continue
        try:
            if name not in candidates:
                continue
            model = revenue_candidates()[name]
            model.fit(refit[MODEL_INPUTS], refit["future_revenue_gbp"].to_numpy(dtype=float))
            pred = np.maximum(model.predict(x_test), 0)
            run_dir = artifact_root / name
            joblib.dump(model, run_dir / "scoring_model.joblib", compress=3)
            scored = final_test[["customer_id", "cutoff"]].copy()
            scored["actual_revenue_gbp"] = y_test
            scored["predicted_revenue_gbp"] = pred
            scored.to_parquet(run_dir / "final_test_predictions.parquet", index=False)
            metric_rows.append(_metric_record("future_revenue_90d", name, "final_test", FINAL_TEST_CUTOFF, revenue_metrics(y_test, pred, CONTACT_CAPACITY)))
            predictions[name] = pred
            print(f"  {name}: frozen September test MAE=GBP {metric_rows[-1]['mae_gbp']:,.2f}, bias={metric_rows[-1]['aggregate_bias_pct']:.1%}", flush=True)
        except Exception as exc:
            statuses[name] = {"status": "refit_failed", "error": f"{type(exc).__name__}: {exc}"}
            print(f"  {name}: scoring refit failed ({type(exc).__name__}: {exc})", flush=True)
    result = pd.DataFrame(metric_rows)
    result.to_csv(root / "reports" / "revenue_90d_model_comparison.csv", index=False)
    score = final_test[["customer_id", "cutoff", "future_revenue_gbp"]].rename(columns={"future_revenue_gbp": "actual_revenue_gbp"}).copy()
    for name, values in predictions.items():
        score[name] = values
    score.to_parquet(root / "data" / "processed" / "final_customer_scores_90d_revenue.parquet", index=False)
    champions_path = root / "artifacts" / "champions.json"
    champions = json.loads(champions_path.read_text(encoding="utf-8")) if champions_path.exists() else {}
    champions["future_revenue_90d"] = {
        "task": "90-day gross merchandise revenue forecast in GBP",
        "metric": "mean MAE across March and June development origins; mean absolute origin bias is the tie-breaker",
        "development_cutoffs": [str(cutoff.date()) for cutoff in development_cutoffs],
        "selected_model": champion,
        "test_cutoff": FINAL_TEST_CUTOFF,
        "test_used_for_selection": False,
        "all_candidates_retained": True,
        "source_sha256": manifest["source_sha256"],
        "model_status": statuses,
        "baseline_comparison_included": True,
    }
    champions_path.write_text(json.dumps(champions, indent=2, allow_nan=False), encoding="utf-8")
    print(f"90-day revenue champion selected across March and June development origins: {champion}", flush=True)
    return result


def fit_bg_nbd_comparison(root: Path) -> pd.DataFrame:
    """Fit behavioral repeat-purchase/spend models from each forecast origin."""
    from .config import FORECAST_CUTOFFS

    features, targets90, targets6m, manifest = load_prepared(root)
    sales = pd.read_parquet(root / "data" / "interim" / "purchase_lines.parquet")
    reports = []
    model_root = root / "artifacts" / "runs" / "behavioral_clv"
    model_root.mkdir(parents=True, exist_ok=True)
    for cutoff_text in FORECAST_CUTOFFS:
        cutoff = pd.Timestamp(cutoff_text)
        summary = customer_day_summary(sales, cutoff)
        run_dir = model_root / cutoff_text
        run_dir.mkdir(parents=True, exist_ok=True)
        gamma = None
        gg_status = "not_fit"
        try:
            repeat = summary["frequency"].gt(0) & summary["repeat_mean_daily_spend"].notna()
            gamma = GammaGamma().fit(summary.loc[repeat, "frequency"], summary.loc[repeat, "repeat_mean_daily_spend"])
            gg_status = "completed" if gamma.finite_population_mean_ else f"q={gamma.q_:.5g} <= 1; population mean not finite"
        except Exception as exc:
            gg_status = f"failed: {type(exc).__name__}: {exc}"
        try:
            bgnbd = BetaGeoNBD().fit(summary["frequency"], summary["recency_months"], summary["age_months"])
            if gamma is None:
                raise ValueError(gg_status)
            mean_spend = summary["repeat_mean_daily_spend"].fillna(0.0)
            conditional_spend = pd.Series(
                gamma.expected_average_spend(summary["frequency"], mean_spend),
                index=summary.index,
            )
            for horizon_name, horizon_end, targets in (
                ("90d", cutoff + pd.Timedelta(days=90), targets90),
                ("6m", cutoff + pd.DateOffset(months=6), targets6m),
            ):
                labels = targets.loc[
                    targets["cutoff"].eq(cutoff) & targets["label_mature"].eq(True)
                ].set_index("customer_id")
                if labels.empty:
                    continue
                horizon_days = (horizon_end - cutoff).days
                frequency = summary["frequency"].reindex(labels.index).to_numpy(dtype=float)
                recency = summary["recency_months"].reindex(labels.index).to_numpy(dtype=float)
                age = summary["age_months"].reindex(labels.index).to_numpy(dtype=float)
                mean_spend_for_label = conditional_spend.reindex(labels.index).to_numpy(dtype=float)
                expected_days = bgnbd.expected_purchases(horizon_days / MONTH_DAYS, frequency, recency, age)
                predicted = np.maximum(np.nan_to_num(expected_days * mean_spend_for_label, nan=0, posinf=0), 0)
                paired = labels.copy()
                paired["predicted_revenue_gbp"] = predicted
                metrics = revenue_metrics(paired["future_revenue_gbp"], paired["predicted_revenue_gbp"], CONTACT_CAPACITY)
                spend_model = (
                    "BG_NBD_x_Gamma_Gamma" if gamma.finite_population_mean_
                    else "BG_NBD_x_Gamma_Gamma_repeat_buyers_plus_empirical_one_time_mean"
                )
                reports.append(_metric_record(
                    "future_revenue_" + horizon_name, spend_model, "forecast_origin", cutoff_text,
                    metrics,
                    bg_nbd_a=float(bgnbd.a_), bg_nbd_finite_expectation=bool(bgnbd.finite_lifetime_expectation_),
                    gamma_gamma_q=float(gamma.q_), gamma_gamma_finite_mean=bool(gamma.finite_population_mean_),
                    gamma_gamma_population_mean_gbp=(float(gamma.population_mean_) if np.isfinite(gamma.population_mean_) else None),
                    gamma_gamma_spend_mode=("population_mean_and_individual_posterior" if gamma.finite_population_mean_ else "repeat_buyer_posterior_plus_empirical_one_time_mean"),
                    repeat_buyers_for_gamma=int(gamma.n_customers_),
                ))
                paired.reset_index().assign(cutoff=cutoff, horizon=horizon_name).to_parquet(
                    run_dir / f"{horizon_name}_predictions.parquet", index=False,
                )
            joblib.dump({"bg_nbd": bgnbd, "gamma_gamma": gamma}, run_dir / "model.joblib", compress=3)
            (run_dir / "fit_summary.json").write_text(json.dumps({
                "cutoff": cutoff_text,
                "bg_nbd": {"r": float(bgnbd.r_), "alpha_per_month": float(bgnbd.alpha_), "a": float(bgnbd.a_), "b": float(bgnbd.b_), "negative_log_likelihood": bgnbd.negative_log_likelihood_, "optimizer_success": bgnbd.optimizer_success_, "finite_lifetime_expectation": bgnbd.finite_lifetime_expectation_},
                "gamma_gamma": {"p": gamma.p_, "q": gamma.q_, "v_gbp": gamma.v_, "population_mean_gbp": gamma.population_mean_ if np.isfinite(gamma.population_mean_) else None, "optimizer_success": gamma.optimizer_success_, "finite_population_mean": gamma.finite_population_mean_, "repeat_buyers": gamma.n_customers_},
                "source_sha256": manifest["source_sha256"],
            }, indent=2, allow_nan=False), encoding="utf-8")
            stale_failure = run_dir / "fit_failure.json"
            if stale_failure.exists():
                stale_failure.unlink()
            print(f"  BG/NBD + Gamma-Gamma {cutoff_text}: a={bgnbd.a_:.3g}, q={gamma.q_:.3g}, finite population mean={gamma.finite_population_mean_}", flush=True)
        except Exception as exc:
            failure = {
                "task": "behavioral_clv", "model": "BG_NBD_x_Gamma_Gamma",
                "split": "forecast_origin", "cutoff": cutoff_text,
                "status": "failed_or_invalid_assumption",
                "error": f"{type(exc).__name__}: {exc}",
                "gamma_gamma_status": gg_status,
            }
            reports.append(failure)
            (run_dir / "fit_failure.json").write_text(json.dumps(failure, indent=2, allow_nan=False), encoding="utf-8")
            print(f"  BG/NBD + Gamma-Gamma {cutoff_text}: skipped ({type(exc).__name__}: {exc})", flush=True)
    result = pd.DataFrame(reports)
    result.to_csv(root / "reports" / "behavioral_clv_comparison.csv", index=False)
    return result

