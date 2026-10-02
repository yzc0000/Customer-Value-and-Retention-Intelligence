"""PCA/K-Means segment discovery and a soft GMM challenger."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .config import SEED
from .pipeline import load_prepared

SEGMENT_FEATURES = [
    "recency_days", "purchase_day_count", "purchase_count", "lifetime_revenue_gbp",
    "avg_order_value_gbp", "avg_basket_units", "product_diversity",
    "mean_days_between_purchase_days", "spend_90d_gbp", "spend_365d_gbp",
]


def _log_nonnegative(values):
    return np.log1p(np.maximum(values, 0))


def _preprocessor(use_pca: bool, variance: float = 0.9) -> Pipeline:
    steps = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("log", FunctionTransformer(_log_nonnegative, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=variance, svd_solver="full", random_state=SEED)))
    return Pipeline(steps)


def fit_segmentation(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features, targets90, _, manifest = load_prepared(root)
    calibration_cutoff = pd.Timestamp("2011-06-01")
    score_cutoff = pd.Timestamp("2011-09-01")
    training = features.loc[features["cutoff"].eq(calibration_cutoff)].reset_index(drop=True)
    scoring = features.loc[features["cutoff"].eq(score_cutoff)].reset_index(drop=True)
    x_train = training[SEGMENT_FEATURES]
    x_score = scoring[SEGMENT_FEATURES]
    root_artifact = root / "artifacts" / "runs" / "segments"
    root_artifact.mkdir(parents=True, exist_ok=True)
    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows = []
    choices: list[tuple[float, int, Pipeline, KMeans, np.ndarray]] = []
    for use_pca in (True, False):
        prep = _preprocessor(use_pca)
        z = prep.fit_transform(x_train)
        for k in range(2, 9):
            labels_by_seed = []
            sils = []
            dbs = []
            fit_models = []
            for seed in (SEED, SEED + 1, SEED + 2):
                model = KMeans(n_clusters=k, n_init=20, random_state=seed, max_iter=500)
                labels = model.fit_predict(z)
                share = pd.Series(labels).value_counts(normalize=True)
                if share.min() < 0.025:
                    sil, dbi = np.nan, np.nan
                else:
                    sil = float(silhouette_score(z, labels, sample_size=min(len(labels), 3500), random_state=SEED, n_jobs=1))
                    dbi = float(davies_bouldin_score(z, labels))
                labels_by_seed.append(labels)
                sils.append(sil)
                dbs.append(dbi)
                fit_models.append(model)
            ari = [adjusted_rand_score(labels_by_seed[0], labels) for labels in labels_by_seed[1:]]
            mean_sil = float(np.nanmean(sils))
            min_group = min(float(pd.Series(labels).value_counts(normalize=True).min()) for labels in labels_by_seed)
            record = {
                "model": "kmeans_pca" if use_pca else "kmeans_no_pca",
                "k": k,
                "pca": use_pca,
                "pca_components": int(z.shape[1]),
                "mean_silhouette": mean_sil,
                "mean_davies_bouldin": float(np.nanmean(dbs)),
                "mean_seed_ari": float(np.mean(ari)),
                "minimum_segment_share": min_group,
                "three_seed_labels": labels_by_seed,
            }
            candidate_rows.append({key: value for key, value in record.items() if key != "three_seed_labels"})
            if np.isfinite(mean_sil) and min_group >= 0.025:
                # Penalize unstable splits slightly while retaining silhouette as the main geometric signal.
                score = mean_sil + 0.10 * float(np.mean(ari))
                choices.append((score, len(candidate_rows) - 1, prep, fit_models[0], labels_by_seed[0]))

    candidate_table = pd.DataFrame(candidate_rows)
    candidate_table.to_csv(report_dir / "segment_candidates.csv", index=False)
    if not choices:
        raise RuntimeError("No K-Means solution passed the minimum segment size rule.")
    _, index, selected_transform, selected_model, train_labels = max(choices, key=lambda choice: choice[0])
    selected_row = candidate_table.iloc[index]
    train_coordinates = np.asarray(selected_transform.transform(x_train))
    score_coordinates = np.asarray(selected_transform.transform(x_score))
    score_labels = selected_model.predict(score_coordinates)
    pca = selected_transform.named_steps.get("pca")
    explained = pca.explained_variance_ratio_ if pca is not None else None
    selected_spec = {
        "algorithm": str(selected_row["model"]),
        "k": int(selected_row["k"]),
        "pca_components": int(selected_row["pca_components"]),
        "projection_variance_ratio": [float(value) for value in explained[:2]] if explained is not None else None,
        "train_cutoff": str(calibration_cutoff.date()),
        "score_cutoff": str(score_cutoff.date()),
        "selection": "mean silhouette + 0.10 * mean three-seed adjusted Rand stability; minimum segment share >=2.5%",
        "source_sha256": manifest["source_sha256"],
    }
    joblib.dump({"transform": selected_transform, "model": selected_model, "spec": selected_spec}, root_artifact / "kmeans_champion.joblib", compress=3)
    (root_artifact / "kmeans_champion.json").write_text(json.dumps(selected_spec, indent=2), encoding="utf-8")

    scored = scoring[["customer_id", "cutoff"]].copy()
    scored["kmeans_segment"] = score_labels.astype(int)
    segmented_history = training.assign(segment=train_labels)
    profiles = segmented_history.groupby("segment", sort=True)[SEGMENT_FEATURES].median()
    profiles.columns = [f"{feature}_median" for feature in profiles.columns]
    profiles.insert(0, "customers", segmented_history.groupby("segment").size())
    profiles["customer_share"] = profiles["customers"] / len(training)
    profiles.reset_index().to_csv(report_dir / "segment_profiles.csv", index=False)
    profile_medians = segmented_history.groupby("segment", sort=True)[SEGMENT_FEATURES].median()
    population_medians = training[SEGMENT_FEATURES].median()
    index_features = [feature for feature in SEGMENT_FEATURES if pd.notna(population_medians[feature]) and population_medians[feature] > 0]
    profile_index = profile_medians[index_features].div(population_medians[index_features], axis="columns")
    profile_index.insert(0, "segment", profile_index.index)
    profile_index.to_csv(report_dir / "segment_profile_index.csv", index=False)

    # GMM on the same PCA training representation supplies soft, not hard, membership.
    pca_transform = _preprocessor(True)
    gmm_train = pca_transform.fit_transform(x_train)
    gmm_rows = []
    gmm_choices = []
    for k in range(2, 7):
        gmm = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-3, n_init=2, max_iter=500, random_state=SEED)
        gmm.fit(gmm_train)
        bic = float(gmm.bic(gmm_train))
        gmm_rows.append({"k": k, "bic": bic, "converged": bool(gmm.converged_), "iterations": int(gmm.n_iter_)})
        gmm_choices.append((bic, gmm))
    gmm_bic, gmm = min(gmm_choices, key=lambda item: item[0])
    gmm_train_probabilities = gmm.predict_proba(gmm_train)
    gmm_probabilities = gmm.predict_proba(pca_transform.transform(x_score))
    scored["gmm_segment"] = gmm_probabilities.argmax(axis=1)
    scored["gmm_max_membership"] = gmm_probabilities.max(axis=1)
    scored.to_parquet(root / "data" / "processed" / "customer_segments.parquet", index=False)

    projection_explained = selected_spec["projection_variance_ratio"] or [None, None]
    training_projection = pd.DataFrame({
        "customer_id": training["customer_id"].to_numpy(),
        "cutoff": calibration_cutoff,
        "snapshot_role": "June 2011 fit/profile",
        "pc1": train_coordinates[:, 0],
        "pc2": train_coordinates[:, 1],
        "kmeans_segment": train_labels.astype(int),
        "gmm_segment": gmm_train_probabilities.argmax(axis=1).astype(int),
        "gmm_max_membership": gmm_train_probabilities.max(axis=1),
    })
    score_projection = pd.DataFrame({
        "customer_id": scoring["customer_id"].to_numpy(),
        "cutoff": score_cutoff,
        "snapshot_role": "September 2011 assignment",
        "pc1": score_coordinates[:, 0],
        "pc2": score_coordinates[:, 1],
        "kmeans_segment": score_labels.astype(int),
        "gmm_segment": gmm_probabilities.argmax(axis=1).astype(int),
        "gmm_max_membership": gmm_probabilities.max(axis=1),
    })
    pd.concat([training_projection, score_projection], ignore_index=True).to_parquet(
        root / "data" / "processed" / "segment_projection.parquet", index=False,
    )

    # Holdout outcomes describe how the June-fitted segments differ later; they
    # are diagnostic only and never feed cluster fitting or selection.
    holdout = targets90.loc[
        targets90["cutoff"].eq(score_cutoff) & targets90["label_mature"].eq(True),
        ["customer_id", "inactive", "future_revenue_gbp"],
    ]
    diagnostic = scored.merge(holdout, on="customer_id", how="inner", validate="one_to_one")
    outcome_tables = []
    for method, label_column in (("kmeans", "kmeans_segment"), ("gmm", "gmm_segment")):
        summary = diagnostic.groupby(label_column, sort=True).agg(
            customers=("customer_id", "size"),
            inactivity_rate=("inactive", "mean"),
            mean_future_revenue_gbp=("future_revenue_gbp", "mean"),
            median_future_revenue_gbp=("future_revenue_gbp", "median"),
            total_future_revenue_gbp=("future_revenue_gbp", "sum"),
        ).reset_index(names="segment")
        summary.insert(0, "method", method)
        outcome_tables.append(summary)
    pd.concat(outcome_tables, ignore_index=True).to_csv(report_dir / "segment_holdout_outcomes.csv", index=False)

    gmm_table = pd.DataFrame(gmm_rows)
    gmm_table.to_csv(report_dir / "gmm_candidates.csv", index=False)
    joblib.dump({"transform": pca_transform, "model": gmm, "bic": gmm_bic, "source_sha256": manifest["source_sha256"]}, root_artifact / "gmm_soft_membership.joblib", compress=3)
    print(f"Segments: selected {selected_spec['algorithm']} k={selected_spec['k']} from June 2011 histories; September customers scored.", flush=True)
    print(f"GMM soft membership: BIC selected k={gmm.n_components}.", flush=True)
    return candidate_table, scored
