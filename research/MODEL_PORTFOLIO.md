# Model portfolio, selection and preservation

This records the model portfolio, the experiments that have been run, and the candidates still worth testing. Results and current project choices are in the [README model results](../README.md#model-results). The weekly LSTM is the selected inactivity lead because the project is testing temporal sequence learning; the highest development P@10 metric winner is recorded separately. All successful fitted competitors and comparison metrics are retained in the local `artifacts/` and `reports/` outputs.

## Portfolio status

The project has distinct tasks. Revenue winners are selected separately by horizon, and segmentation is assessed by stability and useful profiles rather than predictive scores. The table distinguishes completed comparisons from future extensions.

| Family | Status and task | Why it fits | Main limitation |
| --- | --- | --- | --- |
| Logistic GAM | Implemented inactivity benchmark | Smooth nonlinear effects of recency, frequency, spend and purchase gaps | Limited interactions; validate effects and calibration |
| Pooled discrete-time purchase hazard | Implemented statistical inactivity benchmark | Estimates a horizon-specific next-purchase probability from time bins | It is not Cox or Random Survival Forest; the short follow-up limits survival extensions |
| Weekly MLP, GRU and LSTM | Implemented sequence comparison; LSTM is the project lead | Weekly activity can encode recency dynamics and quiet periods | Sparse, short histories; LSTM did not win development P@10 |
| LSTM with hurdle LogNormal output | Implemented revenue experiment for both horizons | Separates purchase probability from heavy-tailed positive spend using weekly and static history | Seasonal instability and distributional assumptions; separate from the selected inactivity LSTM |
| Hurdle NGBoost | Implemented probabilistic revenue competitor | Learns positive-spend distributions and combines them with purchase probability | Predictive intervals need coverage checks; no automatic accuracy advantage |
| Pareto/NBD with seasonal/customer covariates | Implemented finite-horizon statistical revenue competitor | Models latent purchase/dropout rates and a known Q4 purchase multiplier | Point fit; Gamma-Gamma spending remains time constant; first observed purchase can be left truncated |
| GAM hurdle / Tweedie GLM | Implemented revenue alternatives | Conditional spend model or direct zero-compatible nonnegative response | Current fits were unstable on some horizons; inspect value scale and bias |
| BG/NBD × Gamma-Gamma | Implemented per-customer finite-horizon statistical challenger | Models repeat-purchase frequency and spend for noncontractual customers | Early parameter fits reach an optimizer boundary; not a lifetime-value claim |
| Gaussian Mixture Model | Implemented soft segmentation challenger | Gives soft membership rather than a single sharp assignment | Gaussian geometry after transformation can still be a poor description |

Retain recency-only and regularized logistic baselines, Random Forest, BG/NBD + Gamma-Gamma, CatBoost and XGBoost. The customer purchase-frequency model is purpose-built for noncontractual repeat purchasing, so it is already a substantial alternative to boosting, rather than just another generic regressor. Diagnose its assumptions as specified in the main design.

## GAM experiment

Model inactivity with a penalized spline term for a small, nonredundant set of numeric predictors, factor terms for country groups, and initially additive effects. Fit imputation and transforms using training examples. Start with low spline complexity and tune smoothing using the historical development snapshot, rather than pyGAM's default internal objective alone.

Add at most a small development-selected interaction set, such as recency with frequency. Feature-effect plots show modeled associations on the log-odds/probability scale with explicitly stated conditioning; they do not establish that changing a feature causes retention. Correlated monetary/frequency inputs can make individual effects unstable, so compare reduced feature sets. Avoid presenting default confidence bands as if they accounted for customer dependence across snapshots.

For revenue, fit a Gamma GAM with a log link only to **positive returning-customer revenue**, and multiply its conditional mean by a return probability once. Zero-revenue rows belong in the return model, not in the Gamma response. A Tweedie GLM with `1 < power < 2` is a separate direct unconditional-revenue baseline that accommodates zeros and positive continuous values; do not multiply its output by return probability again.

Sources: [LogisticGAM and spline/factor terms](https://pygam.readthedocs.io/en/latest/reference/_autosummary/pygam.pygam.LogisticGAM.html), [pyGAM distributions and links](https://pygam.readthedocs.io/en/latest/notebooks/tour_of_pygam.html), [TweedieRegressor response support](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TweedieRegressor.html).

## Future extension: Cox or Random Survival Forest

The built statistical inactivity comparator is a pooled discrete-time purchase hazard. Cox and Random Survival Forest remain possible future benchmarks if their event/censoring definitions and temporal evaluation are implemented. The guidance below applies to that extension.

Use the forecast cutoff as the time origin, with historical features frozen at that cutoff. The event is the **first qualifying purchase in the future interval**, rather than permanent churn. For the initial fair comparison, cap follow-up at 90 days and use exactly the mature snapshots and customer keys used by classifiers.

- A purchase within `[cutoff, cutoff + 90 days)` is an observed event with elapsed time to that purchase.
- A customer with no purchase in that interval is administratively right-censored at 90 days; they are not known to have permanently left.
- A purchase at the horizon end is excluded, matching the target builder's end-exclusive rule. Preserve this boundary convention when converting survival curves to horizon risk.
- Horizon survival probability estimates no purchase through the follow-up horizon. Report it as 90-day inactivity probability using the matching boundary convention. `1 - S(90)` estimates return probability.

Fit penalized Cox as a simple survival benchmark and Random Survival Forest as the nonlinear challenger. Cox needs a baseline survival estimate to produce probabilities, not merely relative-risk scores; validate proportional-hazard adequacy. Forest raw risk scores are likewise not a substitute for survival probabilities.

At each fitting/forecast origin, use only event and censoring information available then. Do not label earlier rows as nonreturners using the workbook's final date, or start durations at first purchase while ignoring delayed entry into a snapshot risk set. Cutoff-based origins avoid that target mismatch. Same customers across dates still create dependence: bootstrap customers for uncertainty and report the existing new-to-training slice.

Compare `S(90)` against the classifiers with the same capacity metrics, Brier score, AP and calibration. Add survival-specific Brier curves/concordance as diagnostics; concordance alone does not select the best 90-day classifier. Censoring-aware metrics must use appropriately estimated censoring distributions and supported follow-up intervals. Applying ordinary classification calibration at 90 days requires mature labels; it does not automatically preserve a coherent calibrated curve across every horizon.

Partially observed recent cutoffs could be used in a later censoring-aware experiment, but separate that data-availability benefit from architecture comparisons. A longer 30/60/90-day survival-curve product also needs its own horizon validation rather than being assumed correct from one score.

Sources: [Random Survival Forest and survival predictions](https://scikit-survival.readthedocs.io/en/stable/user_guide/random-survival-forest.html), [penalized Cox models](https://scikit-survival.readthedocs.io/en/stable/user_guide/coxnet.html), [survival evaluation and censoring](https://scikit-survival.readthedocs.io/en/stable/user_guide/evaluating-survival-models.html).

## Neural revenue experiment

Start with a compact MLP on the tabular customer features, with a zero-inflated lognormal (ZILN) output, rather than making LSTM multitask immediately. This tests the revenue distribution independently of sequence architecture. The head estimates return probability and lognormal location/scale for positive revenue; its unconditional expected revenue is `p_return * exp(mu + sigma^2 / 2)`. Check positive finite scale, numerical stability, tail behavior and predicted/actual totals.

Use a constant/linear ZILN baseline or the GAM hurdle comparison to distinguish distribution choice from neural complexity. The positive-revenue probability corresponds to purchase return only because the merchandise target is strictly positive whenever a qualifying purchase occurs. This equivalence would change for a future net-revenue target with offsets from credits.

Compare the exact same 90-day revenue target with BG/NBD + Gamma-Gamma, simple spend estimates, Tweedie/GAM hurdle and boosting. Do not select a distribution solely for likelihood fit: evaluate mean GBP prediction, aggregate bias and customer ranking. Predictive intervals are model-based until held-out coverage is checked. LSTM/GRU revenue heads remain a later extension if the simple sequence experiment earns that complexity.

Source: [Wang et al., A Deep Probabilistic Model for Customer Lifetime Value Prediction](https://arxiv.org/abs/1912.07753). This paper supplies the ZILN approach; its results do not establish a winner for this workbook.

## Models to leave out of the initial portfolio

Do not add models only to enlarge the comparison table. Per-customer ARIMA/Prophet is poorly matched to many short, intermittent histories and a customer-level inactivity target. Retailer-wide time-series forecasting is explicitly outside this project's scope. A large Temporal Fusion Transformer targets richer multi-horizon forecasting problems and adds substantial machinery before there is evidence that a small network helps here. The brief's inputs have no images, text corpus or relationship graph supporting a vision model, LLM predictor or graph network.

A small tabular ResNet/FT-Transformer could be technically applicable, but is not a priority recommendation before GAM, survival and the compact neural controls. Small customer populations do not make every transformer impossible; this is a scoped experiment choice, not a universal claim about model performance. Likewise, SVM and ExtraTrees are possible secondary benchmarks but offer less new project direction than the recommended families.

Sources: [Temporal Fusion Transformer task and architecture](https://research.google/pubs/temporal-fusion-transformers-for-interpretable-multi-horizon-time-series-forecasting/), [tabular ResNet/FT-Transformer benchmark and absence of a universally superior family](https://arxiv.org/abs/2106.11959). Suitability and prioritization here are our inference from the audited dataset and project targets.

## Selection and preservation policy

Use time-ordered, mature outcomes for development and keep the final future-period test frozen. The 90-day revenue pipeline uses March and June development origins and selects by mean MAE; its September origin is test only. The six-month pipeline has one development origin because later labels are not mature before its June test. Choose metric winners using development evidence, and record an explicitly user-selected research lead separately when it differs from that winner. Choosing a model from September test performance would turn the test into selection data. If a lead model falls short on the primary metric, report that result and seek a fresh temporal evaluation before making a new validated performance claim.

Before fits, record one primary selection metric per task and supporting guardrails. Inactivity uses precision at a fixed contact capacity, with AP/recall and calibrated Brier/log loss reported alongside it. The original revenue portfolio selected on MAE. The [expanded expected-revenue experiment](REVENUE_CHALLENGER_RESULTS.md) instead fixes mean Tweedie deviance at power 1.5 before its run, with MAE, RMSE, aggregate bias and revenue capture as supporting diagnostics. Mean versus median forecast objectives can favor different models; retain their separate selection records. Six-month selection uses December only; the additional January-May origins have outcome windows overlapping the final June cohort and are retrospective robustness checks. Cluster selection uses stability, useful group sizes and interpretable profiles.

Keep every experiment's code, configuration, dataset/cleaning version, split manifest, seed, dependency versions, runtime, metrics and per-customer predictions. Keep fitted preprocessors, weights and calibrators for successful valid runs. Record failed fits and invalid outputs with reasons; preserve their metadata without labeling them usable models. Store large/customer-level artifacts outside Git history as already planned.

Proposed run structure:

```text
artifacts/runs/<task>/<run_id>/
    manifest.json
    model/                 # fitted estimator/weights and preprocessing
    metrics.json
    predictions.parquet
    diagnostics/
reports/model_comparison.csv
reports/model_cards/<run_id>.md
artifacts/champions.json    # project lead, metric winner and task-specific model status
```

The Streamlit app reads the selected project lead and exposes the full comparison report. Losing successful runs remain available for review, reproduction and later comparisons; the champion registry does not remove their files.

## Next experiments

1. Integrate BG/NBD × Gamma-Gamma into the six-month revenue development comparison after checking parameter sensitivity.
2. If time permits, test a regularized two-part customer revenue model and compact ZILN on the same temporal splits.
3. Keep every model family, report the metric winner and project lead distinctly, and retest on a newer source before operational use.

Do not add time-series dependencies for retailer-wide forecasting; that work is outside the project scope.
