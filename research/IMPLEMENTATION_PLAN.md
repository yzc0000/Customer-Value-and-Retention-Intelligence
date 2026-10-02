# Implementation plan and status

The historical local prototype runs end to end, including the combined customer retention review. This note records the implementation and saved evidence; it does not imply live deployment or measured campaign impact. Detailed results and caveats are in the [README model results](../README.md#model-results).

## Delivered

1. **Data policy and ingestion.** Reads both worksheets, preserves row-level provenance, reconciles exact overlaps by maximum observed multiplicity across worksheets, and keeps repeated identical lines within each worksheet. Merchandise sales, credits, anonymous candidates, cancellations, and excluded service-like codes are accounted for in the preparation manifest.
2. **Cutoff snapshots and targets.** Produces 24,866 customer-cutoff rows across six quarterly snapshots; builds RFM, recency, order/day, product diversity, spend windows, credit proxies, and 26-by-6 weekly sequences. Future targets are independently built using end-exclusive windows and explicit maturity flags.
3. **Predictive comparisons.** Fits prevalence, recency logistic, logistic regression, spline GAM, Random Forest, XGBoost, pooled weekly purchase hazard, MLP, GRU, and LSTM for inactivity. Revenue models compare training-only baselines, Tweedie GLM, Gamma hurdle GLM, Random Forest, and XGBoost at 90-day and six-calendar-month horizons. BG/NBD and Gamma-Gamma are fit by forecast origin.
4. **Segmentation.** Compares PCA and non-PCA K-Means candidates across three random seeds; selects a stable two-group PCA K-Means solution. Fits a Gaussian Mixture challenger with five soft components by BIC.
5. **Saved outputs and interface.** Keeps model files, run metadata, customer predictions, comparison CSVs, a champion registry, and a local Streamlit reader. The raw source workbook and brief remain unchanged.
6. **Expanded revenue comparison.** Executes eleven statistical, additive and probabilistic recipes across 132 model/origin evaluations; preserves every candidate and keeps expected-revenue selection separate from the original MAE registry. Six-month selection excludes labels unavailable by June.
7. **Combined retention review.** Joins 5,224 September customers with LSTM risk, expected 90-day revenue and June-fitted segment assignments. Adds risk/value quadrants, filtered capacity, four deterministic rankings, customer inspection and matching CSV exports. No observed future outcomes enter review records or ranking. See research/RETENTION_REVIEW.md.

## Reproduce

Use Python 3.14 and the pinned packages in requirements-project.txt. The completed end-to-end pipeline is:

    python scripts/project.py all

Run stages independently with prepare, inactivity, revenue, clv, segments, or retention. The optional expanded revenue comparison runs with `python scripts/revenue_challengers.py`; run `python scripts/project.py retention` afterward to refresh the combined view. Launch the interface with:

    python -m streamlit run app/app.py

CatBoost is an optional conditional benchmark; it was unavailable in the active environment because the package is not installed. The other model families ran.

## Current champion choices

- 90-day inactivity: weekly LSTM is the project lead for explicit temporal modeling. Recency-only logistic regression remains the March 2011 winner on precision among the top 10% risk-ranked customers; the lead-model choice does not claim a metric win.
- 90-day gross revenue: original MAE champion XGBoost Tweedie; expanded expected-revenue candidate enriched monthly XGBoost. The combined review uses the expanded candidate in the current workspace. September bias improves from -42.7% to -25.5%, but remains material; this later comparison is retrospective.
- Six-calendar-month gross revenue: original MAE champion recent-180-day spend; expanded expected-revenue candidate BG/NBD × Gamma-Gamma. Corrected June MAE is GBP 574.59 for recent spending, GBP 585.21 for XGBoost and GBP 570.55 for BG/NBD. These June forecasts are not joined to September review records.
- Segmentation: PCA plus K-Means with two clusters; five-component GMM remains as the soft-membership alternative. The app visualizes the PCA map, profiles and retrospective outcomes. Segment membership also filters the combined review, but is not a predictive model input.
- Review ranking: default inactivity probability × historical average order value. Risk-only, forecast-value-only and historical-spend-only rankings are available at the same capacity. There is no measured campaign-response winner.

The selection metrics and frozen-test results are in reports/ and are not reused for model selection.

## Known limits and follow-up work

- The workbook is historical, ends on 9 December 2011, and cannot support a current live model or intervention-lift claim.
- The 90-day revenue champion has material test-period aggregate bias. Add more forward origins or newer retail data before depending on its point estimates.
- BG/NBD × Gamma-Gamma is a promising six-month statistical challenger; its early BG/NBD parameter fit reaches an optimizer boundary, so add sensitivity checks before treating it as champion.
- Retailer-wide ARIMA/ETS forecasting is explicitly out of scope; revenue prediction remains customer-level for value estimation and retention prioritization.
- BG/NBD fitted a below one, which rules out finite infinite-horizon expected transactions. The implementation reports only finite 90-day and six-month forecasts. Some early fits reach the lower parameter bound; review sensitivity before operational use.
- Revenue is gross sales rather than profit. Credits are not joined back to sales, and anonymous purchases cannot be scored.
- CatBoost was not installed. Its inactivity candidate is marked unavailable in the run status.
- Per-customer model explanations and narrative modeling notebooks remain follow-up work. The repository has not been initialized or pushed from this workspace.
- A deployment API, automated retraining, causal campaign evaluation and external model tracking remain optional extensions; they were not part of this local historical prototype.

See the [README model results](../README.md#model-results) for metrics, data counts, interpretation and output locations.
