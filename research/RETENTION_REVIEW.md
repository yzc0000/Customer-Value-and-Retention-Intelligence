# Combined customer retention review

The local app combines customer segments, predicted inactivity and expected customer revenue into a capacity-limited review workflow. This layer consumes saved predictions; it neither retrains models nor removes competitors.

## Population and sources

One row represents one identified customer at the 1 September 2011 cutoff. All 5,224 eligible customers have complete, unique keys in customer history, inactivity predictions, revenue predictions and segment assignments. The join rejects missing customers, duplicates and mismatched forecast dates.

| Signal | Current source | Meaning |
| --- | --- | --- |
| Inactivity | Calibrated weekly LSTM | Probability of no qualifying purchase over the next 90 days |
| Expected revenue | Enriched monthly XGBoost, expanded expected-value selection | Unconditional expected gross merchandise sales in GBP over the same 90 days |
| Segment | June-fitted PCA/K-Means, two groups | Purchase-history profile assigned to September customers |
| Alternative segment | Five-component GMM | Assignment and maximum membership, available in customer inspection |
| Historical value | Customer features strictly before cutoff | Average qualifying order value and recent spending |

The shared outcome window is `[2011-09-01, 2011-11-30)`. Separate six-calendar-month predictions originate in June and are not joined to September records. The review builder selects the expanded expected-revenue candidate when its selector and scores exist; otherwise it uses the original 90-day revenue champion. The interface names the source models.

The artifact selects input columns explicitly. Future inactivity labels, actual future revenue and retrospective segment outcome summaries do not enter review records, filtering, ranking or ranking comparisons. Model evaluation remains available in the other tabs.

## Review rule and groups

The default rule is:

`review_priority_score = calibrated inactivity probability × historical average order value`

This prioritizes a combination of predicted inactivity and a historical order-value proxy. It is not expected lost revenue, campaign recovery, treatment response or profit. Expected 90-day revenue remains a separate forecast; no division by an independently modeled purchase probability creates a conditional spending estimate.

Four selectable rankings use the same filtered population and capacity:

- Risk × historical average order value.
- Inactivity probability only.
- Expected 90-day revenue only.
- Historical 365-day spend only.

Sorting is deterministic: selected score descending, remaining risk and expected-revenue tie-breakers descending, then customer ID ascending. Capacity applies **after** filters: the queue contains `ceil(filtered customers × capacity fraction)` customers. Default capacity is 10%; zero capacity preserves the filtered population with an empty queue.

Four risk/value groups compare inactivity probability with 0.5 and historical average order value with GBP 416.26. The value boundary is the June 2011 cohort's 75th percentile, computed before September and held fixed during filtering. Boundary values belong to the higher group. Both thresholds can be adjusted, and their actual values are included in exports. Groups describe risk and prior order size; their names do not assert permanent churn or campaign responsiveness.

## Interface and export behavior

The **Retention review** tab supports segment, country, risk/value group, minimum inactivity probability and literal customer-ID filters, plus a reset control. Metrics, scatter points, group counts, review rows, alternative rankings and CSV exports all use the resulting population. Larger scatter points indicate queued customers; tooltips retain customer-level values and segment details. The logarithmic value axis accommodates the broad order-value range without truncating large customers.

The on-screen queue shows its first 100 customers. The queue download contains every selected customer; the population download contains every filtered customer, with ranks and queue flags. Both preserve customer/cutoff keys, horizon, model identities, prior history, predictions, actual group thresholds, selected ranking, capacity and filtered population count. A customer inspector explains the saved fields for an individual queued customer.

## Current default comparison

These summaries use the complete September cohort and 523 customers per ranking. They describe predictions and history, not observed intervention results.

| Ranking | Mean predicted inactivity | Median historical order | Sum expected 90-day gross revenue |
| --- | ---: | ---: | ---: |
| Risk × historical order value | 79.6% | GBP 748.23 | GBP 151,859.57 |
| Risk only | 96.3% | GBP 176.10 | GBP 14,049.59 |
| Expected revenue only | 16.3% | GBP 504.36 | GBP 1,298,952.74 |
| Historical 365-day spending | 18.9% | GBP 525.88 | GBP 1,276,898.27 |

The default rule includes customers with larger historical orders than risk-only ranking, at a lower mean predicted inactivity probability. Forecast-value ranking concentrates on customers the revenue model expects to buy. These are explicit review-policy tradeoffs; the data cannot identify which policy would produce the best incremental campaign return.

## Reproduction and provenance

After preparing data, fitting inactivity/revenue and assigning segments:

```powershell
python scripts/project.py retention
python -m streamlit run app/app.py
```

`python scripts/project.py all` also builds this layer last. After the optional expanded comparison, rerun `retention` to consume its scores. Saved outputs are:

- `data/processed/customer_retention_review_90d.parquet`: complete ranked cohort.
- `data/processed/retention_review_priority_90d.csv`: default review queue.
- `reports/retention_policy_comparison.csv`: same-capacity ranking summaries.
- `reports/retention_review_summary.csv`: segment/group aggregates.
- `artifacts/runs/retention/manifest.json`: forecast dates, grain, model identities, definitions, defaults, source workbook identity and input/output SHA-256 hashes.

The app rejects changed or missing source files and directs the user to rebuild. It also verifies the combined output hash. Existing fitted models and their registries remain unchanged.

Seven targeted tests cover key-based joins, outcome exclusion, date/coverage rejection, filtered capacity, deterministic ties, group boundaries, empty/zero-capacity states and invalid probabilities. The existing eight tests also pass. Streamlit interaction checks verify filters, alternative ranking, reset and empty states; native Chrome verifies desktop/narrow charts and a fresh 523-row download.

This is a historical retrospective workflow for a workbook ending in December 2011. Revenue forecasts retain material bias, and there are no intervention, margin or campaign-cost data. Live use or incremental profit optimization requires newer data and campaign-response evidence.
