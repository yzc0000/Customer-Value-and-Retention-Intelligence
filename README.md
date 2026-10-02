# Customer Value & Retention Intelligence

A historical customer analytics project using UCI Online Retail II. It combines customer segmentation, repeat-purchase inactivity prediction, and fixed-horizon customer revenue forecasts. The first Streamlit tab brings these outputs together in a capacity-limited retention review.

The project uses time-based snapshots and future outcomes so model inputs precede the periods being predicted. It preserves competing models and reports their comparisons rather than presenting the project lead as the winner of every metric.

## Get the data

The source workbook is not included in this repository. Download [Online Retail II from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii), then place `online_retail_II.xlsx` at the repository root. The workbook is distributed under CC BY 4.0; retain source attribution when sharing derived work.

The preparation pipeline reads both worksheets, reconciles exact cross-sheet overlap by maximum observed multiplicity, and preserves repeated identical invoice lines within a sheet. Merchandise sales are modeled separately from credits, anonymous purchases and excluded service-like codes. The workbook and customer-level data remain outside Git.

## Run the project

Python 3.14 is used in the active environment. From the repository root, create an environment and install the main requirements:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-project.txt
python scripts/project.py all
python -m streamlit run app/app.py
```

The `all` stage prepares features and targets, fits the original inactivity, revenue and behavioral customer lifetime value comparisons, fits segments, and builds the retention review. To run the expanded statistical and probabilistic revenue comparison, install its additional dependency and run the challenger script. Refresh the review afterward so it uses the expanded revenue selection:

```powershell
.venv\Scripts\python -m pip install -r requirements-revenue-challengers.txt
python scripts/revenue_challengers.py
python scripts/project.py retention
```

Individual pipeline stages are `prepare`, `inactivity`, `revenue`, `clv`, `segments` and `retention`. CatBoost is optional and was not installed in the evaluated environment.

The Streamlit app reads saved results. The source workbook, fitted models and customer-level predictions are excluded from Git; the full pipeline recreates them locally. The project does not provide a live prediction service.

## Data and evaluation design

| Item | Implemented scope |
| --- | --- |
| Source data | 1,067,371 worksheet rows; 22,523 excess exact overlaps reconciled; 1,044,848 retained source lines |
| Modeled purchase mart | 788,187 identifiable qualifying merchandise-sale lines |
| Customer snapshots | Six quarterly cutoffs, June 2010 through September 2011; 24,866 customer-cutoff rows |
| Final cohort | 5,224 identified customers with qualifying history by 1 September 2011 |
| Weekly sequence | 26 time steps and six channels per customer snapshot |
| Inactivity target | No qualifying purchase in the next 90 days |
| Revenue targets | Gross merchandise revenue over 90 days or exactly six calendar months |

Development, calibration and evaluation dates respect target availability. Inactivity models train on June, September and December 2010 snapshots, select on March 2011, calibrate probabilities on June 2011, and report the September 2011 frozen historical comparison. Revenue models use rolling forecast origins; target windows must finish by the forecast date before those labels may be used for fitting or selection. The September 2011 90-day test and June 2011 six-month test have already been examined in this project, so later model comparisons against them are retrospective.

## Model results

### Inactivity prediction

The model family includes prevalence and recency baselines, logistic regression, a spline GAM, Random Forest, pooled discrete-time purchase hazard, XGBoost, and weekly MLP, GRU and LSTM. CatBoost runs only when installed. The positive class is 90-day purchase inactivity.

The weekly LSTM is the project lead because it explicitly learns from customer purchase sequences. This is a modeling choice, not a claim that it won each metric. Recency-only logistic regression remains the March development winner for precision among the top 10% risk-ranked customers (89.4%). The LSTM's March average precision was 0.844 and P@10 was 88.7%; its September retrospective average precision was 0.819 and P@10 was 90.6%. The app keeps the metric comparison visible.

### Customer-level revenue forecasting

The target is expected gross merchandise revenue per identified customer over a specified future horizon. Retailer-wide time-series forecasting is outside scope. Each customer has only 26 weekly history bins; at the September cohort, 72.6% have at most one active week and 90.2% have at most three. That history is too sparse for stable customer-specific seasonal ARIMA or ETS fits.

The original portfolio compares recent-spend baselines, Tweedie GLMs, Gamma hurdle GLMs, Random Forest, XGBoost, and BG/NBD with Gamma-Gamma spending. It separately evaluates 90-day and six-calendar-month horizons. The original 90-day XGBoost Tweedie model was selected on mean MAE across March and June development origins: GBP 270.39 versus GBP 280.86 for recent 90-day spending. On the September retrospective test its MAE was GBP 373.47 versus GBP 391.44 for the recent-spend baseline, while aggregate revenue was underforecast by 42.7%.

The expanded comparison adds additive Tweedie and Gamma spline models, base and seasonal/covariate Pareto/NBD with Gamma-Gamma spending, hurdle NGBoost and a probabilistic revenue LSTM. Eleven recipes completed 132 model-origin evaluations. Because expected customer value calls for a mean forecast, this comparison selects with a fixed mean-oriented Tweedie criterion; it keeps the original MAE selections separate.

| Horizon | Expanded selection evidence | Retrospective cohort comparison |
| --- | --- | --- |
| 90 days: enriched monthly XGBoost | Across four eligible development origins, mean deviance 45.89 versus 47.23 for the original reference; mean MAE GBP 390.47 versus GBP 399.47 | MAE GBP 362.57 versus GBP 373.47; RMSE GBP 1,832.57 versus GBP 2,119.86; aggregate bias -25.5% versus -42.7% |
| Six months: BG/NBD x Gamma-Gamma | December is the only six-month selection outcome complete by the final June date. MAE GBP 656.77 versus GBP 728.95 for recent spending | June MAE GBP 570.55 versus GBP 574.59; RMSE GBP 2,542.49 versus GBP 3,059.67; aggregate bias -12.9% versus -24.3% |

The six-month forecasts originate in June; they are not joined to the September 90-day retention review. January-May six-month comparisons are retrospective because their outcome windows extend past the June forecast origin and overlap the test target. Seasonal Pareto/NBD performs well across those broader retrospective checks but cannot be selected from them. Revenue LSTM and GAM recipes were unstable across origins.

A separate 42-configuration study found a nonlinear Gamma hurdle with 5.6% lower mean development MAE than the original reference, but worse RMSE. Monthly training improved mean absolute origin bias to 1.3% while keeping MAE near the reference. These experiments did not change the expanded selection or receive a new September test.

### Customer segmentation

PCA plus K-Means with two groups was selected using June 2011 history. June silhouette was 0.321 and mean adjusted Rand stability across three seeds was 0.995. A five-component Gaussian Mixture remains as a soft-membership alternative. June-fitted assignments are applied to September customers; future outcomes do not form or select the clusters. Segments describe purchasing history and do not imply causal customer types.

## Combined retention review

The first Streamlit tab joins all 5,224 September customers by customer ID and forecast cutoff. It combines the calibrated weekly LSTM inactivity probability, expanded enriched monthly XGBoost expected 90-day revenue, and the June-fitted segments. The forecast window is 1 September through 29 November 2011 inclusive. The separate June six-month scores stay in the Revenue tab.

The default priority score is:

```
inactivity probability x historical average order value
```

This is a review heuristic based on predicted risk and a historical order-value proxy. It does not estimate intervention response, recovered revenue, profit or expected lost revenue. Expected 90-day revenue is displayed separately as gross sales without an assumed campaign effect.

Users can filter by segment, country, risk/value group, minimum risk and customer ID; change the review capacity; and rank by risk x historical average order value, inactivity risk, expected 90-day revenue, or historical 365-day spend. Capacity is applied after filtering. The default 10% capacity selects 523 of 5,224 customers; it is an adjustable workflow setting, not a model limit. The default queue's mean predicted inactivity is 79.6%, with median historical order value GBP 748.23. Risk-only ranking at the same capacity has 96.3% mean predicted inactivity and GBP 176.10 median historical order value. These are policy tradeoffs, not evidence of campaign performance.

The default risk/value groups use a 50% inactivity threshold and GBP 416.26 historical average order value threshold, the June cohort's 75th percentile. Thresholds can be changed. The table displays the top 100 queued customers; the queue CSV includes the full selected list, and a second CSV includes all filtered customers. Customer review records and ranking comparisons exclude observed future outcomes.

## Limits and interpretation

- The workbook ends on 9 December 2011. These results describe historical behavior from one retailer, not current customers or a live campaign.
- 90-day inactivity means no qualifying purchase in the next 90 days; it does not establish permanent churn.
- September 90-day revenue forecasts retain material aggregate bias, even after the expanded selection.
- Revenue is gross sales, not margin, profit or net revenue. Credits are separate because the source does not reliably link each credit to its original sale.
- Anonymous purchases cannot be scored. The conservative stock-code policy excludes service-like and accounting-like codes.
- BG/NBD's fitted purchase-shape parameter is below one at the examined origins, so its infinite-horizon expected purchase count is not finite. Implemented forecasts are finite-horizon only; they are not lifetime value.
- There are no intervention, campaign-cost or treatment-response data, so campaign lift and incremental profit cannot be estimated.

## Project files

- `app/app.py`: local Streamlit dashboard.
- `src/customer_intelligence/`: ingestion, features, model fitting, segmentation and review logic.
- `scripts/project.py`: full pipeline and individual stages.
- `scripts/revenue_challengers.py`: expanded statistical and probabilistic revenue comparison.
- `reports/`: compact model comparisons and aggregate analysis outputs.
- `artifacts/` and `data/`: locally generated models, manifests and customer-level outputs; excluded from Git.

### Research and definitions

- [System design and model choices](research/PROJECT_DESIGN.md)
- [Dataset assessment](research/DATASET_ASSESSMENT.md)
- [Research and similar-project reviews](research/RESEARCH_NOTES.md)
- [Model portfolio](research/MODEL_PORTFOLIO.md)
- [Inactivity sequence experiment](research/SEQUENCE_MODEL_EXPERIMENT.md)
- [Revenue improvement study](research/REVENUE_IMPROVEMENT.md)
- [Expanded revenue model results](research/REVENUE_CHALLENGER_RESULTS.md)
- [Combined retention review definitions](research/RETENTION_REVIEW.md)
- [Implementation status](research/IMPLEMENTATION_PLAN.md)
- [Feature dictionary](research/FEATURE_DICTIONARY.md)
- [Input cleaning policy](configs/data_policy.json)
- [Snapshot dates](configs/snapshots.json)
- [Reproducible dataset audit](notebooks/00_dataset_scope_audit.ipynb)
