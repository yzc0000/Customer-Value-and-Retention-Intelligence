# Improving customer-level revenue forecasts

## Recommendation

Develop a model of purchasing and positive spending separately, improve the customer history features, and evaluate expected revenue rather than selecting only by MAE. Keep XGBoost Tweedie and BG/NBD x Gamma-Gamma as reference models. The initial experiments show a useful accuracy/calibration tradeoff, not a universal replacement winner.

The suggested GAMs, Pareto/NBD with seasonal/customer covariates, hurdle NGBoost and probabilistic revenue LSTM are now implemented in the [expanded revenue comparison](REVENUE_CHALLENGER_RESULTS.md). That study uses more historical origins and separately evaluates both horizons. The sections below preserve the initial development study and its recommendations; the linked expanded report contains the newer outcomes. Retailer-wide forecasting remains outside scope.

## What is weak in the current implementation?

The target is each known customer's gross merchandise revenue over the next 90 days, in GBP. Six-calendar-month revenue is a separate task. Zero means no qualifying sale during the future window; it does not mean permanent churn.

The existing XGBoost mean development MAE is 270.39, compared with 280.86 for recent spending: a 3.7% improvement. On the already published September test, the MAE improvement is 4.6%, but total customer revenue is underpredicted by 42.7%. Its top-decile ranking captures 59.7% of revenue, versus 56.7% for recent spending. Ranking quality and monetary calibration are separate properties.

| Diagnostic | March 2011 | June 2011 | September 2011, previously published test |
| --- | ---: | ---: | ---: |
| Customers with zero future revenue | 64.9% | 67.1% | 57.2% |
| Share of revenue from top 1% of customers | 34.2% | 40.1% | 37.3% |
| XGBoost signed error for actual top 1%, GBP | -144,857 | -294,381 | -543,594 |
| XGBoost signed error for all customers, GBP | +159,435 | -245,441 | -1,151,520 |

The historical-spend inputs are strongly skewed. The existing Gamma and Tweedie GLMs put raw, standardized amounts inside an exponential link; large inputs can lead to extreme predictions. For example, the original March Gamma hurdle RMSE exceeds 20,000. Log-transforming nonnegative input features addresses this source of extrapolation without changing the revenue target into a log-scale prediction problem.

The original supervised candidates also lack calendar features and use quarterly snapshots only. These are plausible weaknesses, but the experiments below show that adding calendar features or more rows alone does not reliably help. Only about two years of source data are available, limiting seasonal learning.

Some large revenue changes have little historical signal. Customer 14096 had only GBP 10.67 in observed purchases before September and GBP 45,911.04 in the next 90 days. This is an example of missing predictive information, not proof that every large-customer error is unavoidable. Customer type, acquisition channel, contracted orders, promotions, or planned purchases might help if a future dataset supplies them. The current workbook does not contain these fields.

## The forecast quantity needs to be explicit

For value estimation we want expected revenue. Minimizing absolute error estimates a conditional median; squared error and Poisson/Gamma/Tweedie deviances target a conditional mean. With a large probability mass at zero, a median forecast may be zero even when expected revenue is substantial. Selecting a mean model only by MAE can favor conservative monetary estimates. [Scikit-learn's scoring guidance](https://scikit-learn.org/stable/modules/model_evaluation.html#which-scoring-function-should-you-use) explains this distinction.

Retain MAE as an operational diagnostic, but also evaluate RMSE, a common fixed-power Tweedie deviance, signed and absolute origin-level bias, and top-decile revenue capture. Do not confuse accurate summed customer forecasts with a retailer-wide time-series model: the former remains a diagnostic for this customer-level task.

## Executed development experiments

Reproduce with:

```powershell
python scripts/revenue_research.py
```

The run compared seven model recipes, three feature sets, and quarterly versus monthly training: 42 configurations and 84 origin/model evaluations. Five fixed 50/50 blends were subsequently evaluated on saved predictions. No research candidate was scored on September, and no champion pointer was changed. The original XGBoost development metrics were reproduced exactly.

Model recipes: existing XGBoost Tweedie; log-input Tweedie GLM; log-input logistic/Gamma hurdle GLM; Poisson-count/Gamma-order-value GLMs; nonlinear purchase-propensity/Gamma-spend hurdle; histogram boosting with Poisson mean loss; and an absolute-error histogram model as a median benchmark.

The three feature sets are:

- **Basic:** the original numeric features and country.
- **Calendar:** basic features plus cutoff month sine/cosine and the known share of forecast days in October-December.
- **Enriched:** calendar plus recent average order values, prior-90-day spending, recency relative to purchase gaps, last order value, order-value variability, and observed spending/orders in the corresponding forecast window one year earlier. Prior-year features are missing when the source period is incomplete, rather than filled as observed zero.

Monthly training uses snapshots from June 2010 onward. Training at every origin requires `horizon_end_exclusive <= origin`, in addition to an earlier feature cutoff. This prevents training on unfinished monthly target windows. March has 23,760 monthly training rows; June has 36,974. Repeated customers and overlapping target windows are dependent observations, not new independent customers.

All values below are equal-weight averages of the separate March and June origin metrics. Bias is the average of each origin's absolute signed percentage bias, not the signed errors averaged together.

| Configuration | Mean MAE, GBP | Mean RMSE, GBP | Mean absolute origin bias | Mean Tweedie deviance, p=1.5 | Revenue captured by top 10% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Current XGBoost, original features/quarterly training | 270.39 | 956.79 | 12.8% | 40.07 | 62.1% |
| Nonlinear hurdle, enriched features/quarterly training | **255.20** | 1,045.43 | 13.4% | **38.30** | 63.3% |
| Nonlinear hurdle, enriched features/monthly training | 269.28 | 987.56 | **1.3%** | 38.81 | 63.4% |
| Current XGBoost + monthly enriched hurdle, fixed 50/50 | 266.74 | 946.78 | 6.9% | 38.57 | **63.7%** |
| Frequency/severity GLMs, enriched/quarterly | 279.59 | 1,055.04 | 3.1% | 39.55 | 63.3% |
| Poisson mean boosting, calendar/quarterly | 273.80 | **878.42** | 10.3% | 40.02 | 62.4% |

The quarterly enriched hurdle improves MAE by 5.6% and deviance by 4.4%, but RMSE is 9.3% worse. It is a credible customer-error challenger, not a proven remedy for large-value forecasts. Its MAE is 247.88 in March and 262.51 in June, versus 277.50 and 263.29 for the original model; most of its gain comes from March.

The monthly enriched hurdle substantially improves development total calibration: signed bias is +2.3% in March and +0.4% in June. Its MAE changes by only 0.4%, and RMSE is 3.2% worse than the original model. The balanced 50/50 blend improves all four displayed reference metrics modestly; the MAE improvement is only 1.4%.

The richer GLMs no longer produce the original extreme RMSE failures, but they do not become the overall accuracy winners. Calendar-only and monthly-basic candidates often worsen performance. The median-loss candidate underpredicts totals by approximately 85% or more. These results rule out claiming that more rows, more complexity, or a lower-median objective automatically fixes expected revenue.

Paired resampling by customer preserves repeated appearances across the two origins. For the quarterly enriched hurdle, the mean MAE reduction is GBP 15.20, with a conditional 95% bootstrap interval of GBP 7.25-21.84. For the monthly hurdle, it is GBP 1.11, with interval -6.74 to +8.67. These intervals condition on the two existing origins, do not adjust for selecting among configurations, and do not establish robustness across future seasonal regimes.

The already inspected September result informed this diagnosis. Even though research candidates were not scored there, a later reevaluation on September must be called a retrospective comparison, not a newly untouched test. The current source cannot supply a fresh complete 90-day period after that origin.

## Model architecture to develop

The exact two-part expectation is:

`E[revenue | history] = P(revenue > 0 | history) x E[revenue | revenue > 0, history]`.

The first component should be assessed with probability calibration and log loss. The second needs a distribution or loss appropriate for positive, skewed amounts. Avoid hard-thresholding the first probability before multiplication: expected revenue should include the uncertainty about purchasing.

| Candidate | Fit to this data and next experiment | Status |
| --- | --- | --- |
| Nonlinear hurdle with Gamma spending | Directly separates zeros from positive amounts. Expand temporal validation, investigate the large-customer spend errors, and test feature groups separately. Native Gamma loss is available in [histogram boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html). | Implemented development challenger |
| Poisson or negative-binomial counts + Gamma spending | Separates purchase frequency from order size. Weight the order-value fit by future order counts. Poisson is the implemented starting point; negative binomial is a candidate for overdispersion. The analogous [frequency/severity design](https://scikit-learn.org/stable/auto_examples/linear_model/plot_tweedie_regression_insurance_claims.html) is a useful implementation reference. | Poisson/Gamma pilot implemented; NB extension untested |
| Regularized Gamma/Tweedie GAM | Use smooth functions of logged recency, purchase rate, and recent spend, with controlled extrapolation. More interpretable than a neural model and less rigid than a raw-input GLM. | Implemented and tested in the expanded comparison |
| Pareto/NBD with covariates + monetary model | Forecast repeat purchases using a noncontractual customer process; add known seasonality and customer characteristics. [CLVTools](https://www.clvtools.com/articles/CLVTools.html) demonstrates an apparel-retailer model with seasonal covariates. Python [PyMC-Marketing](https://www.pymc-marketing.io/en/latest/notebooks/clv/pareto_nbd.html) provides Bayesian Pareto/NBD and customer covariates. Check the chosen implementation's dynamic-covariate support separately. | Base and seasonal/covariate point fits implemented and tested; not a Bayesian sampler |
| Hurdle NGBoost with positive LogNormal spend | Learn conditional spending uncertainty as well as expected spend, combined with a purchase-probability model. [NGBoost](https://proceedings.mlr.press/v119/duan20a) predicts distribution parameters; its [documentation](https://stanfordmlgroup.github.io/ngboost/1-useage.html) includes LogNormal. This does not guarantee better point accuracy. | NGBoost 0.5.11 installed locally, implemented and tested |
| Shared LSTM with purchase and Gamma/LogNormal spend heads | Pool customers, combine weekly history with static value/calendar features, and train a distributional loss. The positive spend head is trained on purchasers; the purchase head uses all rows. Reusing a classifier's rankings alone does not solve spend estimation. | LogNormal hurdle LSTM implemented and tested separately from the inactivity LSTM |

DeepAR and deep renewal models show how recurrent networks can pool related sparse series and model occurrence/amount uncertainty. They are relevant research directions, but a complete autoregressive rollout adds more machinery than our fixed-horizon two-part target currently needs. See the original [DeepAR paper](https://arxiv.org/abs/1704.04110) and [deep renewal paper](https://arxiv.org/abs/1911.10416).

The existing BG/NBD x Gamma-Gamma model received a development-only regularization sensitivity check, with Gamma-Gamma held fixed. For the six-month December origin, all five BG/NBD penalties improved MAE over the existing recent-spend development baseline of GBP 733.44:

| BG/NBD penalty | Fitted a | Six-month development MAE, GBP | Signed total bias |
| --- | ---: | ---: | ---: |
| 0 | 0.1640 | 696.44 | +55.6% |
| 0.00001 | 0.0103 | 698.34 | +56.2% |
| 0.0001 | 0.0038 | 700.38 | +56.7% |
| Current 0.001 | 0.0009, lower boundary | 656.77 | +49.1% |
| 0.01 | 0.0009, lower boundary | 675.25 | +54.6% |

The sign of the MAE advantage persists when the boundary is avoided, but parameter sensitivity and substantial overforecasting remain. The default penalty's MAE advantage is 10.4%; across the grid it ranges from 4.5% to 10.4%. This strengthens its status as the leading existing statistical six-month challenger without establishing a well-calibrated monetary forecast. At the March/June 90-day origins, the penalty grid changed MAE much less (March: 309.71-311.51; June: 281.38-284.70), and it did not beat the original supervised reference on MAE. No September origin or six-month June test was evaluated in this sensitivity run.

Gamma-Gamma assumes stable customer mean transaction value and independence from the transaction process; those assumptions still need checking. [Fader and Hardie's model note](https://www.brucehardie.com/notes/025/gamma_gamma.pdf) states them. A low simple correlation is only a diagnostic, not proof of independence. A BG/NBD parameter a below one does not invalidate a finite 90-day or six-month expectation; it rules out a finite infinite-horizon expectation under the model. Do not impose a greater-than-one constraint just to relabel finite-horizon revenue as lifetime value.

## Next development sequence

1. Evaluate the current reference, enriched hurdle, mean-loss challenger and fixed blend over additional historical rolling origins, with target-maturity checks. Separate seasonal regimes and customer-value groups in error reporting. Freeze the candidate recipes before collecting another dataset's final holdout.
2. Ablate recent order value, last order/variability, and prior-year history separately. The current enriched bundle shows usefulness but does not identify which added feature caused it. Add purchase-gap variability and distinguish missing history from inactivity where supported.
3. Choose a mean-forecast selection rule before evaluating candidates again. Use common Tweedie deviance plus explicit calibration/tail diagnostics, and keep MAE and ranking results visible. Any calibration model must be learned from earlier completed out-of-time predictions; using the September actual/predicted ratio would leak test outcomes.
4. Test the interpretable statistical challenger (covariate Pareto/NBD or regularized GAM) before pursuing a large neural portfolio. Retain the pure BG/NBD/Gamma-Gamma benchmark and all prior candidates.
5. Add predictive intervals with empirical coverage and width checks. They communicate residual uncertainty but do not by themselves improve mean accuracy. A probabilistic LSTM or NGBoost can then be assessed against the simpler reference on the same origins.
6. Apply the new purchase/spend architecture separately to the six-month task. The 90-day pilot results do not establish an improvement for six-month revenue. Its wider horizon has less training/validation coverage and different uncertainty; the executed BG/NBD sensitivity check supports the existing statistical challenger while exposing its calibration limitations.

## Files and verification

- `scripts/revenue_research.py`: isolated experiment entry point.
- `src/customer_intelligence/revenue_research.py`: cutoff-safe enrichment, model recipes, experiment runner, blends, and paired uncertainty calculation.
- `reports/revenue_research_development.csv`: all 84 development evaluations.
- `reports/revenue_research_ensembles.csv`: fixed-weight blend evaluations.
- `reports/revenue_research_summary.csv`: configuration summaries.
- `reports/revenue_research_uncertainty.csv`: paired customer bootstrap diagnostics.
- `reports/revenue_error_diagnostics.csv`: diagnostics for the previously published forecasts.
- `reports/revenue_bgnbd_sensitivity.csv`: 15 development-only BG/NBD regularization evaluations.
- Ignored `data/processed/revenue_research/`: quarterly/monthly research panels.
- Ignored `artifacts/runs/revenue_research/`: all fitted research models, customer predictions, blend formulae, and manifest.

The original development reference was reproduced exactly; training-label maturity was asserted at every origin; saved two-part models were reloaded in another process and reproduced their saved customer predictions. The current inactivity, segmentation, revenue champion files, and original score outputs are preserved. Use `python scripts/revenue_research.py --summarize` to rescore saved predictions without refitting, and `python scripts/revenue_research.py --statistical-sensitivity` to reproduce the BG/NBD check. If prepared data changes, rebuild the research panel caches before rerunning experiments.
