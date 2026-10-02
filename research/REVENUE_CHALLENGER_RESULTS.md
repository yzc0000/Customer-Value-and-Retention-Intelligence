# Expanded customer revenue model comparison

The suggested models have been fitted and retained. Temporally eligible selection chooses enriched monthly XGBoost for 90-day expected revenue and BG/NBD x Gamma-Gamma for six months. Seasonal/covariate Pareto/NBD is a promising statistical competitor; the tested revenue LSTM and GAM recipes did not improve consistently.

This experiment compares per-customer 90-day and exact six-calendar-month gross merchandise revenue. Retailer-wide time-series forecasting remains outside scope.

## Evaluation protocol

The recipes, random seed (20261001) and selection criterion were fixed before this expanded run. The primary criterion is the equal-origin mean Tweedie deviance at a common power of 1.5, with mean absolute aggregate bias and RMSE as tie-breakers. MAE, RMSE, signed total bias and top-decile revenue capture remain visible. Expected customer value calls for a conditional mean: MAE alone favors a conditional median, especially with many zero outcomes. This experiment therefore uses a different explicit primary criterion from the original MAE-selected portfolio; it does not relabel the original selections. [Scikit-learn's scoring guidance](https://scikit-learn.org/stable/modules/model_evaluation.html#which-scoring-function-should-you-use) explains the mean/median distinction.

- 90-day development origins: September and December 2010, March and June 2011. These cover four seasons, with different amounts of available training history.
- Six-month selection origin: December 2010, whose outcomes finish on 1 June 2011. January-May 2011 are additional retrospective robustness origins: their outcomes end July-November 2011 and overlap the final June cohort, so they cannot select an as-of June model. The combined six-origin averages are retrospective comparisons, not independent validation folds.
- All supervised training snapshots have an earlier feature cutoff and a target end no later than the forecast origin. Feature construction excludes transactions at or after the cutoff. Preprocessing fits on training rows only.
- New supervised models use fully mature monthly snapshots from June 2010. The original XGBoost reference continues using quarterly snapshots, preserving its old recipe; an enriched monthly XGBoost control helps separate representation/training changes from model-family changes.
- Customer-process models fit only transaction histories observed before each scored origin. They do not fit future-revenue labels.
- Selection excludes any origin whose outcome window ends after the final forecast date. This availability requirement was enforced during the audit and saved selections were recomputed without changing fitted candidate predictions. The final September 2011 (90 days) and June 2011 (six months) cohorts were already inspected in earlier work: the new results are retrospective comparisons, not fresh untouched tests.

Metrics average the individual origin results with equal weight. Tweedie evaluation requires positive means: zero forecasts are floored at GBP 0.000001 for scoring only. A recent-spend baseline can predict zero for a customer who later purchases, creating very large deviance; it remains a useful MAE and RMSE reference. Baseline six-month scaling uses the actual calendar horizon length, correcting the earlier December calculation's 183-day approximation to 182 days. The old baseline output remains retained.

## Implemented recipes

| Candidate | Implementation and assumptions |
| --- | --- |
| Tweedie GAM | Ridge-regularized additive cubic splines of eight logged customer-history features, country factors, calendar features and missingness indicators; five knots, constant extrapolation, Tweedie power 1.5 and ridge alpha 0.3. This is a spline-basis additive model, without automatic smoothing-parameter estimation. |
| Gamma hurdle GAM | Same representation, logistic purchase probability (C=0.3) multiplied by a positive-revenue Gamma mean (alpha=0.3). Zero rows fit the purchase head only. |
| Hurdle NGBoost | Histogram-boosted purchase probability times a conditional LogNormal arithmetic mean learned by NGBoost 0.5.11; 250 boosting steps, learning rate 0.03, depth-three base trees, minimum leaf size 25. Positive revenue is scaled by 1000 for fitting and restored to GBP. |
| Probabilistic LSTM | Shared 26-week, six-channel LSTM (hidden size 24) joined to logged static/calendar features; a 32-unit layer predicts purchase logits and positive LogNormal location/scale. Fixed 30 epochs, dropout 0.15, weight decay 0.003, Adam 0.002. Hurdle likelihood uses all rows for purchase and positive rows for spending. |
| Pareto/NBD x Gamma-Gamma | Gamma heterogeneity in Poisson purchase intensity and exponential dropout, fitted with penalized likelihood and two optimizer starts; finite-horizon expected purchase occasions multiplied by Gamma-Gamma expected daily spend. |
| Seasonal/covariate Pareto/NBD x Gamma-Gamma | Adds an October-December purchase-rate multiplier and UK/other-country effects on purchase and dropout rates. Integrates calendar exposure at the actual customer's dates, splitting at seasonal boundaries; seasonality changes purchasing, while spending remains Gamma-Gamma. |
| Reference controls | Actual-horizon-scaled recent spending; original quarterly XGBoost Tweedie; enriched monthly XGBoost; enriched monthly nonlinear Gamma hurdle; original BG/NBD x Gamma-Gamma. |

The LSTM uses `p * exp(mu + sigma^2 / 2) * 1000`, not the positive LogNormal median `exp(mu)`. NGBoost likewise uses its distribution's arithmetic mean. Both supply central 90% predictive intervals for a mixture with an atom at zero; the intervals are evaluated for coverage and width and have not been conformally calibrated. Because of the atom at zero, 5th/95th percentile intervals can have model-implied coverage above 90%; inspect positive-buyer coverage separately. LSTM output constraints bound log location to [-12, 8] and log scale to [-2, 0.8] for numerical regularization; the revenue targets are not winsorized.

The Pareto implementation is a point estimate with weak regularization, not posterior sampling or an invocation of the complete CLVTools package. It uses purchase days as occasions and excludes the observed first day from repeat frequency. A source's first observed purchase need not be the customer's true acquisition date. Gamma-Gamma assumes stable spending and independence from the purchase process; a seasonal purchase model alone cannot resolve changing basket sizes. Predictions are finite-horizon customer revenue, rather than infinite-horizon lifetime value.

Primary references: [Fader and Hardie's Pareto/NBD derivations](https://www.brucehardie.com/notes/009/pareto_nbd_derivations_2005-11-05.pdf), [CLVTools seasonal customer-model example](https://www.clvtools.com/articles/CLVTools.html), [Gamma-Gamma assumptions](https://www.brucehardie.com/notes/025/gamma_gamma.pdf), [NGBoost LogNormal distribution forecasts](https://stanfordmlgroup.github.io/ngboost/1-useage.html), and the [NGBoost paper](https://proceedings.mlr.press/v119/duan20a).

## Reproduce and inspect

```powershell
.venv\Scripts\python -m pip install -r requirements-revenue-challengers.txt
.venv\Scripts\python scripts/revenue_challengers.py
.venv\Scripts\python scripts/audit_revenue_challengers.py
.venv\Scripts\python -m unittest discover -s tests
.venv\Scripts\python -m streamlit run app/app.py
```

Prepared data from `python scripts/project.py prepare` is required. Completed model/origin fits resume from matching source/code signatures. Selection can be recomputed without refitting using `python scripts/revenue_challengers.py --summarize`. The original fitting runner is retained at `artifacts/runs/revenue_challengers/fitted_runner_source.py`; the protocol separately records the audited selection code hash. A quick partial check is `python scripts/revenue_challengers.py --horizons 90d --origins 2011-03-01 --models lstm_hurdle_lognormal`; partial-origin runs do not select a model and overwrite the corresponding development summary with partial results, so rerun the complete comparison afterward.

Every fitted model, fitted preprocessing object, receipt and customer prediction is retained under ignored `artifacts/runs/revenue_challengers/<horizon>/<model>/<origin>/`. The LSTM also retains its state dictionary and training loss. Protocol and horizon selections are saved beside the runs. Customer forecasts for each selected expanded candidate are in ignored `data/processed/revenue_challenger_scores_<horizon>.parquet`. The original `artifacts/champions.json` and score outputs remain preserved; the Revenue tab shows the expanded expected-value selection separately.

Fit receipts describe the available mature snapshot pool. The replay audit records effective supervised training rows after the original reference's quarterly filtering and identifies models fitted directly to transaction histories.

Reviewable aggregate outputs:

- `reports/revenue_challengers_<horizon>_development.csv`: every rolling comparison, explicitly marked development or retrospective robustness.
- `reports/revenue_challengers_<horizon>_selection_summary.csv`: only origins whose complete outcomes are available before the final forecast date.
- `reports/revenue_challengers_<horizon>_summary.csv`: equal-origin summaries and completeness eligibility.
- `reports/revenue_challengers_<horizon>_retrospective.csv`: final cohort comparisons, excluded from selection.
- `reports/revenue_challengers_replay_audit.csv`: separate-process prediction replay and effective training-row checks.
- `reports/revenue_challengers_quadrature_audit.csv`: 24-node versus 64-node Pareto integration checks.
- `reports/revenue_challengers_interval_diagnostics.csv`: interval coverage for all customers, buyers and nonbuyers.
- `reports/revenue_challengers_uncertainty.csv`: paired customer-cluster bootstrap of selected-versus-reference development differences, conditional on existing dates and without adjustment for model selection.

The mathematical tests check base Pareto likelihood against a closed form and independent adaptive quadrature, finite-horizon expectations including the s=1 limit, seasonal calendar exposure, recovery of the base model when covariate effects vanish, and zero-mass mixture quantiles.

## Measured results

All 11 recipes completed every origin for both horizons: 55 selection-eligible evaluations, 55 additional retrospective robustness evaluations and 22 retrospective final-cohort evaluations. No losing model was removed. Values below average origin-level metrics; the comparison includes the original reference rescored on the same expanded origins, so its averages differ from the earlier two-origin pilot.

### 90-day development, four origins

| Model | Mean MAE, GBP | Mean RMSE, GBP | Mean absolute total bias | Mean Tweedie deviance |
| --- | ---: | ---: | ---: | ---: |
| Enriched monthly XGBoost | 390.47 | 1,365.61 | 41.5% | 45.89 |
| Hurdle NGBoost | 402.77 | 1,514.30 | 41.7% | 46.10 |
| Nonlinear Gamma hurdle | 392.25 | 1,449.55 | 35.6% | 46.30 |
| Seasonal/covariate Pareto/NBD x Gamma-Gamma | 372.06 | 1,333.20 | 26.1% | 46.57 |
| BG/NBD x Gamma-Gamma | 379.26 | 1,330.37 | 31.0% | 46.75 |
| Pareto/NBD x Gamma-Gamma | 376.74 | 1,327.21 | 30.2% | 46.85 |
| Original quarterly XGBoost | 399.47 | 1,403.12 | 45.7% | 47.23 |
| Probabilistic LSTM | 474.21 | 1,917.58 | 70.2% | 47.79 |
| Tweedie GAM | 534.69 | 1,756.89 | 88.5% | 52.91 |
| Gamma hurdle GAM | 419.58 | 1,483.01 | 52.7% | 53.64 |
| Recent spending | 402.71 | 1,641.37 | 34.2% | 283,463.09 |

### Six-month combined rolling checks, six overlapping origins (retrospective)

| Model | Mean MAE, GBP | Mean RMSE, GBP | Mean absolute total bias | Mean Tweedie deviance |
| --- | ---: | ---: | ---: | ---: |
| Seasonal/covariate Pareto/NBD x Gamma-Gamma | 569.05 | 2,088.17 | 21.1% | 50.36 |
| BG/NBD x Gamma-Gamma | 576.79 | 2,089.13 | 26.2% | 50.88 |
| Pareto/NBD x Gamma-Gamma | 589.05 | 2,078.31 | 28.0% | 51.27 |
| Original quarterly XGBoost | 827.07 | 2,631.20 | 76.0% | 57.41 |
| Enriched monthly XGBoost | 863.20 | 2,582.36 | 86.4% | 58.51 |
| Hurdle NGBoost | 913.95 | 3,549.90 | 96.4% | 59.31 |
| Nonlinear Gamma hurdle | 879.39 | 2,615.72 | 86.3% | 59.75 |
| Gamma hurdle GAM | 1,248.85 | 3,253.92 | 156.9% | 72.13 |
| Probabilistic LSTM | 1,615.96 | 6,934.69 | 217.5% | 75.53 |
| Tweedie GAM | 5,965.22 | 20,984.57 | 791.1% | 127.97 |
| Recent spending | 655.83 | 2,582.38 | 45.6% | 150,767.67 |

### What the results support

For **90-day expected revenue**, enriched monthly XGBoost is selected by the fixed deviance rule: 45.89 versus 47.23 for the original reference (2.8% lower), with mean MAE 390.47 versus 399.47 (2.3% lower). NGBoost's deviance is close at 46.10, but its mean MAE is slightly worse than the original reference. This is a modest improvement, not a decisive family advantage.

Seasonal/covariate Pareto/NBD has the **lowest 90-day development MAE** at 372.06, 6.9% below the original XGBoost. Its deviance is 46.57 and mean absolute total bias is 26.1%, versus 45.7% for the original reference. It is a useful statistical alternative and a separate MAE winner; it did not win the preregistered expected-value criterion.

For **six-month expected revenue**, temporally eligible selection chooses **BG/NBD x Gamma-Gamma** on December 2010, the only comparison origin with outcomes complete by the final June date. Its deviance is 55.81, versus 57.09 for seasonal Pareto/NBD. Its MAE is 656.77 versus 728.95 for recent spending (9.9% lower); RMSE is 2,281.51 versus 2,874.53 and signed total bias remains substantial at +49.1%. One selection origin provides limited evidence.

Across the **broader retrospective six-month checks**, seasonal/covariate Pareto/NBD has the lowest mean MAE: 569.05 versus 655.83 for recent spending, a 13.2% reduction. Mean RMSE is 19.1% lower, and mean absolute total bias falls from 45.6% to 21.1%. It beats recent-spend MAE at all six origins. Its advantage over existing BG/NBD is only 1.3% lower MAE and 1.0% lower deviance. These overlapping later outcomes cannot select the final June model or support an independent holdout claim.

The **revenue LSTM did not win across origins**. Its 90-day mean MAE is 474.21 and six-month mean MAE is 1,615.96, with strong overforecasting in several periods. March's 90-day MAE was good at 248.06, but December's was 877.89. That variability is why a favorable single-origin result is insufficient. This does not change the user-selected LSTM lead for the separate inactivity task. The tested GAM recipes also failed to improve consistently; the direct six-month Tweedie spline fit produced particularly extreme forecasts. These fixed recipes are not evidence that all GAMs or all LSTMs are unsuitable: stronger regularization, different positive-spend distributions, and multiple-seed development tuning remain possible experiments.

### Retrospective final-origin comparison

These previously inspected cohorts did not choose the expanded winner. The table shows the development-selected candidate beside its reference.

| Horizon and cohort | Model | MAE, GBP | RMSE, GBP | Signed total bias | Tweedie deviance |
| --- | --- | ---: | ---: | ---: | ---: |
| 90d, September 2011 | Original quarterly XGBoost | 373.47 | 2,119.86 | -42.7% | 51.00 |
| 90d, September 2011 | Enriched monthly XGBoost | 362.57 | 1,832.57 | -25.5% | 45.53 |
| 6m, June 2011 | Recent spending | 574.59 | 3,059.67 | -24.3% | 345,755.66 |
| 6m, June 2011 | BG/NBD x Gamma-Gamma | 570.55 | 2,542.49 | -12.9% | 44.91 |

The selected 90-day candidate improves retrospective MAE by 2.9%, RMSE by 13.6%, and underforecasting from 42.7% to 25.5%. Seasonal Pareto/NBD had lower retrospective deviance (43.51), and the LSTM had the lowest retrospective MAE (360.70); neither result overrides development selection.

For six months, selected BG/NBD and recent spending are close on retrospective MAE: 570.55 versus 574.59, only 0.7% better. The stronger improvement is RMSE (16.9% lower) and total underforecasting (12.9% versus 24.3%). Seasonal Pareto/NBD has MAE 573.05 and bias -10.4%. Neither result supports claiming a large final-origin MAE improvement; the 13.2% seasonal Pareto gain belongs to overlapping retrospective rolling checks.

### Corrected legacy six-month refit

During replay verification, the older production six-month refit was found to use March 2011 labels ending in September 2011 at a June 2011 forecast origin. `label_mature` meant complete at the source's December end, rather than available at the forecast date. The new benchmark already checked `horizon_end_exclusive <= origin`; the original pipeline now does too. Its valid refit uses 10,222 quarterly rows rather than the previous 14,734. A regression test exercises the actual production function with later-complete labels, and the original six-month portfolio has been rerun.

The old XGBoost June MAE of 569.14 was affected by this timing error; the corrected value is **585.21**. Development XGBoost MAE remains 999.19. The recent-spend final baseline is unchanged at 574.59 because it does not learn from future labels. Development baseline scaling was separately corrected from 183 to the actual 182 calendar days, giving MAE 728.95 instead of 733.44. The selected original baseline remains the same.

Every older six-month fitted model, score output, comparison and champion file is preserved under ignored `artifacts/runs/legacy_six_month_refit_with_late_labels/`, with `INVALID_REFIT_NOTICE.json`. Those invalid historical supervised final scores are available for audit and must not be used as valid performance evidence. The 90-day quarterly refit also now explicitly checks training-label end dates; its eligible rows and scores are unchanged.

### Limits and next experiment

Only approximately two years of one retailer's data are available. Temporal regimes and repeated customer rows are dependent, and the final-origin results are retrospective. Single-seed compact recipes compare viable implementations, not the best achievable performance of each family. Buyer-specific basket changes remain hard to infer from transaction history; adding a purchase-process model does not supply missing commercial information. All forecasts are gross positive merchandise sales, with credits held separately.

Use enriched monthly XGBoost as the expanded 90-day expected-revenue candidate and BG/NBD x Gamma-Gamma for six months, while retaining seasonal Pareto/NBD as a promising statistical challenger and all probabilistic competitors. The next defensible improvement study would tune monetary regularization and probabilistic spending heads using inner historical origins, check stability across random seeds, and evaluate the resulting frozen choices on newer, genuinely unseen retail data. Do not calibrate forecasts by multiplying by the final cohort's actual/predicted revenue ratio.

## Verification outcome

All 132 retained fits were loaded in another process and reproduced their saved customer predictions and available interval bounds. The original reference scores also reproduced exactly against the corrected six-month refit. All 24 Pareto model/origin pairs passed 24-node versus 64-node integration checks. Eight tests pass, including production training-date and model-selection outcome-availability regression tests. The app was checked for both horizons.

Paired customer-cluster resampling gives a 90-day selected-versus-reference MAE reduction of GBP 9.01, with conditional 95% interval GBP 3.04-14.68; deviance improves at all four selection origins. MAE improves at two of four. For six-month December selection, BG/NBD's MAE reduction is GBP 72.19, interval GBP 30.53-113.43. These intervals condition on the existing dates, preserve repeated customer appearances and do not adjust for candidate selection or uncertainty in future seasonal regimes. They do not establish a general improvement on new data.

September 90-day interval coverage is 92.6% for NGBoost and 92.1% for LSTM across all customers, with mean widths GBP 1,153.90 and GBP 1,180.38. Among actual buyers, coverage is 82.6% and 81.6%; nonbuyers are almost always covered at zero. Those are unconditional mixture intervals, so buyer-only coverage does not have a separate 90% target, but the breakdown prevents mistaking a high overall figure for uniformly calibrated uncertainty.

The selected six-month BG/NBD December fit remains at the lower bound for its dropout parameter a. The earlier regularization sensitivity study supports its MAE advantage while showing substantial overforecasting. Selection here does not resolve that fit limitation or validate an infinite-horizon value estimate.
