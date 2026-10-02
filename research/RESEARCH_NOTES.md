# Research notes and related projects

Research reviewed on 1 October 2026. Local evidence controls claims about the supplied file. Original papers and official documentation control methodological/library claims. Public project code provides design examples, not verified performance targets for this project. No example repository was executed.

## Findings that changed the design

1. **Two worksheets are not two disjoint years.** The local audit found overlapping December 2010 records. Source reconciliation belongs before customer aggregation.
2. **Time coverage constrains the target.** The final December is incomplete. A September cutoff supports 90 days; an earlier June cutoff supports a complete longer-horizon comparison.
3. **Different meanings of recency/frequency must remain separate.** RFM's time since last order is not BG/NBD's elapsed time between first and last purchase. The probabilistic model's repeat purchase-day count also differs from invoice count.
4. **Archived tutorials need a library decision.** Most public projects use `lifetimes`; its maintainer has archived it and points to PyMC-Marketing. Prefer the successor for the new implementation.
5. **Gamma-Gamma is a hypothesis to test.** Positivity, spend-process independence, stability and finite expectations matter, especially when several customers place very large orders.
6. **A classifier score cannot establish offer effectiveness.** Inactivity risk, expected revenue and incremental treatment response answer different questions. The invoice data can support the first two; campaign experiments would be needed for the third.
7. **Code inspection matters.** Several examples have useful portfolio framing but evaluation or probability/value combinations that should not be copied.

## Similar project comparison

### 1. DuckDB/dbt segmentation and probabilistic CLV

[michaelf108/online-retail-clv](https://github.com/michaelf108/online-retail-clv) uses staging/intermediate/mart layers, daily purchase occasions, clustering, per-segment BG/NBD/Gamma-Gamma and Power BI. Its README explicitly acknowledges full-period segment membership in holdout validation, monetary-fit limitations and the archived CLV library. This is useful inspiration for data lineage and honest error reporting, rather than evidence that separate segment fits will improve our models.

I also inspected [holdout validation code](https://github.com/michaelf108/online-retail-clv/blob/main/models/marts/mart_holdout_validation.py) and [segmentation code](https://github.com/michaelf108/online-retail-clv/blob/main/models/marts/customer_segments.py). The holdout uses historical purchase summaries but joins segment labels from a separate model that clusters full-period features. Our improvement is to fit segment transformations using only the forecast-origin history. Adopt the layered tables and separate count/revenue evaluation; defer dbt until it provides value beyond a small Python/SQL package.

### 2. SQL features, repeat-purchase prediction and revenue ranking

[Kingsley-amg/customer-ltv-prediction](https://github.com/Kingsley-amg/customer-ltv-prediction) uses SQLite feature aggregation, a return classifier, a value regressor and gains charts. It compares targeting by predictions with targeting by historical spend, which is an important baseline to retain.

In [the inspected modeling file](https://github.com/Kingsley-amg/customer-ltv-prediction/blob/main/02_model_ltv.py), customers are randomly split after a single feature cutoff. This tests generalization across customers in one period, not later-period performance. The regressor learns log future revenue from both zero and positive outcomes, then its inverse-transformed prediction is multiplied by purchase probability. Our two-stage design instead trains conditional spend only on returning customers; an unconditional revenue estimate is used directly. Its reported scores are not transferable to our stricter evaluation.

### 3. Churn/value pipeline and customer action list

[Hamim-Susmit/customer-churn-clv-online-retail-ii](https://github.com/Hamim-Susmit/customer-churn-clv-online-retail-ii) separates auditing, cleaning, feature building, modeling and action-list output. Its explicit future-value proxy terminology and modular layout are useful.

I inspected [the final-stage script](https://github.com/Hamim-Susmit/customer-churn-clv-online-retail-ii/blob/main/scripts/run_04.py) and [feature/label code](https://github.com/Hamim-Susmit/customer-churn-clv-online-retail-ii/blob/main/src/features.py). The script uses a feature snapshot later than its label cutoff and splits customers by last purchase date, rather than building train and validation examples at independent forecast dates. Features in the script are not rebuilt at the earlier label cutoff. That design can expose future behavior to the model. Our snapshot builder and target builder have separate, enforced temporal boundaries. Do not assume that a time-related variable alone makes a validation scheme leakage-safe.

### 4. Earlier exploratory segmentation and BTYD study

[marinang/online_retail_analysis](https://github.com/marinang/online_retail_analysis) covers EDA, segmentation, inferred inactivity and probabilistic CLV in one notebook. I inspected [the notebook source](https://github.com/marinang/online_retail_analysis/blob/main/analysis.ipynb). Its inactivity discussion relies on thresholds in latent probability-alive paths and low expected purchase counts. That is exploratory model-based interpretation, rather than supervised evaluation of an observed 90-day no-purchase event.

Adopt the connected narrative between behavior, value and risk. Improve the project through a package-based pipeline, historical snapshot labels, explicit model comparisons and a frozen future-period evaluation.

### 5. Maintained reference implementation

[PyMC-Marketing](https://github.com/pymc-labs/pymc-marketing) supplies maintained probabilistic customer-value implementations and worked examples. Its [CLV quickstart](https://www.pymc-marketing.io/en/latest/notebooks/clv/clv_quickstart.html) demonstrates purchase forecasting, monetary forecasting and discounted value. Treat that implementation as the reference for APIs and units, while testing adequacy on our own retailer. Its tutorial examples are not evidence of performance on the supplied workbook.

## Selected primary sources and their role

| Source | Reviewed contribution | Project implication |
| --- | --- | --- |
| [UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) | Original dataset description, schema, period, cancellation code, currency and license | Use dataset 502; preserve attribution and verify workbook contents locally |
| [Fader, Hardie & Lee: BG/NBD](https://www.brucehardie.com/papers/018/) | Original repeat-purchase model | Appropriate behavioral baseline for noncontractual retail |
| [Hardie: probability alive](https://www.brucehardie.com/notes/021/palive_for_BGNBD.pdf) | Latent active-state probability and model variants | Keep alive probability separate from next-period inactivity |
| [Hardie: Gamma-Gamma derivation](https://www.brucehardie.com/notes/025/gamma_gamma.pdf) | Monetary model and population/conditional expectations | Diagnose monetary assumptions and finite mean behavior |
| [PyMC-Marketing BG/NBD](https://www.pymc-marketing.io/en/latest/notebooks/clv/bg_nbd.html) | Model assumptions, data summary conventions and fitting | Use consistent durations and repeat purchase occasions |
| [PyMC-Marketing Gamma-Gamma](https://www.pymc-marketing.io/en/latest/notebooks/clv/gamma_gamma.html) | Spend-process assumptions and implementation | Check independence and positive repeat-spend input |
| [PyMC-Marketing quickstart](https://www.pymc-marketing.io/en/latest/notebooks/clv/clv_quickstart.html) | Combined forecasts, MAP/posterior workflow and value units | Use a maintained implementation with clear point/uncertainty distinction |
| [Archived lifetimes repository](https://github.com/CamDavidsonPilon/lifetimes) | Maintainer states archive status and names a successor | Avoid an unmaintained dependency as the default new stack |
| [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) | Train-only learned preprocessing | Put learned transforms inside the fitting workflow |
| [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) | Time ordering and sample-based gap | Build date-aware panel validation rather than a row-count embargo |
| [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html) | Disjoint calibration, reliability curves and proper scores | Evaluate probabilities as well as ranking |
| [Average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) | AP differs from trapezoidal PR area | Name the reported metric precisely |
| [Clustering guide](https://scikit-learn.org/stable/modules/clustering.html) | Cluster-quality measures and their limitations | Combine geometry, stability and business profiles |
| [CatBoost categorical processing](https://catboost.ai/docs/en/concepts/algorithm-main-stages_cat-to-numberic) | Native category statistics | Convenient country handling; does not replace temporal validation |
| [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression) | Tweedie requirements and numerical cautions | Match objectives to nonnegative zero-heavy revenue |
| [CatBoost package installation](https://catboost.ai/docs/en/concepts/python-installation) | Supported interpreter/platform prerequisites | Smoke-test the actual Windows environment before locking dependencies |
| [XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html) | Regression/classification objectives and tuning controls | Comparable alternative boosting model |
| [DuckDB Python API](https://duckdb.org/docs/current/clients/python/overview) | Local SQL/DataFrame interaction | Small reproducible analytical store |
| [DuckDB Parquet support](https://www.duckdb.org/docs/current/data/parquet/overview) | Columnar file reads/writes | Parse Excel once, reuse normalized artifacts |
| [SHAP and causal interpretation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html) | Predictive explanation does not establish intervention effects | Describe explanatory relationships as associations |
| [Criteo uplift dataset and erratum](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | Randomized advertising benchmark with a corrected release | Optional separate causal extension if scope changes |

## Research limits and evidence discipline

Repository claims were not reproduced, and third-party metrics have not been used as success targets. Source-code inspections are fingerprinted in [source_registry.json](evidence/source_registry.json). Repository default branches can change; the recorded blob hashes identify the inspected versions. Official documentation may also evolve, so verify APIs again during implementation and pin only a tested environment.

The local audit supports feasibility and exposes data issues. It does not show that BG/NBD assumptions hold, that CatBoost wins, that clusters are stable, or that any retention action increases profit. Those are experiment outcomes or additional-data questions.
