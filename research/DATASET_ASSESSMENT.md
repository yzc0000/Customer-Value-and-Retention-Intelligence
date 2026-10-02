# Supplied workbook — scope and quality assessment

Assessed for the project scope using every worksheet of `online_retail_II.xlsx`. The original files remain unchanged. Aggregate evidence is in [dataset_audit.json](evidence/dataset_audit.json), [monthly_profile.csv](evidence/monthly_profile.csv), [snapshot_feasibility.csv](evidence/snapshot_feasibility.csv), and [special_stock_codes.csv](evidence/special_stock_codes.csv). The exact profiling code is [audit_dataset.py](../scripts/audit_dataset.py).

## Verdict

**The dataset is sufficient for the brief's main modeling project. Keep it.** It supports behavioral segmentation, probabilistic repeat-purchase modeling, supervised future-revenue prediction, and temporally labeled inactivity. It needs scope adjustments for permanent churn, profit-based lifetime value, and causal retention ROI.

The verified workbook size is 45,622,278 bytes; its SHA-256 is `bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980`. Its 1,067,371 line records and date bounds match UCI's description of Online Retail II. The source is a historical UK gift retailer with wholesale customers, not a representative sample of all modern online retail. UCI licenses it under CC BY 4.0. [Original dataset documentation](https://archive.ics.uci.edu/dataset/502/online+retail+ii).

## Dataset and grain

| Verified item | Result | Interpretation |
| --- | --- | --- |
| `Year 2009-2010` | 525,461 lines; 1 Dec 2009–9 Dec 2010 | Load alongside the second worksheet |
| `Year 2010-2011` | 541,910 lines; 1 Dec 2010–9 Dec 2011 | Its early December overlaps the first worksheet |
| Raw total | 1,067,371 lines, 8 business columns | A line is not an order or customer |
| Earliest timestamp | 1 Dec 2009, 07:45 | Earlier customer history is unobserved |
| Latest timestamp | 9 Dec 2011, 12:50 | Final day/month are partial for conservative evaluation |
| Known customer IDs | 5,942 distinct IDs | Includes customers represented only by non-purchase activity |
| Invoice IDs | 53,628 distinct IDs | Includes cancellation/accounting activity |
| Stock codes / countries | 5,304 / 43 | Stock codes include merchandise and other charges |

The exact source headers are `Invoice`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `Price`, `Customer ID`, and `Country`. Neither invoice ID nor customer ID is a line-level primary key. Use workbook/sheet/row lineage for ingestion, and a separately reconciled invoice-line grain for modeling.

## Checks performed

Checked all-sheet coverage and schemas; min/max dates; missingness; distinct IDs; exact repeated rows and cross-sheet record hashes; cancellation/quantity/price rules; invoices with multiple customers, timestamps or countries; customers appearing in multiple countries; product-code patterns; measure quantiles; monthly volume/missingness; dates without rows; repeat-purchase sample size; and availability of complete 90/180-day labels at six historical cutoffs.

The profile establishes the supplied file's observed contents. It cannot prove that every underlying business transaction was recorded correctly, because there is no upstream event log or ingestion timestamp.

## Findings, risks and remedies

| Finding | Measured evidence | Risk / severity | Recommended treatment |
| --- | --- | --- | --- |
| Anonymous activity | 243,007 lines lack customer ID: **22.77%** | High: customer analyses exclude a substantial part of recorded activity | Retain anonymous sales for store totals; exclude from customer models and disclose coverage |
| Worksheet overlap | Both sheets cover 1–9 Dec 2010; **22,202 distinct matching record hashes** occur in both sheets | High: concatenation can double-count purchases and value | Reconcile cross-sheet invoice lines and multiplicities before aggregation |
| Exact repeated rows | **34,335 excess rows**, 3.22% of raw lines; **67,242 lines** participate in duplicate groups | High for unreconciled overlap; uncertain for identical within-invoice lines | Separate overlap duplication from legitimate repeated line items; do not blindly deduplicate on invoice/stock alone |
| Credits / cancellations | 19,494 `C`-prefixed lines; 22,950 negative-quantity lines | High if interpreted as purchases | Preserve and classify separately; use purchase and credit ledgers |
| Signs are not equivalent | 3,457 negative-quantity lines without `C`; 1 `C` line with nonnegative quantity | Medium: one flag cannot represent every event type | Test sign/prefix combinations and quarantine unexplained cases |
| Nonpositive price | 6,207 lines have price ≤ 0, including 5 negative-price lines | Medium/high for revenue/value models | Separate free/internal/adjustment records; require positive spend for Gamma-Gamma |
| Nonstandard stock codes | 6,093 lines fail a five-digit-plus-letters product pattern | Medium: charges mix with products, but the pattern also excludes real merchandise | Use reviewed code categories, not an automatic regex purge |
| Heavy upper tail | In the provisional purchase scenario, median order value £302.55; 99th percentile £3,528.02; maximum £168,469.60 | High for unreviewed monetary fitting; large orders can be real | Inspect extreme orders and related credits; retain legitimate bulk behavior; report robust and aggregate errors |
| Value concentration | Top 1% of customers hold **32.09%** of provisional positive sales; top 10% hold **63.89%** | Material model-selection sensitivity | Evaluate customer-level accuracy and high-value slices, not only whole-base MAE |
| Invoice timestamp variation | 83 invoice IDs contain multiple timestamps; no invoice has multiple known customer IDs or countries | Medium for temporal boundaries | Investigate whether invoicing has revisions or split timestamps; retain the line timestamps until a policy is settled |
| Customer country variation | 13 customers have multiple countries | Low/medium for categorical features | Compute a history-only dominant country and retain a variation flag |
| Missing descriptions | 4,382 lines, 0.41% | Low for behavioral features; relevant if adding product text | Use stock codes for basic diversity; defer NLP features |
| Finite observation window | About two years, ending during 9 Dec 2011 | High for immature targets or lifetime claims | Require complete labels; avoid treating the final partial quarter as 90 days |

Counts are directly observed, while the reason for an individual ambiguous row remains unverified. In particular, a cancellation/credit is not proven to be a physical merchandise return, and a repeated within-sheet line is not proven to be an ingestion error.

### Temporal anomalies

- December 2010 contains 23,023 of the exact excess duplicate rows, consistent with the measured worksheet overlap. Reconcile before interpreting its monthly revenue or customer trends.
- Monthly missing-ID rates vary. For example, October 2010 is 14.45%, while January 2011 is 37.66%. A changing identifiable population can bias comparisons; counts come from the raw monthly profile, before overlap reconciliation.
- There are 135 calendar dates without rows; 104 are Saturdays. Do not interpret every absent day as failed ingestion. Trading schedules and holidays are plausible, but the file does not verify the cause.
- December 2011 is partial. First observed purchase cohorts are also left-censored: someone first seen in December 2009 may already have been an established customer.

## Provisional purchase scenario

For a conservative sufficiency check, require known customer ID, positive quantity and price, no `C` prefix, remove excess identical rows, and retain stock codes matching five digits plus optional letters. This is **not the recommended final cleaning policy**.

| Step | Remaining lines |
| --- | ---: |
| Positive, noncancel lines, including anonymous activity | 1,041,670 |
| Also require known customer | 805,549 |
| Also collapse excess exact repeated rows | 779,425 |
| Also apply provisional product-code pattern | 776,577 |

This scenario contains **36,594 purchase invoices and 5,852 customers**, with **4,234 repeat-order customers** and **1,618 one-order customers**. At the daily purchase-occasion grain there are **4,179 repeat buyers** and **1,673 one-day buyers**. Before the duplicate/product-code sensitivity filters, the identifiable positive noncancel purchase population contains **5,878 customers**. This is ample to attempt classical clustering, boosting and probabilistic CLV. It is several thousand customer histories, not a million independent customers; the effective sample size must guide model complexity.

## Forecast feasibility

The following counts use the same provisional scenario. Population: all customers with at least one qualifying purchase strictly before the cutoff. Future interval: `[cutoff, cutoff + horizon)`. Incomplete horizons have missing labels.

| Cutoff | Historical customers | 90-day inactivity | 180-day outcome end | 180-day label complete? |
| --- | ---: | ---: | --- | --- |
| 1 Jun 2010 | 2,684 | 50.30% | 28 Nov 2010 | Yes |
| 1 Sep 2010 | 3,299 | 41.68% | 28 Feb 2011 | Yes |
| 1 Dec 2010 | 4,239 | 66.81% | 30 May 2011 | Yes |
| 1 Mar 2011 | 4,512 | 64.87% | 28 Aug 2011 | Yes |
| 1 Jun 2011 | 4,908 | 67.05% | 28 Nov 2011 | Yes |
| 1 Sep 2011 | 5,224 | 57.20% | 28 Feb 2012 | **No** |

All six 90-day horizons above are fully observed under the conservative final-day policy. September's 90-day interval ends before 30 November 2011. At the September cutoff, restricting review eligibility to a purchase in the prior 365 days gives **4,324 customers** and **50.46%** inactivity, compared with 57.20% for all historical customers. Thus population eligibility changes the task, and a single fixed churn threshold does not settle the business definition.

These are label-feasibility observations, not classifier scores. Differences across cutoffs combine seasonality, cohort age and population composition; they do not establish a causal seasonal effect.

## What the data can and cannot support

| Desired claim | Fitness | Scope adjustment / additional evidence |
| --- | --- | --- |
| Behavioral customer segmentation | Sufficient | Quantify groups; treat trade/wholesale labels as inferred |
| Next-90-day purchase and revenue forecasts | Sufficient | Time-based testing and explicit purchase/revenue definition |
| Six-month value comparison | Sufficient with limitations | June-to-December calendar windows are fully observed; 180-day diagnostics have different boundaries; fewer independent seasonal tests |
| Probability of no purchase in a future period | Sufficient | Call it inactivity, not observed permanent churn |
| True profit-based customer lifetime value | Insufficient | Costs/margins, longer future observation and explicit discounting assumptions |
| Causal effect of retention offers / optimal campaign ROI | Insufficient | Treatment assignment, control outcomes, contact/redemption costs and preferably randomized experiments |
| Browsing-intent or omnichannel modeling | Insufficient | Sessions, product views, campaigns and channel attribution |
| Predicting value at acquisition for an unseen buyer | Different task | Current design requires observed purchase history; define acquisition features and cohorts separately |

No replacement is needed for the core project. If causal budget allocation becomes a required goal, the corrected Criteo uplift dataset is a documented randomized advertising benchmark with treatment and conversion outcomes. It would be a separate experiment, since it has no joinable retail invoice histories and cannot supply this retailer's CLV. Use the corrected release rather than its original leaked version. [Criteo's dataset and erratum](https://ailab.criteo.com/criteo-uplift-prediction-dataset/).

## Minimal quality gates for implementation

Require source hashes and sheet coverage; no duplicate raw lineage keys; explicit overlap reconciliation; one customer per purchase invoice; qualifying positive purchase values; feature dates strictly before cutoff; targets only after cutoff and only for mature horizons; unique customer/cutoff rows; and consistent counts/revenue before and after aggregation. Keep undefined single-order gaps missing, not zero. Retain cleaning-impact totals and raw-versus-reconciled sensitivity results.

## Reproduce

From the workspace root, with the provided workbook in its original location:

```powershell
python scripts/audit_dataset.py
```

The completed audit used Python 3.14.0, pandas 2.3.3, NumPy 2.3.5 and openpyxl 3.1.5. This is the audit environment, not a claim that all proposed modeling libraries have been tested together. The [companion notebook](../notebooks/00_dataset_scope_audit.ipynb) provides the same audit and bounded result tables. Its ordinary Python cells are executed sequentially by [the generator](../scripts/build_audit_notebook.py); this records CPython execution rather than an IPython/Jupyter kernel run. Jupyter is not installed in the research environment.

Validation status: all five Python cells completed and retain outputs; row totals, sample-population partitions, label bounds and local document links were checked. An HTML reading preview was generated, but visual inspection could not be completed: the first headless Edge attempt reported an unusable GPU process, and a targeted retry produced no screenshot. Inspect the notebook in Jupyter or open the [HTML preview](evidence/audit_notebook_preview.html) locally for presentation review. To run with a Jupyter kernel after installing Jupyter/nbconvert/ipykernel, use `python -m jupyter nbconvert --execute --to notebook --inplace notebooks/00_dataset_scope_audit.ipynb`.
