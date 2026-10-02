# Customer feature definitions

Features are built separately at each configured cutoff from eligible transaction lines with `invoice_date < cutoff`. Future targets are built in a separate transform. Customer ID and cutoff are keys; they are not predictive inputs. Revenue is gross positive merchandise revenue in GBP.

| Feature | Definition | Notes |
| --- | --- | --- |
| `recency_days` | Fractional days from last observed eligible purchase line to cutoff | Same timezone-naive source time basis throughout |
| `tenure_days` | Fractional days from first observed eligible purchase line to cutoff | Observed-history span; not true customer age |
| `history_days` | `max(tenure_days, 1)` | Guards run-rate calculations |
| `purchase_count` | Distinct identified invoices with at least one eligible purchase line before cutoff | Lines are window-filtered before invoice aggregation |
| `purchase_day_count` | Distinct calendar days containing one or more eligible purchase invoices | Probabilistic repeat-purchase unit |
| `repeat_purchase_day_count` | `max(purchase_day_count - 1, 0)` | BG/NBD frequency convention |
| `lifetime_revenue_gbp` | Sum of eligible positive purchase line revenue before cutoff | Credits remain separate |
| `avg_order_value_gbp`, median, max | Invoice merchandise revenue summaries | Partial line inclusion at cutoffs prevents post-cutoff information crossing backward |
| `avg_basket_units`, `avg_basket_lines` | Means over customer invoices | A line count is not unique product count |
| `product_diversity` | Unique eligible stock codes over the observed purchase history | Product-code pattern is a conservative eligibility screen |
| `mean_days_between_purchase_days` | Mean gap across distinct purchase days | Missing for one purchase day |
| `purchase_days_per_30d` | Purchase days divided by observed-history exposure in 30-day units | Exposure bounded below by one month |
| `orders_30d`, `orders_90d`, `orders_180d`, `orders_365d` | Distinct invoice IDs with eligible lines in the end-exclusive trailing window | Same window operation is used for spend features |
| `spend_30d_gbp`, `spend_90d_gbp`, `spend_180d_gbp`, `spend_365d_gbp` | Gross eligible purchase revenue in the trailing window | Credits are not allocated against sales |
| `spend_recent90_over_prior90` | `(spend_90d + 1) / (spend_180d - spend_90d + 1)` | Describes direction relative to the preceding 90 days |
| `return_invoice_rate` | Identified C-prefixed negative-credit invoice count divided by purchase invoice count | A return/credit proxy, not a fully linked returned-item rate |
| `credit_amount_365d_gbp` | Absolute eligible identified C-prefixed negative-credit line amount in trailing year | Predictor only; never added to the purchase target |
| `revenue_per_history_day_gbp` | Lifetime purchase revenue divided by observed-history days | Simple historical run-rate feature |
| `zero_purchase_weeks_26w` | Count of zero-activity bins among 26 consecutive cutoff-anchored seven-day bins | Included in static classifiers |

The neural time series uses six weekly channels for `[cutoff - 182 days, cutoff)`: `orders`, `revenue_gbp`, `units`, `product_codes`, `has_purchase`, and `after_first_observed_purchase`. Counts, spend, units and breadth are log-transformed in the network's training-only preprocessor. True quiet weeks stay present as zero-activity observations.

Gaps before a customer's first observed purchase in the workbook are not treated as pre-acquisition truth. The sequence indicator refers to first observed history, not verified acquisition. Country is the customer's last known invoice country at that cutoff; it is a changing descriptive category and is excluded from customer identity.

The purchase policy in [data_policy.json](../configs/data_policy.json) is a scenario, not a universally correct product taxonomy. Special service/charge codes and negative lines remain visible in the cleaning ledger so that the effect of this policy can be reviewed.
