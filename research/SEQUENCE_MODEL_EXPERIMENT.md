# LSTM and GRU experiment

Added following the request to consider LSTM. This is a concrete experiment specification; no neural model has been trained or benchmarked yet.

The broader [model portfolio](MODEL_PORTFOLIO.md) adds GAM, survival and probabilistic revenue alternatives, and records the user requirement to preserve all runs while choosing scoring champions from development evidence.

## Decision and hypothesis

Include a small, shared LSTM as an additional **90-day inactivity classifier**, plus a GRU comparison. Keep the brief's segmentation, probabilistic value models and tabular benchmarks. The hypothesis is that the order of spending changes and quiet periods provides information beyond aggregated recency/frequency/spend features. A neural architecture alone does not establish that advantage.

For example, two customers with similar six-month totals may have increasing versus declining purchase activity. A sequence model can represent those different histories. Its output remains probability of no qualifying purchase in the next 90 days, rather than a claim of permanent churn.

## Dataset fitness

The conservative audit scenario contains 5,852 identifiable purchasing customers, 36,594 invoices and 4,179 customers with more than one purchase day. These are provisional cleaning-scenario counts, not final reconciled model populations. See [aggregate evidence](evidence/dataset_audit.json) and [cleaning qualifications](DATASET_ASSESSMENT.md).

The raw million-plus rows are product lines within invoices, not independent customer sequences. Many customer histories are short. Repeated customer/cutoff examples add training cases but do not add independent customers, and overlapping histories remain correlated. This supports testing a compact network; it does not justify assuming a large deep model will outperform boosting. A different dataset is not needed before this experiment.

Before fitting, report purchase-day count, nonzero-week count, observed-history length and fraction of quiet weeks by forecast cutoff. Include one-purchase and repeat-purchase slices. Do not exclude short histories to make the network look better; if a repeat-buyer-only experiment is useful, give every comparator the same population and report it separately.

## Input contract

Use one example per eligible customer and forecast cutoff. Build 26 consecutive seven-day bins covering `[cutoff - 182 days, cutoff)`, in chronological order. Bins are anchored to the cutoff, rather than mixing partial calendar weeks. The earliest proposed cutoff, 1 June 2010, has this window starting on 1 December 2009.

Initial weekly channels:

| Channel | Definition |
| --- | --- |
| Purchase orders | Distinct qualifying invoices from historical purchase lines |
| Merchandise spend | Positive qualifying purchase revenue in the bin, GBP |
| Purchased units | Quantity on qualifying purchase lines |
| Product breadth | Distinct qualifying product codes bought in the bin |
| Has purchase | Indicator that at least one qualifying purchase occurs |
| After first observed purchase | Indicator that the bin contains or follows the customer's first observed purchase before cutoff |

The last indicator distinguishes pre-first-observed-purchase time from later quiet periods; first observed purchase is not verified acquisition. Fixed-length bins stay present even when no purchase occurs. Do not mask genuine zero-purchase weeks: inactivity is part of the signal. Use explicit masks only if a later variable-length representation introduces artificial padding. [Keras masking guide](https://www.tensorflow.org/guide/keras/understanding_masking_and_padding).

Apply `log1p` to nonnegative count/spend channels and fit scaling on training inputs only. Keep indicators unscaled. Use static recency at cutoff, observed tenure and historical frequency/spend as a second input branch; it gives the network access to history older than 26 weeks. No customer ID predictor, future-derived features or full-workbook normalizers.

Filter purchase lines by timestamp before aggregation, respecting the audited multi-timestamp invoices. Reuse the established merchandise/credit policy and customer eligibility rules. Credits can become a separate weekly channel later if its definition and development benefit are established.

An alternative is the last N purchase events with elapsed time between events, basket value and breadth. That representation must also include time from last purchase to cutoff: elapsed gaps between purchases alone omit the current quiet period. Start with weekly bins because they represent that period directly.

## Small model and comparison budget

Proposed architecture:

```text
26 weeks x 6 channels -> LSTM(32) ----+
                                     +-> concatenate -> Dense(16, ReLU)
scaled static customer features -----+                  -> Dropout(0.2)
                                                        -> sigmoid(inactive_90d)
```

Use one global network trained across customers, `stateful=False`, binary cross-entropy, Adam and development-only early stopping. Start with no class weighting. Retain temporal order inside examples; training examples may be shuffled within the training partition. LSTM's expected tensor is `(batch, timesteps, features)`. [Official LSTM documentation](https://keras.io/api/layers/recurrent_layers/lstm/).

Fit the same architecture with GRU(32) as the recurrent alternative. [Official GRU documentation](https://keras.io/api/layers/recurrent_layers/gru/). Initial recurrent search: 16 versus 32 units, fixed 26 weeks, three recorded random seeds, maximum 100 epochs and early-stopping patience 10. Record seeds, epochs, runtime and parameter counts. Select configurations using development results across seeds; do not select a lucky seed using final outcomes. For the final refit, fix epoch count from development, for example the median best epoch across those seeds.

| Comparator | Purpose |
| --- | --- |
| Recency heuristic and logistic regression | Establish simple risk baselines |
| Required RF, CatBoost and XGBoost on established customer features | Meet the brief and establish tabular performance |
| CatBoost with flattened weekly channels plus the same static features | Control for the extra temporal input information |
| Small MLP with flattened weekly channels plus static features | Control for adding a neural network without recurrence |
| LSTM and GRU with weekly and static inputs | Test recurrent representations |

Preprocessing is fitted within each training partition. Compare models on identical customer/cutoff keys and labels. A comparison against aggregate-only boosting alone would confound additional weekly information with architecture.

## Validation and acceptance

Reuse the [90-day evaluation schedule](PROJECT_DESIGN.md): train on June/September/December 2010; select features, architecture and epochs on March 2011; refit on matured training plus March examples; calibrate on June 2011; freeze for September 2011 testing. Ensure all fitting labels end before the corresponding forecast origin. Final refitting must not early-stop on calibration or test labels.

Report average precision, ROC-AUC, Brier score, log loss, reliability, and precision/recall at the same development-selected contact capacity. Show seed variability and customer-bootstrap uncertainty for differences, plus short-history and new-to-training-customer slices. Select the candidate before viewing final test performance. A failed final generalization result is evidence, not an invitation to retune on September outcomes.

Keep recurrence in the final system if development evidence supports useful improvements in ranking/calibration at acceptable runtime and the final test corroborates generalization. Otherwise retain it as a documented comparison and use the better-supported baseline. Tree SHAP results do not explain the LSTM; initially use aggregate sequence ablations and clearly labeled feature perturbation diagnostics rather than claiming an equivalent explanation.

## Possible value extension

After the inactivity comparison, consider a separate neural revenue experiment. One option is a hurdle model: estimate return probability and conditional positive revenue, multiplying those components once. Another is a zero-inflated lognormal (ZILN) head representing zero revenue and heavy-tailed positive revenue. The original CLV paper describes ZILN for linear and deep models; it is not evidence that LSTM wins on this workbook. [Wang et al., A Deep Probabilistic Model for Customer Lifetime Value Prediction](https://arxiv.org/abs/1912.07753).

For a lognormal positive component with predicted log-location `mu` and log-scale `sigma`, expected revenue is `p_return * exp(mu + sigma^2 / 2)`, subject to fit and numerical checks. Use the same 90-day merchandise target and revenue metrics as other models. Defer multitask value/risk architectures until the simpler experiment is validated.

## Implementation additions

Add proposed `sequences.py` for cutoff-safe tensors, `sequence_models.py` for model construction, and an experiment notebook that calls those modules. Save preprocessing, channel order, bin boundaries, static feature definitions, model weights and run manifests together. Check that adding a future event cannot change an earlier tensor, every bin is before cutoff, quiet weeks remain valid, and target labels match tabular examples.

Use an isolated compatible modeling environment and smoke-test Keras with a supported backend before locking dependencies. The current Python 3.14 audit environment has not been validated for neural modeling. Start on CPU with a tiny fit; hardware requirements and runtime should be measured rather than guessed.
