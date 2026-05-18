# VN30 Hourly Track A Selection Mismatch Analysis

## Diagnostic Row

- Candidate: L2 Logistic h40.
- Feature set: `regime_feature_v2`.
- Regime method: `global_v2`.
- Status: diagnostic only.
- Selected on validation: no.
- Can be claimed: no.

## Validation Versus Final

- Validation accuracy: 51.18%.
- Validation majority baseline: 48.79%.
- Validation delta vs baseline: +2.39 percentage points.
- Selected XGBoost v2 validation accuracy: 51.67%.
- Validation gap versus selected XGBoost v2: -0.50 percentage points.
- Final accuracy: 65.71%.
- Final majority baseline: 50.44%.
- Final delta vs baseline: +15.27 percentage points.

Validation did not strongly reject L2 Logistic h40. It had a positive validation lift, but the registered validation-accuracy selection rule preferred XGBoost shallow h40 by about 0.50 percentage points.

## Breadth

- Broad-based or concentrated: concentrated_or_mixed.
- Tickers above 60%: 19/30.
- Tickers above 65%: 19/30.
- Months above 60%: 11/15.
- Quarters above 60%: 2/5.
- Worst ticker: VIC.
- Worst month: 2025-01.

The final 65.71% result is cross-sectionally broad across many tickers, but not time-broad across quarters. The quarter-level weakness means it should not be treated as stable final65 evidence.

## Class Balance

- Final target up rate: 49.56%.
- Final prediction up rate: 58.57%.
- Final majority baseline: 50.44%.
- Final delta vs majority baseline: +15.27 percentage points.

The final 65.71% result is not mostly a global class-balance artifact because it beats the final majority baseline by a large margin. However, class balance still matters by slice: several weak ticker and month slices have negative deltas versus the same majority baseline.

## Interpretation

This is best classified as a validation-selection mismatch with material final-window luck risk.

It is not a clean success claim because the candidate was not selected by the registered validation rule. It is also not pure rejection because validation lift was positive and close to the selected model. The final strength may be a real signal from simple regularized linear modeling on v2 features, but the mixed quarter-level pattern and repeated final-window inspection require future validation.

## Future Pre-Registered Rule

A future rule could select this type of candidate without seeing final labels only if it is specified before final scoring. One candidate rule:

- Restrict the eligible family to simple regularized linear models on `regime_feature_v2`.
- Require 30/30 tickers and full coverage.
- Require positive validation lift over the train-majority baseline.
- Require validation accuracy within 0.75 percentage points of the best validation candidate in the same horizon set.
- Prefer h=40 over longer horizons when validation scores are within the tolerance.
- Tie-break by model simplicity: L2 Logistic before ElasticNet, before tree models.
- Final window remains scoring-only.

This rule is not retroactive and cannot make the current 65.71% row claimable. It can only be used in a future pre-registered rerun or future blind test.
