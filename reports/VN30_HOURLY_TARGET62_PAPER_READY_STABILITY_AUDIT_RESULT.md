# VN30 Hourly Target62 Paper-Ready Stability Audit Result

## Selected Candidate

- Model: L2 Logistic.
- Horizon: h=40.
- Feature set: `feature_set_C_closest`.
- Threshold: 0.50.
- Setup: Track A canonical-like VN30 stock-only hourly.
- Coverage: 30/30 stocks, full coverage.
- Training run: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Global Result

- Final accuracy: 61.51%.
- Total rows: 4,074.
- Correct predictions: 2,506.
- Incorrect predictions: 1,568.
- Majority baseline accuracy: 50.44%.
- Majority baseline lift: +11.07 percentage points.
- Delta vs 60.43 baseline: +1.08 percentage points.
- Delta vs 60.31 RF h60 reference: +1.20 percentage points.
- Pass 60: yes.
- Pass 60.43: yes.
- Pass 62: no.
- Pass 65: no.

## Average Accuracy

- Mean ticker accuracy: 65.26%.
- Median ticker accuracy: 65.44%.
- Mean month accuracy: 53.71%.
- Median month accuracy: 59.60%.
- Mean quarter accuracy: 56.29%.
- Median quarter accuracy: 54.60%.

## Rolling And Regime Summary

- Row-level rolling 250/500/1000 accuracy: unavailable because row-level predictions were not saved and this audit did not regenerate predictions.
- Monthly expanding accuracy: generated for future figures.
- Quarterly expanding accuracy: generated for future figures.
- Regime stability classification: regime_unstable.
- Regime slices with positive lift: 3/3.
- Regime slices above 60: 1/3.
- Regime slices above 62: 1/3.

## Robustness

- Bootstrap CI: 54.65% to 68.96%.
- Bootstrap source: ticker-weighted resample.
- Significance result: significant versus 50% and majority baseline by normal-approximate binomial tests.
- Validation-final gap: +9.63 percentage points.
- Validation-final mismatch: high_positive_final_gap.

## Classification

- Ticker stability classification: ticker_moderately_stable.
- Time stability classification: time_concentrated_or_mixed.
- Regime stability classification: regime_unstable.
- Overall stability classification: concentrated_or_mixed.
- Paper-ready claim level: improved_baseline60.
- Suitable for paper as baseline60 evidence: yes, exploratory.
- Suitable for target62 claim: no.
- Suitable for final65 claim: no.

## Recommended Paper Wording

Under the Track A canonical-like VN30 hourly setup, a pre-registered validation-selected L2 Logistic h40 model achieved 61.51% final pooled directional accuracy with full 30-stock coverage, exceeding the 60.43% Logistic h40 baseline by 1.08 percentage points. However, the result did not reach the 62% target and showed mixed time stability with a high positive validation-final gap, so it is reported as exploratory improved baseline60 evidence rather than target62 or final65 evidence.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
