# VN30 Hourly Selected Candidate Stability Audit Result

## Selected Candidate

- Model: L2 Logistic.
- Horizon: h=40.
- Feature set: `feature_set_C_closest`.
- Threshold: 0.50.
- Setup: Track A canonical-like VN30 stock-only hourly.
- Coverage: 30/30 stocks, full coverage.
- Row-level predictions regenerated: yes, for the fixed selected candidate only.
- New model selection: no.
- New tuning: no.
- Broad benchmark sweep: no.
- Data fetch: no.
- Paper/DOCX generated or rewritten: no.
- Trading/profitability/live-deployment claim: no.

## Reproduction Result

- Original final accuracy: 61.51%.
- Reproduced final accuracy: 61.51%.
- Reproduction difference: 0.000000 percentage points.
- Original final rows: 4,074.
- Reproduced final rows: 4,074.
- Original majority baseline accuracy: 50.44%.
- Reproduced majority baseline accuracy: 50.44%.
- Original majority baseline lift: +11.07 percentage points.
- Reproduced majority baseline lift: +11.07 percentage points.
- Reproduction status: passed.

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

## Rolling Stability

Rolling rows are sorted by `datetime`, then `ticker`. Rolling majority baseline is the majority-class rate inside each rolling window.

| Window | Windows | Min Accuracy | Max Accuracy | Mean Accuracy | Median Accuracy | Endpoint Accuracy | Below 50% | Below 55% | Below 60% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 250 rows | 3,825 | 7.60% | 92.80% | 63.14% | 63.60% | 48.00% | 663 | 916 | 1,339 |
| 500 rows | 3,575 | 28.40% | 89.40% | 63.03% | 62.80% | 65.00% | 557 | 736 | 1,259 |
| 1000 rows | 3,075 | 41.50% | 81.90% | 63.30% | 62.10% | 76.20% | 452 | 691 | 1,087 |

Longest below-60 episodes:

- 250-row windows: 666 windows, endpoint dates 2025-04-28 to 2025-09-12.
- 500-row windows: 685 windows, endpoint dates 2025-06-13 to 2025-10-01.
- 1000-row windows: 813 windows, endpoint dates 2025-08-29 to 2025-11-06.

## Average Accuracy

- Mean ticker accuracy: 65.26%.
- Median ticker accuracy: 65.44%.
- Mean month accuracy: 53.71%.
- Median month accuracy: 59.60%.
- Mean quarter accuracy: 56.29%.
- Median quarter accuracy: 54.60%.
- Monthly slices below 60%: 8/15.
- Quarterly slices below 60%: 3/5.

## Robustness

- Bootstrap CI from the earlier paper-ready audit: 54.65% to 68.96%.
- Bootstrap source: ticker-weighted resample.
- Significance result from the earlier paper-ready audit: significant versus 50% and majority baseline by normal-approximate binomial tests.
- Validation-final gap: +9.63 percentage points.
- Validation-final mismatch: high_positive_final_gap.

The final score is 9.63 percentage points above validation accuracy; therefore, the result must be interpreted cautiously.

## Classification

- Ticker stability classification: ticker_moderately_stable.
- Time stability classification: time_concentrated_or_mixed.
- Regime stability classification: regime_unstable.
- Overall stability classification: concentrated_or_mixed.
- Paper-ready claim level: improved_baseline60.
- Suitable for paper as baseline60 evidence: yes, exploratory.
- Suitable for target62 claim: no.
- Suitable for final65 claim: no.

## Boundary Wording

Under the Track A canonical-like VN30 hourly setup, a pre-registered validation-selected L2 Logistic h40 model reproduced the 61.51% final pooled directional accuracy with full 30-stock coverage and 4,074 final rows. Row-level rolling checks over 250, 500, and 1000 rows were generated, but they show substantial time variation and many windows below 60%. The result remains exploratory improved baseline60 evidence rather than target62 or final65 evidence.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
