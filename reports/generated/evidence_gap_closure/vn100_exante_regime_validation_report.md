# VN100 Ex-Ante Regime Validation Report

## Source

- Official artifact directory: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Prediction inputs: `daily/predicted_vs_actual.csv` and `hourly/predicted_vs_actual.csv`.
- No model training, provider fetch, or benchmark rerun was performed.

## Ex-Ante Rule

For each ticker/frequency/model/horizon sequence, the ex-ante regime uses the mean of the prior 20
realized target returns, shifted by one row. At least 5 prior observations are required. The bear
threshold is <= -3.00%; the bull threshold is >= 3.00%; remaining labeled rows are sideways.
Rows without enough prior history are marked `insufficient_history`.

## Key Bear-Regime Diagnostic Recheck

| frequency | exante_regime | model | horizon | n_obs | accuracy | passed_60pct | passed_63pct | reliable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | bear | lightgbm | 20 | 309 | 66.34% | True | True | True |
| daily | bear | xgboost | 20 | 309 | 65.05% | True | True | True |

## Ticker Stability of Key Ex-Ante Bear Slices

| slice | ticker_count | reliable_tickers | tickers_passing_63pct |
| --- | --- | --- | --- |
| daily lightgbm h=20 exante bear | 7 | 7 | 4 |
| daily xgboost h=20 exante bear | 7 | 7 | 4 |

## Required Answers

- Bear-regime 63%+ survives ex-ante validation: yes for the listed reliable key slice(s).
- Regime effect stability across windows: not established because 2022-2024 official windows are unavailable.
- Regime effect stability across tickers: partially checkable from the 2025 rows; see ticker-stability table.
- Regime rules usable before prediction time: the derived rule is lagged and therefore usable before the current prediction row, but it remains a proxy requiring deployment-quality validation.

## Comparison Output

- Ex-ante summary CSV: `reports/generated/evidence_gap_closure/vn100_exante_regime_accuracy_summary.csv`.
- Post-hoc versus ex-ante comparison CSV: `reports/generated/evidence_gap_closure/vn100_regime_posthoc_vs_exante_comparison.csv`.

## Claim Boundary

This analysis can upgrade the regime evidence from purely post-hoc to lagged-rule diagnostic evidence where rows are reliable.
It still does not establish a global 60% pass, multi-window stability, full-market representativeness, or trading readiness.
