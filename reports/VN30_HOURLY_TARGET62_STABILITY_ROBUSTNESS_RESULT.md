# VN30 Hourly Target62 Stability Robustness Result

## Scope

- Selected candidate: L2 Logistic h40, `feature_set_C_closest`, threshold 0.50.
- Final accuracy: 61.51%.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Training run: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Global Result

- Final accuracy: 61.51%.
- Final rows: 4,074.
- Majority/simple baseline: 50.44%.
- Lift over majority/simple baseline: +11.07 percentage points.
- Delta vs 60.43 baseline: +1.08 percentage points.
- Delta vs 60.31 RF h60 reference: +1.20 percentage points.

## Average Stability

- Average ticker accuracy: 65.26%.
- Median ticker accuracy: 65.44%.
- Average month accuracy: 53.71%.
- Median month accuracy: 59.60%.
- Average quarter accuracy: 56.29%.
- Median quarter accuracy: 54.60%.

## Breadth

- Tickers above 60: 18/30.
- Tickers above 62: 18/30.
- Months above 60: 7/15.
- Months above 62: 7/15.
- Quarters above 60: 2/5.
- Quarters above 62: 2/5.
- Regime slices with positive lift: 3/3.
- Regime slices above 60: 1/3.
- Regime slices above 62: 1/3.

## Robustness

- Bootstrap CI, ticker-weighted: 54.65% to 68.96%.
- Bootstrap mean: 61.69%.
- Significance test result: significant versus 50% and majority baseline by normal-approximate binomial tests.
- Validation-final mismatch: high_positive_final_gap.
- Validation-final gap: +9.63 percentage points.
- Drawdown proxy: max 3 consecutive months below 50% accuracy.

## Classification

- Stability classification: concentrated_or_mixed.
- Improved baseline60 claim: yes, exploratory only.
- Target62 claim: no.

The result is strong enough to document as an exploratory improved baseline60 result because it was selected by a pre-registered validation rule, retains 30/30 full coverage, passes leakage controls, and beats the 60.43% baseline. It is not stable enough for a target62 claim because final accuracy is below 62%, monthly and quarterly averages are below 60%, and the validation-final gap is high.

## Recommended Next Step

Run a narrow V2 attempt only if the selection rule remains validation-only and adds a stability gate before final scoring, such as requiring validation lift plus minimum validation stability by month/regime. Final65 remains a future-blind or pre-registered stretch target.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
