# VN30 Hourly Track A Baseline60 Improvement V1 Result

## Scope

- Setup: Track A canonical-like VN30 stock-only hourly setup.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking substitution: no.
- Final-label selection: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Reference Results

- Previous Track A Logistic Regression h=40 result: 60.43%.
- Historical RF h60 reference: 60.31%.
- Previous delta vs 60.31: +0.12 percentage points.
- Claim level for the prior result: exploratory baseline60, not final65.

## Stability Audit

- Stability audit run: yes.
- Global final accuracy audited: 60.43%.
- Validation accuracy audited: 51.95%.
- Final majority baseline: 50.44%.
- Delta vs final majority baseline: +9.99 percentage points.
- Delta vs RF h60 60.31: +0.12 percentage points.
- Broad-based or concentrated: broad_based.
- Worst ticker: SHB.
- Worst month: 2025-01.

## Improvement V1

Best validation-selected improvement candidate:

- Model: Logistic Regression.
- Horizon: h=40.
- Feature set: `feature_set_C_closest`.
- Hyperparams id: `logit_l2_C1.0_none`.
- Threshold: 0.55.
- Validation accuracy: 52.60%.
- Validation majority/simple baseline accuracy: 48.79%.
- Validation delta vs baseline: +3.81 percentage points.
- Final accuracy: 58.96%.
- Final majority/simple baseline accuracy: 50.44%.
- Final delta vs baseline: +8.52 percentage points.
- Final rows: 4,074.
- Final coverage: 100.00%.
- Active ticker count: 30.
- Delta vs 60.31: -1.35 percentage points.
- Delta vs previous 60.43: -1.47 percentage points.
- Pass 60: no.
- Pass 60.31: no.
- Pass 65: no.
- Selected on validation: yes.

Diagnostic final-only rows above 60.31 existed in the candidate table, but they were not selected by validation and are not used as improvement evidence.

## Risk And Claim Level

- Overfit risk: low for the validation-selected improvement candidate.
- Evidence: validation accuracy was 52.60% and final accuracy was 58.96%, but the candidate did not beat 60 or 60.31.
- Audit status: completed.
- Claim level: exploratory negative improvement result.

This improvement attempt does not improve the prior Track A baseline60 result and does not beat the historical RF h60 60.31% reference. The baseline60 claim remains tied to the prior validation-selected Track A Logistic Regression h=40 result at 60.43%, with a narrow exploratory margin.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
