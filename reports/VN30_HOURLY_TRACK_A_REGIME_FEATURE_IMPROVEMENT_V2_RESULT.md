# VN30 Hourly Track A Regime Feature Improvement V2 Result

## Scope

- Setup: Track A canonical-like VN30 stock-only hourly setup.
- Main metric: pooled overall directional accuracy.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage for the selected candidate.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking substitution: no.
- Final-label selection: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## References

- Previous Logistic h40 baseline: 60.43%.
- Historical RF h60 reference: 60.31%.
- Main target: beat 60.43%.
- Aspirational target: 65%.

## Baseline Error Audit

- Logistic h40 error audit run: yes.
- Reproduced Logistic h40 final accuracy: 60.43%.
- Final rows: 4,074.
- Active ticker count: 30.
- Majority baseline: 50.44%.
- Worst ticker drag: SHB.
- Worst month drag: 2025-01.
- Worst quarter drag: 2025Q2.
- Logistic h40 beat the canonical h40 complex-model rows in final accuracy.

## Regime Feature V2

Leakage-safe v2 features included lagged market/index returns and trends, stock-minus-market return, rolling volatility, fixed lagged volatility-ratio regimes, bull/bear/sideway proxies from past index trend, volume shock, range shock, session/hour features, ticker encoding, and existing static sector encoding.

## Selected Candidate

Best validation-selected candidate:

- Model: XGBoost shallow.
- Horizon: h=40.
- Feature set: `regime_feature_v2`.
- Regime method: `global_v2`.
- Validation accuracy: 51.67%.
- Validation baseline accuracy: 48.79%.
- Validation delta vs baseline: +2.89 percentage points.
- Final accuracy: 56.58%.
- Final baseline accuracy: 50.44%.
- Final delta vs baseline: +6.14 percentage points.
- Final rows: 4,074.
- Final coverage: 100.00%.
- Active ticker count: 30.
- Delta vs 60.43: -3.85 percentage points.
- Delta vs 60.31: -3.73 percentage points.
- Pass 60: no.
- Pass 60.31: no.
- Pass 60.43: no.
- Pass 65: no.
- Selected on validation: yes.

Diagnostic final-only rows above 60.43 and 65 existed, led by L2 Logistic h40 at 65.71%, but they were not selected by validation and are not used as improvement or final65 evidence.

## Risk And Claim Level

- Overfit risk: low for the validation-selected candidate.
- Evidence: validation accuracy was 51.67% and final accuracy was 56.58%, but the selected candidate did not beat 60, 60.31, or 60.43.
- Audit status: completed.
- Claim level: exploratory negative improvement result.

Regime-aware and feature-improvement v2 did not improve the clean Track A Logistic h40 60.43% baseline under validation-only selection. Final65 remains not established.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
