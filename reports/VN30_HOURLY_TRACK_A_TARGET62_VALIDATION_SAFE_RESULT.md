# VN30 Hourly Track A Target62 Validation-Safe Result

## Scope

- Setup: Track A canonical-like VN30 stock-only hourly setup.
- Main metric: pooled overall directional accuracy.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking substitution: no.
- Final-label selection: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## References

- Baseline60 Logistic h40: 60.43%.
- Historical RF h60 reference: 60.31%.
- Main target: target62, defined as final accuracy >=62%.
- Final65: stretch only.
- Diagnostic 65.71 row: not retroactively claimable.

## Selected Candidate

Selected by the pre-registered validation-only rule:

- Model: L2 Logistic.
- Horizon: h=40.
- Feature set: `feature_set_C_closest`.
- Threshold: 0.50.
- Validation accuracy: 51.88%.
- Validation majority/simple baseline: 48.79%.
- Validation delta vs baseline: +3.09 percentage points.
- Final accuracy: 61.51%.
- Final majority/simple baseline: 50.44%.
- Final delta vs baseline: +11.07 percentage points.
- Final rows: 4,074.
- Final coverage: 100.00%.
- Active ticker count: 30.
- Delta vs 60.43: +1.08 percentage points.
- Delta vs 60.31: +1.20 percentage points.
- Pass 60: yes.
- Pass 60.43: yes.
- Pass 62: no.
- Pass 65: no.
- Selected by pre-registered validation rule: yes.

## Mandatory Audit

- 30/30 ticker coverage: yes.
- Full coverage: yes.
- Leakage audit passed: yes.
- Overfit risk: low.
- Validation-final mismatch: high_positive_final_gap.
- Validation-final gap: +9.63 percentage points.
- Stability by ticker/month/quarter/regime: concentrated_or_mixed.
- Lift over majority/simple baseline: +11.07 percentage points.

Stability details:

- Tickers above 60%: 18.
- Tickers above 62%: 18.
- Months above 60%: 7.
- Months above 62%: 7.
- Quarters above 60%: 2.
- Quarters above 62%: 2.
- Regime slices with positive lift: 3/3.

## Claim Level

- Claim level: exploratory baseline60 improvement, not target62.
- Evidence: the candidate was selected by the pre-registered validation-only rule and beat 60.43, but did not reach 62 and showed a high positive validation-final gap.
- Target62: not reached.
- Final65: not established.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
