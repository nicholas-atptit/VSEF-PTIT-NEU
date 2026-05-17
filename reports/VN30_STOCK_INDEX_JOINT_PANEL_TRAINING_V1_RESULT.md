# VN30 Stock + Index Joint Panel Training V1 Result

## Joint Universe

This run targets a joint 36-instrument panel:

- 30 active VN30 January 2025 stock tickers.
- 6 supported market indices: `VNINDEX`, `VN30`, `HNXINDEX`, `HNX30`, `UPCOMINDEX`, `VN100`.

Indices are defined as prediction targets and rows in the panel, not merely context features.

## Execution Status

- Readiness audit: failed for a validation-safe 36/36 hourly panel.
- Joint panel training v1: not run; gated by readiness failure.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Readiness Finding

The readiness audit found:

- Stock instruments present: 30.
- Index instruments present: 6.
- Total instruments present: 36.
- Usable stock instruments under the strict h=120 split rule: 0/30.
- Usable index instruments under intraday-hourly frequency checks: 1/6.
- Joint panel can run with 36/36 instruments: false.

Primary blocker:

- Stock cached hourly files contain 111 rows per ticker, which is insufficient for the requested h=120 horizon with train/validation/final splits.
- Five supported index cache files are marked hourly but contain only midnight timestamps, so they are not validation-safe intraday hourly rows for this joint panel.

## Baselines

Deterministic baselines were computed as diagnostics only. They are not a trained joint-panel result.

- Best deterministic combined baseline in the diagnostic table: majority/always-up h=120 at 81.28%, driven by index-only rows because stock-only final rows are unavailable at h=120.
- Stock-only RF h=60 historical reference: 60.31%; reference only, not a joint-panel result and not a replacement for combined 36-instrument scoring.
- Existing index benchmark results are reference-only and cannot replace combined 36-instrument scoring.

## Training Result

- Best validation-selected candidate: none.
- Final combined accuracy: not available.
- Final stock-only accuracy: not available.
- Final index-only accuracy: not available.
- Final coverage: not available.
- Delta vs combined baseline: not available.
- Combined >60 reached: no.
- Combined 65 reached: no.
- Stock-only >60 reached: no.
- Stock-only 65 reached: no.

## Safety And Claim Boundary

Result status: unsafe for benchmark claims.

The current cache cannot support a validation-safe full 36-instrument hourly joint-panel claim. Any future improvement must first fix readiness under protocol, then rerun training with validation-only selection and scoring-only final evaluation.

Because prior final windows have been inspected repeatedly, any future improvement from this branch should remain exploratory unless later verified on future blind data.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
