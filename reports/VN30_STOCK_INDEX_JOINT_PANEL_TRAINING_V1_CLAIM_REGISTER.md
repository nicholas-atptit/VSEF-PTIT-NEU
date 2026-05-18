# VN30 Stock + Index Joint Panel Training V1 Claim Register

## Safe Claims

- The corrected joint panel contains 30 January 2025 VN30 stocks and 6 supported indices.
- The corrected 36/36 readiness gate passed before training.
- Training v1 ran on the full joint panel with validation-only selection and scoring-only final evaluation.
- The selected candidate was `lightgbm__h40__own_plus_market_context`.
- Final combined, stock-only, and index-only results are reported separately.
- The selected candidate did not beat the selected-candidate h=40 combined baseline and did not reach the 60% or 65% combined targets.
- No confidence abstention, instrument subset, top-k/ranking substitution, or daily-to-hourly substitution was used for the main result.

Safe combined-result claims require validation-selected, full-coverage, no-leakage evaluation with baseline comparison included. Stock-only and index-only slices must remain separately labeled.

## Conditional Claims

- Combined improvement claims are conditional because the final window has been inspected repeatedly.
- Any combined result where index rows materially lift the score must disclose that lift and must not be described as stock-only performance.
- A future combined >=60 result would be a combined-panel claim only unless the stock-only slice also supports the statement.
- A future final65 claim would remain exploratory unless selected by validation, audited, and later verified on future blind data.

## Unsafe Claims

- Claiming combined 36-instrument accuracy as VN30 stock-only accuracy.
- Claiming index-only accuracy as stock accuracy.
- Claiming final65 if selected by final score.
- Treating confidence-filtered accuracy as full overall accuracy.
- Treating an instrument subset as the full 36-instrument universe.
- Treating top-k/ranking output as directional accuracy.
- Treating daily results as hourly results.
- Claiming trading readiness, profitability, investment recommendation, or live deployment.

## Current V1 Claim Status

- Combined 36-instrument trained-model result: 49.21%, exploratory.
- Stock-only trained-model result: 48.99%, separately reported.
- Index-only trained-model result: 49.56%, separately reported.
- Combined baseline delta: -4.44 percentage points.
- Combined >=60 claim: not supported.
- Combined final65 claim: not supported.
- Stock-only >=60 claim: not supported.
- Stock-only final65 claim: not supported.
- Trading/profitability/live-deployment claim: not made.
