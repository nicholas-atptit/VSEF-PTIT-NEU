# VN30 Stock + Index Joint Panel Training V1 Claim Register

## Safe Claims

- The protocol defines a joint 36-instrument benchmark with 30 VN30 stocks and 6 supported indices.
- Indices are intended as panel rows and prediction targets, not merely side features.
- The readiness audit found the current cache is not validation-ready for a 36/36 intraday-hourly joint-panel run.
- Training v1 was gated and did not fit models because readiness failed.
- No final combined, stock-only, or index-only trained-model accuracy is claimed.

Safe future claims require all of the following:

- Validation-selected candidate.
- Full 36-instrument coverage.
- No confidence abstention.
- No instrument subset.
- No top-k/ranking substitution.
- Leakage audit passed.
- Baseline comparison included.
- Combined, stock-only, and index-only results reported separately.

## Conditional Claims

- If a future run improves combined accuracy, it must disclose whether index rows materially lift the combined score.
- If a future run reaches >=60 combined accuracy, the claim remains combined-panel only unless stock-only accuracy is separately reported and supported.
- If a future run reaches >=65 combined accuracy, final65 remains exploratory unless selected by validation, audited, and later verified on future blind data.

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

- Combined 36-instrument trained-model result: not available.
- Stock-only trained-model result: not available.
- Index-only trained-model result: not available.
- Combined >=60 claim: not supported.
- Combined final65 claim: not supported.
- Stock-only >=60 claim: not supported by this run.
- Stock-only final65 claim: not supported by this run.
