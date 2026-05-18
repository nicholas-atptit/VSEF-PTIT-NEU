# VN30 Hourly Expanded Model Pool + Stacking V1 Claim Register

## Safe Claims

- The expanded model-pool screening and ensemble/stacking v1 run used the VN30 stock-only hourly target.
- The run used 30/30 active VN30 January 2025 stock tickers.
- The main metric was full-coverage pooled stock-only overall directional accuracy.
- Candidate and ensemble selection used validation metrics only.
- Final-window results were scoring-only.
- The best validation-selected ensemble was `stacking_lightgbm_shallow_oof`.
- The best validation-selected ensemble reached 50.76% final stock-only accuracy, below the locked RF h=60 baseline of approximately 60.31%.
- The run did not reach 65%.
- The result is exploratory negative evidence, not a benchmark success claim.

## Conditional Claims

- Any future improvement from this branch should remain exploratory if it is judged on the already inspected final window.
- A stronger claim would require later verification on future blind data or another pre-registered untouched final window.
- An improvement claim is conditional on being validation-selected, full-coverage, stock-only, and compared against the locked 60.31% baseline.

## Unsafe Claims

- Claiming final65 if the result is selected by final score.
- Presenting confidence-filtered accuracy as overall directional accuracy.
- Presenting a ticker subset as the full VN30 universe.
- Presenting top-k/ranking results as overall directional accuracy.
- Presenting index-only accuracy as stock accuracy.
- Presenting daily results as hourly results.
- Presenting a combined stock+index panel result as stock-only accuracy.
- Claiming trading profitability, investment recommendation value, or live-deployment readiness.

## Explicit Non-Claims

- No trading claim.
- No profitability claim.
- No investment recommendation.
- No live-deployment claim.
- No paper or DOCX claim.

