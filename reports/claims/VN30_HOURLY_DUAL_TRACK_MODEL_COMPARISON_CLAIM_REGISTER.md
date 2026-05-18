# VN30 Hourly Dual-Track Model Comparison Claim Register

## Safe Claims

- Track A claims apply only under the canonical historical setup.
- Track A uses VN30 stock-only hourly overall directional accuracy with 30/30 active January 2025 VN30 tickers.
- Track A validation-selected `logistic_regression` h=40 reached 60.43% final accuracy, narrowly above the locked 60.31% baseline, but remains exploratory because RF h60 was not exactly reproduced by this runner.
- Track B claims apply only under the current broader-pipeline setup.
- Track B reproduced the current RF h60 baseline around 56.12%.
- Track B validation-selected ensemble reached 50.76% final accuracy and did not beat the current 56.12% baseline.
- Track B best-final diagnostic row reached 58.77%, but this is not a validation-selected improvement claim.
- The two tracks are not interchangeable.

## Conditional Claims

- A Track A improvement claim requires stating the canonical setup, the closest feature-set C implementation, and the RF h60 reproduction caveat.
- A Track B improvement claim requires validation-selected improvement over 56.12 under the current broader-pipeline setup.
- Any future stronger claim should be verified on future blind data or a pre-registered untouched final window.

## Unsafe Claims

- Claiming a Track B result as an apples-to-apples improvement over 60.31.
- Claiming Track B as an improvement over 60.31 unless it explicitly beats 60.31 and is labeled non-apples-to-apples.
- Claiming a best-final diagnostic row as a selected model result.
- Claiming confidence-filtered accuracy as overall directional accuracy.
- Claiming a ticker subset as full VN30.
- Claiming top-k/ranking as overall directional accuracy.
- Claiming index-only accuracy as stock accuracy.
- Claiming daily results as hourly results.
- Claiming trading profitability, investment recommendation value, or live-deployment readiness.

## Explicit Non-Claims

- No trading claim.
- No profitability claim.
- No investment recommendation.
- No live-deployment claim.
- No paper or DOCX claim.

