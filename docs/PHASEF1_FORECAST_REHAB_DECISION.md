# Phase F1 Forecast Rehab Decision

## Recommendation

Continue forecast rehab with narrowed feature/model scope.

## Why

- The medium matrix improved the Phase 2.6 picture enough to keep working on the forecast layer, but not enough to unblock Phase 3.
- The edge is still weak and unstable at daily frequency, yet it is not uniformly dead.
- The useful slices are concentrated in:
  - `small_banks` rather than `mixed_large_cap`
  - `horizon=5` and `horizon=10` rather than `horizon=1`
  - compact and technical-heavy feature families rather than broad context expansion
  - tree models plus strong statistical baselines rather than the full model roster

## Evidence From `artifacts/forecast_rehab`

- Best target in the bounded medium sweep: `direction_binary`
- Best horizon by forecast quality: `10`
- Best ticker group: `small_banks`
- Feature-family winners by slice were led by:
  - `technical_core` 3 wins
  - `reduced_compact` 3 wins
  - `long_lag` 2 wins
- Best-slice model winners were led by:
  - `xgboost` 5 wins
  - `random_forest` 3 wins
  - `lightgbm` 2 wins
  - `ets` 1 win
- Positive-sharpe share under the fixed Phase 2.6 baseline:
  - `horizon=10`: 48.46%
  - `horizon=5`: 22.31%
  - `horizon=1`: 6.92%
  - `small_banks`: 46.67%
  - `mixed_large_cap`: 5.13%

## What To Narrow Next

- Freeze the weakest rehab candidates to baseline-only status:
  - `naive`
  - `moving_average`
  - likely `linear`, `ridge`, `lasso` for primary rehab sweeps
- Keep the primary rehab focus on:
  - `xgboost`
  - `random_forest`
  - `lightgbm`
  - `ets`
  - `sarimax` as a guarded statistical comparator
- Prioritize:
  - `small_banks`
  - `horizon=5`
  - `horizon=10`
  - `technical_core`, `reduced_compact`, and selected `long_lag` variants

## Not Supported Yet

- No evidence yet that broader ticker expansion is justified.
- No evidence yet that `horizon=1` is economically viable after costs.
- No evidence yet that policy changes are the main blocker.
- No evidence yet to unblock Phase 3.
