# Repository Refactor Result Summary

## What was refactored

- Centralized strict feature-and-target timestamp split discipline.
- Centralized claim-boundary wording and protected-artifact policy.
- Added canonical direction, return/price, interval/range, and ranking metrics.
- Added reusable direction, return/price, ATR/volatility, and quantile-range
  baselines.
- Added point-in-time-safe common feature builders.
- Added canonical forecast-panel schema and validation-only selectors.
- Added repository path, timestamp, research serialization, and artifact
  manifest helpers.
- Removed duplicated split constants and research-I/O helpers from the QML
  runner; removed indirect Model Universe imports of those helpers.

## Created modules

- `src/governance/`
- `src/evaluation/metrics/`
- `src/evaluation/baselines/`
- `src/features/builders/`
- `src/forecasting/panels/`
- `src/forecasting/selectors/`
- `src/utils/paths.py`, `research_io.py`, `timestamps.py`,
  `artifact_manifest.py`

## Scripts simplified

- `scripts/research/run_vn30_qml_forecasting.py`
- `scripts/research/run_vn30_model_universe_direction_price_benchmark.py`

## Imports updated

See `reports/cleanup/REFACTOR_IMPORT_UPDATE_LOG.md`.

## Files moved or deleted

None. The migration map records the conservative no-move decision. No deprecated
files were deleted.

## Protected evidence

All existing protected roots were preserved. Existing QML and Model Universe
generated evidence, data, outputs, archive, paper evidence exports, and
`reports/_index/ACTIVE_EVIDENCE_INDEX.md` remain accessible.

Named paths that did not exist before or after this refactor:

- `reports/project_review/`
- `reports/paper/qml_kernel_feature_vn30/`
- `reports/generated/vn30_index_group_range_forecast/`
- `reports/generated/vn_forecast_engine_v1/`

## Validation status

Passed:

- runtime preflight: 48 OK, 12 environment/local-service warnings, 0 failures
- provider usage policy
- provider usage policy tests: 2 passed
- VN price gateway contract tests: 7 passed
- directional accuracy metric tests: 12 passed
- new focused refactor tests: 11 passed
- QML and Model Universe runner compile checks
- QML and Model Universe `--help` import/CLI smoke checks
- all new reusable-module compile checks
- repository hygiene check

Repository hygiene initially found one pre-existing local absolute path in
`reports/cleanup/V7_BROKEN_WORKTREE_NOTE.md`; it was normalized and the check was
rerun successfully.

Focused Ruff lint was unavailable because Ruff is not installed in the active
Python environment.

## Smoke forecast status

Skipped because `scripts/research/run_vn_forecast_engine_v1.py` does not exist.
No forecast results were invented.

## Remaining technical debt

- QML and Model Universe runners remain very large and retain QML-specific
  cross-runner imports.
- No offline forecast-engine or index-group range-lab runner currently exists.
- `config/` and `configs/` remain separate roots.
- Historical scripts contain additional duplicated metric/feature/baseline
  logic; migrating them requires evidence-specific regression tests.
- The required GitHub remote push stalled while transmitting the repository's
  large history and must be retried after the final commit.

## Exact claim boundary

See `reports/cleanup/REFACTOR_CLAIM_BOUNDARY.md`.
