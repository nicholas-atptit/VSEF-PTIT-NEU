# Refactor Import Update Log

## Changed imports

- `scripts/research/run_vn30_qml_forecasting.py`
  - imports strict split timestamps from `src.governance.split_policy`
  - imports JSON/CSV/Markdown, finite-float, and relative-path helpers from
    `src.utils.research_io`
  - removes duplicated local implementations of those helpers
- `scripts/research/run_vn30_model_universe_direction_price_benchmark.py`
  - imports strict split timestamps from `src.governance.split_policy`
  - imports stable research-I/O helpers from `src.utils.research_io`
  - no longer obtains those utilities indirectly from the QML runner

## Unchanged imports

- Model Universe still imports QML-specific feature/kernel/model helpers from the
  QML runner. Moving those 4,000+ lines safely requires a dedicated
  evidence-preserving migration.
- Historical one-off research runners remain at their current paths.
- Existing `src/ml/`, `src/forecast/`, and `src/engine/` imports remain valid.

## Unresolved references

- `scripts/research/run_vn_forecast_engine_v1.py` is missing.
- `scripts/research/run_vn30_index_group_price_range_forecast_lab.py` is missing.
- No stale imports of the removed QML-local I/O helper definitions were found.

## Tested after import update

- `python -m py_compile scripts/research/run_vn30_qml_forecasting.py`
- `python -m py_compile scripts/research/run_vn30_model_universe_direction_price_benchmark.py`
- `python scripts/research/run_vn30_qml_forecasting.py --help`
- `python scripts/research/run_vn30_model_universe_direction_price_benchmark.py --help`
- Focused governance/evaluation/features/forecasting test suite.
