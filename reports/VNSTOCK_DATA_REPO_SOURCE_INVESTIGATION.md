# vnstock_data Repo Source Investigation

## Summary

- Current branch contains vnstock_data adapter/wrapper code: yes.
- main/origin-main contains vnstock_data adapter/wrapper code: yes.
- vnstock_data is declared as a dependency: yes, in `pyproject.toml` and `config/requirements/requirements_utf8.txt`.
- Runtime preflight knows about vnstock_data: yes, through repository dependency/import checks.
- Current VN30 hourly research scripts bypass the repo adapter: yes, the VN30 hourly scripts use their own direct provider loading path.
- VN30 scripts should prefer the repo adapter when practical: yes, for canonical runtime/provider governance.

## Evidence

The current branch contains `src/data/adapters/vnstock_adapter.py`, which centralizes lazy `vnstock_data` imports and provides the repository adapter surface. The same adapter path is visible in `main` without checking out or merging main.

Dependency declarations include:

- `pyproject.toml`: `vnstock_data`
- `config/requirements/requirements_utf8.txt`: `vnstock_data`

VN30 hourly research code currently has direct provider logic in `scripts/research/vn30_hourly_vnstock_common.py`, and related fetch/benchmark scripts describe the path as `vnstock_data if importable, otherwise legacy vnstock`. That direct path explains why interpreter selection materially changes provider behavior.

## Recommended Provider Path

1. Use the repo adapter first where it supports the required hourly/index operation.
2. Use `vnstock_data` direct calls second for research-only probes where adapter coverage is incomplete.
3. Use legacy `vnstock` only as an explicit fallback and label it in reports.

The next provider work should run with the intended repo venv and should not use bare `python` for provider verification.
