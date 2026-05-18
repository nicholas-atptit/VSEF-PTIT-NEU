# Code Cleanup Validation Fix Result

## Summary

- Hygiene result: passed.
- Bare preflight result: passed, `ok=48 warn=12 fail=0`.
- Intended-venv preflight result: passed, `ok=44 warn=16 fail=0`.
- Provider policy result: passed.
- Py compile result: passed with `PYTHONPYCACHEPREFIX` set outside the repository.
- Benchmark run: no.
- Data fetch: no.
- Model training: no.
- Paper/DOCX generated: no.
- Tags created: no.
- Tags pushed: no.
- Main touched: no.

## Tests

- `<repo-approved-venv>\Scripts\python.exe -m pytest tests/data/test_provider_usage_policy.py -q`: 2 passed, 1 warning.
- `<repo-approved-venv>\Scripts\python.exe -m pytest tests/data/test_vn_price_gateway_contract.py -q`: 7 passed, 1 warning.
- `<repo-approved-venv>\Scripts\python.exe -m pytest tests/ml/test_directional_accuracy_metrics.py -q`: 12 passed, 1 warning.

## Hygiene Fixes

- Updated `scripts/check_repo_hygiene.py` to allow full-data-backup paths only when Git attributes mark them with the `lfs` filter.
- Kept local path, local file URI, Python bytecode/cache, `node_modules`, egg-info, malformed filename, and unapproved generated-root checks active.
- Redacted machine-local repository paths in backup inventories and output metadata with `<repo>`.
- Removed `scripts/research/__pycache__` and compiled active scripts into a temp pycache prefix outside the repository.

## Notes

Runtime preflight warnings are environmental and unchanged in kind: unreachable local services, unset optional environment variables, and missing `data\market_proxy.csv`. The intended Python 3.13 environment also warns about optional or missing packages such as `arch`, `gymnasium`, `stable-baselines3`, and `google-generativeai`.
