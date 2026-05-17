# Docs And Task Runner Cleanup Result

## Summary

- Manual-review scripts reviewed: 13.
- Scripts moved: 0.
- Scripts kept: 13.
- README updated: yes.
- `docs/USAGE.md` created: yes.
- `docs/RESEARCH_WORKFLOW.md` created: yes.
- `Makefile` created: yes.
- `scripts/dev_tasks.ps1` created: yes.
- Benchmark run: no.
- Data fetch: no.
- Model training: no.
- Paper/DOCX generated: no.
- Tags created: no.
- Tags pushed: no.
- Main touched: no.

## Validation Results

- Hygiene: passed, `Repository hygiene check passed.`
- Bare preflight: passed, `ok=48 warn=12 fail=0`.
- Intended-venv preflight: passed, `ok=44 warn=16 fail=0`.
- Provider policy: passed.
- Targeted tests: passed.
  - `tests/data/test_provider_usage_policy.py`: 2 passed, 1 warning.
  - `tests/data/test_vn_price_gateway_contract.py`: 7 passed, 1 warning.
  - `tests/ml/test_directional_accuracy_metrics.py`: 12 passed, 1 warning.
- PowerShell task runner validation: passed with `powershell -ExecutionPolicy Bypass -File scripts/dev_tasks.ps1 -Task validate-all`.
- Make validation: skipped; `make` was not available in this environment.

## Notes

The task runners include validation-only targets. They do not include data fetch, benchmark, model training, paper, or DOCX generation targets.
