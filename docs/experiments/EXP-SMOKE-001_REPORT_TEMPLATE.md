# EXP-SMOKE-001 Smoke Report Template

## Status

This is a template, not a successful smoke report.

A previous local smoke attempt failed honestly because the active interpreter
could not import the canonical `vnstock_data` provider. That earlier failure was
an interpreter/environment issue, not a provider policy change.

The latest verified smoke run uses:

- Python executable: `C:\Users\luong\.venv\Scripts\python.exe`
- Provider import status: `available`
- Provider fetch: completed successfully
- Baseline predictions: 36 rows
- Status: `completed_with_errors`
- Output folder: `outputs/experiments/EXP-SMOKE-001/`

The current blocker is the missing ETS/SARIMAX runtime dependency:

```text
ModuleNotFoundError: No module named 'statsmodels'
```

The latest smoke generated partial failure evidence only. Successful full
model-backed smoke metrics remain pending until `statsmodels` is installed in
the active venv and the smoke is rerun. Do not claim successful ETS or SARIMAX
performance metrics from the `completed_with_errors` run.

## Required Evidence For A Completed Smoke Report

Before replacing this template with `EXP-SMOKE-001_REPORT.md`, verify that a
real smoke run produced:

- `outputs/experiments/EXP-SMOKE-001/config/resolved_config.yaml`
- `outputs/experiments/EXP-SMOKE-001/manifests/run_manifest.json`
- `outputs/experiments/EXP-SMOKE-001/logs/run.log`
- `outputs/experiments/EXP-SMOKE-001/logs/errors.log`
- `outputs/experiments/EXP-SMOKE-001/metrics/metrics.csv`
- `outputs/experiments/EXP-SMOKE-001/metrics/metrics_summary.json`
- `outputs/experiments/EXP-SMOKE-001/predictions/predictions.csv`
- `outputs/experiments/EXP-SMOKE-001/reports/summary.md`

The manifest status must be `completed` or `completed_with_errors` before any
model or baseline result is discussed as smoke evidence.

## Completion Checklist

- [x] `vnstock_data` is installed in the active venv used for the latest smoke run.
- [x] Daily OHLCV data is available for the configured ticker/date window.
- [x] `vnstock_data` fetch completes in the active venv.
- [ ] `statsmodels` is installed in the active venv for ETS and SARIMAX.
- [ ] `C:\Users\luong\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\EXP-SMOKE-001.yaml` runs without model dependency errors.
- [ ] Metrics include at least one row for `model_type=model`.
- [ ] Metrics include at least one row for `model_type=baseline`.
- [ ] Missing optional artifacts are explained in the manifest warnings and summary.
- [ ] No fake logs, fake CSVs, fake manifests, fake charts, or fake metrics are added.
