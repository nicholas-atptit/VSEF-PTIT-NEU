# Phase 2 - Forecasting Core Validation

## Objective

Phase 2 validates whether the VSEF forecasting layer creates independent value
compared with simple baselines and individual supported models. The phase uses
the Phase 1 `ExperimentOrchestrator` and requires evidence through configs,
logs, manifests, metrics, predictions, and generated report artifacts.

## Governance Boundaries

- Provider remains `vnstock_data`.
- Frequency remains daily OHLCV.
- Supported models remain SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, and
  Stacking.
- Baselines are comparison evidence only.
- Outputs are diagnostic evidence only and must not be represented as BUY /
  SELL / HOLD advice.
- Raw generated experiment outputs under `outputs/experiments/` should remain
  local evidence and should not be committed.

## Experiment Set

- `EXP-FC-001`: compares simple baselines against individual supported
  forecasting models on T+1.
- `EXP-FC-002`: compares individual models against Stacking where runtime
  support is available.
- `EXP-FC-003`: compares behavior across T+1, T+3, and T+5.

## Universe

The controlled Phase 2 universe is defined in
`configs/universe/ticker_universe.yaml`:

- `FPT`
- `ACB`
- `HPG`
- `MWG`
- `DGC`

Failed ticker fetches must remain visible in logs, manifests, and the generated
report.

## How To Run

```powershell
C:\Users\luong\.venv\Scripts\python.exe -m compileall src scripts
C:\Users\luong\.venv\Scripts\python.exe -m pytest tests\engine\test_experiment_orchestrator.py tests\ml\evaluation\test_metrics_engine.py tests\ml\baselines\test_baseline_registry.py -q
C:\Users\luong\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\EXP-FC-001.yaml
C:\Users\luong\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\EXP-FC-002.yaml
C:\Users\luong\.venv\Scripts\python.exe scripts\run_experiment.py --config configs\experiments\EXP-FC-003.yaml
C:\Users\luong\.venv\Scripts\python.exe scripts\generate_forecasting_core_report.py --experiments EXP-FC-001 EXP-FC-002 EXP-FC-003 --output reports\forecasting_core
git diff --check
```

## Report Outputs

The report generator reads actual artifacts from `outputs/experiments/EXP-FC-*`
and writes:

- `reports/forecasting_core/FORECASTING_CORE_VALIDATION_REPORT.md`
- `reports/forecasting_core/forecast_metrics.csv`
- `reports/forecasting_core/model_ranking.csv`
- `reports/forecasting_core/stability_metrics.csv`
- `reports/forecasting_core/horizon_comparison.csv`
- `reports/forecasting_core/error_distribution_summary.csv`

Charts are generated only when the local environment supports them and the
underlying metric rows exist.

## Acceptance Criteria

- Phase 2 configs exist and validate through the Phase 1 orchestrator.
- At least one Phase 2 run produces manifest, metrics, predictions, and summary.
- Baselines and models appear in the same metric table where available.
- Stacking is evaluated or disclosed with a failure/skip reason.
- Multi-horizon evidence covers T+1, T+3, and T+5 or reports exact failures.
- The final report is generated from actual artifacts.
- Missing optional artifacts are disclosed.
- No investment advice or model superiority is claimed without metric evidence.
