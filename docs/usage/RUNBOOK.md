# Offline Research Runbook

## Default mode

Use local historical/cached data only. Do not call provider APIs, fetch live
data, run schedulers, or create live prediction ledgers.

## Safe validation

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
python scripts/check_provider_usage_policy.py
python -m pytest tests/data/test_provider_usage_policy.py -q
python -m pytest tests/data/test_vn_price_gateway_contract.py -q
python -m pytest tests/ml/test_directional_accuracy_metrics.py -q
python -m pytest tests/governance tests/evaluation tests/features tests/forecasting -q
```

## Research runners

QML and Model Universe runners are available but heavy. Run them only under an
approved protocol:

```powershell
python scripts/research/run_vn30_qml_forecasting.py --help
python scripts/research/run_vn30_model_universe_direction_price_benchmark.py --help
```

The requested offline forecast-engine and index-group range-lab runner files are
currently missing. No forecast smoke should be claimed until an actual
local-data-only runner exists and is validated.

## Split discipline

Use `src/governance/split_policy.py`. Both feature and target timestamps must be
inside the same period. Boundary-crossing rows are unassigned. Final rows are
scoring-only and never used for selection.
