# Offline Research Runbook

## Default Mode

Use local historical/cached data only. Do not call provider APIs, fetch live
data, run schedulers, create broker workflows, or create live prediction
ledgers.

All forecast-engine output is offline diagnostic evidence. Final rows are
scoring-only, and model selection must use validation rows only.

## Safe Validation

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
python scripts/check_provider_usage_policy.py
python -m pytest tests/data/test_provider_usage_policy.py -q
python -m pytest tests/data/test_vn_price_gateway_contract.py -q
python -m pytest tests/ml/test_directional_accuracy_metrics.py -q
python -m pytest tests/governance tests/evaluation tests/features tests/forecasting -q
```

These commands should not fetch live data or run heavy research benchmarks.

## Offline Forecast Engine

The runner reads existing local caches and writes to
`reports/generated/vn_forecast_engine_v1/`, with summaries under
`reports/results/` and a claim boundary under `reports/claims/`.

### Forecast latest local cache

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --forecast-latest --frequency hourly --horizons 5 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL
```

### Forecast from a historical as-of timestamp

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --forecast-asof "2025-01-02 10:00:00" --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL
```

Historical-asof mode simulates the information cutoff. Actual fields are filled
only when later local rows exist.

### Full offline run

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --full-run --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL --timeout-seconds 14400
```

This builds local datasets, evaluates bounded candidates, selects using
validation only, and writes forecast panels and evaluation reports.

### Evaluation-only run

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --build-evaluate --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL --timeout-seconds 14400
```

Evaluation-only mode does not create a live prediction or production system.

## Heavy Research Runners

QML and Model Universe runners are available but heavy. Run them only under an
approved written protocol:

```powershell
python scripts/research/run_vn30_qml_forecasting.py --help
python scripts/research/run_vn30_model_universe_direction_price_benchmark.py --help
```

Do not run these during ordinary validation, rerun benchmarks casually, or use
final-period results for selection.

## Output Navigation

- Forecast engine: `reports/generated/vn_forecast_engine_v1/`
- Forecast engine reports: `reports/results/VN_FORECAST_ENGINE_V1_*`
- Forecast engine claim boundary: `reports/claims/VN_FORECAST_ENGINE_V1_CLAIM_BOUNDARY.md`
- QML evidence: `reports/generated/vn30_qml_forecasting/`
- Model Universe evidence: `reports/generated/vn30_model_universe_direction_price/`
- Active evidence index: `reports/_index/ACTIVE_EVIDENCE_INDEX.md`

`VNXALL` is skipped and recorded in coverage audits when no local cache exists.
The dedicated index-group price-range-lab runner/evidence package is not present
on this branch.

## Split Discipline

Use `src/governance/split_policy.py`. Both feature and target timestamps must be
inside the same period. Boundary-crossing rows are unassigned. Final rows are
scoring-only and never used for selection.

## Claim Boundary

Offline diagnostic research only. No trading, profitability, BUY/SELL,
recommendation, investment advice, live deployment, production, or daily T+1
claim is permitted.
