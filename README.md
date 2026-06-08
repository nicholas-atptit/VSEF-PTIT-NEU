# VN Forecast Research Lab

Target repository: <https://github.com/nicholas-atptit/VSEF-PTIT-NEU>

VN Forecast Research Lab is an offline historical/local-data-only research
system for VN30 stocks and explicitly configured Vietnamese indices. It
preserves benchmark evidence, claim boundaries, QML diagnostics, and reusable
forecast research components.

## Research scope

- Offline direction, return/price, price-range/interval, and ranking research.
- Reusable forecast engine components under `src/forecasting/`.
- QML diagnostics and paper evidence.
- Model Universe V1-V7 negative audit evidence; it is not a champion-replacement
  claim.
- VN30 and configured-index joint-panel/range research where local evidence is
  available.
- Provider-governed local data access through `src/data/providers/` and
  `src/data/adapters/`.

## Claim boundary

This is not a trading system. It makes no profitability, BUY/SELL,
recommendation, investment-advice, live-deployment, production, or daily T+1
operation claim. Final rows are scoring-only. Candidate selection must be
future-blind and validation-only.

See:

- `docs/governance/CLAIM_BOUNDARY_POLICY.md`
- `docs/governance/DATA_AND_ARTIFACT_POLICY.md`
- `reports/_index/ACTIVE_EVIDENCE_INDEX.md`

## Current runner status

- QML diagnostics: `scripts/research/run_vn30_qml_forecasting.py`
- Model Universe direction/price audit:
  `scripts/research/run_vn30_model_universe_direction_price_benchmark.py`
- Offline forecast-engine runner: `scripts/research/run_vn_forecast_engine_v1.py`
  is currently missing; do not invent forecast results.
- Index-group price-range runner:
  `scripts/research/run_vn30_index_group_price_range_forecast_lab.py` is
  currently missing.

The QML and Model Universe runners are heavy research runners. Do not execute
them during ordinary cleanup or validation.

## Validation

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
python scripts/check_provider_usage_policy.py
python -m pytest tests/data/test_provider_usage_policy.py -q
python -m pytest tests/data/test_vn_price_gateway_contract.py -q
python -m pytest tests/ml/test_directional_accuracy_metrics.py -q
python -m pytest tests/governance tests/evaluation tests/features tests/forecasting -q
```

These commands do not fetch live data or run new model training.

## Operating policy

- Work on branches; do not touch or merge `main` during refactors.
- Push normal branches only. Do not use `git push --mirror`.
- Do not create tags by default.
- Do not fetch data or rerun benchmarks without an explicit protocol.
- Do not delete raw/cache data, generated evidence, paper evidence exports, or
  protected active evidence.
- Do not generate DOCX during cleanup.

Start with `docs/usage/RUNBOOK.md`, `docs/architecture/REPOSITORY_STRUCTURE.md`,
and `docs/workflows/RESEARCH_WORKFLOW.md`.
