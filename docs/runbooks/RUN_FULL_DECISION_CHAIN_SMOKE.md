# Run Full Decision Chain Smoke
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Runbook |
| Created / authored | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-05-05 00:00:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Timestamp source | Local deterministic decision-chain documentation refactor |
| Status | Active |

## Purpose

Run a lightweight end-to-end smoke of the active deterministic
decision-diagnostic chain:

```text
Quant Core
-> Scenario Evaluation Engine v1
-> Risk Governance Layer v1
-> Decision Lane v2
-> Portfolio Allocator v1
-> Phase 3 Router v1
```

The run is diagnostic-only. It must not be interpreted as BUY or SELL authority,
live execution readiness, production trading readiness, or learned meta-model
validation.

## PowerShell Command

Use a one-line PowerShell command:

```powershell
python scripts/run_quant_core.py --preset smoke --run-mode research_core --enable-scenario-engine --enable-risk-governance --enable-portfolio-allocator --enable-phase3-router --output-dir artifacts/quant_core_router_smoke
```

If splitting across lines in PowerShell, use the PowerShell backtick character
at the end of each continued line. Do not use CMD `^` continuation in
PowerShell.

## Expected Artifacts

Base Quant Core:

- `run_manifest.json`
- `summary.md`
- `scenario_matrix.csv`
- `model_governance.csv`
- `full_model_predictions.csv`
- `forecast_summary.csv`
- `forecast_summary_by_horizon.csv`
- `window_summary.csv`
- `risk_summary.csv`
- `regime_summary.csv`
- `signals.csv`
- `positions.csv`
- `trades.csv`
- `strategy_metrics.csv`
- `equity_curve.csv`
- `policy_summary.csv`
- `model_execution_log.csv`
- `model_consensus_summary.csv`
- `model_health_summary.csv`
- `analysis_packets.jsonl`
- `decision_lane_candidates.csv`

Scenario Evaluation Engine v1:

- `scenario_probability.csv`
- `scenario_rankings.csv`
- `scenario_dominance_summary.csv`
- `scenario_uncertainty_summary.csv`
- `scenario_calibration_summary.csv`
- `scenario_manifest.json`

Risk Governance Layer v1:

- `risk_governance_summary.csv`
- `risk_adjusted_candidates.csv`
- `risk_override_log.csv`
- `risk_manifest.json`

Decision Lane v2:

- `decision_lane_enriched_candidates.csv`
- `decision_lane_manifest.json`

Portfolio Allocator v1:

- `portfolio_allocation.csv`
- `portfolio_summary.csv`
- `portfolio_risk_summary.csv`
- `portfolio_decision_cards.jsonl`
- `allocator_manifest.json`

Phase 3 Router v1:

- `router_decisions.csv`
- `router_summary.csv`
- `router_manifest.json`

## Validation Commands

Run lightweight schema and suite checks separately from the smoke:

```powershell
python -m compileall src/phase3_router
pytest tests/phase3_router -q
pytest tests/portfolio_allocator -q
pytest tests/decision_lane -q
pytest tests/risk_governance -q
pytest tests/scenario -q
pytest tests/quant_core -q
```

Check that generated artifacts are not staged:

```powershell
git status --short
```

Generated files under `artifacts/quant_core_router_smoke` must remain local and
must not be committed.

## Smoke Review Checklist

- `run_manifest.json` contains artifact paths for all enabled layers.
- `router_manifest.json` lists canonical router artifacts only.
- `router_decisions.csv` uses only `route_allocation_candidate`, `hold`,
  `reject`, or `no_candidate`.
- `portfolio_allocation.csv` uses only `allocation_candidate` or
  `no_allocation`.
- Manifests preserve diagnostic-only and no BUY/SELL authority fields.
- No generated artifact is staged for commit.
