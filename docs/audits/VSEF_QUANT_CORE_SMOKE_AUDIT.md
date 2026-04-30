# VSEF Quant Core Smoke Audit
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Audit report |
| Created / authored | Thursday, 2026-04-30 11:57:00 ICT (UTC+07:00) |
| Last updated | Thursday, 2026-04-30 11:57:00 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `5ceed8162b4658cd1ee3402fb147aa5246c1f540` |
| Timestamp source | Local smoke audit run |
| Status | Active |

## Purpose

This audit validates that the governed Quant Core runner can execute a bounded smoke workflow and generate the expected artifact surface. It checks orchestration, run-mode filtering, governed model metadata, manifest output, forecast output, fallback provenance, consensus summaries, analysis packets, decision-lane candidate files, and model-health files.

This audit does not claim production trading readiness, robust edge, or final buy recommendations.

## Scope

The smoke preset used one scenario:

- ticker group: `small_banks`
- tickers: `ACB`, `BID`, `CTG`, `MBB`
- horizon: `5`
- target: `forward_return`
- windowing: `train_size=252`, `test_size=21`, `step_size=21`, `max_windows=1`

The successful smoke outputs were written under ignored local paths:

- `artifacts/quant_core_smoke_audit/research_core_smoke`
- `artifacts/quant_core_smoke_audit/baseline_only_smoke`
- `artifacts/quant_core_smoke_audit/decision_core_smoke`

No generated artifacts from `artifacts/` are intended for commit.

## Quant Core Surface Inspection Summary

Available run modes:

- `full_forecast`: full supported forecast zoo, including primary research, comparators, baselines, and shadow models.
- `research_core`: primary research plus comparator models.
- `decision_core`: decision-enabled primary research models.
- `baseline_only`: baseline checkpoint models.

Governed model registry and current roles:

| Model | Family | Role | Status | `research_core` | `decision_core` | `baseline_only` |
| --- | --- | --- | --- | ---: | ---: | ---: |
| `lightgbm` | boosting | `primary_research` | active | 1 | 1 | 0 |
| `xgboost` | boosting | `primary_research` | active | 1 | 1 | 0 |
| `random_forest` | tree | `primary_research` | active | 1 | 1 | 0 |
| `ets` | statistical | `comparator` | active | 1 | 0 | 0 |
| `sarimax` | statistical | `comparator` | active | 1 | 0 | 0 |
| `naive` | baseline | `baseline_only` | baseline | 0 | 0 | 1 |
| `moving_average` | baseline | `baseline_only` | baseline | 0 | 0 | 1 |
| `linear` | linear | `shadow_only` | shadow | 0 | 0 | 0 |
| `ridge` | linear | `shadow_only` | shadow | 0 | 0 | 0 |
| `lasso` | linear | `shadow_only` | shadow | 0 | 0 | 0 |

Expected output artifacts are the runner manifest, markdown summary, scenario matrix, model governance table, forecast outputs, forecast summaries, window summary, risk and regime summaries, signal/position/trade/policy outputs, execution log, consensus summary, model health summary, analysis packet JSONL, and decision-lane candidate CSV.

Current test coverage includes run-mode registry filtering, governance-table schema consistency, manifest governance metadata, analysis-feed packet normalization, packet provenance, cross-link generation, deterministic summaries, retrieval metadata alignment, and idempotency. Existing tests also cover analysis-packet candidate construction and model-health aggregation.

Known out-of-scope items remain unchanged: no new forecast model families, no Phase 3 routing, no stacking or meta-model decision logic, no portfolio allocator, no deep sequence-model rollout, and no claims of broad robust edge.

## Commands Run

Branch and status checks:

```powershell
git branch --show-current
git status --short
```

Smoke commands:

```powershell
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core.py --preset smoke --run-mode research_core --output-dir artifacts\quant_core_smoke_audit\research_core_smoke
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core.py --preset smoke --run-mode baseline_only --output-dir artifacts\quant_core_smoke_audit\baseline_only_smoke
C:\Users\luong\.venv\Scripts\python.exe scripts/run_quant_core.py --preset smoke --run-mode decision_core --output-dir artifacts\quant_core_smoke_audit\decision_core_smoke
```

Initial smoke attempts exposed two operational issues:

- The first run failed with `KeyError: 'date'` because the raw OHLCV fallback path computed date bounds from `raw_df["date"]`, while the tracked daily CSVs use a `time` column. A narrow loader fix now accepts `date`, `time`, `timestamp`, or `trading_date` for raw OHLCV bounds.
- Normal sandboxed execution hit `vnstock_data` license verification SSL permission errors. The smoke commands completed when rerun with normal local/network access.

The successful runs still emitted non-fatal provider warnings:

- `market_proxy_not_found`
- `Sentiment integration is explicitly disabled for the main forecasting pipeline.`
- provider package update notices
- Windows cp1252 logging errors for Unicode provider status messages

## Quant Core Components Verified

- role-aware model governance in `src/core/model_governance.py`
- governed forecast registry in `src/forecast/registry.py`
- scenario orchestration in `src/evaluation/quant_core.py`
- consensus summaries in `src/evaluation/consensus.py`
- analysis packet and decision candidate generation in `src/reporting/analysis_packets.py`
- model-health summaries in `src/reporting/model_health.py`
- quant-core manifest and markdown summary rendering in `src/reporting/quant_core.py`
- top-level runner behavior in `scripts/run_quant_core.py`

## Run Mode Behavior Observed

| Run mode | Requested models | Evaluated models | Skipped models |
| --- | --- | --- | --- |
| `research_core` | `lightgbm`, `xgboost`, `random_forest`, `ets`, `sarimax` | `lightgbm`, `random_forest`, `weighted_ensemble`, `xgboost` | `ets`, `sarimax` skipped because the statistical model classes reported missing `statsmodels` support |
| `baseline_only` | `naive`, `moving_average` | `moving_average`, `naive`, `weighted_ensemble` | none |
| `decision_core` | `lightgbm`, `xgboost`, `random_forest` | `lightgbm`, `random_forest`, `weighted_ensemble`, `xgboost` | none |

The observed model sets match the governed role filters: research mode includes primary research plus comparators, baseline mode includes baseline-only models, and decision mode includes decision-enabled primary research models.

## Artifact Existence Table

| Artifact | `research_core` | `baseline_only` | `decision_core` |
| --- | --- | --- | --- |
| `run_manifest.json` | yes | yes | yes |
| `summary.md` | yes | yes | yes |
| `scenario_matrix.csv` | yes | yes | yes |
| `model_governance.csv` | yes | yes | yes |
| `full_model_predictions.csv` | yes | yes | yes |
| `forecast_summary.csv` | yes | yes | yes |
| `forecast_summary_by_horizon.csv` | yes | yes | yes |
| `window_summary.csv` | yes | yes | yes |
| `risk_summary.csv` | yes | yes | yes |
| `regime_summary.csv` | yes | yes | yes |
| `signals.csv` | yes | yes | yes |
| `positions.csv` | yes | yes | yes |
| `trades.csv` | yes | yes | yes |
| `strategy_metrics.csv` | yes | yes | yes |
| `equity_curve.csv` | yes | yes | yes |
| `policy_summary.csv` | yes | yes | yes |
| `model_execution_log.csv` | yes | yes | yes |
| `model_consensus_summary.csv` | yes | yes | yes |
| `model_health_summary.csv` | yes | yes | yes |
| `analysis_packets.jsonl` | yes | yes | yes |
| `decision_lane_candidates.csv` | yes | yes | yes |

## Manifest Sanity Check

| Run mode | `manifest_type` | Scenario count | Forecast rows | Risk rows | Regime rows | Analysis packet rows | Strategy rows |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `research_core` | `quant_core_run_manifest_v1` | 1 | 336 | 84 | 84 | 84 | 4 |
| `baseline_only` | `quant_core_run_manifest_v1` | 1 | 252 | 84 | 84 | 84 | 3 |
| `decision_core` | `quant_core_run_manifest_v1` | 1 | 336 | 84 | 84 | 84 | 4 |

Each manifest recorded the run mode, run-mode spec, requested models, evaluated models, skipped models, row counts, governance output path, and artifact paths.

## Forecast Output Sanity Check

| Run mode | Prediction rows | Tickers | Horizons | Target types | Governance columns |
| --- | ---: | --- | --- | --- | --- |
| `research_core` | 336 | `ACB`, `BID`, `CTG`, `MBB` | `5` | `forward_return` | present |
| `baseline_only` | 252 | `ACB`, `BID`, `CTG`, `MBB` | `5` | `forward_return` | present |
| `decision_core` | 336 | `ACB`, `BID`, `CTG`, `MBB` | `5` | `forward_return` | present |

The prediction tables include the expected governance and scenario fields: `model_family`, `model_role`, `model_status`, `research_priority`, `supports_policy_eval`, `core_run_id`, `preset`, `group_name`, `target_name`, `target_column`, `target_family`, `target_tradable`, `ticker_group_members`, and `run_mode`.

## Risk/Regime Fallback Interpretation

All successful smoke runs used explicit fallback provenance:

- risk `source_model`: `var_cvar_drawdown_fallback`
- risk `risk_model`: `var_cvar_drawdown_fallback`
- risk `distribution`: `historical_fallback`
- regime `source_model`: `markov_switching_threshold_fallback`

The regime output contained `bear`, `bull`, and `sideway` labels. These fallback labels are acceptable for smoke validation because the goal is to prove that fallback provenance is surfaced, not to prove that the fallback models are superior or production-ready.

## Consensus And Analysis Packet Verification

| Run mode | Consensus rows | Agreement buckets | Sign-conflict rows | Analysis packet rows | Packet metadata |
| --- | ---: | --- | ---: | ---: | --- |
| `research_core` | 84 | `high`, `low` | 26 | 84 | present |
| `baseline_only` | 84 | `high`, `medium` | 32 | 84 | present |
| `decision_core` | 84 | `high`, `low` | 26 | 84 | present |

`analysis_packets.jsonl` was generated for all three runs. Packet identity fields and core metadata were present, including `packet_id`, `packet_generated_at`, `timestamp`, `ticker`, `horizon`, `target_type`, `run_mode`, `core_run_id`, `risk_summary`, `regime_summary`, `policy_summary`, `retrieval_metadata`, `model_by_model_predictions`, and `model_ranks`.

## Decision-Lane Candidate Interpretation

`decision_lane_candidates.csv` was generated for all three runs:

| Run mode | Candidate rows |
| --- | ---: |
| `research_core` | 22 |
| `baseline_only` | 16 |
| `decision_core` | 22 |

Candidate columns include `packet_id`, `timestamp`, `ticker`, `group_name`, `horizon`, `target_type`, `run_mode`, `primary_model_name`, `primary_prediction`, `model_agreement_score`, `agreement_bucket`, `regime_label`, `volatility_bucket`, `active_signal_count`, `top_policy_model`, `top_policy_sharpe`, and `candidate_score`.

These rows are smoke-mode review candidates only. They are not final buy recommendations, not a live decision policy, and not evidence of robust trading edge.

## Model Health Verification

| Run mode | Health rows | Health labels | Notes |
| --- | ---: | --- | --- |
| `research_core` | 6 | `failing`, `weak` | `ets` and `sarimax` have `run_success_rate=0.0` because they were skipped; evaluated primary models and ensemble are labeled `weak` in the smoke slice. |
| `baseline_only` | 3 | `healthy`, `weak` | `moving_average` is labeled `healthy`; `naive` and `weighted_ensemble` are labeled `weak`. |
| `decision_core` | 4 | `weak` | all evaluated decision-core models and the ensemble have `run_success_rate=1.0` and smoke-slice `weak` labels. |

The health files include execution, forecast-quality, drift-style, and policy fields such as `run_success_rate`, `mean_rmse`, `mean_directional_accuracy`, `policy_eval_count`, `mean_sharpe`, and `positive_policy_frequency`.

## Tests Run

Targeted regression check after the loader fix:

```powershell
C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/phase1/test_walkforward.py -q
```

Result: `3 passed`.

Required validation suite:

```powershell
C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/quant_core -q
C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/phase1/test_forecast_contracts.py -q
C:\Users\luong\.venv\Scripts\python.exe -m compileall src
```

Results: `15 passed`, `3 passed`, and compileall completed successfully.

Because the loader source file was changed, the broader ML test target was also run:

```powershell
C:\Users\luong\.venv\Scripts\python.exe -m pytest tests/ml -q
```

Result: `230 passed, 2 skipped`.

## What This Smoke Audit Proves

- The Quant Core runner can execute the `smoke` preset for `research_core`, `baseline_only`, and `decision_core`.
- Role-aware model governance filters the requested model set by run mode.
- The runner writes the expected artifact surface for successful smoke runs.
- Manifest, governance, forecast, risk, regime, consensus, packet, candidate, and health outputs are present and internally readable.
- Fallback risk and regime provenance is explicit in generated outputs.
- Decision-lane candidate files are generated and schema-readable.

## What This Smoke Audit Does Not Prove

- It does not prove production trading readiness.
- It does not prove robust edge.
- It does not prove that decision candidates are final buy recommendations.
- It does not validate full-market, long-window, or 15-year behavior.
- It does not validate deep sequence models, stacking, or meta-model decision logic.
- It does not prove that fallback risk/regime labels are preferable to full statistical model outputs.

## Remaining Risks

- `ets` and `sarimax` were skipped in `research_core` because the model classes reported missing `statsmodels` support.
- Risk and regime outputs used fallback providers in the smoke runs.
- The smoke path still depends on provider availability checks when local market proxy artifacts are absent.
- Windows console encoding emitted non-fatal logging errors for Unicode provider messages.
- Smoke-mode health labels are diagnostic only and should not be interpreted as durable model quality rankings.

## Recommended Next Task

Add a provider-independent smoke fixture or preflight mode for Quant Core so orchestration smoke tests can run without external provider authentication, while preserving explicit fallback provenance and avoiding changes to model-governance semantics.
