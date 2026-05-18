# Phase 7 Runtime Path Registry Evidence

Date: 2026-05-11

## Scope

Phase 7 created a canonical runtime path registry and added warning labels to
ambiguous demo, experimental, and legacy modules. The phase did not modify model
training logic, feature engineering behavior, VN100 universe logic, allocator
logic, API governance behavior, benchmark acceptance policy, repository hygiene
rules, command registry semantics, or frontend removal behavior.

## Inventory Summary

Canonical governed runtime ownership is now documented for:

- Quant Core: `scripts/run_quant_core.py`, `src/evaluation/quant_core.py`,
  `src/reporting/quant_core.py`, and analysis-packet support.
- Scenario engine: `src/scenario/**`.
- Risk governance: `src/risk_governance/**`.
- Decision Lane v2: `src/reporting/decision_lane.py`.
- Portfolio Allocator v1: `src/portfolio_allocator/**`.
- Phase 3 Router v1: `src/phase3_router/**`.
- Governed provider adapter: `src/data/adapters/vnstock_adapter.py`.
- Runtime provenance/preflight: `src/core/runtime_mode.py` and
  `scripts/check_runtime_preflight.py`.
- Active diagnostic API: `src/api/main.py`, `src/api/routes.py`,
  `src/api/schemas.py`, and `src/api/tracing.py`.

Ambiguous non-canonical surfaces are now separated as demo, research,
experimental, legacy, deprecated, maintenance, or test-support paths in
`docs/RUNTIME_PATH_REGISTRY.md`.

## Classification Rules

- `canonical`: authoritative governed diagnostic runtime path.
- `research`: experiment, benchmark, backtest, or report-generation path.
- `demo`: non-authoritative demonstration surface.
- `experimental`: placeholder or unfinished branch; not governed runtime.
- `legacy`: retained for historical compatibility or migration reference.
- `deprecated`: retained only behind gates or pending replacement/removal.
- `maintenance`: operational support, data operations, service bootstrap, or
  governance support.
- `test_support`: tests, fixtures, smoke checks, and validation-only helpers.

## Files Labeled

Experimental labels were added to:

- `src/ml/training_pipeline/train_tft.py`
- `src/ml/training_pipeline/train_cnn.py`
- `src/ml/training_pipeline/export_to_onnx.py`
- `src/ml/training_pipeline/rl_env.py`
- `src/ml/training_pipeline/rl_env_portfolio.py`
- `src/ml/training_pipeline/train_rl_allocator.py`
- `scripts/train_ppo_allocator.py`

Demo labels were added to:

- `src/api/routes_v2.py`
- `src/api/ui/dashboard.py`
- `src/api/ui/chat_terminal.py`
- `scripts/run_agent_decision.py`

Legacy labels were added to:

- `src/ml/models/agent.py`
- `src/adapters/__init__.py`
- `src/adapters/vnstock_adapter.py`
- `src/allocation/__init__.py`
- `src/allocation/portfolio_allocator.py`
- `scripts/autonomous_news_agent.py`
- `scripts/per_session_predict.py`
- `scripts/legacy/benchmark_local_baseline.py`
- `scripts/legacy/scrub_links.py`
- `scripts/legacy/run_phase1_benchmark.py`
- `scripts/legacy/run_phase2_benchmark.py`
- `scripts/legacy/run_phase25_hardening.py`
- `scripts/legacy/run_phase26_calibration.py`
- `scripts/legacy/research/check_sync.py`
- `scripts/legacy/research/generate_csv.py`
- `scripts/legacy/research/list_gemini_models.py`
- `scripts/legacy/research/verify_vnstock_fix.py`

## Canonical Paths Classified

- `scripts/run_quant_core.py`
- `src/evaluation/quant_core.py`
- `src/reporting/quant_core.py`
- `src/reporting/analysis_packets.py`
- `src/evaluation/backtest.py`
- `src/evaluation/walkforward.py`
- `src/evaluation/targets.py`
- `src/evaluation/consensus.py`
- `src/forecast/**`
- `src/regime/**`
- `src/risk/**`
- `src/strategy/**`
- `src/scenario/**`
- `src/risk_governance/**`
- `src/reporting/decision_lane.py`
- `src/portfolio_allocator/**`
- `src/phase3_router/**`
- `src/data/adapters/vnstock_adapter.py`
- `src/data/universe.py`
- `src/core/runtime_mode.py`
- `src/core/model_governance.py`
- `src/api/main.py`
- `src/api/routes.py`
- `src/api/schemas.py`
- `src/api/schemas_v2.py`
- `src/api/tracing.py`
- `scripts/check_runtime_preflight.py`
- `docs/COMMAND_REGISTRY.md`
- `docs/DECISION_DIAGNOSTIC_CHAIN.md`
- `docs/AUTHORITY_BOUNDARY.md`
- `docs/governance/**`

## Non-Canonical Paths Classified

Demo:

- `src/api/routes_v2.py`
- `src/api/ui/**`
- `scripts/run_agent_decision.py`

Research:

- `src/ml/benchmark/system_benchmark.py`
- `src/ml/benchmark/stress_test.py`
- `src/ml/benchmark/risk_tuning.py`
- `src/ml/benchmark/final_report.py`
- `src/ml/benchmark/evaluator.py`
- `src/ml/benchmark/baselines.py`
- `src/ml/benchmark/run.py`
- `src/ml/backtest/**`
- `src/ml/statistics/**`
- `src/evaluation/repeated_seed_runner.py`
- `src/evaluation/forecast_rehab.py`
- `src/evaluation/forecast_rehab_narrow.py`
- `src/evaluation/hardening.py`
- `src/evaluation/calibration.py`
- `src/reporting/forecast_rehab.py`
- `src/reporting/forecast_rehab_narrow.py`
- `src/reporting/hardening.py`
- `src/reporting/calibration.py`
- `scripts/run_backtest_real_data.py`
- `scripts/run_backtest_forward_return.py`
- `scripts/run_backtest_model_comparison.py`
- `scripts/run_dual_task_backtest.py`
- `scripts/run_strategy_backtest.py`
- `scripts/backtest_portfolio_multi_agent.py`
- `scripts/run_combined_signal_analysis.py`
- `scripts/run_regime_aware_analysis.py`
- `scripts/run_signal_effectiveness_backtest.py`
- `scripts/run_candidate_research.py`
- `scripts/run_walk_forward_regime_robustness.py`
- `scripts/run_walkforward_all_models_stacking_eval.py`
- `scripts/run_quant_core_repeated_seeds.py`
- `scripts/run_forecast_rehab.py`
- `scripts/run_forecast_rehab_narrow.py`
- `scripts/run_meta_selector.py`
- `scripts/run_context_meta_selector.py`
- `scripts/generate_forecasting_core_report.py`
- `scripts/generate_risk_aware_report.py`
- `scripts/generate_regime_analysis_report.py`
- `scripts/generate_feature_analysis_report.py`
- `scripts/generate_robustness_report.py`
- `scripts/generate_regime_labels.py`
- `scripts/join_regime_to_predictions.py`
- `scripts/compare_experiment_runs.py`
- `scripts/debug_compare_train_infer.py`
- `scripts/train_ml_tickers.py`
- `scripts/run_news_crawler.py`

Experimental:

- `src/ml/training_pipeline/**`
- `scripts/train_ppo_allocator.py`

Legacy:

- `src/ml/models/agent.py`
- `src/allocation/**`
- `src/routing/**`
- `src/adapters/**`
- `scripts/legacy/**`

Deprecated:

- `scripts/per_session_predict.py`
- `scripts/autonomous_news_agent.py`
- `src/api/routes.py` deprecated `/execute` and `/paper-trade` gates
- `src/api/routes_v2.py` deprecated `/debate` gate

Maintenance and test support:

- `src/ml/benchmark/acceptance.py`
- `src/api/streaming/**`
- `src/streaming/**`
- `src/data/database/**`
- `scripts/sync_*`, `scripts/extract_*`, `scripts/fetch_*`, data curation,
  provider audit, service bootstrap, and validation scripts
- `tests/**`
- `tests/fixtures/**`

## Tests Run

| Command | Result |
| --- | --- |
| `python scripts/check_repo_hygiene.py` | passed |
| `python -m pytest tests -q` | passed: `824 passed, 5 skipped, 33 warnings in 294.19s` |
| `python scripts/check_repo_hygiene.py` | passed after pytest |
| `git status --short` | collected after tests |

## Full Pytest Result

`python -m pytest tests -q` completed successfully:

```text
824 passed, 5 skipped, 33 warnings in 294.19s (0:04:54)
```

Observed warnings were pre-existing environment/package/test warnings. Pytest
also reported that it could not write one local cache path, but this did not
affect test success.

## Hygiene Check Result

`python scripts/check_repo_hygiene.py` completed successfully before and after
pytest:

```text
Repository hygiene check passed.
```

## Final Git Status

Final status before staging showed only Phase 7 edits plus the pre-existing
untracked research audit report:

```text
M reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md
M scripts/autonomous_news_agent.py
M scripts/legacy/benchmark_local_baseline.py
M scripts/legacy/research/check_sync.py
M scripts/legacy/research/generate_csv.py
M scripts/legacy/research/list_gemini_models.py
M scripts/legacy/research/verify_vnstock_fix.py
M scripts/legacy/run_phase1_benchmark.py
M scripts/legacy/run_phase25_hardening.py
M scripts/legacy/run_phase26_calibration.py
M scripts/legacy/run_phase2_benchmark.py
M scripts/legacy/scrub_links.py
M scripts/per_session_predict.py
M scripts/run_agent_decision.py
M scripts/train_ppo_allocator.py
M src/adapters/__init__.py
M src/adapters/vnstock_adapter.py
M src/allocation/__init__.py
M src/allocation/portfolio_allocator.py
M src/api/routes_v2.py
M src/api/ui/chat_terminal.py
M src/api/ui/dashboard.py
M src/ml/models/agent.py
M src/ml/training_pipeline/export_to_onnx.py
M src/ml/training_pipeline/rl_env.py
M src/ml/training_pipeline/rl_env_portfolio.py
M src/ml/training_pipeline/train_cnn.py
M src/ml/training_pipeline/train_rl_allocator.py
M src/ml/training_pipeline/train_tft.py
?? docs/RUNTIME_PATH_REGISTRY.md
?? reports/cleanup/PHASE7_RUNTIME_PATH_REGISTRY_EVIDENCE.md
?? reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
```

`reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md` was already untracked before
Phase 7 work and remains untouched.

Post-commit status:

```text
?? reports/results/RESEARCH_TOPIC_AND_CODE_AUDIT_REPORT.md
```

## Unresolved Or Deferred Risks

- Final repository signoff is moved to Phase 8.
- `scripts/run_portfolio_allocator.py` still imports the older compatibility
  allocator path. Phase 7 documents this without changing command behavior.
- Historical docs and reports can still contain older language; active
  authority remains governed by current docs and the runtime registry.
- Benchmark/research paths remain non-canonical unless future phases promote
  them through explicit governed acceptance evidence.
