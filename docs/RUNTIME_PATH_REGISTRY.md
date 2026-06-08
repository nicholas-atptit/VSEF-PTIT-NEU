# Runtime Path Registry

Status: canonical runtime path registry
Last updated: 2026-05-11

This registry classifies major runtime-relevant paths by ownership and authority.
It does not change command behavior, runtime behavior, model behavior, allocator
logic, API governance behavior, benchmark acceptance policy, or repository hygiene
rules.

Production-safe means safe for the governed diagnostic runtime boundary. It does
not mean trading advice, live execution authority, account routing, or investment
recommendation authority.

## Classification Rules

- `canonical`: authoritative governed diagnostic runtime path.
- `research`: experiment, benchmark, backtest, or report-generation path; not a production runtime authority.
- `demo`: non-authoritative demonstration surface.
- `experimental`: placeholder or unfinished branch; not governed runtime.
- `legacy`: retained for historical compatibility or migration reference.
- `deprecated`: retained only behind gates or pending replacement/removal.
- `maintenance`: operational support, data operations, service bootstrap, or governance support.
- `test_support`: tests, fixtures, smoke checks, and validation-only helpers.

## Registry

| Path | Status | Purpose | Production-safe? | Notes |
| ---- | ------ | ------- | ---------------- | ----- |
| `scripts/run_quant_core.py` | canonical | top-level diagnostic-chain runner | yes | preferred end-to-end Quant Core, scenario, risk governance, decision lane, allocator, and router orchestration |
| `src/evaluation/quant_core.py` | canonical | Quant Core scenario orchestration helpers | yes | governed forecast, risk, regime, policy, consensus, model-health, and packet diagnostics |
| `src/reporting/quant_core.py` | canonical | Quant Core manifest and markdown reporting | yes | writes diagnostic summaries only |
| `src/reporting/analysis_packets.py` | canonical | analysis packet and base decision-lane candidate construction | yes | upstream canonical diagnostic artifacts |
| `src/evaluation/backtest.py` | canonical | cost-aware diagnostic policy evaluation support | yes | used inside Quant Core; not live execution |
| `src/evaluation/walkforward.py` | canonical | deterministic walk-forward evaluation support | yes | governed evaluation primitive for Quant Core |
| `src/evaluation/targets.py` | canonical | forecast target contracts | yes | target definitions only; no training-policy change in Phase 7 |
| `src/evaluation/consensus.py` | canonical | consensus and disagreement summaries | yes | diagnostic summary support |
| `src/forecast/**` | canonical | governed forecast model registry and implementations | yes | active Quant Core forecast surface |
| `src/regime/**` | canonical | regime labeling/model support for diagnostics | yes | used by Quant Core and research analyses |
| `src/risk/**` | canonical | risk model support for diagnostics | yes | VaR/CVaR, drawdown, GARCH, and Monte Carlo model support |
| `src/strategy/**` | canonical | deterministic policy evaluation support | yes | diagnostic policy outputs only |
| `src/scenario/**` | canonical | Scenario Evaluation Engine v1 | yes | probability, dominance, uncertainty, calibration, and manifest outputs |
| `src/risk_governance/**` | canonical | Risk Governance Layer v1 | yes | risk scoring, action fields, adjusted candidates, override logs, and manifests |
| `src/reporting/decision_lane.py` | canonical | Decision Lane v2 enrichment | yes | `src/decision_lane/` is absent; this is the active implementation |
| `src/portfolio_allocator/**` | canonical | Portfolio Allocator v1 | yes | emits `allocation_candidate` or `no_allocation` diagnostics only |
| `src/phase3_router/**` | canonical | Phase 3 Router v1 | yes | emits `route_allocation_candidate`, `hold`, `reject`, or `no_candidate` diagnostics only |
| `src/data/adapters/vnstock_adapter.py` | canonical | governed provider adapter | yes | canonical `vnstock_data` adapter and provenance path |
| `src/data/universe.py` | canonical | VN100 universe support | yes | active universe contract; Phase 7 does not modify universe behavior |
| `src/core/runtime_mode.py` | canonical | runtime mode and mock provenance policy | yes | governs demo/research/audit mock boundaries |
| `src/core/model_governance.py` | canonical | run-mode and model governance metadata | yes | active model governance support |
| `src/api/main.py` | canonical | FastAPI diagnostic application entrypoint | yes | diagnostic research API metadata and root response |
| `src/api/routes.py` | canonical | active v1 diagnostic API routes | yes | active API surface; high-risk legacy routes are gated |
| `src/api/schemas.py` | canonical | active API schema contracts | yes | includes public diagnostic schemas and internal compatibility schemas |
| `src/api/schemas_v2.py` | canonical | terminal payload schema support | yes | used by active v1 prediction response and v2 demo routes |
| `src/api/tracing.py` | canonical | API tracing middleware/helpers | yes | observability support only |
| `scripts/check_runtime_preflight.py` | canonical | runtime reproducibility preflight | yes | read-only import, service, artifact, and environment check |
| `docs/COMMAND_REGISTRY.md` | canonical | command classification registry | yes | command semantics are not changed by Phase 7 |
| `docs/DECISION_DIAGNOSTIC_CHAIN.md` | canonical | diagnostic chain ownership contract | yes | architecture source of truth |
| `docs/AUTHORITY_BOUNDARY.md` | canonical | diagnostic-only authority boundary | yes | forbids recommendation/execution authority |
| `docs/governance/**` | canonical | governed schema and policy docs | yes | source of truth for active schemas and governance |
| `src/ml/trainer.py` | maintenance | manifest-driven ML training/inference support | yes | active support path, but not the canonical Quant Core ownership boundary |
| `src/ml/models/factory.py` | maintenance | supported ML algorithm factory | yes | supports maintained model implementations |
| `src/ml/models/cart.py` | maintenance | supported CART model implementation | yes | governed diagnostic ML support |
| `src/ml/models/lstm.py` | maintenance | supported LSTM model implementation | yes | governed diagnostic ML support |
| `src/ml/models/bilstm.py` | maintenance | supported BiLSTM model implementation | yes | governed diagnostic ML support |
| `src/ml/models/xgboost_model.py` | maintenance | supported XGBoost model implementation | yes | governed diagnostic ML support |
| `src/ml/models/lightgbm_model.py` | maintenance | supported LightGBM model implementation | yes | governed diagnostic ML support |
| `src/ml/models/sarimax.py` | maintenance | supported SARIMAX model implementation | yes | governed diagnostic ML support |
| `src/ml/models/ets.py` | maintenance | supported ETS model implementation | yes | governed diagnostic ML support |
| `src/ml/models/stacking.py` | maintenance | supported stacking model implementation | yes | governed diagnostic ML support |
| `src/ml/models/base.py` | maintenance | shared model base contracts | yes | support code only |
| `src/ml/models/price.py` | maintenance | database model for price records | yes | persistence support, not recommendation logic |
| `src/ml/models/company.py` | maintenance | database model for company records | yes | persistence support |
| `src/ml/models/macro.py` | maintenance | database model for macro records | yes | persistence support with provider provenance metadata |
| `src/ml/models/watchlist.py` | maintenance | database models for watchlist/blacklist records | yes | operational metadata support |
| `src/api/streaming/**` | maintenance | streaming producers, consumers, scheduler, and Kafka helpers | no | service-backed operations; not canonical diagnostic-chain authority |
| `src/streaming/**` | maintenance | streaming session support | no | operational support outside governed diagnostic claims |
| `src/data/database/**` | maintenance | database connections, repositories, migrations, and decision-card persistence | yes | persistence support; not an authority source by itself |
| `scripts/sync_all_data.py` | maintenance | provider/database sync | yes | modifies local/database state; not a canonical diagnostic claim |
| `scripts/run_backdate.py` | maintenance | historical provider/database backfill | yes | data operation only |
| `scripts/sync_raw_to_adjusted.py` | maintenance | local/database price adjustment sync | yes | data operation only |
| `scripts/sync_predictions_to_db.py` | maintenance | prediction database sync | yes | operational sync only |
| `scripts/live_heartbeat_sync.py` | maintenance | service heartbeat/provider sync | yes | service-backed operation |
| `scripts/import_historical_csv.py` | maintenance | historical CSV import | yes | data operation only |
| `scripts/extract_daily_csv.py` | maintenance | daily market CSV extraction | yes | generated data artifact operation |
| `scripts/extract_market_csv_single.py` | maintenance | hourly market CSV extraction | yes | data operation, not canonical runtime |
| `scripts/extract_market_csv_per_ticker.py` | maintenance | per-ticker hourly CSV extraction | yes | data operation, not canonical runtime |
| `scripts/extract_llm_jsonl.py` | maintenance | LLM JSONL extraction | yes | data operation, not canonical runtime |
| `scripts/generate_training_dataset.py` | maintenance | local training dataset generation | yes | data operation only |
| `scripts/generate_sentiment_features.py` | maintenance | sentiment feature artifact generation | yes | data operation only |
| `scripts/compute_market_proxy.py` | maintenance | market proxy artifact generation | yes | data operation only |
| `scripts/compute_sector_proxies.py` | maintenance | sector proxy artifact generation | yes | data operation only |
| `scripts/curate_foreign_flow_provider_artifact.py` | maintenance | provider-backed foreign-flow curation | yes | curation support; validate coverage before interpretation |
| `scripts/audit_ohlcv_cache_coverage.py` | maintenance | OHLCV cache coverage audit | yes | data coverage check only |
| `scripts/audit_foreign_flow_coverage.py` | maintenance | foreign-flow coverage audit | yes | data coverage check only |
| `scripts/fetch_sectors.py` | maintenance | sector metadata fetch | yes | provider/data operation |
| `scripts/fetch_fundamentals.py` | maintenance | fundamentals fetch | yes | provider/data operation |
| `scripts/prepare_vip_list.py` | maintenance | listing/universe artifact preparation | yes | data operation only |
| `scripts/prepare_fundamental_features.py` | maintenance | fundamental feature artifact preparation | yes | data operation only |
| `scripts/run_stream.py` | maintenance | local streaming runtime operation | no | service operation outside canonical diagnostic chain |
| `scripts/run_consumers.py` | maintenance | local consumer runtime operation | no | service operation outside canonical diagnostic chain |
| `scripts/run_local_ollama.ps1` | maintenance | local service bootstrap | no | infrastructure operation |
| `scripts/run_local_ollama.sh` | maintenance | local service bootstrap | no | infrastructure operation |
| `src/ml/benchmark/acceptance.py` | maintenance | benchmark acceptance policy support | no | governs benchmark promotion evidence; not runtime production authority |
| `src/ml/benchmark/system_benchmark.py` | research | system benchmark reporting | no | governed by statistical acceptance status, not production runtime |
| `src/ml/benchmark/stress_test.py` | research | deterministic benchmark stress testing | no | research evidence only |
| `src/ml/benchmark/risk_tuning.py` | research | risk tuning benchmark support | no | research evidence only |
| `src/ml/benchmark/final_report.py` | research | benchmark report writer | no | report generation only |
| `src/ml/benchmark/evaluator.py` | research | benchmark metric evaluation | no | research support |
| `src/ml/benchmark/baselines.py` | research | benchmark baselines | no | research support |
| `src/ml/benchmark/run.py` | research | benchmark CLI support | no | research command support |
| `src/ml/backtest/**` | research | backtest and comparison research modules | no | not canonical governed runtime |
| `src/ml/statistics/**` | research | statistical research tests and confidence intervals | no | research evidence support |
| `src/evaluation/repeated_seed_runner.py` | research | repeated-seed diagnostic research | no | reproducibility/stability research only |
| `src/evaluation/forecast_rehab.py` | research | forecast rehabilitation analysis | no | research report support |
| `src/evaluation/forecast_rehab_narrow.py` | research | narrow forecast rehabilitation analysis | no | research report support |
| `src/evaluation/hardening.py` | research | hardening matrix analysis support | no | research evidence support |
| `src/evaluation/calibration.py` | research | calibration matrix analysis support | no | research evidence support |
| `src/reporting/forecast_rehab.py` | research | forecast rehabilitation report writer | no | research report support |
| `src/reporting/forecast_rehab_narrow.py` | research | narrow rehabilitation report writer | no | research report support |
| `src/reporting/hardening.py` | research | hardening report writer | no | research report support |
| `src/reporting/calibration.py` | research | calibration report writer | no | research report support |
| `scripts/run_backtest_real_data.py` | research | real-data backtest runner | no | non-canonical comparison workflow |
| `scripts/run_backtest_forward_return.py` | research | forward-return backtest runner | no | non-canonical comparison workflow |
| `scripts/run_backtest_model_comparison.py` | research | model-comparison backtest runner | no | non-canonical comparison workflow |
| `scripts/run_dual_task_backtest.py` | research | dual-task backtest runner | no | non-canonical comparison workflow |
| `scripts/run_strategy_backtest.py` | research | strategy backtest runner | no | non-canonical comparison workflow |
| `scripts/backtest_portfolio_multi_agent.py` | research | multi-agent portfolio backtest | no | non-canonical comparison workflow |
| `scripts/run_combined_signal_analysis.py` | research | combined signal analysis | no | research workflow |
| `scripts/run_regime_aware_analysis.py` | research | regime-aware analysis | no | research workflow |
| `scripts/run_signal_effectiveness_backtest.py` | research | signal-effectiveness backtest | no | research workflow |
| `scripts/run_candidate_research.py` | research | risk-aware candidate research | no | research workflow |
| `scripts/run_walk_forward_regime_robustness.py` | research | walk-forward regime robustness | no | research workflow |
| `scripts/run_walkforward_all_models_stacking_eval.py` | research | all-model stacking walk-forward evaluation | no | research workflow |
| `scripts/run_quant_core_repeated_seeds.py` | research | repeated-seed Quant Core stability | no | research workflow |
| `scripts/run_forecast_rehab.py` | research | forecast rehabilitation runner | no | research workflow |
| `scripts/run_forecast_rehab_narrow.py` | research | narrow rehabilitation runner | no | research workflow |
| `scripts/run_meta_selector.py` | research | meta-selector research runner | no | research workflow |
| `scripts/run_context_meta_selector.py` | research | context meta-selector research runner | no | research workflow |
| `scripts/generate_forecasting_core_report.py` | research | forecasting-core report generation | no | research report writer |
| `scripts/generate_risk_aware_report.py` | research | risk-aware report generation | no | research report writer |
| `scripts/generate_regime_analysis_report.py` | research | regime-analysis report generation | no | research report writer |
| `scripts/generate_feature_analysis_report.py` | research | feature-analysis report generation | no | research report writer |
| `scripts/generate_robustness_report.py` | research | robustness/statistical report generation | no | research report writer |
| `scripts/generate_regime_labels.py` | research | regime label diagnostics | no | research workflow |
| `scripts/join_regime_to_predictions.py` | research | joins regime labels to prediction artifacts | no | research workflow |
| `scripts/compare_experiment_runs.py` | research | experiment run comparison | no | research workflow |
| `scripts/debug_compare_train_infer.py` | research | train/inference diagnostic comparison | no | debugging workflow |
| `scripts/train_ml_tickers.py` | research | model training research command | no | writes model/report artifacts; not canonical runtime |
| `scripts/run_news_crawler.py` | research | news crawler research command | no | research/data collection workflow |
| `src/ml/training_pipeline/**` | experimental | unfinished CNN, TFT, RL, and ONNX training/export branches | no | labeled experimental; not governed runtime |
| `scripts/train_ppo_allocator.py` | experimental | PPO allocator proof-of-concept trainer | no | labeled experimental; not governed runtime |
| `src/ml/models/agent.py` | legacy | unused agentic SQLAlchemy model branch | no | labeled legacy |
| `src/api/routes_v2.py` | demo | retained API v2 demo/static route surface | no | labeled demo; includes deprecated debate gate |
| `src/api/ui/**` | demo | terminal UI/chat demo helpers | no | labeled demo; not served as governed web runtime |
| `scripts/run_agent_decision.py` | demo | local agent-orchestrator demonstration | no | labeled demo; not canonical chain |
| `src/allocation/**` | legacy | older allocator compatibility package | no | labeled legacy; active canonical allocator is `src/portfolio_allocator/**` |
| `src/routing/**` | legacy | older router compatibility package and legacy artifact names | no | already labeled legacy; active canonical router is `src/phase3_router/**` |
| `src/adapters/**` | legacy | old adapter import-path shim | yes | labeled legacy; delegates to `src/data/adapters/**` |
| `scripts/legacy/**` | legacy | historical phase benchmark, cleanup, and provider research scripts | no | labeled legacy |
| `scripts/per_session_predict.py` | deprecated | old per-session prediction runner | no | labeled legacy; replaced by governed diagnostic chain/API paths |
| `scripts/autonomous_news_agent.py` | deprecated | old autonomous news loop | no | labeled legacy; retained outside canonical runtime |
| `src/api/routes.py` deprecated endpoints | deprecated | `/execute` and `/paper-trade` legacy/demo gates | no | retained as gated diagnostic responses only |
| `src/api/routes_v2.py` deprecated endpoints | deprecated | `/debate` legacy debate gate | no | retained as gated diagnostic response only |
| `docs/archive/**` | legacy | historical documentation archive | no | not active runtime source of truth |
| `docs/prompt_runs/archive/**` | legacy | historical prompt/run archive | no | not active runtime source of truth |
| `reports/**` historical outputs | research | historical reports and generated evidence packs | no | see `docs/REPORTS_GOVERNANCE.md` |
| `reports/PHASE*_EVIDENCE.md` | maintenance | controlled remediation evidence | yes | canonical only for remediation status and verification |
| `reports/cleanup/AUDIT_REMEDIATION_CHECKLIST.md` | maintenance | remediation tracking checklist | yes | controlled audit-remediation status record |
| `tests/**` | test_support | unit, integration, smoke, and governance tests | yes | validation only |
| `tests/fixtures/**` | test_support | controlled test fixtures | yes | validation fixtures only |
| `scripts/check_repo_hygiene.py` | test_support | repository hygiene verification | yes | validation-only check |
| `scripts/verify_ml_baseline.py` | test_support | ML baseline smoke verification | yes | writes generated artifacts; validation only |
| `scripts/verify_stability.py` | test_support | stability smoke verification | yes | validation only |
| `scripts/verify_db.py` | test_support | database smoke verification | yes | validation only |
| `scripts/validate_database_hardening.py` | test_support | database hardening validation | yes | validation only |
| `scripts/check_db_coverage.py` | test_support | database coverage check | yes | validation only |
| `scripts/check_rest_price.py` | test_support | price endpoint/provider smoke check | yes | validation only |

## Explicit Non-Authority Notes

- Canonical diagnostic paths do not produce BUY or SELL recommendations.
- Research and benchmark paths cannot promote runtime claims without governed
  acceptance evidence.
- Experimental modules are excluded from governed runtime claims even if they
  import successfully.
- Legacy compatibility paths are retained to avoid breaking historical tests,
  artifacts, or imports; they are not the source of canonical runtime ownership.
