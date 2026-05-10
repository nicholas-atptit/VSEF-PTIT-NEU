# Command Registry

Status: canonical command registry for reproducibility and local operation
Last updated: 2026-05-11

This registry defines which commands are canonical, research-only, smoke/test,
legacy, deprecated, or maintenance/data operations. It does not change runtime
semantics. Generated outputs should remain untracked unless a remediation or
review task explicitly curates them.

## Operating Assumptions

Run commands from the repository root.

Primary environment contract:

- Python: 3.11 or newer.
- Package contract: `pyproject.toml` is authoritative; `requirements.txt` mirrors runtime dependencies.
- Canonical provider: `vnstock_data`.
- Local services are optional for many tests but required for service-backed runtime/data workflows.
- Generated outputs normally go under `artifacts/`, `models/`, `data/`, or specific historical `reports/` paths.

## Preflight And Verification

| Command | Class | Output | Generates ignored artifacts | External services |
| --- | --- | --- | --- | --- |
| `python scripts/check_runtime_preflight.py` | canonical runtime | stdout summary | no | checks configured services by socket only |
| `python scripts/check_repo_hygiene.py` | smoke/test | stdout summary | no | no |
| `python -m pytest tests -q` | smoke/test | pytest stdout/cache | may update local pytest cache | no mandatory services for the governed suite |
| `python scripts/verify_ml_baseline.py` | smoke/test | `models/ci_smoke`, `reports/ci_smoke_cart.csv` | yes | no |
| `python scripts/verify_stability.py` | smoke/test | stdout | no | no |
| `python scripts/verify_db.py` | smoke/test | stdout | no | database |
| `python scripts/validate_database_hardening.py` | smoke/test | stdout | no | database |
| `python scripts/check_db_coverage.py` | smoke/test | stdout | no | database |
| `python scripts/check_rest_price.py` | smoke/test | stdout | no | API/provider depending on target |

## Canonical Runtime Commands

These are the preferred operator-facing commands for the active diagnostic chain.

| Command | Purpose | Output | Generates ignored artifacts | External services |
| --- | --- | --- | --- | --- |
| `python scripts/run_quant_core.py --preset smoke --run-mode research_core --enable-scenario-engine --enable-risk-governance --enable-portfolio-allocator --enable-phase3-router --output-dir artifacts/quant_core_router_smoke` | End-to-end deterministic diagnostic-chain smoke | `artifacts/quant_core_router_smoke/` | yes | no mandatory service; provider/data availability affects real-data variants |
| `python scripts/run_quant_core.py --output-dir artifacts/quant_core` | Base quant-core diagnostic run | `artifacts/quant_core/` | yes | provider/data availability depending options |
| `python scripts/run_portfolio_allocator.py --input-dir <quant-core-artifact-dir>` | Create allocation-candidate diagnostics from saved quant-core outputs | input dir or `--output-dir` | yes | no |
| `python scripts/run_phase3_router.py --input-dir <allocator-artifact-dir>` | Create route-decision diagnostics from saved allocator outputs | input dir or `--output-dir` | yes | no |
| `python scripts/run_analysis_feed.py --input-dir <router-artifact-dir> --output-dir artifacts/analysis_feed` | Build analysis-feed artifacts from router outputs | `artifacts/analysis_feed/` | yes | no |
| `python scripts/run_retrieval_prep.py --analysis-feed-dir artifacts/analysis_feed --output-dir artifacts/retrieval_prep` | Prepare retrieval documents from analysis-feed artifacts | `artifacts/retrieval_prep/` | yes | no |
| `python scripts/run_retrieval_ingest.py --retrieval-prep-dir artifacts/retrieval_prep --output-dir artifacts/retrieval_ingest` | Build local retrieval index artifacts | `artifacts/retrieval_ingest/` | yes | optional vector backend depending adapter |
| `python scripts/run_retrieval_query.py --index-dir artifacts/retrieval_ingest` | Query local retrieval index | stdout / optional smoke artifact | may generate local smoke artifact | no |
| `python scripts/run_experiment.py --config <config-yaml>` | Config-driven experiment wrapper | config-defined | yes | provider/data depending config |

## Service Bootstrap And Runtime Operations

These commands operate local services or service-backed flows. Use the preflight first.

| Command | Class | Output | Generates ignored artifacts | External services |
| --- | --- | --- | --- | --- |
| `powershell -ExecutionPolicy Bypass -File scripts/run_local_ollama.ps1` | maintenance | Docker services and pulled model | service state, not repo artifact | Docker, Timescale/Postgres, Chroma, Kafka, Redis, Ollama |
| `bash scripts/run_local_ollama.sh` | maintenance | Docker services and pulled model | service state, not repo artifact | Docker, Timescale/Postgres, Chroma, Kafka, Redis, Ollama |
| `python scripts/run_stream.py` | maintenance | streaming stdout/service state | no tracked output expected | Kafka/Redis/database/provider |
| `python scripts/run_consumers.py` | maintenance | consumer stdout/service state | no tracked output expected | Kafka/Redis/database/LLM |
| `python scripts/live_heartbeat_sync.py` | maintenance | heartbeat sync stdout/service state | may update database | database/provider |

## Data Operations

Use these to build or audit local data artifacts. They may modify local data roots or database state.

| Command | Class | Output | Generates ignored artifacts | External services |
| --- | --- | --- | --- | --- |
| `python scripts/sync_all_data.py` | maintenance | database rows / stdout | no file by default | database, provider |
| `python scripts/run_backdate.py` | maintenance | database rows / stdout | no file by default | database, provider |
| `python scripts/sync_raw_to_adjusted.py` | maintenance | database rows / stdout | no file by default | database |
| `python scripts/sync_predictions_to_db.py` | maintenance | database rows / stdout | no file by default | database |
| `python scripts/list_db_tables.py` | maintenance | stdout | no | database |
| `python scripts/import_historical_csv.py` | maintenance | database rows / stdout | no file by default | database, local CSV inputs |
| `python scripts/extract_daily_csv.py` | maintenance | `data/daily_market_split/` by default | yes | database/local source depending args |
| `python scripts/extract_market_csv_single.py` | maintenance | `stock_hourly_unified_data.csv` by default | yes | database/local source depending args |
| `python scripts/extract_market_csv_per_ticker.py` | maintenance | `stock_hourly_market_data.csv` by default | yes | database/local source depending args |
| `python scripts/extract_llm_jsonl.py` | maintenance | `stock_hourly_training_data.jsonl` by default | yes | local source data |
| `python scripts/generate_training_dataset.py` | maintenance | local training dataset | yes | local source data |
| `python scripts/generate_sentiment_features.py` | maintenance | `data/sentiment_features.csv` / stdout | yes | database |
| `python scripts/compute_market_proxy.py` | maintenance | `data/market_proxy.csv` | yes | local daily market CSVs |
| `python scripts/compute_sector_proxies.py` | maintenance | `data/sector_proxies.csv` | yes | local market CSVs |
| `python scripts/curate_foreign_flow_provider_artifact.py` | maintenance | `data/foreign_flow_curated.csv` by default | yes | provider |
| `python scripts/audit_ohlcv_cache_coverage.py` | maintenance | stdout | no | local OHLCV cache |
| `python scripts/audit_foreign_flow_coverage.py` | maintenance | stdout | no | local foreign-flow artifacts |
| `python scripts/fetch_sectors.py` | maintenance | `data/ticker_sectors.csv` or stdout | yes | provider |
| `python scripts/fetch_fundamentals.py` | maintenance | local fundamentals output/stdout | yes | provider |
| `python scripts/prepare_vip_list.py` | maintenance | `data/listing/` artifacts | yes | local/provider inputs |
| `python scripts/prepare_fundamental_features.py` | maintenance | local feature artifacts | yes | local/provider inputs |

## Research-Only Commands

These commands are retained for experiments, audits, report generation, or non-canonical comparisons. They are not the default runtime path.

| Command family | Scripts | Output | Generates ignored artifacts | External services |
| --- | --- | --- | --- | --- |
| Backtests | `run_backtest_real_data.py`, `run_backtest_forward_return.py`, `run_backtest_model_comparison.py`, `run_dual_task_backtest.py`, `run_strategy_backtest.py`, `backtest_portfolio_multi_agent.py` | `artifacts/backtest*`, `artifacts/dual_task`, `artifacts/strategy_backtest`, report paths | yes | provider/local data depending command |
| Signal/research analysis | `run_combined_signal_analysis.py`, `run_regime_aware_analysis.py`, `run_signal_effectiveness_backtest.py`, `run_candidate_research.py` | `artifacts/combined_signal`, `artifacts/regime_aware_analysis`, required output dirs | yes | local saved artifacts |
| Robustness and walk-forward | `run_walk_forward_regime_robustness.py`, `run_walkforward_all_models_stacking_eval.py`, `run_quant_core_repeated_seeds.py` | `artifacts/walk_forward_regime_robustness`, configured output dirs | yes | provider/local data depending command |
| Forecast rehabilitation | `run_forecast_rehab.py`, `run_forecast_rehab_narrow.py` | `artifacts/forecast_rehab*` | yes | provider/local data depending command |
| Meta-selector research | `run_meta_selector.py`, `run_context_meta_selector.py` | `artifacts/meta_selector*`, `artifacts/context_meta_selector` | yes | local saved artifacts |
| Report generation | `generate_forecasting_core_report.py`, `generate_risk_aware_report.py`, `generate_regime_analysis_report.py`, `generate_feature_analysis_report.py`, `generate_robustness_report.py` | required report output directories | yes | local saved artifacts |
| Label/feature diagnostics | `generate_regime_labels.py`, `join_regime_to_predictions.py`, `compare_experiment_runs.py`, `debug_compare_train_infer.py` | configured CSV/JSON/stdout | yes depending args | local saved artifacts |
| Model training research | `train_ml_tickers.py`, `train_ppo_allocator.py` | `models/`, `reports/ml_benchmark.csv`, local RL model path | yes | provider/local data; RL extras for PPO |
| News/data research | `run_news_crawler.py`, `autonomous_news_agent.py` | `reports/news_keywords_baseline.csv` or local outputs | yes | web/provider/LLM depending command |

## Legacy And Deprecated Commands

Legacy commands are retained for historical reproduction or migration reference. Do not use them as canonical runtime entrypoints.

| Command family | Scripts | Replacement |
| --- | --- | --- |
| Legacy benchmark scripts | `scripts/legacy/run_phase1_benchmark.py`, `scripts/legacy/run_phase2_benchmark.py`, `scripts/legacy/run_phase25_hardening.py`, `scripts/legacy/run_phase26_calibration.py`, `scripts/legacy/benchmark_local_baseline.py` | `run_quant_core.py` and current research report generators |
| Legacy cleanup/research scripts | `scripts/legacy/scrub_links.py`, `scripts/legacy/research/*.py` | Current docs/report governance and provider preflight |
| Deprecated root runners | `run_agent_decision.py`, `per_session_predict.py`, `autonomous_news_agent.py` | Diagnostic chain commands and governed API routes |
| Ad hoc hourly extractors | `extract_market_csv_single.py`, `extract_market_csv_per_ticker.py`, `extract_llm_jsonl.py` | Data operations only; not canonical runtime |

## Output Policy

- `artifacts/`: generated workflow outputs, ignored by default.
- `models/`: generated model bundles; commit only when explicitly curated.
- `data/`: mixed local data/cache/source artifacts; commit only governed small reference data or explicitly curated provider artifacts.
- `reports/`: governed remediation records plus historical reports. See `docs/REPORTS_GOVERNANCE.md`.
- pytest cache and local service state must remain untracked.
