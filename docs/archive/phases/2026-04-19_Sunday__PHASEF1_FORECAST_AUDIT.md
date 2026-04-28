# Phase F1 Forecast Layer Audit
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-04-19 06:23:07 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

| module/file | role | keep / audit / refactor / ablate | reason |
| --- | --- | --- | --- |
| `src/ml/feature_engineering.py` | Builds the governed prepared feature surface and inventory metadata | keep + audit | Already contains the canonical feature inventory, category metadata, and filtered feature-column helpers needed for rehab. |
| `src/ml/features/registry.py` | Resolves approved and task-specific feature sets | keep + audit | The current walk-forward path depends on the registry-selected baseline; Phase F1 needs to compare against it, not replace it. |
| `src/ml/features/feature_registry.json` | Stores the approved feature sets, task-specific selections, and governance notes | keep + audit | This is the source of truth for the current working baseline and for current feature-selection evidence. |
| `data/processed/ml_5y` | Prepared ticker-level dataset backbone | keep | The prepared `ml_5y` CSVs already contain the required OHLCV, context, regime, and wide feature surface. |
| `src/evaluation/walkforward.py` | Leakage-safe time split, target attachment, feature resolution, forecast evaluation | extend | It already owns the correct split/evaluation backbone, but previously only supported forward-return targets. |
| `src/evaluation/backtest.py` | Cost-aware downstream strategy evaluation | keep | Phase F1 needs a fixed execution baseline, not a new backtest engine. |
| `src/forecast/base.py` | Shared forecast contract | keep | The current models already share a usable numeric forecast interface. |
| `src/forecast/registry.py` | Forecast model factory | keep | Reused to instantiate the existing model family under new target/feature ablations. |
| `src/forecast/statistical/*` | Statistical baselines under the shared contract | keep + audit | They remain in scope and are still useful for exposing whether ML models actually beat simple baselines. |
| `src/forecast/ml/*` | ML baselines under the shared contract | keep + audit | They remain in scope and need side-by-side rehab benchmarking, not replacement. |
| `scripts/run_phase1_benchmark.py` | Small end-to-end Phase 1 benchmark | keep + audit | Useful reference for the minimal forecast-to-policy path and manifest writing. |
| `scripts/run_phase2_benchmark.py` | Phase 2 forecast/risk/regime benchmark | keep + wrap | Reused for the risk/regime window builders and as the reference for the fixed downstream policy context. |
| `scripts/run_phase26_calibration.py` | Phase 2.6 policy-ablation runner | keep + audit | Important because Phase F1 must hold the Phase 2.6 default policy fixed rather than re-optimizing policy. |
| `src/reporting/hardening.py` | Stability and cost-sensitivity summaries | keep + wrap | Reused as the pattern for bounded matrix summaries. |
| `src/reporting/calibration.py` | Policy-ablation reporting | keep + audit | Important reference for forecast-vs-policy comparison logic and the Phase 2.6 default policy candidate. |

## Canonical Entry Points

- Data preparation: `data/processed/ml_5y/*.csv` plus the fallback `DualModelTrainer.prepare_ticker_data()` path inside `WalkForwardEvaluator._load_prepared_frame()`.
- Model training/evaluation: `src/evaluation/walkforward.py` via `WalkForwardEvaluator.evaluate()`.
- Phase 1 benchmark entry: `scripts/run_phase1_benchmark.py`.
- Phase 2 benchmark entry: `scripts/run_phase2_benchmark.py`.
- Phase 2.6 policy calibration entry: `scripts/run_phase26_calibration.py`.
- Reporting/manifests: `src/reporting/manifests.py`, `src/reporting/summary.py`, `src/reporting/hardening.py`, and `src/reporting/calibration.py`.

## Key Audit Findings

- The current working forecast baseline does **not** use the whole prepared feature surface. By default, walk-forward evaluation resolves the registry-selected `regression_forecasting` feature set, which currently contains 14 columns.
- The prepared `ml_5y` surface is much wider than the active forecast baseline. One current sample (`ACB.csv`) contains 527 columns, 512 registry-candidate numeric columns, and 467 canonical filtered feature columns.
- The repo already contains time-safe regression, direction-classification, and future-volatility label utilities, but the current shared walk-forward path only attaches the forward-return target.
- The Phase 2.6 default execution candidate is `regime_threshold_adaptive_drawdown::adaptive_current`, and the least-bad calibrated threshold was `0.010`. Phase F1 uses that as a fixed downstream baseline instead of re-opening policy optimization.
