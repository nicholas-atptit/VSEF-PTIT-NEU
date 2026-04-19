# Quant Core Surface Audit

Date: 2026-04-19
Branch: `rebuild-regime-risk-aware-forecasting`

## Scope

This audit maps the current quant-core surface before adding governance and
orchestration. The goal is to preserve the existing forecast, regime, risk,
evaluation, reporting, and benchmark stack while identifying the narrow seams
to extend for:

- full model-zoo governance
- explicit run modes
- analysis-ready forecast packets
- consensus and disagreement summaries
- model-health reporting
- future RAG/LLM/human-analyst integration

## Current Surface Map

| module/file | role | keep / extend / refactor | reason |
| --- | --- | --- | --- |
| `src/forecast/base.py` | Shared forecast model contract and validated forecast frame construction | keep | This is the stable forecast output contract already used by the walk-forward evaluator and downstream summaries. |
| `src/forecast/registry.py` | Current quant-core forecast model registry and factory | extend | This is the natural insertion point for role-aware governance because every benchmark runner already resolves models here. |
| `src/forecast/statistical/*` | Statistical baseline/comparator forecast implementations | keep | These are part of the existing supported forecast zoo and must remain available in full-forecast mode. |
| `src/forecast/ml/*` | Current walk-forward ML forecast implementations | keep | These models already satisfy the forecast contract and belong in the governed quant-core zoo. |
| `src/risk/base.py` | Shared risk model contract | keep | Stable contract surface; governance and orchestration can consume its outputs without changing it. |
| `src/risk/garch.py` | Phase 2 risk forecast generator with volatility, VaR/CVaR, and drawdown context | extend | This is already the primary risk context provider for Phase 2+ and should feed quant-core packets and health summaries. |
| `src/risk/var_cvar.py` | Simple tail-risk estimator for Phase 1 backtests | keep | Useful baseline risk context; no orchestration rewrite needed. |
| `src/risk/monte_carlo.py` | Monte Carlo risk baseline | keep | Baseline risk context remains useful for comparative reporting. |
| `src/risk/drawdown.py` | Drawdown context provider | keep | Already reused by GARCH and sizing logic; should remain unchanged. |
| `src/regime/base.py` | Shared regime model contract and validated regime frame | keep | Stable output contract already suitable for packet generation and summaries. |
| `src/regime/markov_switching.py` | Primary regime model with deterministic fallback | extend | Existing regime context should flow into packet, consensus, and health outputs, but the model logic should remain intact. |
| `src/evaluation/walkforward.py` | Core leak-safe evaluator, target application, forecast summaries, and dataset loading | extend | This is the orchestration backbone; it should remain the execution engine while gaining richer metadata and multi-model run support. |
| `src/evaluation/forecast_rehab.py` | Matrix config for broad rehab analysis | extend | Useful precedent for matrix-driven orchestration and bounded research presets. |
| `src/evaluation/forecast_rehab_narrow.py` | Narrow-scope research-lane model and feature selection | extend | Important evidence source for model-role defaults such as primary/comparator/baseline/parked. |
| `src/evaluation/hardening.py` | Sweep configuration for Phase 2.5 hardening | keep | Existing sweep matrix logic should remain reusable by future quant-core presets. |
| `src/evaluation/calibration.py` | Phase 2.6 calibration helpers | keep | Downstream policy calibration remains separate from quant-core orchestration. |
| `src/evaluation/backtest.py` | Cost-aware backtest and strategy metric generation | keep | Stable policy-evaluation contract that quant-core summaries can consume directly. |
| `src/reporting/manifests.py` | Git/runtime/dependency metadata and manifest writers | extend | Existing manifest layer should record run mode, model roles, packet outputs, and analysis metadata instead of being replaced. |
| `src/reporting/summary.py` | Phase 1/2 summary table and markdown builders | extend | Existing comparison reporting should remain compatible while quant-core adds new summary artifacts. |
| `src/reporting/forecast_rehab.py` | Forecast quality, stability, and forecast-vs-policy reporting helpers | extend | Strong precedent for model-health and analyst-facing summary outputs. |
| `src/reporting/forecast_rehab_narrow.py` | Narrow-lane assessment and comparison reporting | extend | Contains current research-lane logic that should inform governance defaults rather than be discarded. |
| `src/reporting/hardening.py` | Stability, cost-sensitivity, and readiness reporting | extend | Existing stability summaries are directly relevant to model-health and drift-style quant-core reporting. |
| `scripts/run_phase1_benchmark.py` | End-to-end Phase 1 benchmark runner | keep | Existing entry point should remain runnable and compatible. |
| `scripts/run_phase2_benchmark.py` | End-to-end Phase 2 forecast+risk+regime runner | extend | This is the closest current precursor to the top-level quant-core runner and should remain reusable. |
| `scripts/run_forecast_rehab.py` | Broad rehab analysis runner | keep | Useful research lane runner that should coexist with quant-core orchestration. |
| `scripts/run_forecast_rehab_narrow.py` | Narrow rehab research runner | keep | Important focused lane runner that should stay intact and benefit from shared governance primitives. |
| `src/strategy/execution_policy.py` | Policy execution config and signal/position annotation | extend | Existing policy annotations should be captured cleanly in quant-core packets and manifests. |
| `src/core/contracts.py` | Forecast/regime/signal/position schema validation | keep | Existing contracts are the right base layer for reproducible packet construction. |
| `src/ml/models/factory.py` | Broader legacy model-zoo registry outside the walk-forward quant-core lane | keep, reference only | This preserves wider repo model-zoo context, but it is not the current walk-forward quant-core execution registry and should not be destructively merged into this phase. |
| `src/ml/backtest/walk_forward_all_models_stacking.py` | Legacy all-model walk-forward + stacking experiment lane | keep, reference only | Useful evidence that the repo has broader model families, but out of scope for the current quant-core governance phase because no new stacking/meta-model rollout is allowed. |

## Current Entry Points

### `scripts/run_phase1_benchmark.py`

- Uses `supported_forecast_models()` from `src/forecast/registry.py`.
- Executes one model at a time through `WalkForwardEvaluator`.
- Adds weighted ensemble output after per-model evaluation.
- Builds Phase 1 risk summaries with Monte Carlo, VaR/CVaR, and drawdown.
- Produces forecasts, summaries, signals, positions, trades, strategy metrics,
  `run_manifest.json`, and `summary.md`.

### `scripts/run_phase2_benchmark.py`

- Reuses the walk-forward forecast backbone and weighted ensemble output.
- Adds GARCH risk outputs and Markov-switching regime outputs.
- Evaluates explicit strategy modes:
  `forecast_only`, `forecast_plus_risk`, `forecast_plus_risk_and_regime`.
- Produces forecasts, forecast summaries, risk summaries, regime summaries,
  signals, positions, trades, strategy metrics, comparison summaries, manifest,
  markdown summary, and Phase 2 report.

### `scripts/run_forecast_rehab.py`

- Builds a bounded matrix across ticker groups, horizons, target framings, and
  feature families.
- Reuses current forecast registry and Phase 2 policy baseline.
- Produces aggregate forecast slices, quality summaries, model stability
  summaries, forecast-vs-policy summaries, assessment output, manifest, and
  reports.

### `scripts/run_forecast_rehab_narrow.py`

- Encodes the current evidence-backed narrow edge cluster.
- Explicitly distinguishes `PRIMARY_MODELS`, `COMPARATOR_MODELS`,
  `BASELINE_ONLY_MODELS`, and `DEMOTED_MODELS`.
- Produces narrow-scope summaries, cost-sensitivity outputs, model-stability
  reporting, manifest, and report artifacts.
- This is the strongest current precedent for future governance roles.

## Existing Output Schemas Worth Reusing

### Forecast outputs

- Built through `ForecastModel._build_forecast_frame`.
- Validated by `validate_forecast_frame`.
- Core fields already present:
  `timestamp`, `ticker`, `y_true`, `y_pred`, `model_name`, `target_type`,
  `horizon`, `window_id`, optional `target_timestamp`.

### Regime outputs

- Built through `RegimeModel._build_regime_frame`.
- Validated by `validate_regime_frame`.
- Core fields already present:
  `timestamp`, `ticker`, `regime_label`, `regime_prob_bull`,
  `regime_prob_bear`, `regime_prob_sideway`, `source_model`, `window_id`.

### Policy/backtest outputs

- `execute_policy_configuration` already annotates signals and positions with
  policy metadata.
- `CostAwareBacktester.run()` already emits trades, strategy metrics, and equity
  curves.
- These outputs can be linked into quant-core packets without inventing new
  policy execution logic.

### Manifest outputs

- `build_run_manifest` and `build_batch_manifest` already capture git, runtime,
  dependencies, command, evaluated/skipped models, and artifact paths.
- The manifest layer is ready to grow into a quant-core manifest by adding
  run-mode, role-filter metadata, packet outputs, and health/consensus outputs.

## Governance and Orchestration Implications

### Keep

- Walk-forward evaluation as the forecast execution backbone.
- Current forecast contract and supported forecast model implementations.
- Current regime and risk model contracts and outputs.
- Current manifest and summary writing conventions.
- Existing rehab and hardening reporting patterns.

### Extend

- `src/forecast/registry.py` into a role-aware governed registry.
- `src/reporting/manifests.py` to record run mode, role filters, and richer
  output inventories.
- Evaluation/reporting modules to build consensus, disagreement, packet, and
  health summaries.
- Runner layer with an explicit quant-core orchestration entry point.

### Avoid Refactoring In This Phase

- The underlying forecast model implementations.
- Risk/regime model algorithms.
- Phase 3 routing or meta-model logic.
- Legacy all-model stacking experiments in `src/ml`.

## Key Findings

1. The current quant-core already has a strong reusable backbone:
   walk-forward evaluation, model registry, risk/regime overlays, manifests,
   and benchmark reporting.
2. The narrow rehab lane already expresses model-role semantics informally:
   primary, comparator, baseline-only, demoted.
3. The missing piece is not model complexity. It is a unified governance and
   orchestration layer that formalizes those roles and writes richer analysis
   artifacts.
4. The cleanest implementation path is to extend the existing forecast registry
   and reporting layers, then add a dedicated `run_quant_core.py` entry point
   that reuses the current execution path.
