# Phase 2.6 Decision Layer Audit
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Sunday, 2026-04-19 00:50:04 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

## Decision Path

Current policy flow:

1. Forecast outputs enter `generate_threshold_signals()` or `generate_regime_aware_signals()`.
2. Signals are sized in `size_positions()`.
3. `execute_strategy_variant()` selects the policy path for each conditioning mode.
4. `CostAwareBacktester.run()` converts sized positions into trades, daily returns, and strategy metrics after explicit costs.
5. `run_phase25_hardening.py` sweeps thresholds, sizing modes, and cost modes across ticker groups and horizons.
6. `src/reporting/hardening.py` aggregates stability, cost, regime, and risk summaries.

## Audit Map

| module | role | keep / calibrate / simplify / ablate | reason |
| --- | --- | --- | --- |
| `src/strategy/thresholding.py` | Converts forecasts to directional signals with one symmetric threshold | calibrate | The logic is simple and reusable, but Phase 2.5 did not test enough threshold levels to tell whether participation is too loose or too tight. |
| `src/strategy/regime_thresholding.py` | Joins regime and risk context, then adjusts thresholds by regime | calibrate / ablate | The module is mechanically sound, but its value over plain thresholding is still unproven. It needs explicit regime-on versus regime-off ablations rather than hidden use through one rich mode. |
| `src/strategy/sizing.py` | Applies fixed-fraction or adaptive sizing using conviction, volatility, regime, and drawdown multipliers | calibrate / simplify / ablate | This is the most likely pressure point. The adaptive path compounds several penalties at once, which makes it hard to tell whether volatility sizing, regime multipliers, or drawdown haircuts are causing over-restriction. |
| `src/strategy/execution_policy.py` | Wraps thresholding plus sizing into basic and regime-aware execution policies | extend / ablate | The contract is reusable, but the current policy classes are too coarse for Phase 2.6. We need explicit policy variants so regime, volatility sizing, and drawdown control can be toggled independently. |
| `src/evaluation/backtest.py` | Applies explicit entry/exit costs and computes strategy metrics after forecast generation | keep as-is | The backtest already keeps costs explicit and separate from forecast evaluation. It should remain stable so policy calibration does not move the accounting target. |
| `scripts/run_phase2_benchmark.py` | Core Phase 2 forecast, regime, risk, and strategy execution path | extend | It already separates `run_phase2_core()` from `run_phase2_strategy_suite()`. That makes it the right reuse point for a Phase 2.6 runner with richer policy specifications. |
| `scripts/run_phase25_hardening.py` | Bounded matrix runner over groups, horizons, thresholds, cost modes, and sizing modes | wrap / extend | The matrix and manifest structure are reusable, but Phase 2.6 needs policy ablations and calibration-specific summaries rather than only conditioning-mode summaries. |
| `src/reporting/hardening.py` | Stability, regime, risk, cost, and readiness summaries | extend | The current reporting tells us the stack is unstable, but not whether thresholding, sizing, or regime conditioning is the main source of failure. Phase 2.6 reporting needs policy-level attribution. |
| `tests/phase25/*` | Validates matrix sizing and high-level hardening summaries | keep and extend alongside new tests | These tests cover the existing matrix/reporting foundation. Phase 2.6 needs additional tests for policy matrix generation, calibration summaries, and schema consistency. |

## Calibration Questions

These questions should drive explicit benchmarkable ablations rather than ad hoc tweaks:

1. Is the base threshold too low, producing noisy post-cost trading?
2. Is the base threshold too high, suppressing the little usable edge that exists?
3. Is adaptive volatility sizing too restrictive relative to fixed-fraction sizing?
4. Is drawdown control adding useful protection, or just compounding existing size cuts?
5. Does regime-aware thresholding add value over a plain threshold after costs?
6. Does a simpler policy outperform the richer conditioning stack?
7. Are some models forecast-adequate but monetization-poor, implying policy failure rather than forecast failure?

## Required Phase 2.6 Ablations

The runner should make the following policy modes explicit:

- `forecast_only`
- `fixed_threshold_fixed_fraction`
- `fixed_threshold_adaptive`
- `regime_threshold_fixed_fraction`
- `fixed_threshold_adaptive_drawdown`
- `regime_threshold_adaptive_drawdown`
- `risk_only_no_regime`
- `regime_only_fixed_fraction`

## Initial Working Hypothesis

Phase 2.5 evidence points to policy calibration, not regime-model collapse, as the first thing to test:

- regime fallback remained low, so the regime layer is not obviously broken
- adaptive sizing was much more restrictive than fixed-fraction sizing
- fixed-fraction looked more robust in multiple slices
- median strategy quality remained poor even when turnover was reduced

That means Phase 2.6 should first isolate thresholding and sizing behavior before concluding that the forecast edge itself is unmonetizable.
