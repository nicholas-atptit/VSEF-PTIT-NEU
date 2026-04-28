# Phase 2.5 Entry Point Audit
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Saturday, 2026-04-18 22:45:24 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

This document captures the repo audit completed before the Phase 2.5 hardening
work started on branch `rebuild-regime-risk-aware-forecasting`.

## Reuse Decisions

| Existing module/script | Purpose | Action | Reason |
| --- | --- | --- | --- |
| `scripts/run_phase2_benchmark.py` | Canonical Phase 2 end-to-end benchmark over prepared data, forecast models, GARCH risk, Markov-switching regime labels, strategy conditioning, backtest, and reporting | extend | Already the right orchestration path; Phase 2.5 should sweep and aggregate it rather than replace it |
| `src/evaluation/walkforward.py` | Leakage-safe data loading, target generation, and expanding/rolling walk-forward evaluation | keep as-is | This is already the canonical backbone and should remain the single source of train/test window truth |
| `src/evaluation/backtest.py` | Explicit post-forecast backtest with cost/slippage handling and strategy metrics | extend | Phase 2 added `strategy_variant` support; Phase 2.5 can build robustness summaries on top without changing the execution semantics |
| `src/regime/base.py` | Regime-model contract for single-ticker time-ordered history | keep as-is | The Phase 2 contract is already explicit and suitable for stability analysis |
| `src/regime/labels.py` | Regime probability normalization and state-to-label mapping helpers | keep as-is | Already reusable for sanity checks and fallback statistics |
| `src/regime/markov_switching.py` | Markov-switching regime inference with deterministic fallback labeling | extend | This is the main regime generator and the hardening phase needs to measure fallback frequency, switch behavior, and probability stability |
| `src/risk/base.py` | Shared return-distribution risk model contract | keep as-is | Already sufficient for stress-testing risk generation behavior |
| `src/risk/garch.py` | GARCH-based volatility, VaR/CVaR, and drawdown-state forecasting | extend | This is the Phase 2 risk generator and needs stability/exposure analytics rather than a contract rewrite |
| `src/risk/drawdown.py` | Realized drawdown analytics over return histories | keep as-is | Already provides the drawdown state inputs needed for hardening summaries |
| `src/strategy/thresholding.py` | Phase 1 threshold-to-signal conversion | keep as-is | Still needed as the unconditioned baseline policy |
| `src/strategy/regime_thresholding.py` | Phase 2 regime-aware thresholding with risk/regime context joins | extend | Best place to analyze threshold sensitivity and regime-conditioned signal behavior |
| `src/strategy/sizing.py` | Position sizing with volatility/risk/regime/drawdown inputs | extend | Hardening needs to compare adaptive sizing against simpler fallback sizing and measure exposure reduction frequency |
| `src/strategy/execution_policy.py` | Phase 1 and Phase 2 policy orchestration | extend | Existing policy objects are the correct hook for adding bounded sizing-mode stress paths |
| `src/reporting/manifests.py` | Run manifest generation with git/runtime/dependency metadata | extend | Phase 2.5 needs batch-level manifests and benchmark-matrix metadata, not a new manifest system |
| `src/reporting/summary.py` | Comparison tables and Phase 2 markdown summaries | extend | Current summaries cover per-run performance; Phase 2.5 needs cross-run stability and sensitivity summaries |
| `tests/phase2/test_regime_layer.py` | Validates regime schema and label mapping behavior | extend | Good base for adding regime stability tests instead of starting a separate suite |
| `tests/phase2/test_garch_risk.py` | Validates GARCH forecast interface and tail metrics | extend | Good base for adding risk hardening checks |
| `tests/phase2/test_strategy_conditioning.py` | Validates regime-aware thresholds and hostile-state sizing reduction | extend | Directly relevant to threshold sensitivity and sizing stress testing |
| `tests/phase2/test_backtest_variants.py` | Validates strategy-variant separation in the backtest | keep as-is | Already guards the comparison semantics that Phase 2.5 relies on |
| `tests/phase2/test_reporting_phase2.py` | Validates Phase 2 comparison and delta summary logic | extend | Good base for hardening summary and cost-sensitivity tests |
| `tests/phase2/test_walkforward_regression.py` | Guards leakage-safe split behavior after Phase 2 changes | keep as-is | Still the right regression guard for no-leakage guarantees |

## Canonical Entry Points

- Phase 2 benchmark execution: `scripts/run_phase2_benchmark.py`
- Walk-forward evaluation: `src/evaluation/walkforward.py`
- Regime generation: `src/regime/markov_switching.py`
- GARCH/risk generation: `src/risk/garch.py`
- Strategy thresholding and sizing: `src/strategy/regime_thresholding.py`, `src/strategy/sizing.py`, `src/strategy/execution_policy.py`
- Manifests and summaries: `src/reporting/manifests.py`, `src/reporting/summary.py`
- Current Phase 2 regression tests: `tests/phase2/*`

## Phase 2.5 Direction

- Keep `scripts/run_phase2_benchmark.py` as the execution backbone and build a matrix runner around it.
- Reuse the existing evaluator, risk, regime, and strategy layers rather than creating alternate implementations.
- Add batch-level aggregation, stability summaries, cost-sensitivity summaries, and regime/risk sanity outputs as small extensions around the current contracts.
- Treat hardening as evidence gathering: more tickers, more windows, more horizons, more thresholds, and more cost assumptions under the same leakage-safe backbone.
