# Phase 2 Reuse Map
## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Historical archive |
| Created / authored | Saturday, 2026-04-18 21:38:25 ICT (UTC+07:00) |
| Last updated | Tuesday, 2026-04-28 22:28:23 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `8800ce6e3780c7978856737e70cb5e3b999eacee` |
| Timestamp source | Git history |
| Status | Historical reference |

This document captures the repo audit completed before the Phase 2 regime,
risk, strategy, and evaluation extensions started on branch
`rebuild-regime-risk-aware-forecasting`.

## Reuse Decisions

| Existing module | Purpose | Action | Reason |
| --- | --- | --- | --- |
| `src/core/contracts.py` | Shared forecast, signal, and position schemas plus walk-forward window metadata | extend | Phase 2 can keep the existing forecast/signal/position contracts and add regime validation beside them |
| `src/evaluation/walkforward.py` | Canonical leakage-safe data loading, target generation, and expanding/rolling walk-forward evaluation | extend | Already the correct Phase 1 backbone; Phase 2 should add regime/risk context on top of these windows instead of creating a second evaluator |
| `src/evaluation/backtest.py` | Explicit cost-aware post-forecast backtest and separated strategy metrics | extend | The cost/slippage and strategy metric separation are already correct; Phase 2 only needs conditioning-aware comparisons and richer position metadata |
| `src/risk/base.py` | Shared fit/forecast contract for return-distribution risk models | keep as-is | The current contract is already suitable for GARCH and keeps Phase 2 risk models aligned with Phase 1 |
| `src/risk/monte_carlo.py` | Monte Carlo return-path simulation and tail metrics | keep as-is | Still useful as a complementary risk overlay and as a comparison point against GARCH forecasts |
| `src/risk/var_cvar.py` | Historical VaR/CVaR estimation from return distributions | keep as-is | Already produces the tail metrics Phase 2 needs; GARCH output should integrate with it rather than replace it |
| `src/risk/drawdown.py` | Realized drawdown analytics over return series | keep as-is | Directly reusable for drawdown-aware exposure reduction and reporting |
| `src/strategy/thresholding.py` | Transparent threshold-to-signal conversion from forecast frames | extend | This is the natural hook for regime-aware thresholds without rewriting the signal contract |
| `src/strategy/sizing.py` | Risk-budget-based position sizing with risk-frame joins | extend | The current join and sizing path already accepts risk context; Phase 2 should add volatility-aware and drawdown-aware scaling here |
| `src/strategy/execution_policy.py` | Strategy orchestration for forecast -> signal -> positions | extend | Best place to coordinate forecast-only, forecast+risk, and forecast+risk+regime policies while keeping one execution path |
| `src/reporting/manifests.py` | Run manifest generation with git metadata and artifact paths | extend | Phase 2 needs runtime, dependency, regime-policy, and risk-policy metadata added to the same manifest writer |
| `src/reporting/summary.py` | Comparison tables and markdown summary generation | extend | Phase 2 needs conditioning comparisons, but the Phase 1 summary path is already the right output surface |
| `src/forecast/statistical/sarimax.py` | Shared-contract SARIMAX wrapper | keep as-is | Runtime alignment fixed the blocker; no structural rewrite is needed before Phase 2 benchmarking |
| `src/forecast/statistical/ets.py` | Shared-contract ETS wrapper | keep as-is | Same as SARIMAX: now usable in the canonical benchmark interpreter |
| `scripts/run_phase1_benchmark.py` | Canonical end-to-end Phase 1 benchmark over prepared data, risk models, strategy, backtest, and reporting | extend | Best foundation for a Phase 2 validation benchmark because it already exercises the current contracts end to end |
| `src/ml/feature_engineering.py` | Reusable prepared-feature builder, including rule-based regime and volatility state features | wrap / extend | Useful source of leakage-safe regime inputs and volatility features, but Phase 2 regime inference should live under the new contract-first `src/regime/*` layer |
| `src/ml/features/registry.py` | Approved feature-set registry and task feature selection | keep as-is | Existing feature governance is still the safest way to keep Phase 2 feature usage conservative |
| `src/ml/regime/regime_detector.py` | Legacy rule-based `NORMAL/HIGH_VOL/CRISIS` detector with probability outputs | wrap / selective refactor | Helpful fallback ideas and probability-shaping patterns, but the state space does not match the required bull/bear/sideway Phase 2 output contract |
| `src/ml/risk/risk_engine.py` | Legacy rolling risk engine with VaR/CVaR/CoVaR/drawdown summaries | wrap / mine for logic | Useful formulas and naming conventions remain, but the runner is too coupled to the old ML stack to become the Phase 2 core risk path |
| `src/ml/benchmark/system_benchmark.py` | Legacy mode-comparison benchmark for forecast-only vs risk/regime/allocation variants | wrap / mine for logic | The comparison framing matches Phase 2 well, but execution is tied to the old trainer and should not drive the new contract-based benchmark |
| `src/ml/backtest/regime_aware_analysis.py` | Legacy regime-conditioned artifact analysis and reporting | replace for execution, mine for reporting ideas | Useful for naming, regime summaries, and leakage checks, but it operates on old dual-task artifacts instead of Phase 1 forecast frames |

## Canonical Entry Points Found During Audit

- Data preparation: `scripts/train_ml_tickers.py --prepare-only`, backed by `DualModelTrainer.prepare_ticker_data(...)`
- Prepared dataset backbone used by evaluation: `data/processed/ml_5y/*.csv`, loaded by `WalkForwardEvaluator.load_ticker_data(...)`
- Canonical benchmark/evaluation/reporting path: `scripts/run_phase1_benchmark.py`
- Canonical walk-forward engine: `src/evaluation/walkforward.py`
- Canonical strategy backtest engine: `src/evaluation/backtest.py`
- Legacy training entry point still present and now runtime-aligned: `scripts/train_ml_tickers.py`

## Phase 2 Architectural Direction

- Keep the Phase 1 contract-first stack as the primary execution path.
- Add the new regime layer under `src/regime/*` and feed its outputs into the existing strategy and reporting surfaces.
- Add GARCH under `src/risk/*` using the existing risk contract, then merge its outputs with VaR/CVaR and drawdown context.
- Extend the current benchmark rather than routing Phase 2 through the old `src/ml/*` benchmark stack.
- Reuse old regime/risk modules only for formulas, labels, and reporting ideas where they fit the new contracts cleanly.
