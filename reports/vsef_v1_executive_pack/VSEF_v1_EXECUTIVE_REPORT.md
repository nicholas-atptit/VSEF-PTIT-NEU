# VSEF v1 Executive Report

## 1. Executive Summary

VSEF v1 progressed through five evidence-building stages before this consolidation pack.

- Phase 0 froze governance and scope.
- Phase 1 standardized reproducible experiments.
- Phase 2 validated the forecasting core against baselines.
- Phase 3 tested risk-aware diagnostic candidates.
- Phase 4 added regime-aware diagnostics, model health gates, and by-regime evaluation.

VSEF v1 does not prove that any single ML forecasting model dominates all conditions. The strongest evidence supports a governed, reproducible diagnostic framework where model behavior, risk-layer utility, and horizon quality are regime-dependent.

The practical conclusion is direct: VSEF v1 is stronger as an auditable research framework than as a pure prediction engine. Baselines remain hard to beat on MAE/RMSE, risk-aware ranking does not improve aggregate candidate utility, and regime-aware analysis gives the strongest academic framing for what the system can test next.

## 2. System Purpose

VSEF v1 is a governed stock forecasting research framework, not a trading system.

It produces diagnostic artifacts:

- forecasts
- baselines
- metrics
- risk summaries
- candidate diagnostics
- regime labels
- reports

It does not produce:

- BUY / SELL / HOLD advice
- capital allocation
- broker execution
- guaranteed trading signals

Diagnostic candidates are review artifacts only. They are not recommendations.

## 3. Phase-by-Phase Summary

| Phase | Objective | Key Deliverables | Evidence | Main Finding | Status |
| --- | --- | --- | --- | --- | --- |
| Phase 0 | Freeze VSEF v1 governance and scope | Architecture freeze, model registry, data policy, evaluation protocol, project tracker | `docs/architecture/VSEF_v1_ARCHITECTURE.md`, `docs/governance/*.md` | Scope control worked; v1 excludes live trading, broker execution, portfolio allocation, autonomous LLM decisions, and unsupported models | Complete |
| Phase 1 | Standardize reproducible experiments | `ExperimentOrchestrator`, config schema, metrics engine, baseline registry, smoke evidence | `docs/experiments/PHASE1_EXPERIMENT_STANDARDIZATION.md`, `EXP-SMOKE-001_REPORT.md`, source modules | Config-driven runs produce comparable predictions, metrics, manifests, logs, and summaries | Complete |
| Phase 2 | Validate forecasting core against baselines | Forecast metrics, ranking, stability, horizon comparison, validation report | `reports/forecasting_core/*` | The current evidence does not prove consistent model superiority over simple baselines on MAE/RMSE | Complete |
| Phase 3 | Test risk-aware diagnostic candidate ranking | Candidate policies, candidate comparison, top-N basket metrics, risk-aware report | `reports/risk_aware/*` | Risk-aware ranking did not improve aggregate candidate utility over forecast-only ranking | Complete |
| Phase 4 | Test whether model/risk/horizon behavior depends on regimes | Regime detector, regime labels, health gate, by-regime metrics and reports | `reports/regime_analysis/*`, `src/ml/regime/regime_detector.py` | Regime-aware analysis supports a no-universal-best-model thesis under tested definitions, while persistence remains a strong baseline | Complete |
| Phase 4.99 | Consolidate v1 evidence for executive review and defense | Executive report, evidence matrix, slide outline, defense script, limitations, roadmap | `reports/vsef_v1_executive_pack/*` | Consolidates evidence without adding new experiments or claims | Complete |

## 4. Architecture and Governance Summary

Phase 0 froze the v1 boundary. The frozen system contains data, feature engineering, forecasting, ensemble/consensus, evaluation, risk summary, existing strategy diagnostics, artifact/manifest, and documentation/governance layers.

Supported v1 models are SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, and Stacking. Baselines are comparison evidence, not official forecasting models.

The data provider policy remains `vnstock_data` with daily OHLCV only. Required schema is `date`, `ticker`, `open`, `high`, `low`, `close`, and `volume`.

Evaluation policy requires time-series-safe evaluation. Random row-level splits are not accepted for forecasting claims. Claims must be backed by artifact paths, date windows, ticker/horizon context, model lists, and metric definitions.

The diagnostic-only boundary is central. VSEF v1 does not provide investment authority, trade instructions, capital allocation, broker execution, or autonomous decisions.

## 5. Experiment Standardization Summary

Phase 1 established the reproducibility pattern:

```text
CONFIG IN -> ORCHESTRATED RUN -> STANDARD ARTIFACTS OUT
```

`src/engine/experiment_orchestrator.py` loads YAML configs, validates governance boundaries, creates `outputs/experiments/{experiment_id}/`, writes original and resolved config copies, runs configured models and baselines, and records metrics, predictions, logs, manifests, and summaries.

`src/ml/evaluation/metrics_engine.py` standardizes long-form metrics. `src/ml/baselines/baseline_registry.py` standardizes persistence, zero-return, random-direction, and moving-average baselines.

`EXP-SMOKE-001` provides provider-backed smoke evidence. It generated non-empty model and baseline predictions and metrics under the standardized output contract.

## 6. Forecasting Core Findings

Phase 2 showed that the forecasting core can run reproducible model/baseline comparisons across `ACB`, `DGC`, `FPT`, `HPG`, and `MWG`, with T+1, T+3, and T+5 horizon coverage where configured.

The current evidence does not prove that the forecasting layer consistently outperforms simple baselines.

Baseline evidence is strong. In the Phase 2 model ranking, persistence frequently wins MAE/RMSE contexts. Some models win limited directional-accuracy contexts, and SARIMAX beat the best baseline for ACB T+3 on MAE/RMSE, but those wins are bounded and do not establish general superiority.

Stacking ran, which validates ensemble runtime execution, but stacking did not prove superiority. Error distribution evidence also exposed unstable outliers in some model outputs.

## 7. Risk-Aware Decision Findings

Phase 3 compared forecast-only diagnostic candidate ranking against risk-aware ranking using generated Phase 2 forecast artifacts and local daily OHLCV evidence.

The current evidence does not prove that risk-aware ranking improves candidate utility over forecast-only ranking in aggregate.

The Phase 3 report recorded negative aggregate deltas for average realized return, return/volatility proxy, drawdown reduction, and hit ratio. Some metric-specific improvements appeared in selected top-N and horizon rows, but those are context-specific and do not justify a broad risk-layer value claim.

Risk-aware ranking remains useful as a diagnostic process because it exposed where the current risk features are too basic and where candidate scoring inherits forecast instability.

## 8. Regime-Aware Findings

Phase 4 generated `2,495` regime-label rows for `ACB`, `DGC`, `FPT`, `HPG`, and `MWG` over `2023-01-03` to `2024-12-31`.

The evidence supports a no-universal-best-model thesis under the tested regime definitions.

Key Phase 4 findings:

- Trend labels included bull `1,057`, sideway `949`, bear `439`, and insufficient-history `50` rows.
- Volatility labels included low-vol `1,276`, high-vol `1,124`, and insufficient-history `95` rows.
- Persistence remained a stable MAE winner in many regime contexts, supporting baseline competitiveness rather than universal ML model dominance.
- Risk-aware diagnostics showed mixed context-specific improvements by regime, not universal improvement.
- The health gate exposed outliers and weak evidence instead of hiding them: eligible `742`, flagged `608`, excluded `300` health rows.

## 9. What This Means

VSEF v1 should be presented as a governed diagnostic research system, not as an investment recommendation system.

Its strongest contribution is not that it found a dominant model. Its strongest contribution is that it creates a reproducible way to test model behavior, baseline competitiveness, risk-layer utility, and horizon behavior under explicit governance and regime definitions.

Future improvement should focus on regime-aware filtering, model health gates, outlier control, risk feature design, and stricter candidate eligibility. Adding more models blindly is not the next best step.

## 10. Limitations

See `reports/vsef_v1_executive_pack/LIMITATIONS_AND_THREATS_TO_VALIDITY.md`.

## 11. Future Work

See `reports/vsef_v1_executive_pack/FUTURE_WORK_ROADMAP.md`.

## 12. Diagnostic-Only Disclaimer

All VSEF v1 outputs are research and diagnostic decision-support artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of guaranteed profitable trading.
