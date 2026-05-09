# Evidence Matrix

## Evidence Matrix Summary

This supervisor package uses the Phase 4.99 evidence matrix as the primary claim-to-artifact map. The CSV copy in `evidence/EVIDENCE_MATRIX.csv` contains 54 rows covering Phase 0 through Phase 4.99. No requested source artifact was missing when the matrix was assembled.

## Strongest Evidence Areas

- Governance freeze and scope control.
- Config-driven experiment standardization.
- Forecasting model and baseline comparison.
- Risk-aware aggregate utility comparison.
- Regime-aware diagnostics across trend, volatility, and combined regimes.

## Weakest Evidence Areas

- Consistent ML-model superiority over simple baselines.
- Stacking superiority over baselines.
- Aggregate risk-aware utility improvement.
- Risk feature sophistication.
- Forecast outlier control and health-gate maturity.

## Missing Evidence

No requested source artifact was missing from the Phase 0 through Phase 4.99 evidence set used for this package. Raw local experiment outputs under `outputs/experiments/` remain outside this package and should not be treated as committed report artifacts.

## How Evidence Supports the Final Conclusion

The evidence supports a bounded conclusion: VSEF v1 is a governed, reproducible diagnostic research framework. The evidence does not support a universal model superiority claim or an investment-readiness claim. Phase 4 provides the strongest academic contribution because it reframes the result around regime-dependent model behavior, risk-layer utility, and horizon quality.

This matrix links Phase 0 through Phase 4.99 claims to reviewable artifacts. Missing artifacts must remain visible; in the current requested source set, all listed source artifacts exist.

| Phase | Artifact Type | Artifact Path | Exists | Evidence Role | Key Content | Supports Claim | Limitations | Reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 0 | architecture freeze | `docs/architecture/VSEF_v1_ARCHITECTURE.md` | true | governance scope control | Freezes v1 layers, inclusions, exclusions, and diagnostic candidate boundary | VSEF v1 has governed scope | Governance evidence only | Phase 0 |
| Phase 0 | model registry | `docs/governance/MODEL_REGISTRY.md` | true | model scope control | Freezes SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, Stacking | Supported model set is bounded | Runtime evidence comes from later phases | Phase 0 |
| Phase 0 | data policy | `docs/governance/DATA_POLICY.md` | true | data governance | Freezes `vnstock_data`, daily OHLCV, and schema | Provider and schema are governed | Provider/API changes can affect reproducibility | Phase 0 |
| Phase 0 | evaluation protocol | `docs/governance/EVALUATION_PROTOCOL.md` | true | evaluation governance | Requires time-series-safe evaluation and artifact-backed claims | Evaluation claims require evidence | Protocol is policy | Phase 0 |
| Phase 0 | project tracker | `docs/governance/PROJECT_TRACKER.md` | true | scope tracking | Tracks done, deferred, and out-of-scope work | Scope creep is controlled | Sign-off still depends on committed artifacts | Phase 0 |
| Phase 1 | ExperimentOrchestrator | `src/engine/experiment_orchestrator.py` | true | runtime standardization | YAML config loading, validation, outputs, logs, manifests | Experiments are auditable | Environment still matters | Source |
| Phase 1 | metrics engine | `src/ml/evaluation/metrics_engine.py` | true | metric standardization | Long-form model and baseline metrics | Metrics are comparable | Metrics are context-bound | Source |
| Phase 1 | baseline registry | `src/ml/baselines/baseline_registry.py` | true | baseline standardization | Persistence, zero-return, random-direction, moving-average baselines | Baselines are explicit | Baselines are not official v1 models | Source |
| Phase 1 | smoke config | `configs/experiments/EXP-SMOKE-001.yaml` | true | smoke definition | Provider-backed smoke experiment | Smoke can be rerun | Config is not result evidence | EXP-SMOKE-001 |
| Phase 1 | smoke report | `docs/experiments/EXP-SMOKE-001_REPORT.md` | true | smoke evidence | Completed run, 72 prediction rows, 48 metric rows | Standard artifact contract works | Optional missing artifacts disclosed | Run ID `EXP-SMOKE-001-20260508T204336Z-6c409f61` |
| Phase 2 | forecasting core report | `reports/forecasting_core/FORECASTING_CORE_VALIDATION_REPORT.md` | true | forecasting synthesis | Aggregates EXP-FC-001/002/003 | Forecast comparisons ran | Does not prove consistent model superiority | Phase 2 |
| Phase 2 | forecast metrics | `reports/forecasting_core/forecast_metrics.csv` | true | metric evidence | 1,295 metric rows | Models and baselines share schema | Contains outliers | Phase 2 |
| Phase 2 | model ranking | `reports/forecasting_core/model_ranking.csv` | true | ranking evidence | 1,295 ranking rows | Baselines frequently win MAE/RMSE | Context-specific only | Phase 2 |
| Phase 2 | horizon comparison | `reports/forecasting_core/horizon_comparison.csv` | true | horizon evidence | 185 horizon rows | T+1/T+3/T+5 can be compared | Outliers affect some rows | Phase 2 |
| Phase 2 | stability metrics | `reports/forecasting_core/stability_metrics.csv` | true | stability evidence | Prediction/error stability | Instability is visible | Does not prove forecast value | Phase 2 |
| Phase 2 | error distribution | `reports/forecasting_core/error_distribution_summary.csv` | true | error evidence | Residual summaries | Outlier risk is visible | Requires health controls | Phase 2 |
| Phase 3 | candidate policies | `configs/policies/candidate_policy_forecast_only.yaml`; `configs/policies/candidate_policy_risk_aware.yaml` | true | policy control | Forecast-only and risk-aware ranking rules | Candidate policies are explicit | Policies are not recommendation rules | Phase 3 |
| Phase 3 | candidate comparison | `reports/risk_aware/candidate_comparison.csv` | true | candidate diagnostics | 1,780 diagnostic rows | Candidate rows can be compared | Retrospective diagnostics only | Phase 3 |
| Phase 3 | basket metrics | `reports/risk_aware/topn_basket_metrics.csv` | true | basket evidence | 18 top-N rows | Aggregate utility can be tested | Baskets are not portfolios | Phase 3 |
| Phase 3 | risk-aware report | `reports/risk_aware/RISK_AWARE_DECISION_RESEARCH_REPORT.md` | true | risk synthesis | Aggregate risk-aware utility not proven | Weakens risk-layer value claim | Context-specific rows still require review | Phase 3 |
| Phase 3 | risk ranking and comparisons | `reports/risk_aware/risk_adjusted_ranking.csv`; `drawdown_comparison.csv`; `hit_ratio_comparison.csv` | true | risk details | Risk-aware ranking, drawdown, hit ratio | Risk effects are measurable | Aggregate value not proven | Phase 3 |
| Phase 4 | regime detector | `src/ml/regime/regime_detector.py` | true | implementation | Rule-based rolling return and volatility labels | Transparent labels are reproducible | Thresholds are subjective | Phase 4 |
| Phase 4 | regime policy | `configs/policies/regime_policy.yaml` | true | policy | Return window 20, volatility window 20, quantile thresholds | Regime definitions are explicit | Robustness is not exhaustive | Phase 4 |
| Phase 4 | regime labels | `reports/regime_analysis/regime_labels.csv` | true | label evidence | 2,495 label rows | Regime dataset exists | Labels are diagnostics, not ground truth | Phase 4 |
| Phase 4 | model by regime | `reports/regime_analysis/regime_model_metrics.csv`; `model_ranking_by_regime.csv`; `model_ranking_consistency.csv` | true | by-regime model evidence | 1,650 metric rows; ranking and consistency rows | No universal ML dominance is shown | Persistence baseline remains strong | Phase 4 |
| Phase 4 | risk by regime | `reports/regime_analysis/regime_risk_metrics.csv`; `regime_risk_policy_comparison.csv` | true | by-regime risk evidence | 196 risk metric rows; 98 comparison rows | Risk-aware value is context-specific | No universal risk-layer improvement | Phase 4 |
| Phase 4 | horizon by regime | `reports/regime_analysis/regime_horizon_metrics.csv`; `horizon_ranking_by_regime.csv` | true | by-regime horizon evidence | 198 metric rows; 33 ranking rows | Horizon behavior can be evaluated by regime | T+1 outliers distort aggregate rows | Phase 4 |
| Phase 4 | health gate | `reports/regime_analysis/model_health_by_regime.csv`; `eligibility_flags.csv` | true | control layer | Eligible 742, flagged 608, excluded 300 | Outliers and weak evidence are exposed | Rules need future refinement | Phase 4 |
| Phase 4 | regime report | `reports/regime_analysis/REGIME_AWARE_ANALYSIS_REPORT.md` | true | synthesis | Phase 4 objective, policy, findings, limitations | Regime-aware framing is strongest academic contribution | Rule-based regimes remain exploratory | Phase 4 |
| Phase 4.99 | executive report | `reports/vsef_v1_executive_pack/VSEF_v1_EXECUTIVE_REPORT.md` | true | executive synthesis | Phase 0-4 narrative and conclusion | Defense narrative is coherent | Synthesis only | Phase 4.99 |
| Phase 4.99 | evidence matrix | `reports/vsef_v1_executive_pack/EVIDENCE_MATRIX.csv`; `EVIDENCE_MATRIX.md` | true | evidence inventory | Artifact-by-artifact source map | Claims tie to paths | Depends on artifacts remaining available | Phase 4.99 |
| Phase 4.99 | defense pack | `DEFENSE_SLIDE_OUTLINE.md`; `DEFENSE_SCRIPT.md` | true | defense preparation | 12-slide outline and short/full scripts | Defense-ready material exists | Not a slide deck file | Phase 4.99 |
| Phase 4.99 | limitations and roadmap | `LIMITATIONS_AND_THREATS_TO_VALIDITY.md`; `FUTURE_WORK_ROADMAP.md` | true | review and planning | Validity threats and next-step roadmap | Future work is disciplined | Does not approve Phase 5 | Phase 4.99 |

## Summary

- Total CSV matrix rows: 54.
- Missing requested source artifacts: none found during the Phase 4.99 evidence check.
- Strongest evidence areas: governance freeze, config-driven experiment standardization, baseline comparison, risk-aware aggregate evaluation, regime-aware diagnostics.
- Weakest evidence areas: consistent ML superiority over baselines, aggregate risk-aware utility, deep learning/stacking value, risk feature sophistication, forecast outlier control.
