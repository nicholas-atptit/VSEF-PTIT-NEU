# Source Artifact Map

This appendix lists the major source files used to assemble the final supervisor package. It is an orientation aid, not a replacement for the evidence matrix.

## Governance and Architecture

- `docs/architecture/VSEF_v1_ARCHITECTURE.md`
- `docs/governance/MODEL_REGISTRY.md`
- `docs/governance/DATA_POLICY.md`
- `docs/governance/EVALUATION_PROTOCOL.md`
- `docs/governance/PROJECT_TRACKER.md`

## Experiment Standardization

- `docs/experiments/PHASE1_EXPERIMENT_STANDARDIZATION.md`
- `docs/experiments/EXP-SMOKE-001_REPORT.md`
- `configs/experiments/EXP-SMOKE-001.yaml`
- `src/engine/experiment_orchestrator.py`
- `src/ml/evaluation/metrics_engine.py`
- `src/ml/baselines/baseline_registry.py`

## Forecasting Core

- `reports/forecasting_core/FORECASTING_CORE_VALIDATION_REPORT.md`
- `reports/forecasting_core/forecast_metrics.csv`
- `reports/forecasting_core/model_ranking.csv`
- `reports/forecasting_core/stability_metrics.csv`
- `reports/forecasting_core/horizon_comparison.csv`
- `reports/forecasting_core/error_distribution_summary.csv`
- `reports/forecasting_core/charts/`
- `configs/experiments/EXP-FC-001.yaml`
- `configs/experiments/EXP-FC-002.yaml`
- `configs/experiments/EXP-FC-003.yaml`

## Risk-Aware Diagnostics

- `reports/risk_aware/RISK_AWARE_DECISION_RESEARCH_REPORT.md`
- `reports/risk_aware/candidate_comparison.csv`
- `reports/risk_aware/topn_basket_metrics.csv`
- `reports/risk_aware/risk_summary.csv`
- `reports/risk_aware/risk_adjusted_ranking.csv`
- `reports/risk_aware/drawdown_comparison.csv`
- `reports/risk_aware/hit_ratio_comparison.csv`
- `reports/risk_aware/charts/`
- `configs/policies/candidate_policy_forecast_only.yaml`
- `configs/policies/candidate_policy_risk_aware.yaml`
- `configs/experiments/EXP-RK-001.yaml`
- `configs/experiments/EXP-RK-002.yaml`

## Regime-Aware Analysis

- `reports/regime_analysis/REGIME_AWARE_ANALYSIS_REPORT.md`
- `reports/regime_analysis/regime_labels.csv`
- `reports/regime_analysis/regime_summary.csv`
- `reports/regime_analysis/regime_model_metrics.csv`
- `reports/regime_analysis/regime_risk_metrics.csv`
- `reports/regime_analysis/regime_horizon_metrics.csv`
- `reports/regime_analysis/model_health_by_regime.csv`
- `reports/regime_analysis/eligibility_flags.csv`
- `reports/regime_analysis/model_ranking_by_regime.csv`
- `reports/regime_analysis/model_ranking_consistency.csv`
- `reports/regime_analysis/regime_risk_policy_comparison.csv`
- `reports/regime_analysis/horizon_ranking_by_regime.csv`
- `reports/regime_analysis/charts/`
- `configs/policies/regime_policy.yaml`
- `src/ml/regime/regime_detector.py`

## Phase 4.99 Executive Pack

- `reports/vsef_v1_executive_pack/VSEF_v1_EXECUTIVE_REPORT.md`
- `reports/vsef_v1_executive_pack/EVIDENCE_MATRIX.csv`
- `reports/vsef_v1_executive_pack/EVIDENCE_MATRIX.md`
- `reports/vsef_v1_executive_pack/WHAT_WORKED_FAILED_MEANS.md`
- `reports/vsef_v1_executive_pack/DEFENSE_SLIDE_OUTLINE.md`
- `reports/vsef_v1_executive_pack/DEFENSE_SCRIPT.md`
- `reports/vsef_v1_executive_pack/LIMITATIONS_AND_THREATS_TO_VALIDITY.md`
- `reports/vsef_v1_executive_pack/FUTURE_WORK_ROADMAP.md`
- `reports/vsef_v1_executive_pack/PHASE_4_99_SYNTHESIS_SUMMARY.md`
