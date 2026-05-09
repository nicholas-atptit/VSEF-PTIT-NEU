# Phase 5 Feature Analysis Limitations

- Feature importance and ablation results are diagnostic evidence, not causal proof.
- Ablation can be affected by correlated features and model re-fitting variance.
- Statistical wrappers such as ETS may ignore exogenous features, so feature ablation is mainly informative for feature-aware model wrappers.
- Regime labels are rule-based diagnostics and are sensitive to policy thresholds and small samples.
- Top-N decision-quality deltas are left blank unless feature-specific candidate metrics exist.
- SHAP status: SHAP was not generated because shap is unavailable in the active environment: No module named 'shap'
- Chart status: Generated 7 chart artifact(s) from actual Phase 5 tables.
- Ablation rows generated: 324.
- Tree importance rows generated: 1188.
- SHAP rows generated: 0.
- Decision-quality rows generated: 108.
- Raw experiment outputs under `outputs/experiments/` are local run evidence and are not intended for staging.

- EXP-FA-000: status=completed; errors=0; warnings=7.
- EXP-FA-001: status=completed; errors=0; warnings=10.
- EXP-FA-002: status=completed; errors=0; warnings=10.
- EXP-FA-003: status=completed; errors=0; warnings=10.
- EXP-FA-004: status=completed; errors=0; warnings=10.
- EXP-FA-005: status=completed; errors=0; warnings=10.
- EXP-FA-006: status=completed; errors=0; warnings=10.
- EXP-FA-007: status=completed; errors=0; warnings=11.
- EXP-FA-008: status=completed; errors=0; warnings=12.

All Phase 5 outputs are feature-analysis and interpretability research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, causal proof, or proof of guaranteed profitable trading.
