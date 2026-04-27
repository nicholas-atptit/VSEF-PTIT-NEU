# Improvement Roadmap - Quant-Core Forecasting

## P0: Operational Reliability (Immediate)
- [x] **Interpreter Lock**: Standardize on `.venv_py313` for all quantitative execution.
- [x] **Statistical Calibration**: Verify `ETS` and `SARIMAX` execution on real market data.
- [x] **Risk/Regime Stability**: Ensure `GARCH` and `MarkovSwitching` execute without universal fallbacks.

## P1: Performance & UX (Short-Term)
- **Fallback Diagnostics**: Improve the `model_execution_log.csv` to explicitly capture the exact numerical error or data gap triggering a fallback.
- **Convergence Optimization**: Review initialization methods for `SARIMAX` to reduce optimization failures.
- **Reporting Clarity**: Add visual flags in `summary.md` for scenarios where statistical comparators were out-performed by baselines.

## P2: Long-Term Enhancements
- **Dynamic Ensemble Weighting**: Move from static weighted ensemble to a Kalman-filter or variance-based dynamic weight allocator in `src/ensemble/`.
- **Regime-Specific Risk Budgets**: Wire the `regime_multiplier_strength` directly into a more granular portfolio allocator.
- **CI/CD Quant-Gating**: Implement a "Quant-Smoke" check in the main GitHub Action to catch dependency regressions early.

---
*Signed: Antigravity - Senior Software Architect*
