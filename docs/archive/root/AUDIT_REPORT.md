# Repository Audit Report - Quant-Core Operational Validation

## 1. Current Architecture Analysis

### Entry Points & Orchestration
- **Main Entry**: `scripts/run_quant_core.py` is the primary orchestrator for governed model execution.
- **Core Orchestrator**: `src/evaluation/quant_core.py` provides the logic for scenario matrix builds and multi-model execution.
- **Model Governance**: `src/core/model_governance.py` and `src/forecast/registry.py` define the "Model Zoo" roles (primary, comparator, baseline, shadow).

### Data Flow (End-to-End)
1. **Scenario Matrix**: Built in `src/evaluation/quant_core.py` based on presets (smoke, medium, full_forecast_daily).
2. **Walk-Forward Evaluator**: `src/evaluation/walkforward.py` handles the time-series splitting.
3. **Model Execution**: Forecasters from `src/forecast/` are instantiated and fitted.
4. **Risk & Regime Layer**: 
   - **Risk**: `GARCHRiskModel` (primary) or `VaRCVaRRiskModel` (fallback) in `src/risk/`.
   - **Regime**: `MarkovSwitchingRegimeModel` (primary) or deterministic threshold (fallback) in `src/regime/`.
5. **Reporting**: Results aggregated into analysis packets (`src/reporting/analysis_packets.py`) and manifests.

### Environment & Dependencies
- **Validated Runtime**: `.venv_py313` (Python 3.13.5)
- **Verified Packages**: `statsmodels 0.14.6`, `arch 8.0.0`, `scikit-learn 1.8.0`, `xgboost 3.2.0`, `lightgbm 4.6.0`.

## 2. Completed vs. Incomplete Components

| Component | Status | Location |
| :--- | :--- | :--- |
| Model Governance & Zoo | Fully Implemented | `src/core/`, `src/forecast/` |
| Statistical Forecasters (ETS/SARIMAX) | Operationally Validated | `src/forecast/statistical/` |
| Risk Layer (GARCH) | Operationally Validated | `src/risk/garch.py` |
| Regime Layer (Markov Switching) | Operationally Validated | `src/regime/markov_switching.py` |
| Analysis Packet Synthesis | Fully Implemented | `src/reporting/analysis_packets.py` |
| Full-Forecast Validation | **Verified** | 27 scenarios in `.venv_py313` |

## 3. Technical Debt & Risks

- **Convergence Stability**: `SARIMAX` and `MarkovSwitching` occasionally trigger `ConvergenceWarning` on small or volatile data segments.
- **PowerShell Stderr Sensitivity**: External runners (like CI/CD or Antigravity) may interpret warnings as errors due to PowerShell's default behavior.
- **Data Density Requirements**: Markov Switching requires at least 80 observations (hard constraint in `markov_switching.py`).

## 4. Code Smells & Architectural Weaknesses
- **Fallback Transparency**: While the manifest tracks fallbacks, the summary reporting could make the "Reason for Fallback" more prominent.
- **Script Coupling**: `scripts/run_quant_core.py` is quite large; reporting logic is partially centralized in `src/reporting/` but also present in the runner.

## 5. Major Operational Risks
- **Interpreter Drift**: Using `.venv` or `.venv313` (which are misconfigured or missing packages) breaks statistical models.
- **Memory Usage**: The full zoo (10+ models) across many tickers/horizons can be memory-intensive.

---
*Signed: Antigravity - Senior Software Architect*
