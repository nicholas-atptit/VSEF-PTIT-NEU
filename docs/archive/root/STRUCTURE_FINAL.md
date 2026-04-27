# Structure Final - Quant-Core Architecture

## File Structure Overview

```text
./
├── scripts/
│   └── run_quant_core.py      <-- Main Entry Point
├── src/
│   ├── core/
│   │   └── model_governance.py <-- Governance Tables & Roles
│   ├── forecast/
│   │   ├── registry.py        <-- Model Factory & Specs
│   │   └── statistical/       <-- ETS, SARIMAX, Naive
│   ├── risk/
│   │   └── garch.py           <-- Primary Risk Model
│   ├── regime/
│   │   └── markov_switching.py <-- Primary Regime Model
│   ├── evaluation/
│   │   └── quant_core.py      <-- Orchestration Helpers
│   └── reporting/
│   │   ├── analysis_packets.py <-- Data Synthesis
│   │   └── manifests.py       <-- Metadata Writing
├── artifacts/
│   └── quant_core/            <-- Production Outputs (Manifests, Charts, CSVs)
└── docs/
    └── prompt_runs/           <-- Historical Traceability
```

## Folder Responsibilities

- **`src/core/`**: Foundations and governance definitions.
- **`src/forecast/`**: Predictive models. Statistical models live in `statistical/`, ML models in `ml/`.
- **`src/risk/` & `src/regime/`**: Conditioning layers that adjust forecast signals based on market context.
- **`src/reporting/`**: Converges forecasts, risk, and regime into trading-ready "Analysis Packets".

---
*Signed: Antigravity - Senior Software Architect*
