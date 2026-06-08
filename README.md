# VSEF — Vietnam Stock Evaluation and Forecasting Framework

**Research-only** | **Offline historical/local-data-only** | **Diagnostic framework** | **Proprietary / not open source** | **No BUY/SELL / no trading authority**

VSEF is the Vietnam Stock Evaluation and Forecasting Framework: a governed research environment for Vietnamese stock-market forecasting, diagnostic evaluation, and evidence review.

## 1. Executive Summary

VSEF is a proprietary Vietnamese stock-market research and diagnostic framework. It supports structured offline diagnostics across forecasting, Quantum Machine Learning (QML) representation experiments, model-universe benchmarking, price-range and interval evaluation, scenario and risk-governance concepts, and forecast-panel generation.

VSEF is for research and diagnostic review only. It does not issue BUY or SELL recommendations, execute trades, provide production trading authority, or provide investment advice. Current default operation is offline historical/local-data-only.

## 2. Research Attribution

Luong Minh Quan — Posts and Telecommunications Institute of Technology

Contact: [luongminhquan.working.research@gmail.com](mailto:luongminhquan.working.research@gmail.com)

Nguyen Nguyet Ha — National Economics University

Contact: [nghnguyetha.workspace@gmail.com](mailto:nghnguyetha.workspace@gmail.com)

The project is developed in collaboration with, and financially supported by, the Risk Management Department — Viettel Global.

Market data is connected and retrieved through Vnstock, a Python package for Vietnamese stock market analysis.

> Vnstock by thinh-vu on GitHub. Copyright © 2022–2026.

## 3. Proprietary Notice and Access Restrictions

VSEF is proprietary. It is **not open source** and is not released under an open-source license. All source code, documentation, schemas, diagnostic logic, research workflows, generated structures, naming conventions, architecture, governance logic, and related project materials are protected intellectual property unless explicitly stated otherwise in writing.

Access to this repository or any project material does **not** grant a license, ownership, reuse, redistribution, publication, commercial-usage, derivative-work, or model-training right.

Without prior written permission from the project owners, prohibited actions include:

- copying, cloning, or redistributing the framework;
- reusing its code, architecture, diagnostic chain, schemas, or governance logic;
- modifying and republishing it as a derivative system;
- using it for commercial, production, advisory, or trading services;
- extracting materials for external publication, benchmarking, or model training; and
- reverse engineering, reproducing, or imitating its protected design.

All rights are reserved by the project authors and authorized institutional collaborators. Unauthorized use, reproduction, redistribution, or derivative implementation may violate intellectual property rights and may be subject to legal action.

Permission requests, research inquiries, and institutional correspondence:

- Luong Minh Quan: [luongminhquan.working.research@gmail.com](mailto:luongminhquan.working.research@gmail.com)
- Nguyen Nguyet Ha: [nghnguyetha.workspace@gmail.com](mailto:nghnguyetha.workspace@gmail.com)

## 4. What This Repository Is

- A Vietnamese stock-market research framework.
- A VN30 stock-group research environment.
- A configured Vietnamese-index research environment.
- An offline forecast engine for direction, return/price, range/interval, and ranking diagnostics.
- A QML diagnostic research package.
- A Model Universe V1-V7 benchmark-audit workspace. V1-V6 closeout evidence is present; the requested dedicated V7 result summary is not present on this branch.
- An evidence-governance and claim-boundary repository.
- A paper and research-artifact workspace.

## 5. What This Repository Is Not

- Not a trading system, broker workflow, or execution workflow.
- Not an investment-recommendation engine or BUY/SELL signal generator.
- Not a profitability guarantee.
- Not a live deployment or daily T+1 production system.
- Not open source and not licensed for reuse.
- Not an active VN100-scope claim unless explicitly configured and supported by evidence.

## 6. Current Research Evidence Snapshot

Results below retain their exact target and claim boundaries. Raw accuracy across different targets is not directly comparable.

| Track | Scope / Target | Main result | Baseline / limitation | Status | Evidence path |
|---|---|---|---|---|---|
| Classical absolute-direction champion | VN30 hourly, `absolute_direction`, L2 Logistic, `feature_set_C_closest`, h40 | 61.61% final accuracy; +10.90 pp lift; 4,074 rows | Separate existing benchmark context; not replaced by later QML, Model Universe, or forecast-engine diagnostics | Claimable within its exact existing scope | `reports/results/VN30_FULL_MODEL_TUNING_V3_RESULT_SUMMARY.md` |
| QML V4 bounded discovery | Bounded `market_relative_vn30` h40 sample | 69.44% final diagnostic accuracy | Focused bounded discovery; later larger-sample and drift checks weakened broader interpretation | `qml_diagnostic_only`; exploratory beyond exact scope | `reports/results/VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_RESULT_SUMMARY.md` |
| QML V8 drift-aware kernel feature | `market_relative_vn30` h40 | 64.44% final diagnostic accuracy; strongest QML architecture result in current evidence | Different target/scope from classical champion; future-blind confirmation required | Diagnostic-only; `future_blind_required` | `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md` |
| BiLSTM 72.50% | `market_relative_vn30` h40 | 72.50% raw final accuracy | Same-target always-down baseline reached 79.69%; repaired metrics and seed stability were weak | Demoted; not claimable | `reports/results/VN30_MODEL_UNIVERSE_V1_V6_CLOSEOUT_REPORT.md` |
| 74.57% market-relative artifact | Model Universe exploratory `market_relative_vn30` final-ranked row | 74.57% final-ranked artifact | Baseline/class-imbalance and final-selection artifact; not confirmed model edge | `exploratory_not_claimable` | `reports/results/VN30_MODEL_UNIVERSE_DIRECTION_PRICE_RESULT_SUMMARY.md` |
| Model Universe V1-V7 audit | Broad direction and price/return model-universe benchmark | Broad audit and negative evidence; no claimable replacement | Dedicated V7 result summary is not present on this branch; present closeout evidence covers V1-V6 | Negative audit / robustness appendix | `reports/results/VN30_MODEL_UNIVERSE_V1_V6_CLOSEOUT_REPORT.md` |
| VN30 + index-group price-range lab | Requested scope: VNINDEX, VN30, HNXINDEX, HNX30, UPCOMINDEX, VNXALL | Dedicated result, claim, and generated package are not present on this branch | Do not cite requested 79.84% coverage or 0.040533 average width as repository evidence; VNXALL depends on local cache | Not evidenced as a dedicated track on this branch | See `reports/results/` and `reports/claims/`; requested dedicated files are absent |
| Forecast Engine v1 | VN30 stocks and configured Vietnamese indices from local cache | Offline panel engine for direction, return/price, range/interval, and ranking output | Local-cache coverage and timestamp granularity vary by asset | Diagnostic-only | `reports/results/VN_FORECAST_ENGINE_V1_EVALUATION_SUMMARY.md` |
| Return/price point forecasts | Model Universe relock and Forecast Engine v1 | Did not robustly beat random walk / last close in current evidence | Forecast Engine v1 selects the random-walk return baseline | Weak under current setup | `reports/results/VN_FORECAST_ENGINE_V1_EVALUATION_SUMMARY.md` |
| Ranking / relative strength | Forecast Engine v1 validation diagnostics | Positive selected validation Spearman IC | Final transfer remains weak; separate from overall directional accuracy | Useful diagnostic; not claimable as overall edge | `reports/results/VN_FORECAST_ENGINE_V1_EVALUATION_SUMMARY.md` |

Start evidence review at [`reports/_index/ACTIVE_EVIDENCE_INDEX.md`](reports/_index/ACTIVE_EVIDENCE_INDEX.md). The requested `reports/project_review/` master-review package and `reports/paper/qml_kernel_feature_vn30/` package are not present on this branch.

## 7. System and Research Architecture

```text
Local historical/cache data
-> data loaders / provider contracts
-> point-in-time feature builders
-> target builders
-> split policy
-> model/evaluation layer
-> validation-only selector
-> forecast panel generator
-> reports / generated evidence
-> claim boundary
```

| Path | Responsibility |
|---|---|
| `src/data/` | Local provider contracts, adapters, loaders, and validation |
| `src/features/` | Point-in-time-safe feature construction |
| `src/evaluation/` | Metrics, baselines, stability, and evaluation logic |
| `src/forecasting/` | Offline forecast engine, panels, and validation-only selectors |
| `src/governance/` | Split, claim-boundary, and artifact policies |
| `scripts/research/` | Research, audit, and benchmark orchestrators |
| `reports/generated/` | Preserved generated diagnostic evidence |
| `reports/results/` | Result summaries |
| `reports/claims/` | Claim-boundary and claim-register files |
| `reports/project_review/` | Requested project-review group; not present on this branch |
| `reports/paper/` | Paper-source materials and evidence maps where present |

## 8. Authority Boundary

VSEF has diagnostic authority only.

**Allowed outputs**

- forecast, scenario, risk, range/interval, and ranking diagnostics;
- diagnostic candidates;
- `allocation_candidate`, `no_allocation`;
- `route_allocation_candidate`, `hold`, `reject`, `no_candidate`;
- offline forecast panels and validation reports.

**Prohibited outputs**

- BUY or SELL recommendations;
- live execution instructions or broker/execution workflows;
- production trading authority;
- investment advice as actionable instruction;
- learned meta-model trading authority;
- profitability guarantees; and
- portfolio-allocation advice as an actionable recommendation.

The system supports research interpretation, validation, and governance review, not direct trading execution.

## 9. Quick Start

```powershell
git clone https://github.com/nicholas-atptit/VSEF-PTIT-NEU.git
cd VSEF-PTIT-NEU
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Optional QML dependencies are listed in `requirements-qml.txt`.

## 10. Safe Validation Commands

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
python scripts/check_provider_usage_policy.py
python -m pytest tests/data/test_provider_usage_policy.py -q
python -m pytest tests/data/test_vn_price_gateway_contract.py -q
python -m pytest tests/ml/test_directional_accuracy_metrics.py -q
python -m pytest tests/governance tests/evaluation tests/features tests/forecasting -q
```

These are safe validation commands. They should not fetch live data or run heavy model training.

## 11. Offline Forecast Engine Runbook

All commands below read existing local historical/cache data. They do not authorize provider calls, live operation, or BUY/SELL output.

### 11.1 Forecast Latest Local Cache

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --forecast-latest --frequency hourly --horizons 5 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL
```

Reads the latest existing local cache, generates a latest forecast panel, and writes to `reports/generated/vn_forecast_engine_v1/`. It performs no live-data fetch and emits no BUY/SELL output.

### 11.2 Forecast From Historical As-Of Timestamp

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --forecast-asof "2025-01-02 10:00:00" --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL
```

Simulates what the engine would have forecast from a historical timestamp. If actual future data exists locally, scoring fields can be filled; otherwise outputs remain pending/offline.

### 11.3 Full Offline Forecast-Engine Run

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --full-run --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL --timeout-seconds 14400
```

Builds local panel data; evaluates direction, return/price, range, and ranking diagnostics; selects models by validation only; and writes forecast panels and evaluation reports.

### 11.4 Evaluation-Only Run

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --build-evaluate --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL --timeout-seconds 14400
```

Evaluates models on historical local data. It does not create a live system, and final rows are scoring-only.

`VNXALL` may be skipped and recorded in coverage audits when its local cache is missing.

## 12. Heavy Research Runners

These are heavy research runners and should only be run under a written protocol:

```powershell
python scripts/research/run_vn30_qml_forecasting.py --help
python scripts/research/run_vn30_model_universe_direction_price_benchmark.py --help
```

Do not run them during ordinary validation, rerun benchmark experiments without a protocol, or use final-period results for model selection.

## 13. Main Artifact Groups

### Forecast Engine

- `reports/generated/vn_forecast_engine_v1/`
- `reports/results/VN_FORECAST_ENGINE_V1_*`
- `reports/claims/VN_FORECAST_ENGINE_V1_CLAIM_BOUNDARY.md`

### QML Diagnostics

- `reports/generated/vn30_qml_forecasting/`
- `reports/results/VN30_QML*`
- `reports/claims/VN30_QML*`
- `reports/paper/qml_kernel_feature_vn30/` — not present on this branch

### Model Universe

- `reports/generated/vn30_model_universe_direction_price/`
- `reports/results/VN30_MODEL_UNIVERSE*`
- `reports/claims/VN30_MODEL_UNIVERSE*`

### VN30 + Index Group Price-Range Lab

- `reports/generated/vn30_index_group_range_forecast/` — not present on this branch
- `reports/results/VN30_INDEX_GROUP_PRICE_RANGE*` — not present on this branch
- `reports/claims/VN30_INDEX_GROUP_PRICE_RANGE*` — not present on this branch

### Project Review

The following requested project-review files are not present on this branch:

- `reports/project_review/MASTER_EVIDENCE_INDEX.md`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_FULL_PROGRESS_REVIEW.md`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_EVIDENCE_MATRIX.csv`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_CLAIM_STATUS_REGISTER.csv`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_NEXT_ACTION_ROADMAP.md`

## 14. Documentation Map

- [`docs/architecture/REPOSITORY_STRUCTURE.md`](docs/architecture/REPOSITORY_STRUCTURE.md)
- [`docs/usage/RUNBOOK.md`](docs/usage/RUNBOOK.md)
- [`docs/workflows/RESEARCH_WORKFLOW.md`](docs/workflows/RESEARCH_WORKFLOW.md)
- [`docs/governance/CLAIM_BOUNDARY_POLICY.md`](docs/governance/CLAIM_BOUNDARY_POLICY.md)
- [`docs/governance/DATA_AND_ARTIFACT_POLICY.md`](docs/governance/DATA_AND_ARTIFACT_POLICY.md)
- [`reports/_index/ACTIVE_EVIDENCE_INDEX.md`](reports/_index/ACTIVE_EVIDENCE_INDEX.md)
- `reports/project_review/MASTER_EVIDENCE_INDEX.md` — not present on this branch

## 15. Repository Structure

```text
src/        Reusable code: data, features, evaluation, forecasting, governance
scripts/    Research runners, validation scripts, maintenance tools
tests/      Unit, integration, and governance tests
docs/       Architecture, usage, workflow, and governance docs
reports/    Results, claims, protocols, generated evidence, paper materials
configs/    Research, forecasting, data, and validation configuration
data/       Protected local/cache data
outputs/    Protected output artifacts
archive/    Preserved historical snapshots
```

## 16. Development Rules

- Work on branches; do not touch or merge `main`.
- Do not create tags by default or use `git push --mirror`.
- Do not generate DOCX during cleanup or documentation work.
- Do not fetch live data or call provider APIs without a written protocol.
- Do not rerun benchmarks without a written protocol.
- Do not delete active evidence.
- Do not tune or select on final-period rows.
- Do not use BUY/SELL wording or create trading-authority claims.
- Update documentation when schemas or contracts change.

## 17. Known Limitations

- No live production system, daily T+1 system, or broker/execution workflow.
- QML is diagnostic-only.
- Model Universe V1-V7 did not produce a claimable replacement; dedicated V7 result evidence is not present on this branch.
- Return/price point forecasts remain weak under current evidence.
- Ranking final transfer remains weak.
- Market-relative raw accuracy can be misleading because of class imbalance.
- Index coverage depends on local cache; `VNXALL` may be skipped if cache is missing.
- Dedicated project-review, QML kernel-feature paper-package, and index-group price-range-lab paths requested above are not present on this branch.

## 18. Disclaimer

VSEF is a proprietary research and diagnostic framework. Outputs are intended for structured analysis, validation, and governance review. Nothing in this repository constitutes investment advice, financial advice, trading advice, or a recommendation to buy, sell, hold, allocate, or execute any financial instrument. Use is restricted by the proprietary notice and access restrictions above.
