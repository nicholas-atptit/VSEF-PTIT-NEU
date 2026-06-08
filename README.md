# VN Forecast Research Lab

VN Forecast Research Lab is an offline historical/local-data-only research
system for VN30 stocks and explicitly configured Vietnamese indices.

Target repository: <https://github.com/nicholas-atptit/VSEF-PTIT-NEU>

## 1. Repository Purpose

The repository supports reproducible research on:

- direction forecasting
- return and price forecasting
- price-range and interval forecasting
- ranking and relative-strength diagnostics
- Quantum Machine Learning (QML) diagnostics
- Model Universe benchmark audits

The repository is not a trading system, investment recommendation engine, live
deployment, or daily T+1 production system. It makes no BUY/SELL,
profitability, portfolio-allocation, or broker-execution claim.

## 2. What This Repo Contains

- Offline forecast-engine components under `src/forecasting/`
- Split, claim-boundary, and artifact governance under `src/governance/`
- Reusable metrics and baselines under `src/evaluation/`
- Point-in-time-safe feature builders under `src/features/`
- Local provider contracts, adapters, and loaders under `src/data/`
- QML diagnostic runner and preserved evidence
- Model Universe V1-V7 audit evidence
- VN30 and configured-index research evidence where locally available
- Active evidence indexes and paper-source materials

The QML paper-package path and project-review master-index package named in
older plans are not present on this branch; see **Known missing or optional
files** below.

## 3. What This Repo Does Not Do

- No live-data fetch by default
- No provider/API pull unless an explicit protocol authorizes it
- No production scheduler, trading dashboard, API service, or broker workflow
- No BUY/SELL signal or investment advice
- No profitability guarantee
- No live prediction ledger
- No daily T+1 production system

## 4. Quick Start

```powershell
git clone https://github.com/nicholas-atptit/VSEF-PTIT-NEU.git
cd VSEF-PTIT-NEU
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

QML-specific optional dependencies are listed in `requirements-qml.txt`.

## 5. Safe Validation

```powershell
python scripts/check_repo_hygiene.py
python scripts/check_runtime_preflight.py
python scripts/check_provider_usage_policy.py
python -m pytest tests/data/test_provider_usage_policy.py -q
python -m pytest tests/data/test_vn_price_gateway_contract.py -q
python -m pytest tests/ml/test_directional_accuracy_metrics.py -q
python -m pytest tests/governance tests/evaluation tests/features tests/forecasting -q
```

These commands validate repository policy and focused contracts. They should not
fetch live data or run new heavy model training.

## 6. How to Rerun Offline Diagnostic Simulations

All commands below use existing local historical/cache data only.

### 6.1 Forecast latest local cache

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --forecast-latest --frequency hourly --horizons 5 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL
```

This uses the latest locally cached timestamp and writes artifacts to
`reports/generated/vn_forecast_engine_v1/`. It does not fetch live data.

### 6.2 Forecast from historical as-of timestamp

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --forecast-asof "2025-01-02 10:00:00" --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL
```

This simulates forecasts from a historical cutoff. Evaluation fields may be
filled when later local rows exist; otherwise actual fields remain null.

### 6.3 Full offline forecast-engine run

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --full-run --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL --timeout-seconds 14400
```

This builds the dataset, evaluates bounded models, selects by validation only,
writes forecast panels, and writes evaluation summaries. It performs no live
data fetch.

### 6.4 Evaluation-only run

```powershell
python scripts/research/run_vn_forecast_engine_v1.py --offline-historical-only --build-evaluate --frequency hourly --horizons 5,10,20,40,60 --index-codes VNINDEX,VN30,HNXINDEX,HNX30,UPCOMINDEX,VNXALL --timeout-seconds 14400
```

This evaluates available local historical data. It does not create a live
prediction or production system.

## 7. Heavy Research Runners

These runners are heavy and should run only under a written protocol:

```powershell
python scripts/research/run_vn30_qml_forecasting.py --help
python scripts/research/run_vn30_model_universe_direction_price_benchmark.py --help
```

Do not run them during ordinary validation, rerun benchmark experiments without
a protocol, or use final-period rows for model selection.

## 8. Main Research Findings

| Track | Existing evidence | Status / boundary |
|---|---|---|
| Classical absolute-direction champion | L2 Logistic, `feature_set_C_closest`, h40, 61.61% final accuracy context | Separate existing benchmark claim; not replaced by QML, Model Universe, or current forecast-engine diagnostics |
| QML V4 | Validation-selected `market_relative_vn30` h40 bounded-sample candidate reached 69.44% final accuracy | Focused QML discovery; `qml_diagnostic_only`; broader QML expansion not justified and later checks weakened generalization |
| QML V8 | Drift-aware kernel-feature architecture reached 64.44% final diagnostic accuracy on `market_relative_vn30` h40 | Strongest QML architecture result; diagnostic-only; `future_blind_required`; does not replace the classical champion |
| BiLSTM | `market_relative_vn30` h40 reached 72.50% raw final accuracy | Demoted and not claimable: class imbalance, 79.69% always-down/majority baseline, weak repaired metrics, and seed instability |
| 74.57% market-relative artifact | Model Universe exploratory final-ranked market-relative row | Class-imbalance/final-ranked artifact, not confirmed model edge, `exploratory_not_claimable` |
| Model Universe V1-V7 | Broad model-family audit and negative evidence | No claimable replacement; useful for robustness, failure analysis, and negative evidence |
| VN30 + index-group price-range lab | Requested result/claim files and generated package are absent on this branch | Do not cite the requested 79.84% coverage or 0.040533 width as repository evidence; current forecast-engine interval evidence is separate and diagnostic-only |
| Ranking / relative strength | Validation diagnostics and positive signals exist | Useful diagnostic; final transfer remains weak; not claimable as an overall directional result |
| Return/price point forecast | Model Universe relock and Forecast Engine v1 did not robustly beat random walk / last close | Weak under current evidence; Forecast Engine v1 selects the random-walk return baseline |

Primary sources include:

- `reports/results/VN30_FULL_MODEL_TUNING_V3_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_RESULT_SUMMARY.md`
- `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`
- `reports/results/VN30_MODEL_UNIVERSE_V1_V6_CLOSEOUT_REPORT.md`
- `reports/results/VN30_MODEL_UNIVERSE_DIRECTION_PRICE_RESULT_SUMMARY.md`
- `reports/results/VN_FORECAST_ENGINE_V1_EVALUATION_SUMMARY.md`

## 9. Output Artifacts

- `reports/generated/vn_forecast_engine_v1/`
- `reports/generated/vn30_qml_forecasting/`
- `reports/generated/vn30_model_universe_direction_price/`
- `reports/generated/vn30_model_universe_benchmark/`
- `reports/results/`
- `reports/claims/`
- `reports/protocols/`
- `reports/paper/`

Treat generated outputs as preserved evidence, not disposable build output.

## 10. Evidence Navigation

Start with:

- `reports/_index/ACTIVE_EVIDENCE_INDEX.md`
- `reports/_index/ACTIVE_CODE_MAP.md`
- `reports/results/VN_FORECAST_ENGINE_V1_EVALUATION_SUMMARY.md`
- `reports/results/VN_FORECAST_ENGINE_V1_LATEST_FORECAST_REPORT.md`
- `reports/claims/VN_FORECAST_ENGINE_V1_CLAIM_BOUNDARY.md`
- `reports/results/VN30_MODEL_UNIVERSE_V1_V6_CLOSEOUT_REPORT.md`
- `reports/results/VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md`

### Known missing or optional files

The following requested navigation/evidence paths are not present on this
branch. Do not invent or cite them as available evidence:

- `reports/project_review/MASTER_EVIDENCE_INDEX.md`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_FULL_PROGRESS_REVIEW.md`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_EVIDENCE_MATRIX.csv`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_CLAIM_STATUS_REGISTER.csv`
- `reports/project_review/VN30_QML_MODEL_UNIVERSE_NEXT_ACTION_ROADMAP.md`
- `reports/paper/qml_kernel_feature_vn30/`
- `reports/generated/vn30_index_group_range_forecast/`
- `reports/results/VN30_INDEX_GROUP_PRICE_RANGE_FORECAST_RESULT_SUMMARY.md`
- `reports/claims/VN30_INDEX_GROUP_PRICE_RANGE_FORECAST_CLAIM_BOUNDARY.md`
- `reports/results/VN30_MODEL_UNIVERSE_V7_EXHAUSTIVE_EXPANSION_RESULT_SUMMARY.md`

## 11. Claim Boundary

- Offline diagnostic research only
- Final rows are scoring-only
- Model selection uses validation only
- No trading, profitability, BUY/SELL, recommendation, or investment advice
- No live deployment, production, or daily T+1 operation
- No VN100 assumption unless explicitly configured and locally available
- No index-as-stock claim
- No champion replacement without comparable future-blind evidence

See `docs/governance/CLAIM_BOUNDARY_POLICY.md`.

## 12. Repository Structure

| Path | Purpose |
|---|---|
| `src/data/` | Local data contracts, adapters, loaders, and validation |
| `src/features/` | Point-in-time-safe reusable feature construction |
| `src/evaluation/` | Metrics, baselines, stability, and evaluation logic |
| `src/forecasting/` | Offline forecast engine, panels, and selectors |
| `src/governance/` | Split, claim-boundary, and artifact policies |
| `scripts/research/` | One-off research and orchestration runners |
| `tests/` | Automated contracts and validation |
| `reports/` | Results, claims, protocols, generated evidence, and paper sources |
| `docs/` | Architecture, usage, workflows, and governance |
| `configs/` | Research, feature, policy, and universe configuration |
| `data/`, `outputs/`, `archive/` | Protected local data and evidence roots |

## 13. Development / Git Policy

- Work on normal branches; do not touch or merge `main`
- Do not create tags by default
- Do not use `git push --mirror`
- Do not generate DOCX during cleanup/documentation work
- Do not fetch live data without an explicit protocol
- Do not delete active evidence, raw/cache data, outputs, or archives
- Commit and push normal branches only

## 14. Known Limitations

- No live forecast or daily T+1 production system
- Return/price point forecasting remains weak
- Ranking final transfer remains weak
- QML remains diagnostic-only
- Model Universe did not replace the existing champion
- Index coverage depends on local cache availability
- `VNXALL` may be skipped when its local cache is missing
- Project-review master-index, dedicated QML paper-package, and index-group
  price-range-lab paths listed above are absent on this branch
