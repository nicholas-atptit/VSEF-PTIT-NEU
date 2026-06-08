# NCKH Results Claim Register V2

## Purpose

This V2 claim register updates the original claim governance after the evidence-gap-closure phase. It preserves the original claim boundaries while allowing carefully labeled derived-diagnostic claims.

## A. Safe Claims Unchanged

| Claim | Evidence source | Allowed wording | Forbidden wording | Abstract? | Placement |
|---|---|---|---|---|---|
| The official benchmark uses a 2024-12-31 training-label cutoff and 2025 held-out evaluation. | `run_config.json`; `manifest.json`; `reports/generated/vn100_artifact_date_schema_verification.md` | The official run enforces `target_timestamp <= train_cutoff` with a 2025 held-out evaluation window. | The model was trained with all data through 2025. | Yes | Chapter 3 and abstract if concise. |
| The official global benchmark did not pass 60%. | `daily/benchmark_summary.json`; `hourly/benchmark_summary.json` | The official daily and hourly benchmark summaries both report `passed = false`. | The VN100 benchmark passed the global 60% threshold. | Yes | Chapter 4 and conclusion. |
| The official evaluated set contains seven tickers. | `benchmark_summary.json`; `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` | The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII. | The evidence represents the full VN100 universe. | Yes, as limitation | Chapter 3, Chapter 4 limitations, conclusion. |
| Practical trading readiness is not established. | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md`; `reports/generated/vn100_cost_slippage_readiness_review.md` | Cost/slippage proxy diagnostics exist, but executable trading readiness is not established. | The strategy is ready for live trading or profitable deployment. | Yes, as limitation | Chapter 4 limitations, appendix, conclusion. |

## B. New Safe Derived-Diagnostic Claims

| Claim | Evidence source | Allowed wording | Forbidden wording | Abstract? | Placement |
|---|---|---|---|---|---|
| The v2 confidence sweep covers all available prediction-row combinations. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md` | A derived v2 confidence sweep covers 32 available frequency/model/horizon combinations from 154,048 official prediction rows. | The official benchmark was rerun across all combinations. | Conditional yes | Chapter 4; mention in abstract only as derived diagnostics. |
| The best >=50% coverage derived confidence candidate is daily XGBoost h=20 threshold 0.86. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_summary.csv` | In the derived v2 sweep, daily XGBoost h=20 at threshold 0.86 reaches 60.55% accuracy with 53.02% coverage. | The global VN100 benchmark now passes 60%. | Conditional yes | Chapter 4 only unless supervisor approves abstract wording. |
| Lagged ex-ante bear-regime diagnostics support the 2025 daily h=20 result. | `reports/generated/evidence_gap_closure/vn100_exante_regime_validation_report.md` | Lagged ex-ante bear-regime diagnostics reach 66.34% for daily LightGBM h=20 and 65.05% for daily XGBoost h=20 over 309 rows. | The model has a stable full-market 63% method. | Conditional yes | Chapter 4; maybe conclusion as cautious support. |
| Cost/slippage proxy artifacts now exist. | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md` | The evidence package now includes cost/slippage proxy diagnostics, including turnover, drawdown, profit factor, trade count, exposure, and equity curve. | The trading system is profitable after realistic costs. | No | Appendix or Chapter 4 limitations only. |

## C. Conditional Upgraded Claims

| Claim | Evidence source | Allowed wording | Forbidden wording | Abstract? | Placement |
|---|---|---|---|---|---|
| Confidence filtering evidence is stronger than in submission package v1. | `reports/superseded/NCKH_EVIDENCE_UPGRADE_ADDENDUM_V2.md`; `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md` | Derived v2 diagnostics broaden confidence-filter evidence beyond the original hourly stacking h=1 selected slice. | Confidence filtering proves robust profitability or global superiority. | Maybe, only with "derived diagnostics" qualifier | Chapter 4. |
| Bear-regime evidence is less purely post-hoc after ex-ante proxy testing. | `reports/generated/evidence_gap_closure/vn100_exante_regime_validation_report.md` | A lagged ex-ante proxy supports the 2025 daily bear-regime h=20 diagnostic. | The regime rule is validated for deployment. | Maybe, if paired with limitation | Chapter 4. |
| Cost/slippage evidence moves from missing to preliminary proxy. | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md` | The project now has preliminary cost/slippage proxy diagnostics, but they are not execution-ready backtests. | Trading readiness is established. | No | Appendix or limitations. |
| Some v2 selected candidates are less concentrated than the original hourly slice. | `reports/generated/evidence_gap_closure/vn100_full_confidence_sweep_report.md`; `reports/generated/vn100_ticker_concentration_summary.md` | Higher-coverage daily XGBoost h=20 v2 candidates show lower prediction-count concentration than the original selected hourly confidence slice. | The results are representative of the full VN100 universe. | No | Chapter 4 concentration discussion. |

## D. Still Unsafe Claims

| Claim | Evidence source | Allowed wording | Forbidden wording | Abstract? | Placement |
|---|---|---|---|---|---|
| Global benchmark success | `daily/benchmark_summary.json`; `hourly/benchmark_summary.json` | The official global benchmark did not pass 60%. | The official VN100 benchmark passed the global 60% threshold. | No | Avoid except as a denied claim. |
| Full-market VN100 representativeness | `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.md` | Cache coverage limits representativeness; only seven tickers are benchmark-usable for 2025. | The evidence is definitive for all VN100 stocks. | No | Avoid except as limitation. |
| Stable multi-window 63% method | `reports/generated/evidence_gap_closure/vn100_multiwindow_validation_report.md` | Multi-window stability is not established because 2022-2024 official artifacts are unavailable. | The method is stable across years at 63% accuracy. | No | Avoid except as limitation. |
| Trading readiness or profitability | `reports/generated/evidence_gap_closure/vn100_cost_slippage_validation_report.md` | Cost/slippage proxy diagnostics are preliminary and not execution-ready. | The model is profitable after costs or ready for live deployment. | No | Avoid except as limitation. |
| New model contribution | Evidence-upgrade scripts and reports | No new model family was introduced in this phase. | The evidence upgrade improves model architecture or adds a new forecasting family. | No | Methods limitation or appendix note. |

## Abstract Guidance

The abstract may mention V2 evidence only if it uses cautious wording:

> Derived post-benchmark diagnostics broaden confidence-filter evidence and support a lagged ex-ante bear-regime result in the 2025 window, while ticker coverage and multi-window validation remain limited.

The abstract must not mention trading readiness, profitability, full-market representativeness, or stable multi-window 63% performance.
