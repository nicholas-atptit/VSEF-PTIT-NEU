# NCKH Full Paper Draft: VN30 Hourly Available-Window V1 With Figures

## Abstract

This paper reports a VN30 hourly available-window forecasting study using real local hourly data only. The study is an hourly available-window VN30 subset analysis rather than a full-constituent VN30 historical evaluation. The design does not claim to satisfy the 2005-2026 full-history requirement, does not use daily evidence, and does not reuse old VN100 evidence.

## 1. Scope and Claim Boundary

- Study label: VN30 hourly available-window.
- Selected tickers: 27/30.
- Selected ticker list: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, LPB, MBB, MSN, MWG, PLX, SAB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIC, VJC, VNM, VPB, VPL, VRE.
- Excluded ticker list: HPG, SHB, VIB.
- Full VN30 representativeness: false.
- Trading readiness: not claimed.
- Full-history 2005-2026 VN30 evidence: not claimed.

## 2. Methods

The study is an hourly available-window VN30 subset analysis rather than a full-constituent VN30 historical evaluation.

The benchmark uses hourly OHLCV rows from the local hourly cache. The target is directional classification. Training labels obey the leakage boundary `target_timestamp <= train_cutoff`.

- Actual training period: 2025-12-15 09:00:00 to 2026-03-13 11:00:00.
- Actual evaluation period: 2026-03-13 13:00:00 to 2026-05-11 14:00:00.
- Frequency: hourly only.

## 3. Data

The audit found that a full 2005-2026 hourly VN30 design is not supported by local data. The available-window design was selected from the local hourly coverage audit.

See `reports/generated/vn30_hourly_available_window/audit/vn30_hourly_available_window_audit.md` for per-ticker coverage.

## 4. Benchmark Results

- Prediction rows: 75756.
- Overall accuracy: 0.5118670468345742.
- Benchmark status: completed.
- Models executed: lightgbm, random_forest, stacking, xgboost.

Detailed benchmark tables are in `reports/generated/vn30_hourly_available_window/paper_tables/`.

## 5. Diagnostics

- Confidence-filtered diagnostics are included as diagnostic evidence only.
- Ex-ante regime labels use prior information only.
- Cost/slippage outputs are proxy diagnostics and do not establish live execution readiness.

## 6. Limitations

- External hourly data is still required for the original 2005-2026 full design.
- The selected hourly window is short relative to a full historical study.
- VN30 representativeness is not claimed unless the selected design includes all 30 tickers.
- The paper does not use old VN100 evidence.
- The paper does not use daily evidence.
- The paper does not fabricate data.

## Figures

![Figure 1: Research pipeline](generated/vn30_hourly_available_window/paper_figures/figure1_research_pipeline.png)

![Figure 2: Available-window walk-forward design](generated/vn30_hourly_available_window/paper_figures/figure2_available_window_walk_forward_design.png)

![Figure 3: Hourly accuracy by model/horizon](generated/vn30_hourly_available_window/paper_figures/figure3_hourly_accuracy_by_model_horizon.png)

![Figure 4: Confidence threshold vs coverage/accuracy](generated/vn30_hourly_available_window/paper_figures/figure4_confidence_threshold_vs_coverage_accuracy.png)

![Figure 5: Regime-specific accuracy](generated/vn30_hourly_available_window/paper_figures/figure5_regime_specific_accuracy.png)

![Figure 6: Ex-ante regime accuracy](generated/vn30_hourly_available_window/paper_figures/figure6_exante_regime_accuracy.png)

![Figure 7: Cost/slippage equity curve](generated/vn30_hourly_available_window/paper_figures/figure7_cost_slippage_equity_curve.png)
