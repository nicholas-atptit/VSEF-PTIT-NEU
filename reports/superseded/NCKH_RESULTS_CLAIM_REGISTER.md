# NCKH Results Claim Register

## A. Safe Claims

| Claim | Evidence source | Paper location | Allowed wording | Forbidden wording |
|---|---|---|---|---|
| The official benchmark uses a 2024-12-31 training-label cutoff and 2025 held-out evaluation. | `run_config.json`; `manifest.json`; `reports/generated/vn100_artifact_date_schema_verification.md` | Chapter 3.4, Figure 2 | The official run enforces `target_timestamp <= train_cutoff` with a 2025 held-out evaluation window. | The model was trained with all data through 2025. |
| The official global benchmark did not pass 60%. | `daily/benchmark_summary.json`; `hourly/benchmark_summary.json`; Table 3 | Chapter 4.1 | The official daily and hourly benchmark summaries both report `passed = false`. | The VN100 benchmark passed the global 60% threshold. |
| The current official run evaluates seven tickers. | `benchmark_summary.json`; `usable_cache_summary.csv`; Table 1 | Chapter 3.1, Chapter 3.2 | The official evaluated tickers are ANV, BCM, BID, BMP, BVH, BWE, and CII. | The current evidence represents the full VN100 universe. |
| Practical trading readiness is not established. | `reports/generated/vn100_cost_slippage_readiness_review.md`; Table 8 | Chapter 4.7, Chapter 5 | The selected VN100 slices do not yet have official cost-adjusted return, slippage, turnover, drawdown, or profit-factor artifacts. | The strategy is ready for live trading or profitable deployment. |

## B. Conditional Claims

| Claim | Evidence source | Paper location | Allowed wording | Forbidden wording |
|---|---|---|---|---|
| A selected confidence-filtered strategy-level diagnostic passes 60%. | `confidence_threshold_sweep_summary.csv`; Table 5; Figure 4 | Chapter 4.3 | Hourly stacking h=1 at threshold 0.57 reaches 60.03% filtered accuracy with 31.30% coverage. | The global benchmark passes 60%. |
| Some model/horizon rows outperform simple directional baselines. | `baseline_delta_summary.csv`; Table 4 | Chapter 4.2 | Some model/horizon slices beat selected simple baselines in directional accuracy. | The models are generally superior across all VN100 conditions. |
| Bear-regime daily h=20 diagnostics exceed 63%. | `daily/regime_accuracy_summary.csv`; Table 6; Figure 5 | Chapter 4.4 | Bear-regime daily h=20 diagnostics exceed 63% accuracy in the official 2025 artifact. | The system has a stable full-market 63% method. |
| Several rows are statistically above a 50% null. | `significance_summary.csv`; Table 7 | Chapter 4.5 | Several model/horizon rows are statistically above a 50% directional null. | Statistical significance proves trading profitability. |
| Selected confidence-slice evidence is concentrated. | `reports/generated/vn100_ticker_concentration_summary.md` | Chapter 4.6 | The selected hourly confidence slice is concentrated in five tickers, with the top three contributing most selected rows. | The selected confidence result is representative of the whole VN100 universe. |

## C. Unsafe Claims

| Claim | Evidence source | Paper location | Allowed wording | Forbidden wording |
|---|---|---|---|---|
| Global benchmark success | Table 3 shows `passed_60pct_global = false` | Avoid | State that the global benchmark did not pass. | The official VN100 benchmark passed the global 60% threshold. |
| Full-market representativeness | Table 1 and cache usability evidence show seven evaluated tickers | Avoid | State that cache coverage limits representativeness. | The result is definitive for all VN100 stocks. |
| Stable 63% method | Table 6 shows regime-specific rows only | Avoid | State that 63%+ evidence is bear-regime diagnostic only. | The model achieves a stable full-market 63% accuracy. |
| Trading profitability | Cost/slippage review shows missing trading-readiness artifacts | Avoid | State that practical trading readiness is not established. | The model is profitable after costs or ready for live deployment. |
| New-model contribution | This phase adds docs, tables, figures, and a report generator only | Avoid | State that no new model family was introduced. | This phase improves model performance or adds a new forecasting architecture. |
