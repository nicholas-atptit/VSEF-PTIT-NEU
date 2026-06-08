# VN30 Hourly 2015 Benchmark Claim Register

Scope: VN30 hourly 2015 Jan-2025-universe benchmark and post-benchmark diagnostics. This register is a claim boundary for reports and paper drafting.

## Safe Claims

| Claim | Status | Evidence |
| --- | --- | --- |
| Benchmark uses VN30 January 2025 universe. | safe | Frozen active universe source is VN30 January 2025 review. |
| 30/30 active tickers represented. | safe | Benchmark audit passed and all active tickers appear in predictions. |
| Global accuracy is 51.34%. | safe | Base benchmark summary reports 77,692 predictions and 51.34% global directional accuracy. |
| Global 60% threshold not passed. | safe | Base benchmark `passed=false`; best model/horizon is below 60%. |
| Benchmark is hourly only. | safe | Audit passed frequency check; no daily or resampled markers. |
| No data fetch was run for post-benchmark diagnostics. | safe | Diagnostics read existing prediction artifact only. |

## Conditional Claims

| Claim | Status | Required Boundary |
| --- | --- | --- |
| Some model/horizon rows may outperform baselines. | conditional | Must cite model, horizon, baseline, observation count, and delta; this is not a trading claim. |
| Some horizon 20 model rows are statistically above 50%. | conditional | Must state multiple-testing limitation; only 3 rows remain significant under Bonferroni adjustment. |
| Some confidence-filtered slices may approach or pass 60%. | conditional | Must state coverage, threshold, post-hoc nature, and whether coverage floor is met. Current sweep found no coverage-ok slice at or above 60%. |
| Regime-sliced diagnostics may show stronger/weaker rows. | conditional | Must state labels are existing benchmark-internal regime proxies, not independently validated market regimes. |
| Cost/slippage proxy rows may be positive under simplified assumptions. | conditional | Must state proxy is non-executable and not a profitability, execution, or live deployment claim. |

## Unsafe Claims

| Claim | Status | Reason |
| --- | --- | --- |
| Trading readiness. | unsafe | No execution system, liquidity validation, sizing policy, or live robustness evidence. |
| Profitability. | unsafe | Cost/slippage diagnostics are proxy-only and overlap horizon returns. |
| Full-market investment suitability. | unsafe | Scope is VN30 frozen universe and hourly benchmark diagnostics only. |
| Stable 60%+ method. | unsafe | Global benchmark is 51.34%; best base model/horizon is 54.58%; confidence sweep is post-hoc. |
| Live deployment. | unsafe | No live data, monitoring, execution, risk controls, or production validation. |
| Paper/DOCX generated. | unsafe | No paper or DOCX has been generated in this phase. |

## Paper Boundary Recommendation

Use conservative language: "The VN30 hourly benchmark did not pass the 60% global threshold. Horizon 20 tree-based models show the strongest diagnostic signal, but the result is not sufficient for trading readiness or profitability claims."
