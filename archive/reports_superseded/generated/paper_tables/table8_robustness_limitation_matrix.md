# Table 8: Robustness and Limitation Matrix

| risk_or_limitation | current_evidence | paper_handling | remaining_work |
| --- | --- | --- | --- |
| Global benchmark threshold | daily passed=false; hourly passed=false | State that the global 60% benchmark did not pass. | Repeat after broader coverage and additional windows. |
| Usable ticker coverage | Official summaries evaluate ANV, BCM, BID, BMP, BVH, BWE, CII. | Frame as limited-cache VN100 evidence, not a full-market conclusion. | Expand benchmark-usable VN100 cache coverage. |
| Confidence sweep coverage | 31 combined sweep rows; 0 daily sweep data rows. | Mark confidence table/figure as partial. | Generate full daily and all-model threshold sweeps. |
| Selected-slice concentration | Selected hourly confidence slice has five tickers and top-three prediction share near 79.49%. | Treat selected pass as narrow strategy-level diagnostic. | Re-test after ticker coverage broadens. |
| Regime-specific post-hoc risk | Daily bear-regime h=20 rows exceed 63%, but are regime-specific diagnostics. | Do not describe as a stable full-market 63% method. | Define ex-ante regime rules and validate across windows. |
| Trading readiness | Official selected slices have no cost-adjusted return, slippage, turnover, drawdown, or profit-factor artifacts. | State practical trading readiness is not established. | Run cost/slippage-aware backtests with trade and portfolio metrics. |

## Note

- Source artifact: artifact verification, concentration, coverage, cost/slippage readiness, model/source health artifacts.
- Claim supported: The main limitations are limited coverage, selected-slice concentration, partial sweep evidence, and missing trading-cost validation.
- Limitation: The matrix records current evidence gaps; it is not a new experiment.
- Status: ready.
