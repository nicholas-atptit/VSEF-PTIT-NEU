# Index Directional Benchmark Claim Register

- Created at UTC: 2026-05-17.
- Scope: supported index-only directional benchmark.
- Evidence: `outputs/index_directional_benchmark/` and `reports/generated/index_benchmark/`.

## Safe Claims

| Claim | Status | Evidence |
|---|---|---|
| Supported daily index benchmark ran for VNINDEX, HNXINDEX, UPCOMINDEX, VN30, and HNX30. | Safe | `reports/generated/index_benchmark/index_readiness.md`, `outputs/index_directional_benchmark/accuracy_summary.csv` |
| Supported hourly index benchmark ran for VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, and VN100. | Safe | `reports/generated/index_benchmark/index_readiness.md`, `outputs/index_directional_benchmark/accuracy_summary.csv` |
| Daily index benchmark can be described as 2015-start daily where cache/provider supports it. | Safe | earliest daily date is 2015-01-05 for usable daily indices |
| Hourly index benchmark uses actual available hourly cache windows only. | Safe | earliest hourly benchmark cache timestamp is 2022-05-19 |
| At least one exact index/frequency/model/horizon passed 60% final directional accuracy. | Safe | 39 exact results passed 60 |
| Best daily final accuracy was UPCOMINDEX 1D XGBoost h=1 at 97.60%. | Safe with baseline caveat | majority-class baseline was also 97.60% |
| Best hourly final accuracy was VNINDEX 1H XGBoost h=40 at 66.67%. | Safe with coverage caveat | final rows 159; majority-class baseline 63.52% |
| Strongest passing model lift over baseline was UPCOMINDEX 1D Random Forest h=40 at 62.03%, +7.46pp over baseline. | Safe | `outputs/index_directional_benchmark/accuracy_summary.csv` |

## Unsafe Claims

| Claim | Reason |
|---|---|
| Using an index result as a stock benchmark result. | Index and stock benchmarks are separate tracks. |
| Using a stock result as an index benchmark result. | Index-only benchmark uses index OHLCV only. |
| Saying hourly index data starts in 2015. | Hourly cache starts in 2022/2023 depending on index. |
| Presenting daily index results as hourly results. | Frequencies are separate and not resampled. |
| Presenting hourly index results as daily results. | Frequencies are separate and not resampled. |
| Hiding frequency, coverage, horizon, or baseline comparison. | These fields materially affect interpretation. |
| Claiming trading readiness. | No execution, cost, slippage, or portfolio diagnostics were run. |
| Claiming profitability. | Directional classification accuracy is not a profitability result. |
| Claiming live deployment readiness. | This is an offline benchmark only. |

## Final Boundary

- Daily-to-hourly resampling used: no.
- Hourly-to-daily resampling used: no.
- Stock claims made from index data: no.
- Trading/profitability/live-deployment claim: no.
- Paper generated: no.
- DOCX generated: no.
