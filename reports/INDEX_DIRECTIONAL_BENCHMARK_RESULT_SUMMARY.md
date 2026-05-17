# Index Directional Benchmark Result Summary

- Created at UTC: 2026-05-17.
- Branch: `research/vn100-evidence-hardening-v1`.
- Output directory: `outputs/index_directional_benchmark/`.
- Scope: supported Vietnamese market indices only.
- Stock benchmark claims: no.
- Daily-to-hourly resampling: no.
- Hourly-to-daily resampling: no.
- Paper or DOCX generated: no.

## Data Scope

- Supported index codes: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100.
- Daily readiness: yes for 5/6 indices.
- Daily indices benchmarked: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30.
- Daily index not benchmarked: VN100, because provider/cache returned only one daily row.
- Earliest daily index date: 2015-01-05.
- Hourly readiness: yes for 6/6 indices.
- Hourly indices benchmarked: VNINDEX, HNXINDEX, UPCOMINDEX, VN30, HNX30, VN100.
- Earliest hourly index timestamp in the benchmark cache: 2022-05-19.
- 2015 hourly index data exists: no.
- Hourly benchmark label: actual available hourly cache window, not 2015 hourly.

## Benchmark Result

- Any exact index/frequency/model/horizon reached 60%: yes.
- Daily pass60 count: 25 exact results.
- Hourly pass60 count: 14 exact results.
- Total pass60 count: 39 exact results.

## Best Daily Index Result

| index_code | frequency | model | horizon | final accuracy | final rows | best baseline | baseline accuracy | delta |
|---|---|---|---:|---:|---:|---|---:|---:|
| UPCOMINDEX | 1D | XGBoost | 1 | 97.60% | 334 | majority_class | 97.60% | +0.00pp |

Interpretation: the best daily result passes 60%, but it does not beat the strongest simple baseline for that exact setting. The high h=1 UPCOMINDEX score is mostly a final-period class-balance result, not standalone evidence of model lift.

## Best Hourly Index Result

| index_code | frequency | model | horizon | final accuracy | final rows | best baseline | baseline accuracy | delta |
|---|---|---|---:|---:|---:|---|---:|---:|
| VNINDEX | 1H | XGBoost | 40 | 66.67% | 159 | majority_class | 63.52% | +3.14pp |

Interpretation: the best hourly result passes 60% on the available hourly cache window. It uses actual hourly rows starting in 2022, not synthetic or resampled 2015 hourly data.

## Strongest Model Lift Over Baseline

| index_code | frequency | model | horizon | final accuracy | baseline accuracy | delta |
|---|---|---|---:|---:|---:|---:|
| UPCOMINDEX | 1D | Random Forest | 40 | 62.03% | 54.58% | +7.46pp |
| HNXINDEX | 1D | Random Forest | 20 | 60.76% | 56.65% | +4.11pp |
| VNINDEX | 1H | XGBoost | 40 | 66.67% | 63.52% | +3.14pp |

## Baseline Comparison

- Majority-class baseline is close to several high-scoring results.
- The best daily accuracy result, UPCOMINDEX 1D h=1, has zero lift over the majority-class baseline.
- The best hourly accuracy result, VNINDEX 1H h=40, is +3.14pp above its strongest simple baseline.
- The strongest daily lift that also passes 60 is UPCOMINDEX 1D Random Forest h=40 at 62.03%, +7.46pp over baseline.

## Claim Boundary

- Index benchmark result may be reported only by exact index, frequency, model, horizon, final accuracy, row count, and baseline comparison.
- Index results cannot support stock benchmark claims.
- Daily index results cannot be presented as hourly results.
- Hourly index results cannot be presented as 2015 hourly results.
- No trading-readiness claim.
- No profitability claim.
- No live-deployment claim.
