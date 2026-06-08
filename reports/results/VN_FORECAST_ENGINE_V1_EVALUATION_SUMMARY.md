# VN Forecast Engine V1 Evaluation Summary

Selection uses validation rows only; final rows are scoring-only.

## Selected Models

|   horizon | task           | model_id                          | selection_split   |
|----------:|:---------------|:----------------------------------|:------------------|
|         5 | direction      | random_forest_classifier          | validation        |
|         5 | return_price   | random_walk_return_baseline       | validation        |
|         5 | range_interval | rolling_volatility_band           | validation        |
|         5 | ranking        | momentum_rank_baseline            | validation        |
|        10 | direction      | hist_gradient_boosting_classifier | validation        |
|        10 | return_price   | random_walk_return_baseline       | validation        |
|        10 | range_interval | rolling_volatility_band           | validation        |
|        10 | ranking        | momentum_rank_baseline            | validation        |
|        20 | direction      | hist_gradient_boosting_classifier | validation        |
|        20 | return_price   | random_walk_return_baseline       | validation        |
|        20 | range_interval | rolling_volatility_band           | validation        |
|        20 | ranking        | momentum_rank_baseline            | validation        |
|        40 | direction      | random_forest_classifier          | validation        |
|        40 | return_price   | random_walk_return_baseline       | validation        |
|        40 | range_interval | rolling_volatility_band           | validation        |
|        40 | ranking        | momentum_rank_baseline            | validation        |
|        60 | direction      | random_forest_classifier          | validation        |
|        60 | return_price   | random_walk_return_baseline       | validation        |
|        60 | range_interval | rolling_volatility_band           | validation        |
|        60 | ranking        | momentum_rank_baseline            | validation        |

- Direction beats strongest simple baseline on any selected validation row: True
- Return/price beats random walk or last close on any selected validation row: False
- Range interval has selected validation coverage between 70%-90%: True
- Ranking has positive selected validation Spearman IC: True
- Results claimable: offline diagnostic evidence only.

## Limitations and Next Steps

- Local cache coverage and timestamp granularity vary by asset.
- Pooled bounded models are research baselines, not deployed predictors.
- No live data fetch, provider call, or production workflow was used.
