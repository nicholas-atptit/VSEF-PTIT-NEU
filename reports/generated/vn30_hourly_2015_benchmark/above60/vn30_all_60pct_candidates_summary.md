# VN30 Hourly 2015 - Above 60% Audit Summary

- Generated at UTC: `2026-05-16T14:53:09+00:00`.
- Source: `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv`.
- Total predictions in benchmark: 77,692.
- Global accuracy: 51.34%.

## Are there any global model/horizon rows above 60%?

**NO.** No global model or model/horizon combination exceeds 60% accuracy.
- Best global candidates: 0 (all are conditional or exploratory).

## Are there any coverage-qualified confidence slices above 60%?

**YES.** lightgbm h=20 at conf>=0.95 achieved 60.32% with coverage 17.59%.

## Are there any ticker-level rows above 60%?

**YES.** 160 ticker-level candidates pass 60%.
- Best: model=lightgbm, horizon=8, ticker=ACB, min_obs>=50 at 61.73% (162 obs).
- These are exploratory/post-hoc and cannot be claimed as global results.

## Are there any regime rows above 60%?

**NO.** No regime-level rows exceed 60%.

## Which results are claim-safe and which are only exploratory?

- **Safe global claims (>60%):** 0.
- **Conditional claims (>60% with coverage disclosure):** 42.
- **Exploratory/post-hoc observations (>60%):** 1528.

### Conditional Claims (require coverage/row disclosure)

| candidate_type | model | horizon | filter_description | observations | coverage_ratio | accuracy | correct_count | incorrect_count | pass_60pct | min_row_floor_met | coverage_floor_met | post_hoc_warning | allowed_claim_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.775, min_rows>=100, cov_floor>=0.1 | 549 | 0.1194 | 0.619308 | 340 | 209 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.775, min_rows>=300, cov_floor>=0.1 | 549 | 0.1194 | 0.619308 | 340 | 209 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.775, min_rows>=500, cov_floor>=0.1 | 549 | 0.1194 | 0.619308 | 340 | 209 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=100, cov_floor>=0.2 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=100, cov_floor>=0.1 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=300, cov_floor>=0.2 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=300, cov_floor>=0.1 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=500, cov_floor>=0.2 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=500, cov_floor>=0.1 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=1000, cov_floor>=0.2 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.7, min_rows>=1000, cov_floor>=0.1 | 1251 | 0.272 | 0.617106 | 772 | 479 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.725, min_rows>=100, cov_floor>=0.2 | 964 | 0.2096 | 0.613071 | 591 | 373 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.725, min_rows>=100, cov_floor>=0.1 | 964 | 0.2096 | 0.613071 | 591 | 373 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.725, min_rows>=300, cov_floor>=0.2 | 964 | 0.2096 | 0.613071 | 591 | 373 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.725, min_rows>=300, cov_floor>=0.1 | 964 | 0.2096 | 0.613071 | 591 | 373 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.725, min_rows>=500, cov_floor>=0.2 | 964 | 0.2096 | 0.613071 | 591 | 373 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | random_forest | 20 | model=random_forest, horizon=20, conf>=0.725, min_rows>=500, cov_floor>=0.1 | 964 | 0.2096 | 0.613071 | 591 | 373 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | xgboost | 20 | model=xgboost, horizon=20, conf>=0.875, min_rows>=100, cov_floor>=0.2 | 1060 | 0.2305 | 0.610377 | 647 | 413 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | xgboost | 20 | model=xgboost, horizon=20, conf>=0.875, min_rows>=100, cov_floor>=0.1 | 1060 | 0.2305 | 0.610377 | 647 | 413 | yes | yes | yes | yes | conditional |
| model_horizon_confidence | xgboost | 20 | model=xgboost, horizon=20, conf>=0.875, min_rows>=300, cov_floor>=0.2 | 1060 | 0.2305 | 0.610377 | 647 | 413 | yes | yes | yes | yes | conditional |
| ... |  |  |  |  |  |  |  |  |  |  |  |  |  |

### Exploratory/Post-Hoc Observations

| candidate_type | model | horizon | filter_description | observations | coverage_ratio | accuracy | correct_count | incorrect_count | pass_60pct | min_row_floor_met | coverage_floor_met | post_hoc_warning | allowed_claim_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=100, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=100, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=100, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=100, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=100, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=300, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=300, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=300, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=300, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | xgboost | 4 | model=xgboost, horizon=4, conf>=0.95, min_rows>=300, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| ... |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Top 20 Passing Candidates Overall

| candidate_type | model | horizon | filter_description | observations | coverage_ratio | accuracy | correct_count | incorrect_count | pass_60pct | min_row_floor_met | coverage_floor_met | post_hoc_warning | allowed_claim_level |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=100, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=300, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=500, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.5 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.4 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.3 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.2 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| model_horizon_confidence | random_forest | 4 | model=random_forest, horizon=4, conf>=0.875, min_rows>=1000, cov_floor>=0.1 | 1 | 0.0002 | 1.0 | 1 | 0 | yes | no | no | yes | exploratory |
| ... |  |  |  |  |  |  |  |  |  |  |  |  |  |

## Boundary

- No trading-readiness, profitability, or live deployment claim is made.
- All confidence-filtered and combined results are post-hoc diagnostics.
- Global claims require all 30 active tickers and full evaluation coverage.
- No new data was fetched. No prediction labels were edited.
