# VN30 Full Model Tuning V3 Protocol

## Scope

- Final evaluation target: VN30 stock hourly directional benchmark only.
- Index data usage: lagged market-state/context features only.
- Index benchmark performance is not claimed as stock benchmark performance.
- Target variants are evaluated and reported separately; target claims are not mixed.
- Out of scope: trading, profitability, BUY/SELL, investment recommendation, live deployment, DOCX, paper generation, git tags.

## Split Discipline

- Train rows require feature_timestamp <= `2023-12-31 23:59:59` and target_timestamp <= `2023-12-31 23:59:59`.
- Validation rows require feature_timestamp and target_timestamp from `2024-01-01 00:00:00` through `2024-12-31 23:59:59`.
- Final rows require feature_timestamp and target_timestamp >= `2025-01-01 00:00:00`.
- Candidate, model, target, feature, and threshold selection for claimable rows uses validation only.
- Final-ranked rows are exploratory only and require re-lock or future-blind confirmation.

## Search Design

- Target variants: absolute_direction, market_relative_vn30, market_relative_vnindex, excess_return_direction, thresholded_direction, neutral_removed_direction.
- Horizons: [20, 30, 35, 40, 45, 50, 55, 60].
- Thresholds: 0.400 to 0.650 step 0.005.
- Feature groups: feature_set_C_closest, compact_stable_features, feature_set_C_closest_plus_index_context, feature_set_C_closest_plus_relative_strength, feature_set_C_closest_plus_volatility_regime, feature_set_C_closest_plus_volume_shock, compact_stable_plus_index_context, market_context_features, low_noise_features, all_stable_features, regime_interaction_features.
- Model families: logistic_regression, elasticnet_logistic, calibrated_logistic, random_forest, extra_trees, xgboost, lightgbm, hist_gradient_boosting, soft-vote ensemble, regime-gated ensemble, historical replay rows.
- Full grid is represented in `candidate_grid.csv` as model-family parameter specs with theoretical expanded counts; fitted screening candidates are also listed in the same file.

## Validation Composite Score

score = 0.30 * validation_lift + 0.25 * validation_accuracy + 0.15 * quarterly_stability + 0.10 * ticker_stability + 0.10 * prediction_balance + 0.05 * row_count_score + 0.05 * simplicity_score.
