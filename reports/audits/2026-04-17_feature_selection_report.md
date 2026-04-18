# Feature Selection Report

## Method
- Selection scope: `walk_forward_train_only`
- Fold count: `3`
- Duplicate key count in analysis frame: `0`
- Future-join leakage count in analysis frame: `0`

## Final Task Feature Sets
- `regression_forecasting`: ema_50, range_20, breadth_member_count, close_to_sma_200, turnover_ma_60, pct_above_ma20, rolling_max_60, dist_ma_60, rolling_volatility_60, close_return_10d, macd_signal, bb_width, dist_ma_20, market_return_60d
- `directional_classification`: m_ret_20d, breadth_thrust_10, market_return_60d, new_high_low_spread_5, rolling_min_5, pct_above_ma20, pct_above_ma50, up_volume, m_ret_5d, declining_share, down_volume, up_down_volume_ratio_5
- `regime_detection`: market_return_20d, sector_return_60d, pct_above_ma50, new_high_low_ratio, market_return_60d, pct_above_ma20, sector_return_20d, new_high_low_spread_5, m_ret_5d, new_high_low_spread
- `risk_layer`: rolling_volatility_60, bb_width, dist_ma_60, close_mean_10, sma_200, volume_std_60, volume_ma_20, turnover_ma_60, range_20, volume_max_20

## Demotion Candidates
- `relative_strength_sector_20` (regression_forecasting): demote because weak_low_stability
- `log_return` (regression_forecasting): demote because weak_low_stability
- `adx_14` (regression_forecasting): demote because weak_low_stability
- `close_std_5` (regression_forecasting): demote because weak_low_stability
- `close_std_10` (regression_forecasting): demote because weak_low_stability
- `up_down_volume_ratio_5` (regression_forecasting): demote because weak_low_stability
- `high_low_range_pct` (regression_forecasting): demote because weak_low_stability
- `breadth_thrust_5` (regression_forecasting): demote because weak_low_stability
- `relative_strength_sector_60` (regression_forecasting): demote because weak_low_stability
- `rsi_14_lag_5` (regression_forecasting): demote because weak_low_stability
- `volume_min_60` (regression_forecasting): demote because weak_low_stability
- `rolling_corr_market_20` (regression_forecasting): demote because weak_low_stability
- `breadth_momentum_5` (regression_forecasting): demote because weak_low_stability
- `stoch_d_14` (regression_forecasting): demote because weak_low_stability
- `momentum_20` (regression_forecasting): demote because weak_low_stability
- `rsi_14_lag_2` (regression_forecasting): demote because weak_low_stability
- `vroc_20` (regression_forecasting): demote because weak_low_stability
- `abnormal_gap_sigma_20` (regression_forecasting): demote because weak_low_stability
- `up_volume` (regression_forecasting): demote because weak_low_stability
- `pct_return_lag_1` (regression_forecasting): demote because weak_low_stability