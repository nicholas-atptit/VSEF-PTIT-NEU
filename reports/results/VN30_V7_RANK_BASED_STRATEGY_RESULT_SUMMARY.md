# VN30 V7 Rank Based Strategy Result Summary

## Locked Validation Strategy

- Selection status: validation_selected_rank_strategy.
- Strategy: `v7rank_00681` / top_quantile_rotation.
- Validation return/Sharpe/drawdown: 11.45% / 0.36917250419508074 / -62.69%.
- Validation trades/exposure/turnover: 241 / 0.6923180726621808 / 36.099614505297374.
- Validation strongest baseline/lift: always_up_directional_baseline / -19.54 pp.

## Final Locked Result

- Total return after cost: 22.35%.
- Sharpe: 0.708986604931575.
- Max drawdown: -61.26%.
- Trade count/exposure/turnover: 53 / 0.4677482787741967 / 8.380902607513093.
- Strongest final baseline: buy_and_hold_vn30_index; baseline delta: -31.10 pp.

## Required Answers

1. Score-rank correlation verdict: validation_non_positive_final_weak_positive; spearman rows=[{'split': 'validation', 'metric': 'spearman_rank_vs_realized_return', 'value': -0.0134232483201921, 'rows': 28830}, {'split': 'final', 'metric': 'spearman_rank_vs_realized_return', 'value': 0.0368721321719448, 'rows': 4074}].
2. Best score deciles: [{'split': 'final', 'score_decile': 9, 'decile_label': '9', 'rows': 481, 'avg_model_score': 0.6082732444083438, 'avg_realized_return': 0.0636138697769263, 'median_realized_return': 0.0116279069767442, 'hit_rate': 0.5343035343035343, 'prediction_up_ratio': 1.0}, {'split': 'validation', 'score_decile': 3, 'decile_label': '3', 'rows': 2883, 'avg_model_score': 0.3076041875092229, 'avg_realized_return': 0.0082162288120525, 'median_realized_return': 0.0030774491366045, 'hit_rate': 0.5209850849809227, 'prediction_up_ratio': 0.0}].
3. Validation winner: [{'rank_strategy_id': 'v7rank_00681', 'model_family': 'calibrated_logistic', 'target_variant': 'absolute_direction', 'feature_group': 'compact_stable_features', 'horizon': 40, 'strategy_template': 'top_quantile_rotation', 'top_n': nan, 'top_quantile': 0.1, 'w_model': nan, 'w_rs': nan, 'score_spread_threshold': nan, 'market_regime_filter': 'off', 'max_positions': 10, 'max_exposure': 0.7, 'cost_bps': 0, 'slippage_bps': 0, 'uses_absolute_probability_threshold': False, 'strategy_family_key': 'top_quantile_rotation__n__q0p1__wm__wrs__spread__mp10__ex0p7', 'split': 'validation', 'total_return': 0.1145223414565361, 'annualized_return': 0.1148543774724319, 'sharpe': 0.3691725041950807, 'sortino': 0.1181844863343562, 'max_drawdown': -0.6269293057671788, 'calmar': 0.1832014812768781, 'win_rate': 0.4854771784232365, 'profit_factor': 1.4420520765194926, 'average_trade_return': 0.0067684005685703, 'trade_count': 241, 'exposure_ratio': 0.6923180726621808, 'turnover': 36.099614505297374, 'cost_adjusted_return': 0.1145223414565361, 'claim_label': 'rank_strategy_candidate', 'reason_not_claimable': 'validation rank-strategy diagnostic', 'strongest_baseline': 'always_up_directional_baseline', 'strongest_baseline_return': 0.3099711619191527, 'baseline_delta': -0.1954488204626165, 'drawdown_score': 0.3730706942328212, 'trade_count_score': 1.0, 'turnover_score': 0.0269544579730663, 'validation_score': 0.254640832449915, 'passes_trade_count': True, 'passes_exposure': True, 'passes_positive_after_cost': True, 'passes_drawdown_reported': True, 'passes_baseline_preferred': False, 'validation_selected_eligible': True}].
4. Locked validation-selected rank strategy generates final trades: true.
5. Final total return after cost: 22.35%.
6. Final Sharpe: 0.708986604931575.
7. Final max drawdown: -61.26%.
8. Final trade count and exposure: 53 / 0.4677482787741967.
9. Beats buy-and-hold VN30: false (baseline=53.45%).
10. Beats equal-weight VN30: true (baseline=18.39%).
11. Beats random same-turnover: false (baseline=31.90%).
12. Rank-based selection solves fixed-threshold validation infeasibility: true.
13. Claimable results: none as a strategy claim; locked row is a validation-governed offline rank_strategy_candidate only.
14. Exploratory only: all final-ranked leaderboard rows and any final-period strategy comparison.
15. Paper-safe wording: offline diagnostic only; no BUY/SELL, profitability, investment advice, live deployment, or claimable strategy claim.

Paper-safe wording:

> VN30 V7 replaced fixed absolute probability thresholds with validation-governed rank-based strategy rules for the frozen calibrated-logistic compact-stable h40 score family. Rank selection removes the V5 validation infeasibility caused by the 0.54-0.56 threshold band. The locked strategy is selected on validation only and final results are offline diagnostics; future-blind confirmation is required before stronger strategy claims.
