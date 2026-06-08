# VN30 V7 Rank Based Strategy Protocol

## Scope

- VN30 stock hourly strategy diagnostics only.
- Frozen score family: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Model probabilities are ranking scores only; fixed absolute probability thresholds are not the main selector.
- Validation-only selection governs the locked rank strategy.
- Final-ranked leaderboards are exploratory_not_claimable.
- No broad model tuning, final-performance claimable selection, DOCX, push, merge, tag, VN100, or index-as-stock claim.

## Strategy Grid

- top_n_rotation: N=[1, 3, 5, 10].
- top_quantile_rotation: quantiles=[0.05, 0.1, 0.2, 0.3].
- score_rank_plus_relative_strength: weights=[(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5)]; N=3,5,10.
- market_regime_rank_filter: top-N when risk-on/neutral.
- score_spread_filter: spread thresholds=[0.0, 0.005, 0.01, 0.02, 0.03].
- cost_bps=[0, 5, 10, 20, 30]; slippage_bps=[0, 5, 10, 20]; max_exposure=[0.5, 0.7, 1.0].

## Validation Score

0.30 * validation_total_return + 0.25 * validation_sharpe + 0.15 * max_drawdown_score + 0.15 * baseline_delta + 0.10 * trade_count_score + 0.05 * turnover_score.
