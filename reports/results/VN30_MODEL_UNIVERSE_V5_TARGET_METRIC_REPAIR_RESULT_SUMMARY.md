# VN30 Model Universe V5 Target and Metric Repair Result Summary

## Required Answers

1. Was the BiLSTM 72.50% final result mostly class-imbalance driven: yes. Best BiLSTM repaired row `v4__bilstm_relock__BiLSTM__market_relative_vn30__h40__combined_strategy_features__seed42` has final accuracy 72.50%, final balanced accuracy 47.78%, final macro F1 46.08%, final MCC -0.0603, final predicted-positive ratio 9.69%, and same split majority baseline 79.69%.
2. Does BiLSTM still look strong under balanced accuracy, macro F1, MCC, and lift over strongest baseline: no. Validation balanced-accuracy lift is +2.87 pp, macro-F1 lift is +0.54 pp, and MCC lift is 0.0981.
3. Best direction candidate under baseline-gated repaired metrics: `v4__bilstm_relock__BiLSTM__market_relative_vn30__h40__combined_strategy_features__seed42` with validation balanced accuracy 54.09%, macro F1 50.61%, MCC 0.1265, baseline gate `true`.
4. Least class-imbalance-contaminated target: `absolute_direction` with weighted split imbalance gap 0.0440.
5. Does any candidate beat the 61.61% absolute-direction champion on comparable scope under repaired metrics: no.
6. Does any market-relative candidate beat QML V8 64.44 on comparable scope under repaired metrics: no.
7. Does any price/return model beat random walk / last price robustly: yes. Best repaired price/return row `price__ridge__volatility_adjusted_return_h__h20__relative_strength` has validation improvement +11.76 pp and final improvement +11.76 pp.
8. Which candidates remain future-blind worthy: 6 rows are listed in `v5_future_blind_candidate_registry.csv`.
9. Exact claim boundary: offline diagnostic-only VN30 stock hourly repair audit; no result is claimable now; no trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, index-as-stock, DOCX, tag, merge, push --mirror, main-branch, or champion-replacement claim is made.

## Artifact Index

- `v5_class_balance_audit.csv`
- `v5_metric_repair_results.csv`
- `v5_baseline_gated_leaderboard.csv`
- `v5_bilstm_metric_repair.csv`
- `v5_target_repair_results.csv`
- `v5_price_return_metric_repair.csv`
- `v5_future_blind_candidate_registry.csv`
