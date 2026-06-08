# VN30 Model Universe Direction + Price Forecasting Result Summary

## Required Answers

1. Which direction target performed best: `absolute_direction` h10 with `naive_bayes` on `relative_strength`.
2. Which price/return target performed best: `forward_log_return_h` h20 with `xgboost_regressor` on `relative_strength`.
3. Which model family performed best for direction: `naive_bayes` with validation accuracy 66.00% and final accuracy 39.86%.
4. Which model family performed best for price/return: `xgboost_regressor` with validation RMSE 0.0245612 and final RMSE 0.130293.
5. Did any model beat the 61.61% absolute-direction classical champion on comparable absolute-direction scope: exploratory final-ranked rows did (true, best comparable final accuracy 69.86%), but no claimable replacement is made because final-ranked rows are `exploratory_not_claimable`.
6. Did any model beat the 64.44% QML V8 market-relative result on comparable market_relative_vn30 scope: exploratory final-ranked rows did (true, best comparable final accuracy 74.57%), but no claimable replacement is made because final-ranked rows are `exploratory_not_claimable`.
7. Did any model forecast price/return better than random walk / last price baseline: validation-screening yes (true, locked validation error improvement +12.34 pp); final transfer is reported separately and is not a trading or production claim.
8. Which models failed or were skipped: 1177 candidate/model rows were skipped; see `skipped_models.csv`.
9. Is there evidence that stock direction can be forecast: diagnostic evidence exists when validation-selected direction candidates beat simple validation baselines, but this is not a trading claim.
10. Is there evidence that stock price/return can be forecast: diagnostic evidence exists if validation RMSE improves over random-walk/last-price or historical-return baselines; this is reported separately from direction accuracy.
11. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, VN100, index-as-stock, merge, tag, DOCX, or production claim is made.

## Locked Direction Candidate

- Candidate: `direction__naive_bayes__absolute_direction__h10__relative_strength`.
- Validation accuracy: 66.00%.
- Final accuracy: 39.86%.
- Lift over strongest validation baseline: +7.43 pp.
- Claim label: `direction_candidate`.

## Locked Price/Return Candidate

- Candidate: `price__xgboost_regressor__forward_log_return_h__h20__relative_strength`.
- Validation RMSE: 0.0245612.
- Final RMSE: 0.130293.
- Validation sign accuracy: 64.00%.
- Final sign accuracy: 39.57%.
- Claim label: `price_return_candidate`.
