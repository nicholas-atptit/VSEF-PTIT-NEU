# VN30 Model Universe V4 BiLSTM Relock Result Summary

## Required Answers

1. Reconstructed BiLSTM candidate: `v4__bilstm_relock__BiLSTM__market_relative_vn30__h40__combined_strategy_features__seed42` using market_relative_vn30 h40, combined_strategy_features, validation 64.06%, final 72.50%.
2. Split/leakage audit: `pass`; feature_timestamp and target_timestamp boundaries passed for train, validation, and final.
3. Ticker stability: see `v4_bilstm_ticker_stability.csv`; reconstruction final ticker accuracy mean is 73.34%.
4. Quarter/month stability: see `v4_bilstm_quarter_stability.csv`; minimum rolling/window accuracy was 51.25%.
5. Prediction/class balance: validation predicted-positive 10.94%, final predicted-positive 9.69%.
6. Rolling-origin replay: see `v4_bilstm_rolling_origin.csv`; V4 keeps this as diagnostic because early/late windows are replay checks, not new final selection.
7. Seed sensitivity: mean 59.56%, std 0.0797; stability warning is `unstable`; see `v4_bilstm_seed_sensitivity.csv`.
8. Architecture ablation: best validation ablation was `v4__bilstm_relock__BiLSTM__market_relative_vn30__h40__combined_strategy_features__seed42` with validation 64.06% and final 72.50%.
9. Comparison against QML V8 64.44: reconstructed BiLSTM final 72.50% vs QML V8 context 64.44%; label `future_blind_required` and future-blind confirmation required.
10. Comparison against same-target baselines: best same-target baseline was `direction__always_down__market_relative_vn30__h40__combined_strategy_features` with validation 63.12% and final 79.69%; BiLSTM beats strongest same-target baseline on final: false.
11. Relock decision: `bilstm_relock_unstable_future_blind_required`.
12. Exact claim boundary: offline diagnostic-only; no daily T+1 system, trading, profitability, BUY/SELL, recommendation, live deployment, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or replacement claim is made.

## QML Context

- QML V8 context row: `qml_v8_context_market_relative_vn30_h40` final 64.44%.
- V4 BiLSTM is a market-relative diagnostic candidate only and does not replace the absolute-direction 61.61% champion.
