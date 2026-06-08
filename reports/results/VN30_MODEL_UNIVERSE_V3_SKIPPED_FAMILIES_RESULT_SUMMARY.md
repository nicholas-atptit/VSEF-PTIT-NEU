# VN30 Model Universe V3 Skipped Families Result Summary

## Required Answers

1. Which skipped families successfully ran: deep_sequence, ensemble, qml_integration, statistical.
2. Which skipped families remained unavailable: PennyLane hybrid QNN, V4 pure quantum kernel replay, catboost_classifier, catboost_regressor, direction-price_joint_diagnostic_ensemble, qml_v8_kernel_features_l2, qml_v8_kernel_features_lightgbm, stacking_logistic.
3. Did statistical models improve price/return forecasting: true on validation; best statistical row is `v3__statistical__GARCH-assisted__forward_log_return_h__h40`.
4. Did deep/sequence models improve direction forecasting: true on validation.
5. Did deep/sequence models improve price/return forecasting: true on validation.
6. Did CatBoost improve over XGBoost/LightGBM if available: CatBoost unavailable or skipped.
7. Did QML integration improve over QML V8 or same-target classical baselines: false for the best QML integration row `direction__qml_v8_kernel_features_l2__market_relative_vn30__h40__qml_kernel_features`; this remains diagnostic-only.
8. Did ensembles improve validation-to-final transfer: best ensemble `v3__ensemble__model_family_ensemble__market_relative_vn30__h40__market_context` validation 63.12%, final 24.69%.
9. Did any validation-governed candidate beat the 61.61 absolute-direction champion on comparable scope: false for `v3__deep__MLP__absolute_direction__h40__combined_strategy_features` with final 54.69%; future-blind confirmation is still required.
10. Did any validation-governed candidate beat the 64.44 QML V8 market-relative result on comparable scope: true for `v3__deep__BiLSTM__market_relative_vn30__h40__combined_strategy_features` with final 72.50%; future-blind confirmation is still required.
11. Did any price/return model beat random walk/last price on final: true for `v3__statistical__ETS__forward_simple_return_h__h20` with final improvement +1.70 pp.
12. Which candidates require future-blind confirmation: 49 rows are listed in `v3_future_blind_candidate_registry.csv`.
13. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or replacement claim is made.

## Best Rows

- Best direction validation row: `v3__deep__BiLSTM__market_relative_vn30__h40__combined_strategy_features`; validation 64.06%, final 72.50%.
- Best price/return validation row: `v3__statistical__GARCH-assisted__forward_log_return_h__h40`; validation improvement +6.73 pp, final improvement +0.18 pp.
- Best ensemble row: `v3__ensemble__model_family_ensemble__market_relative_vn30__h40__market_context`; validation 63.12%, final 24.69%.
- Best QML integration row: `direction__qml_v8_kernel_features_l2__market_relative_vn30__h40__qml_kernel_features`; validation 57.14%, final 21.43%.
