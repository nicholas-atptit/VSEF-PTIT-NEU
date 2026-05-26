# VN30 QML Forecasting V8 Drift-Aware Kernel-Feature Result Summary

## Required Answers

1. Did kernel-feature drift explain validation-final decay: partially; mean final PSI versus validation across kernel-feature audits was 1.5096 where finite, and final label/ticker distribution still differed from validation.
2. Which QML kernel features were most useful: positive_centroid_similarity, top_eigen_projection_1, negative_centroid_similarity.
3. Did drift-aware calibration improve over V7: true.
4. Did the final result improve over 58.89%: true; selected final accuracy was 64.44%.
5. Did the model beat same-target Logistic/RBF/LightGBM: validation wins versus Logistic, RBF SVM, LightGBM.
6. Does any result replace the 61.61% classical champion: no; target/scope differs and future-blind confirmation is required.
7. Is a new QML paper still justified: yes, as a diagnostic representation-and-drift study, not as a replacement claim.
8. Exact claim boundary: V8 is experimental and diagnostic-only; selection is validation-governed; final rows are scoring-only; no trading, profitability, BUY/SELL, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Locked Validation-Selected Candidate

- Candidate: `qml_v8__v4_sized_distribution_matched__relative_market_context_topk4__minmax_0_pi__qml_kernel_features_plus_relative_strength_market_context__l2_logistic__robust_threshold_by_validation_quantile`.
- Sample: v4_sized_distribution_matched.
- Kernel feature source: relative_market_context_topk4.
- Feature set: qml_kernel_features_plus_relative_strength_market_context.
- Meta-model: l2_logistic.
- Drift method: robust_threshold_by_validation_quantile.
- Validation accuracy: 60.56%.
- Final accuracy: 64.44%.
- Delta versus V7 final 58.89%: +5.56 pp.
- Delta versus 61.61% classical champion context: +2.83 pp.
- Claim label: `qml_kernel_feature_rescue_improved_requires_future_blind`.
