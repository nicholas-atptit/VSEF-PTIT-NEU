# VN30 QML Forecasting V6 Scaling Rescue Result Summary

## Required Answers

1. Why did QML weaken under larger samples: sample shift, small-sample overfit, and kernel concentration are the primary diagnosis.
2. Was there label/class balance drift: true.
3. Was there feature distribution drift: false.
4. Was kernel concentration detected: true (1 diagnostic rows).
5. Did feature scaling rescue performance: no durable rescue; best overall scaling `minmax_0_pi` validation 60.56%, final 69.44%; best largest-feasible scaling `standard_zscore` validation 55.33%, final 49.67%.
6. Did feature map reps/entanglement rescue performance: best rescue `qml_v6__largest_feasible__minmax_0_pi__quantum_kernel_classifier__market_relative_vn30__h40__relative_strength_features__topk_availability__k4__r2__full__C1p0__none` validation 51.67%, final 43.33%.
7. Did QSVC regularization rescue performance: best regularized row validation 51.67%, final 43.33%; beat best same-target classical on both validation and final = false.
8. Did QML-as-feature meta-model improve performance: best meta row `l2_logistic` validation 48.33%, final 42.00%.
9. Did any validation-selected QML result beat same-target classical baselines: false.
10. Does any QML result replace the 61.61% classical champion: no; target/scope differs and future-blind confirmation is required.
11. Is QML paper still justified: yes, as a negative/diagnostic QML evidence track, not as a replacement result.
12. Exact claim boundary: V6 is a scaling-failure and kernel-rescue diagnostic only; no trading, profitability, BUY/SELL, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Decision Labels

`class_balance_drift, kernel_concentration_detected, small_sample_overfit, qsvc_regularization_issue, qml_not_rescued`

## Best Rows

- Best scaling row: `qml_v6__v4_sized__minmax_0_pi__quantum_kernel_classifier__market_relative_vn30__h40__relative_strength_features__topk_availability__k4__r2__full__C1p0__none`.
- Best rescue row: `qml_v6__largest_feasible__minmax_0_pi__quantum_kernel_classifier__market_relative_vn30__h40__relative_strength_features__topk_availability__k4__r2__full__C1p0__none`.
- Best regularization row: `qml_v6__largest_feasible__minmax_0_pi__quantum_kernel_classifier__market_relative_vn30__h40__relative_strength_features__topk_availability__k4__r2__full__C1p0__none`.
- Best QML feature meta row: `l2_logistic`.
