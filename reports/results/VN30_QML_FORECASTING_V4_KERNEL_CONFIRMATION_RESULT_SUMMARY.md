# VN30 QML Forecasting V4 Kernel Confirmation Result Summary

## Required Answers

1. Does the v3 quantum-kernel signal persist across feature variants: validation RBF advantage appeared in 4/5 completed feature variants; validation Logistic advantage appeared in 3/5.
2. Does quantum kernel consistently beat RBF SVM: validation-selected row beats RBF on validation and final = true; selected rolling windows won 2/3.
3. Does quantum kernel consistently beat Logistic: validation-selected row beats Logistic on validation and final = false; selected rolling windows won 2/3.
4. Does any validation-selected QML candidate beat the 61.61% classical champion: true.
5. Did the exploratory >61.61 QML row survive confirmation: true; final-ranked rows remain exploratory_not_claimable and do not replace the validation-selected result.
6. Is broader QML expansion justified: not justified beyond focused diagnostics.
7. Exact claim boundary: V4 is a focused experimental quantum-kernel confirmation diagnostic only; no QML result is claimable or replaces the 61.61% L2 Logistic champion.

## Validation-Selected QML Candidate

- Candidate: `qml_v4__quantum_kernel_classifier__market_relative_vn30__h40__relative_strength_features__topk_availability__k4__q4__r2`.
- Feature set: relative_strength_features__topk_availability__k4.
- Qubits/reps: 4 / 2.
- Validation accuracy: 60.56%.
- Final accuracy: 69.44%.
- QML minus RBF SVM: validation +8.33 pp, final +48.33 pp.
- QML minus Logistic: validation -4.44 pp, final +45.56 pp.
- QML minus 61.61% classical champion: +7.83 pp.
- Claim label: `qml_diagnostic_only`.
- Runtime by family: calibrated_logistic=0.13s, l2_logistic=0.02s, quantum_kernel_classifier=1810.88s, svm_linear=0.02s, svm_rbf=0.03s.

## Paper-Safe Wording

VN30 QML v4 ran a focused quantum-kernel confirmation diagnostic on VN30 hourly stock forecasting for market-relative VN30 h40 only. The benchmark used strict feature_timestamp and target_timestamp split discipline, train-only PCA/scaling, balanced train-only sampling, validation-governed selection, and same-sample comparisons against RBF SVM, linear SVM, Logistic Regression, and calibrated Logistic Regression. Final-ranked rows are exploratory_not_claimable. No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.
