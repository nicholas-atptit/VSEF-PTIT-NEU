# VN30 QML Forecasting V3 Targeted Sanity Result Summary

## Required Answers

1. Did quantum kernel beat RBF SVM on the same target/features/sample: true; validation-selected row: true.
2. Did any QML model beat logistic regression: true; validation-selected row: true.
3. Did any QML model beat the 61.61% classical champion: true; validation-selected row: false. Boundary: a non-selected final-scored QML row exceeded 61.61%, but this is exploratory_not_claimable and cannot replace the validation-selected result.
4. Target variant that worked best by validation-selected QML row: market_relative_vn30.
5. Runtime per model family: calibrated_logistic=0.18s, l2_logistic=0.02s, quantum_kernel_classifier=1193.19s, random_forest_small=0.90s, strongest_simple_baseline=0.00s, svm_linear=0.03s, svm_rbf=0.05s.
6. Is QML worth expanding to a larger benchmark: limited expansion may be justified only as a diagnostic follow-up; full validation-governed rerun and future-blind confirmation remain required.
7. Claim boundary: V3 is a bounded experimental sanity diagnostic only; no QML result is claimable or replaces the 61.61% classical champion.

## Final Validation-Selected QML Result

- Candidate: `qml_v3__quantum_kernel_classifier__market_relative_vn30__h40__combined_strategy_features__pca_train_only__k4__q4__r1`.
- QML family/library: quantum_kernel_classifier / qiskit_machine_learning.
- Target/horizon/features: market_relative_vn30 / h40 / combined_strategy_features__pca_train_only__k4.
- Validation accuracy: 52.92%.
- Final accuracy: 42.50%.
- QML minus RBF SVM final accuracy: +20.00 pp.
- QML minus Logistic final accuracy: +21.67 pp.
- QML minus 61.61% classical champion: -19.11 pp.
- Claim label: `qml_kernel_candidate`.

## Paper-Safe Wording

VN30 QML v3 tested a bounded quantum-kernel sanity benchmark on VN30 hourly stock forecasting with train-only transforms and strict feature_timestamp/target_timestamp split discipline. Quantum-kernel results are diagnostic-only and are compared against same-sample RBF SVM, linear SVM, logistic, calibrated logistic, random forest, and simple baselines. No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.
