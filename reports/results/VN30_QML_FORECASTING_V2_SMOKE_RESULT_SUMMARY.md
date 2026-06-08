# VN30 QML Forecasting V2 Smoke Result Summary

## Required Answers

1. QML dependencies installed successfully: true.
2. QML library actually ran: qiskit_machine_learning.
3. QML model family actually ran: quantum_kernel_classifier, variational_quantum_classifier.
4. QML candidates ran vs skipped: 12 ran / 0 skipped.
5. Best QML validation candidate: qml_smoke__variational_quantum_classifier__absolute_direction__h40__combined_strategy_features__pca_train_only__k4__d1.
6. Final smoke result: accuracy=48.75%, lift=-27.50 pp, claim_label=`qml_diagnostic_only`.
7. QML vs classical smoke baseline: false (-0.15 pp).
8. QML vs 61.61% classical champion: false (-12.86 pp).
9. Runtime and circuit complexity: see `qml_smoke_runtime_summary.csv` and `qml_smoke_circuit_summary.csv`; total runtime was 1506.35 seconds.
10. Claimable result: no. V2 is a smoke diagnostic only.
11. Claim boundary: experimental QML smoke only; no trading, profitability, BUY/SELL, recommendation, live deployment, DOCX, VN100, tag, merge, or index-as-stock claim; stronger QML claims require full validation-governed rerun and future-blind confirmation.

## Classical Champion To Beat

- L2 Logistic Regression / feature_set_C_closest / h40 / threshold 0.50.
- Final accuracy: 61.61%.
- Final lift: +10.90 pp.
- Claim label: baseline60_candidate.
