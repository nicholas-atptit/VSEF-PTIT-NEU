# VN30 QML Forecasting Result Summary

## Required Answers

1. QML dependencies available: false. Qiskit=False, qiskit_machine_learning=False, PennyLane=False.
2. QML library/backend used: none.
3. QML model families run: none; QML execution skipped gracefully.
4. Best feature compression: QML not executed; best classical diagnostic compression was topk_availability.
5. Best target variant: QML not executed; best classical diagnostic target was market_relative_vn30.
6. Validation-governed QML candidate locked: none.
7. Final QML accuracy and lift: not evaluated; QML dependencies missing / not evaluated; QML dependencies missing.
8. Did QML beat the 61.61% classical champion: false.
9. Did QML beat comparable classical baselines: false.
10. Runtime and circuit complexity: see `reports/generated/vn30_qml_forecasting/qml_runtime_summary.csv` and `qml_circuit_summary.csv`.
11. Claimable QML result: none. Claim label is `qml_dependency_missing`.
12. Paper-safe wording: VN30 QML forecasting was evaluated as an optional, dependency-guarded diagnostic track using strict feature_timestamp and target_timestamp split discipline. In this run QML did not replace the validation-governed classical champion; no trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, DOCX, tag, merge, or index-as-stock claim is made.

## Classical Benchmark To Beat

- Model: L2 Logistic Regression.
- Feature set: feature_set_C_closest.
- Horizon: h40.
- Threshold: 0.50.
- Final accuracy: 61.61%.
- Final lift: +10.90 pp.
- Claim label: baseline60_candidate.

## Diagnostic Boundary

QML remains experimental and diagnostic-only. Final-period results are scoring-only after validation selection and cannot be used to promote an unvalidated result.
