# VN30 QML Forecasting V5 Full Confirmation Result Summary

## Required Answers

1. Did the v4 QML candidate reproduce: true; replay validation 60.56%, final 69.44%.
2. Did performance survive larger sample sizes: medium survived = false; medium validation 46.67%, final 50.42%; largest feasible validation 51.67%, final 43.33%.
3. Did QML beat RBF SVM under the same target: true for the final evaluated sample stage `largest_feasible`.
4. Did QML beat Logistic under the same target: false for the final evaluated sample stage `largest_feasible`.
5. Did QML beat other same-target classical models: false against the best same-target classical row.
6. Did rolling-origin checks support the signal: 2/6 final-stage windows beat both RBF SVM and Logistic.
7. Is QML expansion justified: limited same-target diagnostics are justified; broad QML search is not.
8. Can any QML result be claimed: no; QML v5 remains diagnostic-only and requires future-blind confirmation for stronger wording.
9. Does QML replace the 61.61% classical champion: no; the champion is a different target/scope benchmark and is not replaced by this market-relative diagnostic.
10. Exact paper-safe wording: VN30 QML v5 replayed a frozen quantum-kernel candidate on VN30 hourly market-relative VN30 h40 forecasting with train-only feature selection/scaling and feature_timestamp/target_timestamp split discipline. Results are diagnostic-only, same-target classical comparisons are reported, and no trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Frozen Candidate

- Candidate: `qml_v5__largest_feasible__quantum_kernel_classifier__market_relative_vn30__h40__relative_strength_features__topk_availability__k4__q4__r2`.
- Feature design: relative_strength_features, top-k 4, ZZFeatureMap reps 2.
- Final evaluated sample stage: largest_feasible.
- Validation accuracy: 51.67%.
- Final accuracy: 43.33%.
- QML minus RBF SVM: validation +4.33 pp, final +20.67 pp.
- QML minus Logistic: validation -12.67 pp, final +12.67 pp.
- QML minus best same-target classical: validation -14.00 pp, final -2.33 pp.
- QML minus 61.61% champion: -18.28 pp.
- Final decision label: `qml_same_target_candidate`.
