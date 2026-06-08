# VN30 QML Forecasting V7 Hybrid Kernel Rescue Result Summary

## Required Answers

1. Did distribution matching reduce class-balance/sample drift: original validation positive ratio 39.44%, matched validation positive ratio 48.33%; distribution-matched samples improved metadata coverage but did not remove all target drift.
2. Did QML recover on medium/largest samples: false.
3. Did kernel health improve: health prefilter rejected or flagged 87/224 kernels; best validation-governed row effective-rank ratio 0.634736410373554.
4. Did hybrid kernels beat pure quantum kernels: true.
5. Did hybrid kernels beat RBF SVM or Logistic: true on validation.
6. Did normalization/shrinkage help: best normalization/shrinkage row validation 60.00%, final 57.78%.
7. Did QSVC regularization help: best regularization revisit validation 58.89%, final 56.67%.
8. Did QML-derived kernel features help: best meta row `calibrated_logistic` validation 61.11%, final 58.89%.
9. Was QML rescued: false.
10. Does any result replace the 61.61% classical champion: no; target/scope differs and future-blind confirmation is still required.
11. Is the new QML paper still justified: yes, as a diagnostic rescue/negative-evidence track with explicit failure modes, not as a replacement result.
12. Exact claim boundary: V7 is experimental and diagnostic-only; final-ranked rows are exploratory_not_claimable; no trading, profitability, BUY/SELL, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Best Validation-Governed Row

- Candidate: `qml_v7__v4_sized_original__kernel_feature_meta__calibrated_logistic__relative_plus_volatility_features__topk_availability__k4`.
- Sample: v4_sized_original.
- Kernel type: qml_kernel_features.
- Validation accuracy: 61.11%.
- Final accuracy: 58.89%.
- QML minus RBF SVM validation: +12.22 pp.
- QML minus Logistic validation: +22.22 pp.
- Claim label: `qml_diagnostic_only`.

## Best Hybrid Row

- Candidate: `qml_v7__v4_sized_original__relative_volatility_topk4__rank_gaussian_to_pi__PauliFeatureMap__r1__hybrid_rbf_0p9`.
- Validation accuracy: 58.89%.
- Final accuracy: 56.67%.
