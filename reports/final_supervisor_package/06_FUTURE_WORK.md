# Future Work Roadmap

This roadmap is research-oriented. It does not propose live trading, broker execution, real portfolio allocation, or investment advice.

## 1. Regime-Aware Filtering

- Evaluate models only in regimes where they show evidence of usefulness.
- Avoid universal model selection.
- Add regime-specific eligibility thresholds.
- Compare regime-specific vs global ranking.
- Test whether regime-filtered evidence improves interpretability without hiding weak rows.

## 2. Model Health and Eligibility Gate

- Minimum prediction count.
- Missing prediction rate.
- Error volatility.
- Extreme prediction filter.
- Baseline gap threshold.
- Stability score.
- Model exclusion/flagging report.
- Clear separation between excluded rows, flagged rows, and eligible rows.

## 3. Risk Feature Design

- Better volatility features.
- Downside deviation.
- Drawdown persistence.
- Tail-risk features.
- Risk score calibration.
- Avoid risk score that selects the same top-N as forecast-only.
- Evaluate whether risk features add information beyond forecast score.

## 4. Forecast Outlier Control

- `y_pred` outlier detection.
- Clipping or winsorization research.
- Model confidence penalty.
- Anomaly flags.
- Report outlier impact before and after filtering.
- Preserve both raw and controlled metrics for review.

## 5. Candidate Policy Improvement

- Require positive evidence vs baseline.
- Require model health pass before candidate generation.
- Require regime eligibility.
- Separate exploration candidates from high-confidence diagnostics.
- Keep diagnostic candidates distinct from recommendations.

## 6. Robustness Testing

- Alternative regime thresholds.
- Alternative date windows.
- Ticker subset sensitivity.
- Walk-forward windows.
- Transaction cost sensitivity for diagnostic baskets.
- Sensitivity to horizon and candidate top-N settings.

## 7. Reporting and Defense Improvements

- Create final slide deck.
- Create appendix tables.
- Create reproducibility checklist.
- Create supervisor-facing summary.
- Add an artifact inventory appendix with checksums where useful.

## 8. Scope Discipline

This roadmap does not approve Phase 5 by itself. Each future item should enter a governed task with evidence requirements, acceptance criteria, and diagnostic-only language before implementation.
