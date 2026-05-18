# VN30 Hourly Track A Diagnostic 65 Row Claim Boundary

## Boundary

- The L2 Logistic h40 65.71% final row cannot be claimed.
- It was not selected by the registered validation rule.
- It cannot establish final65.
- It cannot replace the clean Track A baseline60 result.

## Current Valid Baseline

- Baseline60 remains the Track A Logistic Regression h40 result at 60.43%.
- Historical RF h60 reference remains 60.31%.
- The selected regime-feature v2 candidate remains XGBoost shallow h40 at 56.58%, which failed to improve the baseline.

## Future Requirement

Future final65 requires one of:

- A validation-safe pre-registered selection rule that selects the candidate before final scoring.
- A future blind test where the rule and candidate are fixed before labels are evaluated.

Unsafe claims include selecting the best final row after inspection, describing confidence-filtered or subset results as overall VN30 accuracy, using top-k/ranking as directional accuracy, or making trading/profitability/live-deployment claims.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
