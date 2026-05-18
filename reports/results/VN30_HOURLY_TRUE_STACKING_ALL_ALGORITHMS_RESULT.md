# VN30 Hourly True Stacking All Algorithms Result

## Scope

- Setup: Track A canonical-like VN30 stock-only hourly setup.
- Main metric: pooled overall directional accuracy.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage for the selected stack.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking substitution: no.
- Final-label selection: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## References

- Prior Track A Logistic Regression h40: 60.43%.
- Historical RF h60 reference: 60.31%.
- Main targets: beat 60.31% and beat 60.43%.
- Aspirational target: 65%.

## Base Models Used

The all-algorithm OOF layer generated validation/OOF and final probabilities for:

- Random Forest.
- ExtraTrees.
- Decision Tree / CART.
- XGBoost.
- LightGBM.
- Logistic Regression.
- Baseline signals: previous direction, moving average, majority-train, always-up.

## True Stacking Result

Best validation-selected compliant stack:

- Stacking method: `stack_all_xgboost_shallow_meta`.
- Meta-model: `xgboost_shallow_meta`.
- Horizon: h=80.
- Base models used:
  - `always_up__h80__feature_set_C_closest`
  - `logistic_regression__h80__feature_set_C_closest`
  - `moving_average__h80__feature_set_C_closest`
  - `xgboost__h80__feature_set_C_closest`
  - `previous_direction__h80__feature_set_C_closest`
  - `lightgbm__h80__feature_set_C_closest`
  - `extra_trees__h80__feature_set_C_closest`
  - `random_forest__h80__feature_set_C_closest`
  - `decision_tree_cart__h80__feature_set_C_closest`
  - `majority_train__h80__feature_set_C_closest`
- Validation accuracy: 63.88%.
- Validation baseline accuracy: 54.84%.
- Validation delta vs baseline: +9.04 percentage points.
- Final accuracy: 55.01%.
- Final baseline accuracy: 54.49%.
- Final delta vs baseline: +0.52 percentage points.
- Final rows: 2,874.
- Final coverage: 100.00%.
- Active ticker count: 30.
- Delta vs 60.31: -5.30 percentage points.
- Delta vs 60.43: -5.42 percentage points.
- Pass 60: no.
- Pass 60.31: no.
- Pass 60.43: no.
- Pass 65: no.
- Selected on validation: yes.

Diagnostic stacks with higher validation accuracy at h=100 or h=120 were not compliant full-universe selected stacks because their final ticker counts were below 30. They are not used as success evidence.

## Risk And Claim Level

- Overfit risk: high.
- Evidence: validation accuracy was 63.88%, but final accuracy fell to 55.01%.
- Audit status: completed.
- Claim level: exploratory negative true-stacking result.

True stacking across all available candidate algorithm families did not beat the historical RF h60 60.31% reference, did not beat the prior Track A Logistic Regression h40 60.43% result, and did not establish final65.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
