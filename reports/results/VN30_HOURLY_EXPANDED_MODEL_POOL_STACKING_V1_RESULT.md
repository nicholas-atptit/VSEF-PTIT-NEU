# VN30 Hourly Expanded Model Pool + Stacking V1 Result

## Scope

- Main target: VN30 stock-only hourly overall directional accuracy.
- Universe: 30/30 active VN30 January 2025 stock tickers.
- Coverage: full coverage.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking substitution: no.
- Index-only result as stock result: no.
- Daily result as hourly result: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Locked Baseline

- Locked Random Forest h=60 baseline: approximately 60.31%.
- Main target: beat 60.31%.
- Aspirational target: 65%.

## Model Pool Screened

Executed screening families:

- Random Forest.
- ExtraTrees.
- Decision Tree / CART.
- XGBoost.
- LightGBM.
- Logistic Regression.
- Baselines: majority, always-up, previous direction, moving average, random seeded.

Diagnostic or heavy families were inventoried but not used as main screening candidates unless active/stable in the local pipeline. SARIMAX/ETS remain diagnostic baselines only, and LSTM/BiLSTM were not used because this run kept the stock-only hourly pipeline light and stable.

## Best Validation-Selected Base Model

- Model family: XGBoost.
- Candidate: `xgboost__h40__stock_lagged_rolling`.
- Horizon: h=40.
- Feature set: `stock_lagged_rolling`.
- Validation accuracy: 54.84%.
- Validation majority/simple baseline accuracy: 51.68%.
- Validation delta vs majority/simple baseline: +3.16 percentage points.
- Final stock-only accuracy: 50.57%.
- Final majority/simple baseline accuracy: 47.55%.
- Final delta vs majority/simple baseline: +3.01 percentage points.
- Final rows: 8,757.
- Final coverage: 100.00%.
- Delta vs locked RF h=60 baseline: -9.74 percentage points.
- Pass >60.31: no.
- Pass 65: no.

## Ensemble / Stacking V1

Validation-selected shortlist:

- `xgboost__h40__stock_lagged_rolling`
- `extra_trees__h40__stock_lagged_rolling`
- `decision_tree_cart__h60__stock_lagged_rolling`
- `lightgbm__h40__stock_lagged_rolling_plus_index_context`
- `xgboost__h60__stock_lagged_rolling`
- `random_forest__h40__stock_lagged_rolling`

Best validation-selected ensemble:

- Ensemble method: `stacking_lightgbm_shallow_oof`.
- Horizon: h=40.
- Base models used:
  - `xgboost__h40__stock_lagged_rolling`
  - `extra_trees__h40__stock_lagged_rolling`
  - `lightgbm__h40__stock_lagged_rolling_plus_index_context`
  - `random_forest__h40__stock_lagged_rolling`
- Validation accuracy: 57.65%.
- Validation delta vs majority/simple baseline: +5.98 percentage points.
- Final stock-only accuracy: 50.76%.
- Final delta vs majority/simple baseline: +3.21 percentage points.
- Final rows: 8,757.
- Final coverage: 100.00%.
- Delta vs locked RF h=60 baseline: -9.55 percentage points.
- Pass >60.31: no.
- Pass 65: no.

## Risk And Claim Level

- Overfit risk: medium.
- Evidence: validation-selected ensemble accuracy was 57.65%, but final accuracy was 50.76%.
- Audit status: completed.
- Claim level: exploratory, with no benchmark success claim.

This run does not improve the locked 60.31% VN30 stock-only hourly baseline. It provides controlled negative evidence for the expanded model pool plus validation-selected ensemble/stacking path on the current inspected final window.

No trading, profitability, investment-recommendation, or live-deployment claim is made.

