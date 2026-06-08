# VN30 Hourly Dual-Track Model Comparison Result

## Scope

- Track A: canonical historical setup.
- Track B: current broader-pipeline setup.
- Main metric: VN30 stock-only hourly pooled overall directional accuracy.
- Universe: 30/30 active VN30 January 2025 tickers.
- Confidence abstention: no.
- Ticker subset: no.
- Top-k/ranking substitution: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Track A - Canonical Historical Setup

- Canonical run: yes.
- RF h60 baseline reproduced: no.
- Historical locked baseline accuracy: 60.31%.
- Historical final rows: 3,474.
- Reproduced RF h60 accuracy in this runner: 59.73%.
- Reproduced RF h60 rows: 3,474.
- Best validation-selected model: `logistic_regression`.
- Best validation-selected horizon: h=40.
- Best validation-selected feature set: `feature_set_C_closest`.
- Validation accuracy: 51.95%.
- Final accuracy: 60.43%.
- Final rows: 4,074.
- Delta vs 60.31: +0.12 percentage points.
- Pass 60.31: yes.
- Pass 65: no.
- Claim level: exploratory.

Important boundary: although the validation-selected Track A candidate narrowly beats 60.31, the RF h60 reproduction row itself did not reproduce 60.31 exactly. The historical lock remains valid in its original canonical evidence chain, and this runner is a closest feature-set C implementation rather than proof that every historical implementation detail is identical.

## Track B - Current Broader Pipeline

- Current-pipeline run: yes.
- Current RF h60 baseline reproduced: yes.
- Current RF h60 baseline accuracy: 56.12%.
- Current RF h60 baseline rows: 8,637.
- Validation-selected model: `stacking_lightgbm_shallow_oof`.
- Validation-selected horizon: h=40.
- Validation-selected feature set: `validation_selected_ensemble`.
- Validation accuracy: 57.65%.
- Final accuracy: 50.76%.
- Delta vs 56.12: -5.36 percentage points.
- Pass 56.12: no.
- Pass 60: no.
- Pass 65: no.
- Claim level: exploratory.

Best final diagnostic in Track B:

- Best final model: `xgboost`.
- Best final horizon: h=120.
- Best final feature set: `stock_lagged_rolling`.
- Best final accuracy: 58.77%.
- Delta vs 56.12: +2.65 percentage points.
- Pass 56.12: yes.
- Pass 60: no.
- Pass 65: no.

Important boundary: the best-final Track B row is diagnostic only because final accuracy must not drive selection. The validation-selected Track B candidate did not beat the current RF h60 baseline.

## Audit Summary

- Track A same universe: yes.
- Track A same target: yes.
- Track A no confidence abstention/ticker subset/top-k: yes.
- Track A final-label selection avoided: yes.
- Track B current setup reproduced: yes.
- Track B no confidence abstention/ticker subset/top-k: yes.
- Track B final-label selection avoided: yes.

No trading, profitability, investment-recommendation, or live-deployment claim is made.

