# VN30 Hourly 2015 - Overall Directional Final65 Protocol

## 1. Target Definition

- **Metric**: Pooled overall directional accuracy (micro-average across all tickers and timestamps).
- **Full universe**: All 30 active VN30 tickers.
- **Full final evaluation coverage**: 100% of available rows in final evaluation period.
- **No confidence abstention**: Every row must receive a directional prediction.
- **No ticker filtering**: No subset of tickers.
- **No top-k or ranking metric**: Directional accuracy only.
- **No conditional slice**: No time period subset or regime subset can be used as final65 success.

## 2. Current Locked Result

- **Baseline60**: Random Forest h=60
- **Overall directional accuracy**: 60.31%
- **Coverage**: 100%
- **Rows**: 3,474
- **Status**: Baseline60 passed, Final65 not yet passed.

## 3. Candidate Improvement Paths

### A. Deeper Hyperparameter Tuning
- Random Forest h=60/h=80/h=120
- XGBoost h=60/h=80/h=120
- LightGBM h=60/h=80/h=120
- Stacking v2 if time-series-safe

### B. Feature Engineering
- Stock lagged returns
- Rolling returns
- Rolling volatility
- Rolling volume shock
- Market-relative lagged returns
- Market trend features
- Volatility-normalized momentum
- Ticker identity encoding if safely supported
- No future values

### C. Longer Horizons
- h=40, h=60, h=80, h=100, h=120, h=160

### D. Class Imbalance Handling
- class_weight: None, balanced
- scale_pos_weight if valid
- Balanced subsampling
- Threshold adjustment selected on validation only, must still predict all rows

### E. Full-Coverage Ensemble
- Validation-weighted soft voting
- Per-horizon ensemble
- Per-ticker model router allowed only if it predicts every ticker/event row
- No abstention

## 4. Selection Rule

- Select hyperparameters/model/horizon using 2024 validation only.
- Final evaluation used only once for scoring.
- No final-eval model selection.

## 5. Success

- Final overall directional accuracy >=65%.
- Otherwise report best result and gap.

## 6. Forbidden

- Top-k/ranking metrics as final target
- Confidence abstention for final target
- Ticker subset for final target
- Daily data
- Resampling
- Data fetch
- Universe change
- Final-eval label leakage
- Trading/profitability claims
