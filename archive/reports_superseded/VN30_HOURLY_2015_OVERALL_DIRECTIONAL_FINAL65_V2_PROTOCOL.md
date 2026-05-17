# VN30 Hourly 2015 - Overall Directional Final65 V2 Protocol

## 1. Metric
- **Metric**: Pooled overall directional accuracy (micro-average across all tickers and timestamps).
- **Full universe**: All 30 active VN30 tickers.
- **Full final evaluation coverage**: 100% of available rows in final evaluation period.
- **No confidence abstention**: Every row must receive a directional prediction.
- **No ticker filtering**: No subset of tickers.
- **No top-k or ranking metric**: Directional accuracy only.
- **No conditional slice**: No time period subset or regime subset can be used as final65 success.

## 2. Problem Found in V1
- V1 failed final65 (best result: 59.70%, gap to 65%: 5.30 percentage points).
- Validation accuracy was much lower than final accuracy (~48-50% vs ~59-60%).
- 2024 validation alone may be unstable or non-representative of 2025-2026 final evaluation.
- V1 sweep was runtime-limited (only 20 RF configs + 20 other model configs).
- Single validation window may not provide reliable selection signal.

## 3. Rolling Validation Design

### Window A
- **Train**: 2015-01-01 to 2021-12-31
- **Validation**: 2022-01-01 to 2022-12-31

### Window B
- **Train**: 2015-01-01 to 2022-12-31
- **Validation**: 2023-01-01 to 2023-12-31

### Window C
- **Train**: 2015-01-01 to 2023-12-31
- **Validation**: 2024-01-01 to 2024-12-31

### Selection Rule
- Select candidates by **mean validation accuracy** across windows A/B/C.
- Penalize unstable candidates with high validation variance.
- **Stability score** = mean_validation_accuracy - validation_std.
- **Tie-break** by mean_validation_accuracy.
- Final 2025-2026 evaluation used **only once** after selection.

## 4. Candidate Models
- Random Forest
- XGBoost
- LightGBM
- Stacking only if time-series-safe and already supported

## 5. Horizons
- h=40, h=60, h=80, h=100, h=120, h=160, h=200 (if enough rows remain)

## 6. Feature Sets
- existing
- stock_lagged
- market_lagged
- combined_stock_market
- interaction_features
- volatility_normalized_features

## 7. Success
- Final overall directional accuracy >=65%.
- 100% coverage.
- All 30 tickers.
- No conditional/filter claim.

## 8. Forbidden
- Top-k/ranking metrics as final target
- Confidence abstention for final target
- Ticker subset for final target
- Daily data
- Resampling
- Data fetch
- Universe change
- Final-eval label leakage
- Trading/profitability claims
