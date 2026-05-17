# VN30 Hourly 2015 - Overall Directional Final65 Claim Register

## Safe Claims

### baseline60_overall_directional
- **Claim**: Random Forest h=60 achieves 60.31% overall directional accuracy on full VN30 universe with 100% coverage.
- **Status**: VERIFIED
- **Metric**: Pooled overall directional accuracy (canonical evaluator v1.0.0)
- **Coverage**: 100%
- **Rows**: 3,474
- **Active tickers**: 30/30

### final65_overall_directional
- **Claim**: *(pending)* Model achieves >=65% overall directional accuracy on full VN30 universe with 100% coverage.
- **Status**: NOT YET PASSED
- **Condition**: Must achieve >=65% pooled accuracy with 100% coverage on full final evaluation set.

## Failed Claims

### final65_directional (previous attempts)
- **Claim**: Various models/horizons achieve >=65% directional accuracy.
- **Status**: FAILED
- **Reason**: No model achieved >=65% with sufficient coverage.

### topk_ranking_75 (separate track)
- **Claim**: LightGBM h=120 achieves 75.54% precision@10.
- **Status**: INVALID (below random baseline)
- **Note**: This is a ranking metric, NOT directional accuracy. Out of scope for final65.

## Unsafe Claims

### ranking_as_directional
- **Claim**: Using ranking precision@k as directional accuracy.
- **Status**: UNSAFE
- **Reason**: Different metric, not comparable.

### confidence_filtered_as_overall
- **Claim**: Using confidence-filtered result as overall accuracy.
- **Status**: UNSAFE
- **Reason**: Not full coverage, not overall.

### ticker_subset_as_full_universe
- **Claim**: Using ticker subset result as full universe accuracy.
- **Status**: UNSAFE
- **Reason**: Not full universe.

### profitability_claim
- **Claim**: Strategy is profitable.
- **Status**: UNSAFE
- **Reason**: No trading/backtesting evidence.

### trading_readiness_claim
- **Claim**: Strategy is ready for live trading.
- **Status**: UNSAFE
- **Reason**: No production testing, no risk analysis.

### live_deployment_claim
- **Claim**: Strategy is ready for live deployment.
- **Status**: UNSAFE
- **Reason**: No production testing, no risk analysis.
