# VN30 Hourly 2015 - Claim Register

## Claims

### 1. Baseline60
- **Claim**: RF h=60 achieves 60.31% directional accuracy on final evaluation
- **Type**: Directional accuracy
- **Status**: Verified
- **Coverage**: 100%
- **Rows**: 3,474
- **Evaluator**: Canonical v1.0.0
- **Source**: Final65 Focus v3

### 2. Directional Final65 Failed
- **Claim**: No model/horizon/policy achieves >=65% directional accuracy with >=30% coverage
- **Type**: Negative result
- **Status**: Verified
- **Best**: 61.48% at 31.5% coverage (RF h=40, confidence abstention)
- **Source**: All-Model Final65 Router, Final65 Focus v3

### 3. Top-K Ranking Experiment
- **Claim**: *(pending)* precision@k >=65% or hit_rate@k >=65% on final evaluation
- **Type**: Ranking metric
- **Status**: Pending
- **k values**: 3, 5, 10
- **Models**: LightGBM, XGBoost, Random Forest
- **Horizons**: 20, 40, 60, 80, 120
- **Selection**: 2024 validation only
- **Disclosure**: Ranking metric change from directional accuracy

## Forbidden Claims
- No trading-readiness claim
- No profitability claim
- No live deployment claim
- No comparison of precision@k to directional accuracy without metric change disclosure

## Version History
- v1: Initial register (2026-05-17)
