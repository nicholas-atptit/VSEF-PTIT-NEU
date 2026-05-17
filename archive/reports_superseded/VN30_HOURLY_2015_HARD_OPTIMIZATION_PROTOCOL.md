# VN30 Hourly 2015 - Hard Optimization Protocol v2

## A. No-Leakage Split

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2015-01-01 to 2023-12-31 | Model training |
| Validation | 2024-01-01 to 2024-12-31 | Hyperparameter/threshold/policy selection |
| Final eval | 2025-01-01 to 2026-05-14 | Scoring only |

## B. Target Hierarchy

| Priority | Target | Requirements |
|----------|--------|-------------|
| 1 | global_full_universe_60 | All 30 tickers, no filter, >=60% |
| 2 | global_model_horizon_60 | Specific model/horizon, all 30 tickers, >=60% |
| 3 | coverage_qualified_65 | >=65%, coverage >=30%, rows >=1000, validation-selected |
| 4 | weaker_conditional | >=65%, coverage >=20%, rows >=500 |
| 5 | exploratory_only | Ticker/slice rows <500 or coverage <20% |

## C. Candidate Strategies

1. Per-ticker models - best model/horizon per ticker selected on validation
2. Per-sector/ticker-cluster routing - grouped routing on validation
3. Weighted soft-voting ensemble - validation-learned weights
4. Model-horizon router - per-ticker model/horizon selection on validation
5. Meta-labeling - predict forecast correctness, abstain on low-confidence
6. Confidence calibration - Platt/isotonic on validation only
7. Regime-aware routing - ex-ante regime features only

## D. Forbidden

- Final-eval label selection
- Post-hoc eval-only thresholding
- Dropping weak tickers for global claim
- Daily/resampled data
- Trading/profitability claims

## E. Boundary

- No trading-readiness, profitability, or live deployment claim
- No paper or DOCX generation
- No new data fetching
- No main branch modifications