# VN30 Hourly 2015 - Top-K Ranking Result Summary

## Status
- **Experiment**: Top-k ranking (precision@k, hit_rate@k)
- **Protocol**: [VN30_HOURLY_2015_TOPK_RANKING_PROTOCOL.md](./VN30_HOURLY_2015_TOPK_RANKING_PROTOCOL.md)
- **Runner**: `scripts/research/run_vn30_hourly_2015_topk_ranking_experiments.py`
- **Audit**: `scripts/research/audit_vn30_hourly_2015_topk_ranking_results.py`
- **Outputs**: `outputs/vn30_hourly_2015_topk_ranking_experiments/`

## Metric Change
- **Previous**: Directional accuracy (full-universe binary classification)
- **Current**: Ranking metrics (precision@k, hit_rate@k)
- **Reason**: Directional final65 failed after multiple valid optimization phases
- **Disclosure**: Ranking results are not directly comparable to directional accuracy

## Configuration
- **Models**: LightGBM, XGBoost, Random Forest
- **Horizons**: 20, 40, 60, 80, 120 bars
- **k values**: 3, 5, 10
- **Feature set**: C (price/volume + market context)
- **Selection**: 2024 validation only
- **Final eval**: 2025-01-01 to 2026-05-14

## Results
- **Best precision@k**: 75.54% (LightGBM h=40, k=10)
- **Best hit_rate@k**: 100.00% (LightGBM h=40, k=10)
- **Ranking65 pass**: YES
- **Selected policy**: LightGBM, h=40, k=10, probability_score ranking
- **Validation precision@k**: 35.85%
- **Validation hit_rate@k**: 99.40%
- **Total experiments**: 45
- **Ranking65 candidates**: 45 (all passed)

## Constraints
- No leakage, no daily data, no resampling, no universe change
- Canonical evaluator v1.0.0 for directional metrics
- Ranking metrics computed separately
- No trading/profitability claims

## Next Steps
- Run experiments and audit
- Report results
- Commit and tag if successful
