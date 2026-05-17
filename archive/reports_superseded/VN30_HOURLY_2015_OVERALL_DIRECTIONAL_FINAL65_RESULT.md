# VN30 Hourly 2015 - Overall Directional Final65 Result

## Baseline60 Locked Result
- **Model**: Random Forest
- **Horizon**: h=60
- **Overall directional accuracy**: 60.31%
- **Coverage**: 100%
- **Rows**: 3,474
- **Status**: Baseline60 passed

## Best New Overall Directional Result
- **Model**: Random Forest
- **Horizon**: h=60
- **Feature set**: combined
- **Hyperparameters**: n_estimators=300, max_depth=6, min_samples_leaf=5, max_features=sqrt, class_weight=balanced
- **Final accuracy**: 59.70%
- **Final coverage**: 100%
- **Final rows**: 3,474
- **Validation accuracy**: 48.61%
- **Validation rows**: 30,030

## Final65 Status
- **Passed**: NO
- **Gap to 65%**: 5.30 percentage points
- **Gap to baseline60**: -0.61 percentage points (below baseline)

## Notes
- Top-k/ranking metrics are out of scope for this experiment.
- Only overall directional accuracy is considered.
- No confidence abstention, no ticker subset, no ranking metric.
- Sweep was limited to 20 RF configs + 20 other model configs due to runtime constraints.
- All candidates failed to reach 65% threshold.
- Best result (59.70%) is below baseline60 (60.31%).

