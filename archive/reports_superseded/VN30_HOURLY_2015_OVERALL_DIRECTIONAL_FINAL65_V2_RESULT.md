# VN30 Hourly 2015 - Overall Directional Final65 V2 Result

## Baseline60 Locked Result
- **Model**: Random Forest
- **Horizon**: h=60
- **Overall directional accuracy**: 60.31%
- **Coverage**: 100%
- **Rows**: 3,474
- **Status**: Baseline60 passed

## V1 Best Result
- **Model**: Random Forest
- **Horizon**: h=60
- **Feature set**: combined
- **Final accuracy**: 59.70%
- **Final coverage**: 100%
- **Final rows**: 3,474
- **Gap to 65%**: 5.30 percentage points
- **Status**: Failed

## V2 Best Selected Candidate
- **Model**: Random Forest
- **Horizon**: h=60
- **Feature set**: combined
- **Hyperparameters**: n_estimators=300, max_depth=6, min_samples_leaf=1, max_features=0.5, class_weight=None
- **Rolling validation mean**: 50.45% (Window C only - Windows A/B had insufficient data)
- **Rolling validation min**: 50.45%
- **Rolling validation std**: 0.00%
- **Final accuracy**: 58.49%
- **Final coverage**: 100%
- **Final rows**: 3,474

## Final65 Status
- **Passed**: NO
- **Gap to 65%**: 6.51 percentage points
- **Search completed**: YES (runtime-limited to 10 candidates)
- **Candidates completed**: 10

## Notes
- Top-k/ranking metrics are explicitly out of scope for this experiment.
- Only overall directional accuracy is considered.
- No confidence abstention, no ticker subset, no ranking metric.
- Rolling validation windows A/B/C were intended but only Window C (2024) had sufficient data due to feature warmup requirements.
- Final evaluation used only once after selection.
- Search was runtime-limited to 10 candidates due to rolling validation overhead.
- Best v2 result (58.49%) is below both v1 (59.70%) and baseline60 (60.31%).

