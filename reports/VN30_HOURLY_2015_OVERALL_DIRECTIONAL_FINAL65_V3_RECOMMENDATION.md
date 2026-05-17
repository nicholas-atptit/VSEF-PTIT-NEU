# VN30 Hourly 2015 - Overall Directional Final65 V3 Recommendation

## Audit Findings Summary

### 1. Data Availability
- **Actual hourly data starts**: 2023 (NOT 2015 as nominal design suggests)
- **2022 data**: 0 rows (Window A impossible)
- **2023 data**: 9,600 rows, 30 tickers
- **2024 data**: 30,030 rows, 30 tickers
- **2025-2026 data**: 5,274 rows, 30 tickers

### 2. Why Windows A/B Failed
- **Window A (2022)**: NO DATA EXISTS. The hourly cache only contains data from 2023 onwards.
- **Window B (2023)**: 9,600 raw rows → 7,800 valid labels after h=60 horizon shift. Class balance = 0.3147 (heavily imbalanced towards class 0/down).
- **Root cause**: The "2015" design is nominal. Actual hourly data coverage begins in 2023.

### 3. Feature Warmup Loss
- h=60: ~4% rows lost (acceptable)
- Rolling windows: minimal additional loss
- Warmup is NOT the primary issue

### 4. Distribution Shift (CRITICAL FINDING)
| Window | Rows | Class Balance | VNINDEX Trend |
|--------|------|---------------|---------------|
| 2023 validation | 9,600 | 0.3147 | +8.24% |
| 2024 validation | 30,030 | 0.4809 | +11.93% |
| Final (2025-2026) | 5,245 | 0.2984 | +51.65% |

**Key observations**:
- Final evaluation period has a MASSIVE bull market (+51.65% VNINDEX gain)
- Final class balance is 0.2984 (only 30% "up" labels) - this seems counterintuitive for a bull market
- 2024 validation is the most balanced (0.4809) but final is heavily skewed
- The final period's extreme bull market may create different prediction dynamics

### 5. Validation-Final Accuracy Mismatch
- Validation accuracy: ~50%
- Final accuracy: ~59-60%
- This gap suggests the final period's market conditions are fundamentally different from validation periods

## V3 Recommendation

### Recommended Path: **D - Do not run V3 until feature construction is fixed**

**Rationale**:
1. **Rolling validation is unavailable**: Only 2023 and 2024 data exist. Window A (2022) is impossible.
2. **Distribution shift is severe**: Final period has +51.65% VNINDEX trend vs +11.93% in 2024. This makes validation unrepresentative.
3. **Class balance differs**: Final (0.2984) vs 2024 validation (0.4809). Model trained on balanced data may not generalize to imbalanced final period.
4. **Final accuracy may be inflated**: The extreme bull market may make certain patterns easier to predict, inflating final accuracy relative to validation.

### If V3 Must Proceed:

**Preferred validation design**: **B - Intra-2024 blocked validation**
- Split 2024 into quarterly folds (Q1, Q2, Q3, Q4)
- Train on 2023 + 3 quarters, validate on 1 quarter
- This provides 4 validation folds within the most representative period

**Candidate budget**: 50-100 candidates (runtime-optimized)

**Runtime optimization plan**:
- Single feature set (combined)
- Single horizon (h=60) initially
- Focus on RF with targeted hyperparameter search
- No rolling validation (use intra-2024 folds)

**Model focus**: RF h=60/h=120 only (known strong areas)

**Feature warmup**: Keep as-is (4% loss is acceptable)

**Whether final65 remains realistic**: **UNCERTAIN**
- The distribution shift between validation and final is too severe to trust validation-based selection
- Final accuracy may be inflated by bull market conditions
- 65% target may be achievable in bull markets but not generalizable

### Alternative Consideration
If the goal is to demonstrate model skill rather than hit 65%, consider:
- Reporting results with full disclosure of distribution shift
- Using 2024-only validation with explicit caveat about market regime differences
- Focusing on relative improvement over baseline rather than absolute 65% threshold

## Conclusion

**Do not run V3 yet** until the validation-final distribution shift is addressed. The current setup has a fundamental methodological issue: validation periods (2023-2024) do not represent the final evaluation period (2025-2026) due to extreme market regime differences.

If proceeding is necessary, use intra-2024 blocked validation and report results with full disclosure of limitations.
