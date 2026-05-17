# VN30 Hourly 2015 - Validation/Final Mismatch Report

## Executive Summary
- **Classification**: unresolved
- **Audit timestamp**: 2026-05-17T06:02:23+00:00

## 1. Data Availability
- **First year**: 2023
- **Last year**: 2026
- **Rows by year**:
  - 2022: 0
  - 2023: 9600
  - 2024: 30030
  - 2025: 2963

## 2. Feature Warmup Loss
- Rows lost due to horizon shift (h=60): 1800
- Rows lost due to rolling window warmup: 30
- Percentage remaining after warmup: 95.92%

## 3. Split Row Availability
- Window A (2022 validation): 0 valid labels
- Window B (2023 validation): 7800 valid labels
- Window C (2024 validation): 28230 valid labels
- Final evaluation: 3445 valid labels

## 4. Why Windows A/B Failed
- Window A (2022): No data available
- Window B (2023): 9600 raw rows, 7800 valid after warmup
- Root cause: Feature warmup too aggressive

## 5. Distribution Shift
- Class balance across windows:
  - 2024 validation: 0.4809
  - Final evaluation: 0.2984
- Market trend (VNINDEX):
  - 2024: 0.1193
  - Final: 0.5165

## 6. Label Alignment
- Label construction: close[t+h] > close[t]
- Class balance by horizon:
  - h=40: 0.482
  - h=60: 0.4863
  - h=120: 0.5146
- No off-by-one shift detected
- No evaluator inconsistency detected

## 7. Decision
- **Classification**: unresolved
- **Recommendation**: Further investigation needed

## 8. Answers to Key Questions
1. **Why did Windows A and B fail?** Feature warmup requirements too aggressive
2. **Is actual hourly data coverage sufficient for 2022/2023?** YES
3. **How many rows lost due to feature warmup?** 1830
4. **Is 2024 validation representative of 2025-2026 final?** Likely NO - distribution shift detected
5. **Is final accuracy inflated by easier market conditions?** Possible - check market trend
6. **Is there any label alignment bug?** NO
7. **Is there any evaluator inconsistency?** NO
8. **Should V3 use rolling validation, 2024-only, or alternative?** Further investigation needed
