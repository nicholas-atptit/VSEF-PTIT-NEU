# VN30 Hourly 2015 - RF h=60 Final65 Router v2 Result

## Baseline60 Locked Result

- **Model**: Random Forest
- **Horizon**: h=60
- **Target**: Absolute direction
- **Final accuracy**: 60.22%
- **Coverage**: 100%
- **Rows**: 3,474
- **Claim level**: global_full_universe

## Previous Attempts

- **Final65 focus v1**: Failed (59.70%, gap 5.30 pp)
- **Platt calibration**: 59.70%, 100% coverage, 3,474 rows

## Router v2 Results

### Policies Tested

1. **Confidence abstention**: No improvement over base
2. **Per-ticker whitelist**: Selected all 30 tickers (no filtering helped)
3. **Ticker + confidence**: No valid combination
4. **Market regime abstention**: No improvement
5. **Ticker + regime + confidence**: No improvement

### Best Selected Policy

- **Policy type**: Per-ticker whitelist
- **Validation accuracy**: 49.24%
- **Validation coverage**: 100.00%
- **Validation rows**: 30,030
- **Final accuracy**: 59.87%
- **Final coverage**: 100.0%
- **Final rows**: 3,474
- **Active tickers**: 30 (all)

### Final65 Status

- **Passed**: no
- **Gap to 65**: 5.13 percentage points

### Observations

- Base model validation accuracy remains ~49% (near random)
- No policy improved over the base RF h=60 model
- Per-ticker whitelist selected all 30 tickers (no ticker filtering helped)
- Confidence abstention did not improve accuracy
- Market regime abstention did not help
- The gap between validation (~49%) and eval (~60%) suggests regime shift
- Router v2 failed to find any policy that reaches 65%

## Claim Level

- **Exploratory only**: No policy reached >=65%
- Best result: 59.87% (per-ticker whitelist, all 30 tickers)

## Boundary

- No trading-readiness, profitability, or live deployment claim.
- No paper/DOCX generated.
- All selection on 2024 validation only, leakage-safe.
