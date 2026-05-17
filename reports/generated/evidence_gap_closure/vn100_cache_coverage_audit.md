# VN100 Cache Coverage Audit

## Source

- Official artifact directory: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Local cache directory inspected: `data/market_cache/vnstock_data/vn100`.
- No provider fetch, training run, or benchmark rerun was performed.

## Summary

- VN100 tickers considered in official cache summary: 104.
- Tickers with local daily cache files: 60.
- Tickers with local hourly cache files: 86.
- Standalone daily benchmark-usable tickers for 2025: 0.
- Hourly benchmark-usable tickers for 2025: 7.
- Tickers benchmark-usable in at least one frequency: 7.
- New benchmark-usable tickers beyond the official evaluated set: 0.

## Benchmark-Usable Tickers

ANV, BCM, BID, BMP, BVH, BWE, CII

## Missing/Unusable Reason Concentration

| reason | ticker_count |
| --- | --- |
| daily:ends_before_eval_end_tolerance; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:445<500 | 20 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 \| hourly:cache_missing; pre_eval_rows_below_min:0<500; eval_rows_below_min:0<500 | 19 |
| daily:ends_before_eval_end_tolerance; eval_rows_below_min:0<60 \| hourly:cache_missing; pre_eval_rows_below_min:0<500; eval_rows_below_min:0<500 | 12 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:445<500 | 11 |
| daily:ends_before_eval_end_tolerance; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:240<500 | 11 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:240<500 | 8 |
| daily:ends_before_eval_end_tolerance; eval_rows_below_min:0<60 | 5 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 | 2 |
| daily:ends_before_eval_end_tolerance; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:65<500 | 2 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:155<500 | 1 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:185<500 | 1 |
| daily:cache_missing; pre_eval_rows_below_min:0<120; eval_rows_below_min:0<60 \| hourly:starts_at_or_after_eval_start; pre_eval_rows_below_min:0<500; eval_rows_below_min:443<500 | 1 |

## Expanded Benchmark Readiness Verdict

The existing official cache summary does not show additional 2025 benchmark-usable tickers beyond the
seven already evaluated tickers. The expanded 2025 benchmark is therefore not generated in this phase.

## Output Files

- CSV audit: `reports/generated/evidence_gap_closure/vn100_cache_coverage_audit.csv`.
- Missing-evidence note for expanded benchmark: `reports/generated/evidence_gap_closure/vn100_expanded_benchmark_missing_evidence.md`.

## Claim Boundary

This audit can support a coverage-readiness statement only. It does not add predictions, improve accuracy,
or establish representativeness for the full VN100 universe.
