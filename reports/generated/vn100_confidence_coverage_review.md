# VN100 Confidence Coverage Review

## Source Artifacts

- Official artifact directory: `outputs/vn100_hybrid_official_2025_confidence_sweep_traincutoff`.
- Combined sweep file: `confidence_threshold_sweep_summary.csv`.
- Hourly sweep file: `hourly/confidence_threshold_sweep_summary.csv`.
- Daily sweep file: `daily/confidence_threshold_sweep_summary.csv`.

## Schema Check

The available sweep schema supports coverage-constrained review through these fields: `frequency`, `model`, `horizon`, `threshold`, `total_rows`, `evaluated_rows`, `coverage_ratio`, `filtered_accuracy`, `passed_60pct`, `coverage_ok`, and `selected_candidate`.

The combined sweep file contains 31 data rows, all for `hourly`, `stacking`, horizon `1`. The daily sweep file contains only the header row, so daily confidence-threshold coverage results are missing from the current official artifact family.

## Coverage Floors

| Minimum coverage floor | Best eligible row | Evaluated rows | Coverage ratio | Filtered accuracy | Passed 60% | Selected candidate |
|---|---:|---:|---:|---:|---|---|
| >= 50% | hourly stacking h=1 threshold 0.55 | 3,782 | 0.5153290639051642 | 0.5817028027498677 | false | false |
| >= 40% | hourly stacking h=1 threshold 0.56 | 3,019 | 0.41136394604169507 | 0.5905929115601193 | false | false |
| >= 30% | hourly stacking h=1 threshold 0.57 | 2,297 | 0.3129854203569969 | 0.6003482803656944 | true | true |

## Interpretation

Under a 50% or 40% minimum coverage floor, no available confidence-filtered row reaches 60% accuracy. The only available selected confidence-sweep pass is hourly stacking h=1 at threshold 0.57, with 31.30% coverage and 60.03% filtered accuracy.

This supports a narrow strategy-level diagnostic claim only. It does not support a global benchmark pass, and it does not support a daily confidence-sweep conclusion because the official daily sweep artifact has no data rows.

## Missing Evidence

- Daily confidence-threshold sweep rows are absent from the official artifact family.
- The available threshold sweep covers only hourly stacking h=1, not every model/horizon/frequency combination.
- Coverage-constrained review is based on one 2025 evaluation window and does not establish multi-window stability.
