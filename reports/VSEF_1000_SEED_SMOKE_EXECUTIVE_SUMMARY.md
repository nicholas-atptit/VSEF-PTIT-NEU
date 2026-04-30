# VSEF 1000-Seed Smoke Executive Summary

## Document Metadata

| Field | Value |
| --- | --- |
| Document type | Executive summary |
| Created / authored | Thursday, 2026-04-30 20:05:31 ICT (UTC+07:00) |
| Timezone | Asia/Ho_Chi_Minh / ICT (UTC+07:00) |
| Source output root | `artifacts\quant_core_repeated_seed_1000_smoke` |
| Branch | `vsef-doc-datetime-metadata-standardization` |
| Commit | `4e317fba24f836209488b0b6e23729a04bbe45dd` |
| Run mode | `research_core` |
| Preset | `smoke` |
| Seed count | `1000` |
| Status | Complete summary from saved artifacts |

## What Was Run

The saved output under `artifacts\quant_core_repeated_seed_1000_smoke` represents a 1000-seed Quant Core smoke diagnostic using `preset=smoke` and `run_mode=research_core`. The seed range is 1 to 1000. The report generation step did not rerun the experiment or run any heavy training.

Repeated-seed training is a stability diagnostic. It must not be used to select the single best random seed based on test-period performance.

## Why It Matters

Repeated-seed evaluation tests whether the pipeline is sensitive to random seeds. This is useful for governance because large seed dispersion would weaken claims, while low dispersion indicates that the tested workflow is reproducible under the saved data, configuration, and environment.

## Completion Status

- Seed folders found: 1000
- Completed seed folders by required files: 1000
- Execution-log status counts: 985 `completed`, 15 `skipped_complete`, 0 `failed`
- Root aggregate files found: 12 of 12 expected files

The saved 1000-seed smoke output is complete by the minimum completion-file rule.

## Key Stability Findings

- Model-level forecast metrics are stable across all 1000 seeds at displayed precision.
- Model-horizon metrics are also stable for the available horizon: `5`.
- The largest recorded standard deviation across available stability tables is 0, which is floating-point precision scale rather than meaningful dispersion.
- Seed-level artifact inventory shows all expected seed artifacts present for all 1000 seeds.
- `failure_summary.csv` contains status rows only; no failed seed rows are present in the final execution log.

## Key Limitations

- This is a smoke diagnostic, not a production trading validation.
- The result shows seed stability, not market edge or improved accuracy.
- The available model-horizon output contains only one horizon, so cross-horizon stability is not evaluated here.

## Recommended Next Action

Use this report as a governance summary for the completed smoke diagnostic. The next technical step is to review whether a later `medium` / `decision_core` repeated-seed diagnostic is worth running, while preserving the same rule: summarize mean, standard deviation, and quantiles, and do not select a best seed.
