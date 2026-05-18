# Repository Rename Cleanup Inventory

- Branch inventoried: `research/vn100-evidence-hardening-v1`.
- Scope: documentation, reports, active paper/source indexes, claim-boundary files, and research scripts.
- Excluded: raw data files, cache files, large output CSVs, generated model outputs, archive snapshots, and protected raw data paths.
- LFS/large-artifact policy: no large LFS data/output artifact was selected for rename.

## Rename Inventory

| current file path | proposed new path | category | reason | rename | references update | protected | LFS/large artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reports/PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | `reports/VN30_HOURLY_PAPER_FIGURE_DATA_SOURCE_INVENTORY.md` | paper source report | Add VN30 hourly scope and remove vague paper-only name. | yes | yes | no | no |
| `reports/PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_EN.md` | paper source report | Add VN30 hourly scope and language suffix. | yes | yes | no | no |
| `reports/PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | `reports/VN30_HOURLY_PAPER_TABLE_FIGURE_CAPTIONS_VI.md` | paper source report | Add VN30 hourly scope and language suffix. | yes | yes | no | no |
| `reports/PAPER_MISSING_METRICS_TODO.md` | `reports/VN30_HOURLY_PAPER_MISSING_METRICS_TODO.md` | paper source report | Add VN30 hourly scope. | yes | yes | no | no |
| `reports/PAPER_ROW_LEVEL_FIGURE_TODO.md` | `reports/VN30_HOURLY_PAPER_ROW_LEVEL_FIGURE_TODO.md` | paper source report | Add VN30 hourly scope. | yes | yes | no | no |
| `reports/PAPER_LITERATURE_DATA_TODO.md` | `reports/VN30_HOURLY_PAPER_LITERATURE_DATA_TODO.md` | paper source report | Add VN30 hourly scope. | yes | yes | no | no |
| `reports/PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | `reports/VN30_HOURLY_PAPER_DOCX_MISSING_FOR_FIGURE_INSERTION.md` | paper source report | Add VN30 hourly scope. | yes | yes | no | no |
| `reports/PAPER_WITH_FIGURES_LAYOUT_QA.md` | `reports/VN30_HOURLY_PAPER_WITH_FIGURES_LAYOUT_QA.md` | paper source report | Add VN30 hourly scope. | yes | yes | no | no |
| `reports/VN30_HOURLY_TARGET62_PAPER_READY_CLAIM_BOUNDARY.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_CLAIM_BOUNDARY.md` | claim boundary | Replace target62/paper-ready wording with selected-candidate scope. | yes | yes | no | no |
| `reports/VN30_HOURLY_TARGET62_STABILITY_CLAIM_BOUNDARY.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_CLAIM_BOUNDARY.md` | claim boundary | Clarify selected-candidate stability boundary. | yes | yes | no | no |
| `reports/VN30_HOURLY_TARGET62_STABILITY_ROBUSTNESS_PROTOCOL.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_PROTOCOL.md` | protocol | Remove misleading target62 framing; final result did not reach 62%. | yes | yes | no | no |
| `reports/VN30_HOURLY_TARGET62_STABILITY_ROBUSTNESS_RESULT.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_ROBUSTNESS_RESULT.md` | result | Remove misleading target62 framing; preserve metrics. | yes | yes | no | no |
| `reports/VN30_HOURLY_TARGET62_PAPER_READY_STABILITY_AUDIT_PROTOCOL.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_PROTOCOL.md` | protocol | Use selected-candidate naming. | yes | yes | no | no |
| `reports/VN30_HOURLY_TARGET62_PAPER_READY_STABILITY_AUDIT_RESULT.md` | `reports/VN30_HOURLY_SELECTED_CANDIDATE_STABILITY_AUDIT_RESULT.md` | result | Use selected-candidate naming. | yes | yes | no | no |
| `reports/VN30_DAILY_2015_BENCHMARK_RESULT_SUMMARY.md` | `reports/VN30_DAILY_2015_RESULT_SUMMARY.md` | result summary | Shorter active daily result summary name. | yes | yes | no | no |
| `reports/INDEX_DIRECTIONAL_BENCHMARK_RESULT_SUMMARY.md` | `reports/VN30_INDEX_BENCHMARK_RESULT_SUMMARY.md` | result summary | Align index benchmark with VN30 research identity. | yes | yes | no | no |
| `reports/INDEX_DIRECTIONAL_BENCHMARK_CLAIM_REGISTER.md` | `reports/VN30_INDEX_BENCHMARK_CLAIM_REGISTER.md` | claim register | Align index benchmark with VN30 research identity. | yes | yes | no | no |
| `scripts/research/rerun_vn30_hourly_selected_l2_logistic_h40_row_predictions.py` | `scripts/research/rerun_vn30_hourly_selected_candidate_row_predictions.py` | research script | Use action-based selected-candidate name without model-specific clutter. | yes | yes | no | no |
| `scripts/research/build_paper_empirical_tables.py` | `scripts/research/build_vn30_hourly_paper_empirical_tables.py` | paper builder script | Add VN30 hourly scope. | yes | yes | no | no |
| `scripts/research/build_paper_empirical_figures.py` | `scripts/research/build_vn30_hourly_paper_empirical_figures.py` | paper builder script | Add VN30 hourly scope. | yes | yes | no | no |
| `scripts/research/audit_vn30_hourly_target62_paper_ready_stability.py` | `scripts/research/audit_vn30_hourly_selected_candidate_stability_summary.py` | audit script | Replace target62/paper-ready wording with selected-candidate stability scope. | yes | yes | no | no |
| `scripts/research/audit_vn30_hourly_target62_stability_robustness.py` | `scripts/research/audit_vn30_hourly_selected_candidate_stability_robustness.py` | audit script | Replace target62 wording with selected-candidate robustness scope. | yes | yes | no | no |

## Existing Canonical Names Kept

| current file path | proposed new path | category | reason | rename | references update | protected | LFS/large artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROLLING_STABILITY_RESULT.md` | same | result | Already matches convention. | no | no | no | no |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ROW_LEVEL_RERUN_PROTOCOL.md` | same | protocol | Already matches convention. | no | no | no | no |
| `reports/VN30_HOURLY_SELECTED_CANDIDATE_ARTIFACT_DISCOVERY.md` | same | artifact discovery | Already matches convention. | no | no | no | no |
| `docs/USAGE.md` | same | docs | Already canonical. | no | yes | no | no |
| `docs/RESEARCH_WORKFLOW.md` | same | docs | Already canonical. | no | yes | no | no |
| `docs/REPOSITORY_STRUCTURE.md` | same | docs | Updated in place as current structure guide. | no | yes | no | no |
| `README.md` | same | top-level docs | Current repository landing page. | no | yes | no | no |
| `reports/generated/paper_tables_current/` | same | generated folder | Protected generated artifact folder; not renamed to avoid breaking provenance. | no | index only | yes | yes |
| `reports/generated/paper_figures_current/` | same | generated folder | Protected generated artifact folder; not renamed to avoid breaking provenance. | no | index only | yes | yes |
| `reports/generated/vn30_hourly_selected_candidate_rolling/` | same | generated folder | Already matches selected-candidate convention. | no | no | yes | yes |

## Created Index Files

| current file path | proposed new path | category | reason | rename | references update | protected | LFS/large artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| n/a | `reports/VN30_RESEARCH_CLAIM_REGISTER.md` | claim-register index | Add standardized VN30 research claim-register entry point without merging or changing existing claim registers. | no | yes | no | no |

## Deferred Or Not Renamed

| current file path | proposed new path | category | reason | rename | references update | protected | LFS/large artifact |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `reports/VN30_HOURLY_TRACK_A_TARGET62_VALIDATION_SAFE_*` | same | protocol/result/claim register | Historical protocol scope genuinely used target62 validation-safe language; not renamed in this cleanup. | no | no | no | no |
| `reports/VN30_DAILY_2015_TARGET60_*` | same | protocol/result/claim register | Historical phase names genuinely refer to target60 attempts. | no | no | no | no |
| `scripts/research/run_vn30_daily_2015_target60_*.py` | same | research scripts | Historical target60 scripts; not active rename target. | no | no | no | no |
| `scripts/legacy/research/**/*.py` | same | legacy scripts | Preserved legacy inventory; renaming legacy scripts risks breaking archival references. | no | no | no | no |
| `reports/generated/**/*.csv` | same | generated artifacts | Generated evidence; excluded from rename cleanup. | no | no | yes | yes |
| `outputs/**` | same | output artifacts | Protected output evidence; excluded from rename cleanup. | no | no | yes | yes |
| `data/**` | same | data artifacts | Protected raw/cache data; excluded from rename cleanup. | no | no | yes | yes |
| `archive/generated_data_snapshots/**` | same | archive snapshots | Protected snapshots; excluded from rename cleanup. | no | no | yes | yes |
