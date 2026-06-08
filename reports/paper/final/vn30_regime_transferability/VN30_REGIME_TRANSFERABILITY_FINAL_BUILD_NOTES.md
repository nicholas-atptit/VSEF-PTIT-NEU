# VN30 Regime Transferability Final Build Notes

## Source Manuscript

- `reports/paper/VN30_REGIME_TRANSFERABILITY_FULL_DRAFT.md`
- Title preserved exactly: `Regime-Dependent Transferability of Machine Learning Forecasts in an Emerging Equity Market: Evidence from VN30 Walk-Forward Benchmarks`

## Evidence Pack Used

- Reference evidence commit: `2f344f885eb2b9727cb90e2f7435b1cf9c0c5e26`
- Paper evidence pack: `reports/generated/vn30_regime_transferability_paper_pack/`
- H4 robustness pack: `reports/generated/vn30_regime_distance_h4_strengthening/`
- Citation source: `reports/paper/VN30_REGIME_TRANSFERABILITY_CITATION_TODO.md`
- Evidence-pack status check for both source pack directories returned no local modifications.

## Tables Inserted

1. Table 1. Study Design and Claim Boundary
2. Table 2. Latent-Regime Construction Audit
3. Table 3. Hypothesis Evidence Summary
4. Table 4. Regime Information and Regime-Conditional Accuracy
5. Table 5. Transferability Summary
6. Table 6. H4 Strengthening Audit Summary
7. Table 7. References and Literature Role

Markdown table files were written under `tables/`.

## Figures Generated

1. Figure 1. Research Design Flow
2. Figure 2. Hypothesis Evidence Status Panel
3. Figure 3. Transfer Matrix Heatmap
4. Figure 4. H4 Robustness Count Panel
5. Figure 5. RD/FRD Claim Boundary Diagram

PNG files were written under `figures/`.

## Reference Style

- APA-style author-year prose citations in the manuscript body.
- Reference list title: `## References`
- Reference format: `Author(s). Year. "Title." Journal/Book/Proceedings volume(issue): pages. DOI/ISBN if verified.`
- Fama (1970) has no DOI because the citation TODO did not select a preferred DOI.
- Reference count: 22.

## Render Tools Used

- Markdown and table/figure generation: Python with pandas and matplotlib.
- DOCX rendering: Pandoc 3.9.0.2.
- PDF rendering: Pandoc standalone embedded HTML plus Microsoft Edge headless print-to-PDF using a browser file URI.

## Render Limitations

- No LaTeX, wkhtmltopdf, Typst, Graphviz, LibreOffice CLI, reportlab, python-docx, or WeasyPrint toolchain was available on PATH/import check.
- PDF was rendered through the available browser engine after DOCX rendering succeeded.
- No old DOCX/PDF files were edited; the final DOCX/PDF were newly created in the requested output directory.
- Visual page-overflow inspection was not available programmatically. The layout audit therefore used file-level checks: final Markdown table/figure references, DOCX table count, DOCX embedded media count, and DOCX/PDF nonzero file sizes.

## Claim-Boundary Confirmation

- Benchmark evidence remains discovery/diagnostic only.
- H1 is diagnostically supported.
- H2 is diagnostically supported.
- H3 is mixed / partially supported.
- H4 is partially supported only under a narrow robustness design and is not general support.
- RD and FRD are diagnostic distances, not causal mechanisms.
- Final-window outputs are descriptive scoring-only.
- Router 63.33% is not claim-eligible.
- Soft voting near 62.00% is not claim-eligible.
- Main bounded h40 result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage.
- Vietnam AMH evidence is contextual motivation only and does not replace this paper's VN30 stock-level evidence.
- International AMH evidence motivates time-varying predictability only and does not prove VN30 H4.

## Files Created

- `VN30_REGIME_TRANSFERABILITY_FINAL.md`
- `VN30_REGIME_TRANSFERABILITY_FINAL.docx`
- `VN30_REGIME_TRANSFERABILITY_FINAL.pdf`
- `VN30_REGIME_TRANSFERABILITY_FINAL_BUILD_NOTES.md`
- `tables/table1_study_design_claim_boundary.md`
- `tables/table2_latent_regime_construction_audit.md`
- `tables/table3_hypothesis_evidence_summary.md`
- `tables/table4_regime_information_and_accuracy.md`
- `tables/table5_transferability_summary.md`
- `tables/table6_h4_strengthening_audit_summary.md`
- `tables/table7_references_and_literature_role.md`
- `figures/figure1_research_design_flow.png`
- `figures/figure2_hypothesis_status_panel.png`
- `figures/figure3_transfer_matrix_heatmap.png`
- `figures/figure4_h4_robustness_count_panel.png`
- `figures/figure5_rd_frd_claim_boundary.png`

## Validation Output

- Exact title check: pass.
- Duplicate hypothesis table fix: pass; only Table 3 remains as the hypothesis table, and Section 6.1 now uses prose.
- Forbidden positive-claim phrase search: no matches.
- `[citation needed]` search in final markdown: no matches.
- Raw local repository path / GitHub URL search in final markdown: no matches.
- QML/qml search in final markdown: no matches.
- Table reference count in final Markdown: 7.
- Figure reference count in final Markdown: 5.
- DOCX table count: 7.
- DOCX embedded media count: 5.
- Brier DOI is escaped in Markdown as `10.1175/1520-0493(1950)078\<0001:VOFEIT\>2.0.CO;2`; Pandoc DOCX and HTML-to-PDF rendering completed successfully with this value.
- DOCX exists: true, size 276596 bytes.
- PDF exists: true, size 535568 bytes.
- Abstract word count: 264.
- Reference count: 22.
- Table count: 7.
- Figure count: 5.
- `python -m compileall scripts/research`: passed; output included `Listing 'scripts/research'...`
- `python scripts/check_repo_hygiene.py`: passed; output `Repository hygiene check passed.`
- `git diff --stat`: reports only pre-existing tracked generated-report changes, 15 files changed, 30 insertions(+), 30 deletions(-). New final paper outputs are untracked and therefore not shown in diff stat.
- `git status --short`: shows the pre-existing generated-report modifications, `?? reports/cleanup/V7_BROKEN_WORKTREE_NOTE.md`, and `?? reports/paper/final/`.

## Safety Confirmation

- No commit.
- No push.
- No tag.
- No QML edit.
- No market-data fetch.
- No provider/model/runtime logic change.
- No evidence-pack CSV edit.
