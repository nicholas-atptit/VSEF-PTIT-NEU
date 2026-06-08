# VN30 Regime Transferability Final Complete Build Notes

## Source File Used

- `VN30_REGIME_TRANSFERABILITY_FINAL.md`
- Complete output file: `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.md`
- Title preserved exactly: `Regime-Dependent Transferability of Machine Learning Forecasts in an Emerging Equity Market: Evidence from VN30 Walk-Forward Benchmarks`

## Added Discussion Subsection

- Added `### 7.1 Theoretical Interpretation of Partial Support` under Section 7 Discussion after the H4 discussion paragraph.
- Section 7.1 word count: 528.
- The subsection interprets the pattern H1/H2 supported, H3 mixed/partial, and H4 narrow partial support as bounded theoretical support rather than rejection of the regime-transferability framework.
- It keeps latent regimes as reconstructed diagnostic states, treats Market State Distance as a plausible but incomplete diagnostic mechanism, and identifies future-work directions including liquidity shocks, investor sentiment, macroeconomic conditions, algorithm-specific sensitivity, richer distance measures, and future-blind validation.

## Render Tools

- DOCX rendering: Pandoc 3.9.0.2.
- PDF rendering: Pandoc standalone embedded HTML plus Microsoft Edge headless print-to-PDF through a browser file URI.
- Existing tables and figures were reused; no new tables, figures, metrics, citations, or results were created.

## Files Created

- `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.md`
- `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.docx`
- `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.pdf`
- `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE_BUILD_NOTES.md`

## Validation Results

- Exact title check: pass.
- Section 7.1 exists: pass.
- Table count in final complete Markdown: 7.
- Figure count in final complete Markdown: 5.
- Reference count: 22.
- `[citation needed]` search: no matches.
- QML/qml search: no matches.
- Raw local repository path / GitHub URL search in paper body: no matches.
- Forbidden positive-claim phrase search: no matches after replacing the exact `live deployment` phrase in the new subsection with `production deployment`.
- DOCX exists and is nonzero: true, 277800 bytes.
- PDF exists and is nonzero: true, 544347 bytes.
- `python -m compileall scripts/research`: passed.
- `python scripts/check_repo_hygiene.py`: passed.

## Claim-Boundary Confirmation

- H1 remains diagnostically supported.
- H2 remains diagnostically supported.
- H3 remains mixed / partially supported.
- H4 remains partially supported only under a narrow K=4 ticker-level robustness design and is not general support.
- RD and FRD remain diagnostic distances, not causal mechanisms.
- Final-window outputs remain descriptive scoring-only.
- Router 63.33% remains not claim-eligible.
- Soft voting near 62.00% remains not claim-eligible.
- Main bounded h40 result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage.
- No trading, profitability, investment, live-deployment, unreported score-target, or broad generalization claim was added.

## Safety Confirmation

- No push.
- No tag.
- No QML edit.
- No market-data fetch.
- No provider/model/runtime logic change.
- No evidence-pack CSV edit.
