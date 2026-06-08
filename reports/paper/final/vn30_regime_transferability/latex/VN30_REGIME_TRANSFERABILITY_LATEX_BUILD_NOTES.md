# VN30 Regime Transferability LaTeX Build Notes

## Source File Used

- Source complete paper: `../VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.md`
- LaTeX output file: `VN30_REGIME_TRANSFERABILITY_LATEX_COMPLETE.tex`
- BibTeX output file: `references.bib`
- Title preserved exactly: `Regime-Dependent Transferability of Machine Learning Forecasts in an Emerging Equity Market: Evidence from VN30 Walk-Forward Benchmarks`

## LaTeX Engine

- Render attempt date: 2026-06-01.
- `latexmk`: unavailable on PATH.
- `pdflatex`: unavailable on PATH.
- `bibtex`: unavailable on PATH.
- `biber`: unavailable on PATH.
- `xelatex`: unavailable on PATH.
- `lualatex`: unavailable on PATH.
- Engine used: none; no local LaTeX engine was available.
- PDF created: no.
- PDF file size: unavailable because `VN30_REGIME_TRANSFERABILITY_LATEX_COMPLETE.pdf` was not created.
- PDF render status: `skipped_with_reason`; `latexmk -pdf VN30_REGIME_TRANSFERABILITY_LATEX_COMPLETE.tex` failed because `latexmk` was not recognized, and `pdflatex VN30_REGIME_TRANSFERABILITY_LATEX_COMPLETE.tex` failed because `pdflatex` was not recognized.
- Page count: unavailable because the PDF was not rendered.

## Files Created

- `VN30_REGIME_TRANSFERABILITY_LATEX_COMPLETE.tex`
- `references.bib`
- `VN30_REGIME_TRANSFERABILITY_LATEX_BUILD_NOTES.md`

## Manuscript Structure

- Abstract
- Introduction
- Literature Review
- Conceptual Framework
- Hypotheses
- Data and Methodology
- Results
- Discussion
- Limitations and Future Work
- Conclusion
- References
- Appendices

## Formula Fixes

- Section 3.2 includes proper LaTeX equations for \(p_{i,t+h}\), the regime-conditional probability inequality, and \(FS_{t+h} = g(M, X_t, R_t, \mathcal{D}_t)\).
- Section 3.2 explains \(Y_{t+h}\), \(X_t\), \(R_t\), \(M\), \(\mathcal{D}_t\), and \(FS_{t+h}\).
- Section 3.6 includes proper LaTeX equations for RIG, log-loss, TRR, TG, RD, RD\_cosine, and FRD.
- Section 3.6 explains \(\mathcal{L}^{val}_{global}\), \(\mathcal{L}^{val}_{regime}\), \(y_n\), \(\hat{p}_n\), \(\mathrm{Acc}_{i \rightarrow j}\), \(\bar{z}_i\), \(\bar{z}_j\), \(\hat{\beta}_i\), and \(\hat{\beta}_j\).

## Counts and Checks

- Abstract word count: 307.
- Table count in LaTeX body: 7.
- Figure count in LaTeX body: 5.
- Reference count in `references.bib`: 22.
- Citation-placeholder search: no matches.
- Interface markup filename search: no matches.
- Forbidden positive-claim phrase search: no matches.
- Section 3.2 broken-placeholder check: no `for:`, no `where:`, and no empty display equations.
- Section 3.6 formula presence check: pass.

## Claim-Boundary Confirmation

- Benchmark evidence remains discovery/diagnostic only.
- H1 remains diagnostically supported.
- H2 remains diagnostically supported.
- H3 remains mixed / partially supported.
- H4 remains partially supported only under a narrow K=4 ticker-level robustness design and is not general support.
- RD and FRD remain diagnostic distances, not causal mechanisms.
- Final-window outputs remain descriptive scoring-only.
- Router 63.33% remains not claim-eligible.
- Soft voting near 62.00% remains not claim-eligible.
- Main bounded h40 result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage.
- Vietnam AMH evidence is contextual motivation only and does not replace this paper's VN30 stock-level evidence.
- International AMH evidence motivates time-varying predictability only and does not prove VN30 H4.
- No trading, profitability assertion, investment recommendation, production-deployment, unreported score-target, or broad generalization claim was added.

## Validation Output

- Exact title in `.tex`: pass.
- `.tex` exists: true.
- `.bib` exists: true.
- PDF exists and nonzero: false.
- PDF page count: unavailable.
- Abstract mandatory numeric facts: pass.
- `python -m compileall scripts/research`: passed; output included `Listing 'scripts/research'...`
- `python scripts/check_repo_hygiene.py`: passed; output `Repository hygiene check passed.`

## Limitations

- The target PDF was not rendered because no LaTeX engine was available on PATH.
- Page count could not be measured for the LaTeX manuscript.
- The approximately 35-45 page target could not be verified without a rendered PDF.
- Existing tables and figures were reused; no new empirical tables, figures, metrics, citations, or results were created.
