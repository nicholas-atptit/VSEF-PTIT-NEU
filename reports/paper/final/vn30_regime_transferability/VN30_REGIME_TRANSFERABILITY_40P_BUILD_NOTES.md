# VN30 Regime Transferability 40P Build Notes

## Source File Used

- Source Markdown: `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.md`
- Source DOCX: `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.docx`
- Source PDF: `VN30_REGIME_TRANSFERABILITY_FINAL_COMPLETE.pdf`
- Expanded Markdown output: `VN30_REGIME_TRANSFERABILITY_40P_COMPLETE.md`

## Target Output Files

- `VN30_REGIME_TRANSFERABILITY_40P_COMPLETE.md`
- `VN30_REGIME_TRANSFERABILITY_40P_COMPLETE.docx`
- `VN30_REGIME_TRANSFERABILITY_40P_COMPLETE.pdf`
- `VN30_REGIME_TRANSFERABILITY_40P_BUILD_NOTES.md`
- Intermediate HTML used for browser PDF rendering: `VN30_REGIME_TRANSFERABILITY_40P_COMPLETE.html`

## Page Target and Actual Page Count

- Target: 38-42 rendered PDF pages, preferably close to 40.
- Actual measured PDF page count: 39 pages.
- Page-count method: strict PDF byte-pattern count for page objects, `/Type\s*/Page\b`.
- Page target status: pass, measurable and inside 38-42.

## Render Tools Used

- DOCX rendering: Pandoc 3.9.0.2.
- HTML rendering: Pandoc 3.9.0.2, standalone HTML5 with MathML.
- PDF rendering: Microsoft Edge headless print-to-PDF from the generated HTML file.
- PDF render note: Edge print-to-PDF required an approved escalated run because the sandboxed browser invocation exited without writing the PDF.
- Formatting: A4 print CSS, 12pt Georgia/Times-compatible academic serif font, academic margins, controlled line spacing.
- LaTeX was not used.

## Counts

- Abstract word count: 339.
- Table count: 7.
- Figure count: 5.
- Reference count: 22.

## Formula Checks

- Section 3.2 includes:
  - `p_{i,t+h} = \Pr(Y_{t+h}=1 \mid X_t, R_t=i)`
  - `\exists i \neq j: \Pr(Y_{t+h}=1 \mid X_t, R_t=i) \neq \Pr(Y_{t+h}=1 \mid X_t, R_t=j)`
  - `FS_{t+h} = g(M, X_t, R_t, \mathcal{D}_t)`
- Section 3.6 includes:
  - RIG equation.
  - Log-loss equation.
  - TRR equation.
  - TG equation.
  - RD equation.
  - RD_cosine equation.
  - FRD equation.
- Empty display-equation check: pass.
- Broken placeholder search for `for:`: no matches.

## Claim-Boundary Confirmation

- Benchmark evidence remains discovery and diagnostic only.
- H1 remains diagnostically supported.
- H2 remains diagnostically supported.
- H3 remains mixed / partially supported.
- H4 remains partially supported only under a narrow K=4 ticker-level robustness design and is not general support.
- RD and FRD remain diagnostic distances, not causal mechanisms.
- Final-window outputs remain descriptive scoring-only.
- Router 63.33% remains not claim-eligible.
- Soft voting near 62.00% remains not claim-eligible.
- Main bounded h40 result remains Logistic L2 using the C-closest reference feature set, validation-selected threshold 0.55, final accuracy 61.63%, and full 30-stock coverage.
- Vietnam AMH evidence remains contextual motivation only and does not replace this paper's VN30 stock-level evidence.
- International AMH evidence motivates time-varying predictability only and does not prove VN30 H4.
- No trading, profitability, investment, production-deployment, live-deployment, unreported score-target, or broad generalization claim was added.

## Validation Output

- Exact title check: pass.
- 40P Markdown exists: pass.
- DOCX exists and is nonzero: pass, 289551 bytes.
- PDF exists and is nonzero: pass, 610907 bytes.
- PDF page count: 39.
- Page count 38-42: pass.
- Abstract word count 300-400: pass, 339.
- Table count = 7: pass.
- Figure count = 5: pass.
- Reference count = 22: pass.
- `[citation needed]` search: no matches.
- QML/qml search: no matches.
- Raw local repo path or GitHub URL in paper body: no matches.
- Forbidden positive-claim phrase search: no matches.
- Formula sections have clean equations: pass.
- Empty equations or malformed display-equation placeholders: no matches found.
- `python -m compileall scripts/research`: passed.
- `python scripts/check_repo_hygiene.py`: passed.

## Limitations

- PDF page count was measured programmatically through strict PDF page-object matching because no dedicated `pdfinfo`, `qpdf`, or `mutool` command was available on PATH.
- The generated HTML file is an intermediate render artifact needed for the successful browser PDF path.
- Existing unrelated worktree modifications were present before this task and were not changed by the 40P build.

## Safety Confirmation

- No push.
- No tag.
- No staging.
- No commit.
- No market-data fetch.
- No QML edit.
- No provider/model/runtime logic change.
- No evidence-pack CSV edit.
