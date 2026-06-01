**Table 2. Latent-Regime Construction Audit**

| Audit item | Value | Interpretation |
| --- | --- | --- |
| Method | train-only 3-regime GaussianMixture | Fit on training timestamps only. |
| Horizon | h40 | Primary latent-regime diagnostics in this pack. |
| Training timestamps | 259 | Unique training dates used for GMM fit. |
| State features | Lagged return, volatility, breadth, dispersion, and volume-shock context. | No external economic labels. |
| Lagged variables | yes | State columns are ex-ante or lagged. |
| Leakage audit | no target leakage flags in feature-family audit | No target/future/final leakage flags reported. |
| Final outputs | yes | Scoring-only; not used for selection. |
