**Table 4. Regime Information and Regime-Conditional Accuracy**

| Diagnostic | Value | Role | Interpretation |
| --- | --- | --- | --- |
| RIG (validation) | 0.403333 | primary | Primary diagnostic evidence: same-regime log-loss is lower than global log-loss, giving positive validation RIG. |
| RIG (final) | 0.429032 | descriptive_only | Descriptive final-window scoring only; not selection evidence and not promoted beyond context. |
| Accuracy association (validation) | gap 0.085666; chi-square 121.98; p 3.25e-27 | primary | Primary diagnostic evidence: global-model correctness differs across latent regimes in validation. |
| Accuracy association (final) | gap 0.118453; chi-square 44.28; p 2.4232e-10 | descriptive_only | Descriptive final-window scoring only; not selection evidence. |
