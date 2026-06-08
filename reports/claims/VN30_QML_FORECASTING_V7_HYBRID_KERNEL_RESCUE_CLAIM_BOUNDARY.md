# VN30 QML Forecasting V7 Hybrid Kernel Rescue Claim Boundary

- QML v7 hybrid kernel rescue is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature selection, scaling, PCA, rank transforms, kernel construction, and kernel-feature transforms must be train-only or validation-safe.
- Distribution matching must not use final labels for sample selection.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Final-ranked rows remain exploratory_not_claimable.
- Same-target comparisons are diagnostic and do not replace the 61.61% L2 Logistic classical champion because the target/scope is not directly identical.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require full validation governance and future-blind confirmation.
