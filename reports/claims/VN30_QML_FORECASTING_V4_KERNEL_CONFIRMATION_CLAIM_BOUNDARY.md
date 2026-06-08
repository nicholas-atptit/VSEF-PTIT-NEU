# VN30 QML Forecasting V4 Kernel Confirmation Claim Boundary

- QML v4 kernel confirmation is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature scaling, PCA, compression, and balanced sampling must be train-only or validation-safe.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Final-ranked rows remain exploratory_not_claimable.
- Quantum-kernel rows are compared against same-sample RBF SVM, linear SVM, Logistic Regression, and calibrated Logistic Regression.
- No QML result replaces the 61.61% L2 Logistic classical champion.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require a full validation-governed benchmark and future-blind confirmation.
