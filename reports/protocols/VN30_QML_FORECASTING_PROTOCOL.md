# VN30 QML Forecasting Protocol

- Quantum Machine Learning forecasting is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- QML candidates must compare against the current classical champion: L2 Logistic, feature_set_C_closest, h40, threshold 0.50, 61.61% final accuracy, +10.90pp lift.
- QML dependencies are optional and must be dependency-guarded so the baseline research pipeline remains runnable without QML packages.
- No QML result can replace the classical champion unless the result is validation-governed and split-safe.
- No trading, profitability, BUY/SELL, investment recommendation, live deployment, DOCX, tag, or merge claim is made.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- All QML experiments must enforce feature_timestamp and target_timestamp split discipline.
- QML feature compression must use train-only fit transforms.
- Candidate QML families:
  - quantum kernel classifier
  - variational quantum classifier
  - hybrid QNN, optional
- Required future artifact directory: `reports/generated/vn30_qml_forecasting/`

## V1 Diagnostic Design

- Required split discipline:
  - train rows require `feature_timestamp <= 2023-12-31 23:59:59` and `target_timestamp <= 2023-12-31 23:59:59`;
  - validation rows require both timestamps inside calendar year 2024;
  - final rows require both timestamps on or after `2025-01-01 00:00:00`.
- Target variants must be evaluated separately:
  - `absolute_direction`
  - `market_relative_vn30`
  - `market_relative_vnindex`
- Horizons must be evaluated separately: h20, h40, h50, h60.
- Index data may be used only as lagged market-context features or market-relative target context; this is not an index-as-stock experiment.
- Feature compression must be train-only:
  - top-k by train-split feature availability;
  - mutual-information top-k fit on train only;
  - PCA fit on train only.
- Classical baselines must run even when QML dependencies are missing.
- QML execution must use local CPU simulation only and must skip gracefully when optional APIs are unavailable or incompatible.
- Candidate selection must be validation-governed only; final accuracy must not rank claimable rows.
