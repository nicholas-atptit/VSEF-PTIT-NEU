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
