# VN30 V6 Strategy Validation Repair Protocol

## Scope

- VN30 stock hourly strategy diagnostics only.
- Frozen family: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Validation diagnostics are reported before any final-period result.
- No broad model tuning, final-performance selection, VN100, DOCX, push, merge, or tag.

## Diagnostic Grid

- Thresholds: 0.40 to 0.65 step 0.005.
- Market regime filter: on/off.
- Volatility filter: on/off.
- Drawdown filter: on/off.
- Max positions: [3, 5].
- Diagnostic cost/slippage: 10 bps / 5 bps.

## Relock Repair Rule

- Validation relock is considered repaired only if validation trade generation is possible with >=10 trades, positive after-cost return, and strongest validation baseline comparison is reported.
- Final-period strategy results remain exploratory unless a future run is validation-governed and future-blind confirmed.
