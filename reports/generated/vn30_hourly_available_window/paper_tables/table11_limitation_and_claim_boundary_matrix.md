# Table 11: Limitation and claim boundary matrix

| area | status | evidence | claim_boundary |
| --- | --- | --- | --- |
| Full-history requirement | not_satisfied | External hourly 2005-2026 data is still required. | Do not claim 2005-2026 full-history VN30 evidence. |
| Constituent coverage | subset | 27/30 tickers selected. | No full VN30 representativeness claim unless selected tickers are 30/30. |
| Frequency | hourly_only | Local hourly files only. | Daily data and daily-to-hourly resampling are excluded. |
| Leakage | declared | target_timestamp <= train_cutoff | Training labels are bounded by the selected train cutoff. |
| Trading readiness | not_claimed | Cost/slippage outputs are proxy diagnostics. | No trading readiness claim without execution-ready evidence. |

## Note

Claim boundaries for the available-window VN30 hourly study.
