# Table 1: Dataset and Evaluation Scope

| item | value | detail |
| --- | --- | --- |
| Universe | VN100 | Official benchmark universe. |
| Raw daily cache request range | 2006-01-01 to 2015-12-31 | Manifest raw daily range: 2006-01-01 to 2015-12-31. |
| Raw hourly cache request range | 2016-01-01 to 2025-12-31 | Manifest raw hourly range: 2016-01-01 to 2025-12-31. |
| Training-label cutoff | 2024-12-31 | Rule: target_timestamp <= train_cutoff. |
| Official evaluation window | 2025-01-01 to 2025-12-31 | Held-out 2025 target outcomes. |
| Effective daily evaluation range | 2025-01-02 to 2025-12-31 | Daily predictions: 26104. |
| Effective hourly evaluation range | 2025-01-02 to 2025-12-31 | Hourly predictions: 127944. |
| Evaluated tickers | ANV, BCM, BID, BMP, BVH, BWE, CII | 7 tickers evaluated in official summaries. |
| Daily benchmark-usable cache rows | 0 of 104 | Usable tickers: none; actual range: n/a to n/a. |
| Hourly benchmark-usable cache rows | 7 of 104 | Usable tickers: ANV, BCM, BID, BMP, BVH, BWE, CII; actual range: 2024-01-02 to 2025-12-31. |

## Note

- Source artifact: run_config.json; manifest.json; usable_cache_summary.csv; daily/hourly benchmark_summary.json.
- Claim supported: The official run is a 2025 held-out VN100 benchmark with limited usable-cache coverage.
- Limitation: Standalone daily cache rows are not benchmark-usable; the daily benchmark uses hybrid resampled inputs.
- Status: ready.
