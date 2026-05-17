# VN30 vnstock Hourly Fetch Diagnostics

## Scope

- Frozen universe: `configs/universes/vn30_constituents_frozen.csv`.
- Symbols probed: 31 total (30 VN30 tickers plus VNINDEX).
- Provider sources discovered from installed `vnstock`: Quote registry=FMP, KBS, MSN, VCI; Vnstock client=KBS, VCI, MSN.
- Provider calls use the registered quote provider classes directly; the Vnstock client is used only for the VNINDEX world-index probe.
- Intervals tested: 1H, 60m, 1h, hourly.
- Windows tested: 2024-01-02 to 2024-01-05, 2024-12-02 to 2024-12-06, 2025-01-02 to 2025-01-06, 2026-05-04 to 2026-05-13.
- Full attempt CSV: `reports/generated/vn30_hourly_listing_aware/provider_diagnostics/vn30_vnstock_hourly_fetch_diagnostics.csv`.
- This diagnostic does not treat sample-window support as full-history support.

## Direct Answers

- Which tickers return any hourly rows? 30/30: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE.
- Which tickers return rows in 2024? 29/30: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.
- Which tickers return rows in 2025? 29/30: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.
- Which tickers return rows in 2026? 30/30: ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE.
- Does VNINDEX return hourly rows? yes (returned years: 2023, 2024, 2025, 2026).
- Does the provider appear to limit hourly history to recent dates? Not clearly from these probes. Old-window rows were observed for most tickers and VNINDEX, but this still does not prove full listing-aware history.
- Is full listing-aware benchmark feasible with this provider? Not yet established. Sample coverage is broader than the current partial cache, but the full listing-aware benchmark is not feasible until a complete normalized cache passes the row-count and VNINDEX validation gates.
- Next required action: Do not run the listing-aware benchmark from the current partial cache. Repair the listing-aware fetch path to use the registered quote providers and supported hourly intervals observed here, rerun the hourly-only cache build with conservative throttling, then validate the normalized cache. If that still fails the row-count or VNINDEX gates, acquire an external hourly source rather than using daily data or resampling.

## Stock Coverage Summary

| question | count | tickers |
| --- | --- | --- |
| Any hourly rows | 30/30 | ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE |
| Rows in 2024 | 29/30 | ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE |
| Rows in 2025 | 29/30 | ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE |
| Rows in 2026 | 30/30 | ACB, BID, CTG, DGC, FPT, GAS, GVR, HDB, HPG, LPB, MBB, MSN, MWG, PLX, SAB, SHB, SSB, SSI, STB, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VPL, VRE |

## Provider Combination Summary

| provider | source | entrypoint | interval | successful_symbols | successful_vn30_tickers | diagnoses |
| --- | --- | --- | --- | --- | --- | --- |
| vnstock.ProviderRegistry.quote | KBS | ProviderRegistry.get('quote').history | 1H | 31 | 30 | hourly_rows_returned:34; provider_error:90 |
| vnstock.ProviderRegistry.quote | KBS | ProviderRegistry.get('quote').history | 1h | 31 | 30 | hourly_rows_returned:34; provider_error:90 |
| vnstock.ProviderRegistry.quote | KBS | ProviderRegistry.get('quote').history | 60m | 31 | 30 | hourly_rows_returned:34; unsupported_interval:90 |
| vnstock.ProviderRegistry.quote | VCI | ProviderRegistry.get('quote').history | 1H | 31 | 30 | hourly_rows_returned:121; unsupported_ticker:3 |
| vnstock.ProviderRegistry.quote | VCI | ProviderRegistry.get('quote').history | 1h | 31 | 30 | hourly_rows_returned:121; unsupported_ticker:3 |
| vnstock.ProviderRegistry.quote | FMP | ProviderRegistry.get('quote').history | 1H | 0 | 0 | auth_error:124 |
| vnstock.ProviderRegistry.quote | FMP | ProviderRegistry.get('quote').history | 1h | 0 | 0 | auth_error:124 |
| vnstock.ProviderRegistry.quote | FMP | ProviderRegistry.get('quote').history | 60m | 0 | 0 | auth_error:124 |
| vnstock.ProviderRegistry.quote | FMP | ProviderRegistry.get('quote').history | hourly | 0 | 0 | auth_error:124 |
| vnstock.ProviderRegistry.quote | KBS | ProviderRegistry.get('quote').history | hourly | 0 | 0 | unsupported_interval:124 |
| vnstock.ProviderRegistry.quote | MSN | ProviderRegistry.get('quote').history | 1H | 0 | 0 | unsupported_index:4; unsupported_ticker:120 |
| vnstock.ProviderRegistry.quote | MSN | ProviderRegistry.get('quote').history | 1h | 0 | 0 | unsupported_index:4; unsupported_ticker:120 |
| vnstock.ProviderRegistry.quote | MSN | ProviderRegistry.get('quote').history | 60m | 0 | 0 | unsupported_index:4; unsupported_ticker:120 |
| vnstock.ProviderRegistry.quote | MSN | ProviderRegistry.get('quote').history | hourly | 0 | 0 | unsupported_index:4; unsupported_ticker:120 |
| vnstock.ProviderRegistry.quote | VCI | ProviderRegistry.get('quote').history | 60m | 0 | 0 | unsupported_interval:124 |
| vnstock.ProviderRegistry.quote | VCI | ProviderRegistry.get('quote').history | hourly | 0 | 0 | unsupported_interval:124 |
| vnstock.Vnstock.world_index | MSN | Vnstock.world_index.quote.history | 1H | 0 | 0 | provider_error:4 |
| vnstock.Vnstock.world_index | MSN | Vnstock.world_index.quote.history | 1h | 0 | 0 | provider_error:4 |
| vnstock.Vnstock.world_index | MSN | Vnstock.world_index.quote.history | 60m | 0 | 0 | provider_error:4 |
| vnstock.Vnstock.world_index | MSN | Vnstock.world_index.quote.history | hourly | 0 | 0 | provider_error:4 |

## Diagnosis Counts

| diagnosis | attempts |
| --- | --- |
| auth_error | 496 |
| hourly_rows_returned | 344 |
| provider_error | 196 |
| unsupported_index | 16 |
| unsupported_interval | 462 |
| unsupported_ticker | 486 |

## Interpretation

- Success means the provider returned standardized hourly OHLCV rows for a small probe window only.
- A ticker that succeeds in one sample window is not considered benchmark-usable.
- Benchmark usability still requires the listing-aware normalized cache to pass the training-row, evaluation-row, and VNINDEX gates.
