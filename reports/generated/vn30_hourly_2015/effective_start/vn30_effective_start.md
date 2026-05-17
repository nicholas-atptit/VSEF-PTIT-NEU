# VN30 2015 Effective Starts

- Rule: `effective_start(ticker) = max(2015-01-01, first_trading_date)`.
- Frozen tickers needing listing-date verification: BCM, BVH, TCB, TPB, VCB, VHM, VIB, VIC, VJC, VNM, VPB, VRE.

| ticker | first_trading_date | effective_start | reason | needs_verification |
|---|---|---|---|---:|
| `ACB` | 2006-11-21 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `BCM` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `BID` | 2014-01-24 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `BVH` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `CTG` | 2009-07-16 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `FPT` | 2006-12-13 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `GAS` | 2012-05-21 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `GVR` | 2018-03-21 | 2018-03-21 | `listed_after_2015_use_first_trading_date` | no |
| `HDB` | 2018-01-05 | 2018-01-05 | `listed_after_2015_use_first_trading_date` | no |
| `HPG` | 2007-11-15 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `LPB` | 2017-10-05 | 2017-10-05 | `listed_after_2015_use_first_trading_date` | no |
| `MBB` | 2011-11-01 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `MSN` | 2009-11-05 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `MWG` | 2014-07-14 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `PLX` | 2017-04-21 | 2017-04-21 | `listed_after_2015_use_first_trading_date` | no |
| `SAB` | 2016-12-06 | 2016-12-06 | `listed_after_2015_use_first_trading_date` | no |
| `SHB` | 2009-04-20 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `SSB` | 2021-03-24 | 2021-03-24 | `listed_after_2015_use_first_trading_date` | no |
| `SSI` | 2006-12-15 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `STB` | 2006-07-12 | 2015-01-01 | `listed_before_2015_use_2015_start` | no |
| `TCB` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `TPB` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VCB` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VHM` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VIB` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VIC` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VJC` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VNM` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VPB` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
| `VRE` |  | 2015-01-01 | `missing_first_trading_date_fallback_to_2015_or_provider_first_timestamp` | yes |
