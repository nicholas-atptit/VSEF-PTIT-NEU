# Stock Pipeline Governance Audit (2026-04-17)

## Scope
This phase hardened governance only: canonical feature registry, prune proposal, approved task-specific feature sets, artifact-level QA contracts, and explicit raw-vs-adjusted close semantics.

## Current State
- The pipeline remains vnstock_data-first.
- Feature engineering still computes the broader research surface, including deprecated and experimental columns for backward compatibility.
- Training now uses a governed shared base feature union instead of the unrestricted canonical feature list.

## What Changed
- Added `src/ml/features/feature_registry.json` as the canonical registry.
- Added explicit approved sets for `forecast_core_features`, `classification_signal_features`, `regime_features`, and `risk_features`.
- Added loader-level data quality contracts covering provenance, duplicate keys, date monotonicity, missing ratio, artifact staleness, and unsupported-source flagging.
- Added explicit price semantics: `model_close_reference`, `raw_close`, deprecated alias `close_raw`, and conditional `adjusted_close` only when upstream data is explicitly adjusted.

## Registry Summary
- Total registered features: 381
- Status counts: {'experimental': 195, 'active': 160, 'deprecated': 26}
- Category counts: {'price_volume_core': 236, 'technical_indicator': 76, 'market_context': 25, 'flow_microstructure': 14, 'macro_cross_asset': 12, 'regime_state': 7, 'sentiment_news': 6, 'risk_layer': 5}
- Governed shared base feature count: 46

## Prune Summary
- Primary deprecation target is the automatic `d_*` expansion.
- Legacy aliases and compatibility-only indicators are explicitly deprecated rather than silently tolerated.
- Raw macro levels and weak research-only transforms remain available but are not approved defaults.

## Data Quality Notes
- Local audit universe used 12 cached tickers from `data/daily_market_split_data`.
- Highest local missing ratios were structural warmup/context effects rather than leakage: `rolling_corr_sector_20`=48.4%, `sma_200`=16.1%, `close_to_sma_200`=16.1%.
- No near-constant features were detected in the local numeric scan after excluding unsupported context sources.

## Remaining Limitations
- Macro and foreign-flow artifacts are still optional; approved sets resolve against available columns rather than forcing unsupported inputs.
- The multi-task trainer still uses one governed shared base union for forecast/classification rather than three fully separate runtime matrices.
- Feature generation still emits pandas fragmentation warnings; correctness is unaffected, but performance can be optimized later.
