# VN30 Stock + Index Joint Panel Training V1 Result

## Joint Universe

This run uses the corrected January 2025 VN30 joint panel:

- 30 active VN30 stock tickers: `ACB`, `BID`, `BCM`, `BVH`, `CTG`, `FPT`, `GAS`, `GVR`, `HDB`, `HPG`, `LPB`, `MBB`, `MSN`, `MWG`, `PLX`, `SAB`, `SHB`, `SSB`, `SSI`, `STB`, `TCB`, `TPB`, `VCB`, `VHM`, `VIB`, `VIC`, `VJC`, `VNM`, `VPB`, `VRE`.
- 6 supported market indices: `VNINDEX`, `VN30`, `HNXINDEX`, `HNX30`, `UPCOMINDEX`, `VN100`.
- Total instruments: 36.

Indices are prediction targets and panel rows, not merely context features.

## Execution Status

- Readiness audit: passed for a validation-safe 36/36 hourly panel.
- Joint panel training v1: run.
- Candidate selection: validation-only.
- Final evaluation: scoring-only.
- Confidence abstention: no.
- Instrument subset: no.
- Top-k/ranking substitution: no.
- Data fetch: no.
- Paper/DOCX generated: no.
- Trading/profitability/live-deployment claim: no.

## Baselines

- Best deterministic combined baseline: majority/always-up h=120 at 59.40%.
- Selected-candidate horizon baseline: majority h=40 combined 53.65%, stock-only 52.45%, index-only 55.55%.
- Stock-only RF h=60 historical reference: 60.31%; reference only, not a joint-panel result and not a replacement for combined 36-instrument scoring.
- Existing index benchmark results are reference-only and cannot replace combined 36-instrument scoring.

## Training Result

- Best validation-selected candidate: `lightgbm__h40__own_plus_market_context`.
- Best model: LightGBM.
- Horizon: h=40.
- Feature set: `own_plus_market_context`.
- Validation combined accuracy: 62.66%.
- Final combined accuracy: 49.21%.
- Final stock-only accuracy: 48.99%.
- Final index-only accuracy: 49.56%.
- Final coverage: 100.00%.
- Final combined rows: 14,300.
- Final stock-only rows: 8,757.
- Final index-only rows: 5,543.
- Combined baseline comparison: 49.21% vs 53.65%, delta -4.44 percentage points.
- Stock-only baseline comparison: 48.99% vs 52.45%, delta -3.46 percentage points.
- Index-only baseline comparison: 49.56% vs 55.55%, delta -5.99 percentage points.
- Combined >=60 reached: no.
- Combined >=65 reached: no.
- Stock-only >=60 reached: no.
- Stock-only >=65 reached: no.

## Index-Row Effect

Index rows did not materially lift the result to a useful combined benchmark claim. Final index-only accuracy was 49.56%, slightly above stock-only accuracy at 48.99%, and final combined accuracy was 49.21%; the combined score remained below the selected-candidate h=40 combined baseline.

## Risk And Claim Level

- Overfit risk: high.
- Evidence: validation combined accuracy was 62.66%, but final combined accuracy fell to 49.21%.
- Audit status: exploratory.
- Claim level: exploratory, with no benchmark success claim.

Because prior final windows have been inspected repeatedly, any future improvement from this branch should remain exploratory unless later verified on future blind data.

No trading, profitability, investment-recommendation, or live-deployment claim is made.
