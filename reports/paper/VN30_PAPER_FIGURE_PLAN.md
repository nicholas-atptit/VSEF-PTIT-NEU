# VN30 Paper Figure Plan

## Main paper figures

1. **Benchmark comparison summary across methods and horizons.** Panels: VN30 stock-only selected candidate, supported indices, and joint stock+index panel summary. The joint panel is included only as exploratory summary evidence because valid row-level and per-instrument joint artifacts were not found and the final combined score is below 60%.
2. **Selected best candidate: actual vs predicted over time.** Uses the fixed L2 Logistic h40 selected candidate row-level reproduction. Includes a final-window actual/predicted direction panel and a rolling-accuracy panel.
3. **Forecast-vs-actual comparison on representative instruments.** Representative stocks: MBB, GAS, ACB, MWG, VIC, SHB; representative indices: VNINDEX, VN30. Uses L2 Logistic/logistic regression, Random Forest, XGBoost, and LightGBM where row-level artifacts exist. Stacking is documented as unavailable at row level.
4. **Per-instrument accuracy distribution.** Sorted stock-only and index-only panels.
5. **Stability figure.** Rolling 250/500/1000 row accuracy curves plus monthly and quarterly summaries for the fixed selected candidate.
6. **Model-vs-actual overlay for one representative stock and one representative index.** Uses one representative stock and VNINDEX; no smoothing.

## Appendix figures

- **Appendix A.** Small multiples for all 30 VN30 stocks under the selected stock-only candidate, plus supported indices using the available h40 XGBoost index benchmark proxy.
- **Appendix B.** One overlay figure per representative instrument across available methods.
- **Appendix C.** Stock and index performance ranking figures/tables.

## Style rules

All figures use white backgrounds, readable labels, true binary direction scales for overlays, and explicit benchmark threshold lines where useful. No smoothing is applied beyond rolling windows that are already documented as rolling accuracy summaries.
