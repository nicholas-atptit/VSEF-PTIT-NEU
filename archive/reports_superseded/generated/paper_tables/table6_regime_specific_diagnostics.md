# Table 6: Regime-Specific Diagnostics

| frequency | regime | best_model | horizon | n_obs | accuracy | passed_60pct | reliable | top_ticker_by_contribution | top_ticker_contribution_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| daily | bear | lightgbm | 20 | 444 | 0.6959459459459459 | true | true | BVH | 0.19369369369369369 |
| daily | bull | random_forest | 5 | 629 | 0.5325914149443561 | false | true |  |  |
| daily | high_volatility | random_forest | 20 | 622 | 0.6093247588424437 | true | true |  |  |
| daily | low_volatility | lightgbm | 20 | 413 | 0.5907990314769975 | false | true |  |  |
| daily | sideways | xgboost | 10 | 560 | 0.5732142857142857 | false | true |  |  |
| hourly | bear | random_forest | 20 | 1277 | 0.5559906029757243 | false | true |  |  |
| hourly | bull | stacking | 1 | 1445 | 0.5633217993079584 | false | true |  |  |
| hourly | high_volatility | stacking | 1 | 2012 | 0.5651093439363817 | false | true | BID | 0.16302186878727634 |
| hourly | low_volatility | random_forest | 1 | 2728 | 0.5608504398826979 | false | true |  |  |
| hourly | sideways | stacking | 1 | 4760 | 0.557563025210084 | false | true |  |  |

## Note

- Source artifact: daily/hourly regime_accuracy_summary.csv; generated ticker concentration summary.
- Claim supported: Regime diagnostics show conditional signal, especially daily bear-regime h=20, without proving global performance.
- Limitation: Regime findings are post-hoc diagnostics from one official window unless validated ex ante.
- Status: ready.
