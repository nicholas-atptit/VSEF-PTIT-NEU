# VN30 Hourly RF h60 Baseline Reproduction Result

## Summary

- Reproduced inside current expanded-model pipeline: no.
- Historical canonical RF h60 accuracy: 60.31%.
- Current expanded-pipeline RF h60 closest accuracy: 56.12%.
- Accuracy difference: -4.19 percentage points.
- Historical final rows: 3,474.
- Current final rows: 8,637.
- Row count difference: +5,163 current rows.
- Data fetch: no.
- New model family training: no.
- Broad tuning: no.
- Paper/DOCX generated: no.
- Tags: no.

## Checks

- Same universe: yes, 30/30 active VN30 January 2025 stocks.
- Same horizon: yes, h=60.
- Same target definition: yes, absolute direction, `future_close > current_close`.
- Same final window: no.
- Same row count: no.
- Same feature set: no; the current closest feature set is `stock_lagged_rolling_plus_index_context`, but it is not the historical feature set C implementation.

## Difference Explanation

The locked historical RF h60 baseline is a canonical/historical benchmark with 3,474 final rows and a fixed calendar final window beginning 2025-01-01. The current expanded-model pipeline uses a per-ticker 60/20/20 chronological split over the current available cache, producing 8,637 final rows for RF h60.

The current closest RF h60 configuration includes stock lagged/rolling features plus lagged index context and reaches 56.12%, not 60.31%. Because the split/window, row count, and feature implementation differ, this is not a reproduction of the locked result.

## Baseline Decision

- The 60.31% RF h60 result remains valid as the locked historical baseline in its original canonical setup.
- It should not be treated as apples-to-apples with the current expanded-model pipeline unless the current pipeline is refit to the same calendar split, row set, target construction, and feature set C implementation.
- The expanded model-pool + stacking v1 failure is only partially comparable to the locked baseline: universe, horizon, and target align, but final window, row count, split design, and feature implementation do not.

No trading, profitability, investment-recommendation, or live-deployment claim is made.

