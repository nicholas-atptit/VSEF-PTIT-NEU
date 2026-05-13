# VN30 Hourly Available-Window Market Index Audit

## Scope

- Exact local hourly index codes audited: `VNINDEX`, `VN30INDEX`, `VNXALL`.
- No aliases are used. Local stock symbols such as `VNI` or `VNX` are not treated as market indices.
- Daily data, daily-to-hourly resampling, and fabricated index context are not used.
- Selected stock available-window period: 2025-12-15 09:00:00 to 2026-05-11 14:00:00.

## Result

- Available-window index context ready: 0 of 3 exact codes.
- If exact local index data is missing, this is a limitation rather than a reason to fabricate context.

## Per-Index Audit

| index_code | exact_code_local_file_found | first_available_hourly_timestamp | last_available_hourly_timestamp | selected_window_rows | selected_train_rows | selected_eval_rows | overlaps_selected_window | covers_selected_window | available_window_context_ready | limitation |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| VNINDEX | False |  |  | 0 | 0 | 0 | False | False | False | exact_code_local_hourly_index_data_missing; context_not_fabricated |
| VN30INDEX | False |  |  | 0 | 0 | 0 | False | False | False | exact_code_local_hourly_index_data_missing; context_not_fabricated |
| VNXALL | False |  |  | 0 | 0 | 0 | False | False | False | exact_code_local_hourly_index_data_missing; context_not_fabricated |

## Boundary

- The available-window stock benchmark remains real local hourly evidence.
- Missing exact-code local index data must be disclosed as a market-context limitation.
- Do not reconstruct pre-start or missing index history unless the vendor supplies it and clearly labels it.
