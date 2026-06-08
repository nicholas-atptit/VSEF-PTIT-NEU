# VN30 Paper Figure Data Audit

Local-only audit for the paper-ready forecast-vs-actual figure pack. No data fetch, benchmark sweep, provider change, or model training was performed.

## Active evidence

| benchmark family | active evidence | summary files | row-level prediction files | per-instrument files | rolling outputs | missing files | buildable figures |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VN30 hourly selected candidate | Fixed L2 Logistic h40, `feature_set_C_closest`, threshold 0.50; reproduced final accuracy 61.51% | `reports/generated/vn30_hourly_track_a_target62_validation_safe/selected_candidate_summary.csv`; `reports/generated/vn30_hourly_target62_paper_ready_stability/global_summary.csv` | `outputs/vn30_hourly_selected_l2_logistic_h40_row_predictions/row_predictions.csv` | `reports/generated/vn30_hourly_target62_paper_ready_stability/by_ticker.csv` | `reports/generated/vn30_hourly_selected_candidate_rolling/rolling_accuracy_250.csv`, `_500.csv`, `_1000.csv`, monthly and quarterly summaries | none for selected-candidate stock-only figures | Figures 1, 2, 4, 5, Appendix A stock panel |
| VN30 hourly stock benchmark | Local Jan-2025 hourly benchmark with XGBoost, LightGBM, Random Forest, and stacking context at h1/h4/h8/h20 | `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/accuracy_summary.csv` | `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/predicted_vs_actual.csv` | `outputs/vn30_hourly_2015_jan2025_benchmark/hourly/accuracy_summary.csv` | not found | L2 Logistic h40 is not part of this benchmark artifact | Figure 1 stock benchmark context |
| VN30 hourly all-method base models | Core base model prediction artifact for h40 feature_set_C_closest | `reports/generated/vn30_hourly_track_a_true_stacking_all_algorithms/base_model_scores.csv` | `reports/generated/vn30_hourly_track_a_true_stacking_all_algorithms/base_final_predictions.csv` | derived from row-level predictions | none | row-level stacking predictions are not present | Figures 3, 6, Appendix B for L2 Logistic/logistic regression, Random Forest, XGBoost, LightGBM |
| VN30 daily benchmark | Canonical daily benchmark summary | `outputs/vn30_daily_2015_benchmark/daily/accuracy_summary.csv` | `outputs/vn30_daily_2015_benchmark/daily/predicted_vs_actual.csv` has only datetime/ticker/y_true/y_pred and no method/horizon columns | summary only | `outputs/vn30_daily_2015_target60_v2/daily/rolling_validation_results.csv` | method-specific daily row-level predictions are not available in that row file | Context only; not used for required main forecast-vs-actual hourly figures |
| Supported index benchmark | Supported index directional benchmark | `outputs/index_directional_benchmark/accuracy_summary.csv` | `outputs/index_directional_benchmark/predicted_vs_actual.csv` | derived from accuracy summary by index | none | L2 Logistic and stacking row-level index predictions are not present | Figures 1, 3, 4, 6, Appendix A index proxy |
| Joint stock+index panel | Exploratory joint-panel selected summary, final combined accuracy 49.21% | `outputs/vn30_stock_index_joint_panel_training_v1/selected_candidate_summary.csv` | not found | not found | not found | joint-panel row-level predictions and per-instrument files | Figure 1 summary panel only; no joint forecast-vs-actual figure |

## Planned Figure Buildability

| figure | buildable | basis | limitation |
| --- | --- | --- | --- |
| Figure 1 | yes | summary artifacts | joint panel is summary-only and below benchmark threshold |
| Figure 2 | yes | selected-candidate row-level reproduction | pooled multi-ticker row order used for final-window display |
| Figure 3 | partial yes | stock all-method base predictions and index benchmark predictions | stacking row-level predictions unavailable; index L2 Logistic unavailable |
| Figure 4 | yes | selected ticker slices plus index benchmark summary | stock and index panels use different valid artifacts |
| Figure 5 | yes | regenerated selected-candidate rolling summaries | rolling windows are row-based, not calendar-time smoothing |
| Figure 6 | partial yes | one stock all-method overlay and one index benchmark overlay | index overlay excludes L2 Logistic and stacking |
| Appendix A | partial yes | all VN30 selected-candidate rows; index benchmark proxy | selected stock-only candidate has no index rows |
| Appendix B | partial yes | representative all-method base rows | stacking row-level predictions unavailable |
| Appendix C | yes | per-instrument source table | cross-panel artifact caveat applies |
