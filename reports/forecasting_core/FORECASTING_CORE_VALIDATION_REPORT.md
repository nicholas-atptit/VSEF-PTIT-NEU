# Forecasting Core Validation Report

## Executive Summary

Experiments aggregated: EXP-FC-001=completed, EXP-FC-002=completed, EXP-FC-003=completed.
sarimax beat the best baseline for ACB T+3 on mae (0.276297 vs 0.284094). sarimax beat the best baseline for ACB T+3 on rmse (0.351903 vs 0.35903). These are bounded experiment contexts and do not establish general model superiority.
The report uses generated metrics and predictions only; missing artifacts and model failures remain visible in manifests, logs, and report limitations.

## Phase 2 Objective

Phase 2 validates whether the VSEF forecasting layer creates independent value compared with simple baselines and individual models. The evidence is generated from Phase 1 experiment artifacts: config, logs, manifest, metrics, predictions, and summary files.

## Relation To Phase 0 And Phase 1 Governance

- Phase 0 provider policy is unchanged: `vnstock_data` and daily OHLCV only.
- Phase 0 supported model scope is unchanged: SARIMAX, ETS, XGBoost, LightGBM, LSTM, BiLSTM, and Stacking.
- Phase 1 ExperimentOrchestrator remains the runtime entry point.
- Baselines are comparison evidence only, not official forecasting models.
- Decision or ranking outputs are diagnostic evidence only and must not be presented as BUY / SELL / HOLD advice.

## Experiment Universe

- Selected tickers: ACB, DGC, FPT, HPG, MWG.
- Selection reason: controlled VN equity basket across technology, banking, materials, retail, and chemicals exposure.
- Coverage status: 5 ticker(s) appear in manifests.
- Failed tickers: none recorded in manifests.
- Universe config: `configs\universe\ticker_universe.yaml`.

## Experiment Design

- `EXP-FC-001`: baseline comparison across individual supported models and simple baselines.
- `EXP-FC-002`: ensemble comparison across individual models and stacking where runtime support is available.
- `EXP-FC-003`: multi-horizon comparison across T+1, T+3, and T+5.

## Data And Provider Evidence

| experiment | status | provider | frequency | import_status | date_window | manifest |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | completed | vnstock_data | daily | available | 2023-01-01 to 2024-12-31 | outputs\experiments\EXP-FC-001\manifests\run_manifest.json |
| EXP-FC-002 | completed | vnstock_data | daily | available | 2023-01-01 to 2024-12-31 | outputs\experiments\EXP-FC-002\manifests\run_manifest.json |
| EXP-FC-003 | completed | vnstock_data | daily | available | 2023-01-01 to 2024-12-31 | outputs\experiments\EXP-FC-003\manifests\run_manifest.json |

## Baseline Comparison Results

- Baseline winner contexts: 48.
- Model winner contexts: 27.

| experiment_id | ticker | horizon | metric_name | model_name | model_type | metric_value |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | ACB | 1 | directional_accuracy | ets | model | 0.5271317829457365 |
| EXP-FC-001 | ACB | 1 | mae | persistence | baseline | 0.1475968992248061 |
| EXP-FC-001 | ACB | 1 | rmse | persistence | baseline | 0.2066829200910524 |
| EXP-FC-001 | DGC | 1 | directional_accuracy | lightgbm | model | 0.4961240310077519 |
| EXP-FC-001 | DGC | 1 | mae | persistence | baseline | 1.0748837209302324 |
| EXP-FC-001 | DGC | 1 | rmse | persistence | baseline | 1.6217016615486857 |
| EXP-FC-001 | FPT | 1 | directional_accuracy | ets | model | 0.5658914728682171 |
| EXP-FC-001 | FPT | 1 | mae | persistence | baseline | 1.2013178294573643 |
| EXP-FC-001 | FPT | 1 | rmse | persistence | baseline | 1.5943589997723344 |
| EXP-FC-001 | HPG | 1 | directional_accuracy | bilstm | model | 0.5271317829457365 |
| EXP-FC-001 | HPG | 1 | mae | persistence | baseline | 0.2034883720930232 |
| EXP-FC-001 | HPG | 1 | rmse | persistence | baseline | 0.2645062681515749 |
| EXP-FC-001 | MWG | 1 | directional_accuracy | bilstm | model | 0.4961240310077519 |
| EXP-FC-001 | MWG | 1 | mae | persistence | baseline | 0.6389147286821705 |
| EXP-FC-001 | MWG | 1 | rmse | persistence | baseline | 0.8939607885544358 |
| EXP-FC-002 | ACB | 1 | directional_accuracy | ets | model | 0.5271317829457365 |
| EXP-FC-002 | ACB | 1 | mae | persistence | baseline | 0.1475968992248061 |
| EXP-FC-002 | ACB | 1 | rmse | persistence | baseline | 0.2066829200910524 |
| EXP-FC-002 | DGC | 1 | directional_accuracy | lightgbm | model | 0.4961240310077519 |
| EXP-FC-002 | DGC | 1 | mae | persistence | baseline | 1.0748837209302324 |
| EXP-FC-002 | DGC | 1 | rmse | persistence | baseline | 1.6217016615486857 |
| EXP-FC-002 | FPT | 1 | directional_accuracy | ets | model | 0.5658914728682171 |
| EXP-FC-002 | FPT | 1 | mae | persistence | baseline | 1.2013178294573643 |
| EXP-FC-002 | FPT | 1 | rmse | persistence | baseline | 1.5943589997723344 |
| EXP-FC-002 | HPG | 1 | directional_accuracy | lightgbm | model | 0.5271317829457365 |

## Single-Model Comparison

- `mae` best model row: sarimax on HPG T+1 with 0.20773; worst model row: sarimax on ACB T+1 with 1.12475e+91.
- `rmse` best model row: sarimax on HPG T+1 with 0.268851; worst model row: sarimax on ACB T+1 with 1.05813e+92.
- `directional_accuracy` best model row: ets on ACB T+3 with 0.598425; worst model row: sarimax on MWG T+5 with 0.4.

Model coverage by sample count:

| model_name | sample_size |
| --- | --- |
| bilstm | 4515 |
| ets | 22365 |
| lightgbm | 22365 |
| lstm | 4515 |
| sarimax | 22365 |
| stacking | 4515 |
| xgboost | 22365 |

## Stacking / Ensemble Comparison

Stacking produced metric rows in EXP-FC-002. This validates runtime execution, not model superiority.

| ticker | horizon | metric_name | metric_value | rank | best_baseline_value |
| --- | --- | --- | --- | --- | --- |
| ACB | 1 | coverage_count | 129.0 | 5 | 129.0 |
| ACB | 1 | directional_accuracy | 0.4496124031007752 | 2 | 0.1085271317829457 |
| ACB | 1 | mae | 4.839445862360975e+55 | 6 | 0.1475968992248061 |
| ACB | 1 | mape | 2.2508765978323867e+56 | 6 | 0.7121662491841998 |
| ACB | 1 | missing_prediction_rate | 0.0 | 5 | 0.0 |
| ACB | 1 | prediction_count | 129.0 | 5 | 129.0 |
| ACB | 1 | rmse | 4.552806150929545e+56 | 6 | 0.2066829200910524 |
| DGC | 1 | coverage_count | 129.0 | 5 | 129.0 |
| DGC | 1 | directional_accuracy | 0.4728682170542636 | 4 | 0.0465116279069767 |
| DGC | 1 | mae | 27.44077232124991 | 7 | 1.0748837209302324 |
| DGC | 1 | mape | 25.710937465305072 | 7 | 1.0231099288143952 |
| DGC | 1 | missing_prediction_rate | 0.0 | 5 | 0.0 |
| DGC | 1 | prediction_count | 129.0 | 5 | 129.0 |
| DGC | 1 | rmse | 28.1494671645402 | 7 | 1.6217016615486857 |
| FPT | 1 | coverage_count | 129.0 | 5 | 129.0 |
| FPT | 1 | directional_accuracy | 0.5271317829457365 | 2 | 0.0310077519379844 |
| FPT | 1 | mae | 13.559047957712592 | 4 | 1.2013178294573643 |
| FPT | 1 | mape | 12.085893383665477 | 4 | 1.0520509448195785 |
| FPT | 1 | missing_prediction_rate | 0.0 | 5 | 0.0 |
| FPT | 1 | prediction_count | 129.0 | 5 | 129.0 |

## Multi-Horizon Comparison

- Horizons observed: T+1, T+3, T+5.
- Missing horizons: none.

Best MAE row by horizon:

| horizon | model_name | model_type | mean_metric_value | context_count |
| --- | --- | --- | --- | --- |
| 1 | persistence | baseline | 0.6532403100775193 | 5 |
| 3 | persistence | baseline | 1.2470866141732282 | 5 |
| 5 | persistence | baseline | 1.6002079999999996 | 5 |

## Stability And Worst-Window Discussion

Worst missing prediction rate rows:

| experiment_id | ticker | horizon | model_name | model_type | metric_value |
| --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | ACB | 1 | bilstm | model | 0.0 |
| EXP-FC-001 | ACB | 1 | ets | model | 0.0 |
| EXP-FC-001 | ACB | 1 | lightgbm | model | 0.0 |
| EXP-FC-001 | ACB | 1 | lstm | model | 0.0 |
| EXP-FC-001 | ACB | 1 | moving_average_rule | baseline | 0.0 |
| EXP-FC-001 | ACB | 1 | persistence | baseline | 0.0 |
| EXP-FC-001 | ACB | 1 | sarimax | model | 0.0 |
| EXP-FC-001 | ACB | 1 | xgboost | model | 0.0 |
| EXP-FC-001 | ACB | 1 | zero_return | baseline | 0.0 |
| EXP-FC-001 | DGC | 1 | bilstm | model | 0.0 |

Prediction-count coverage:

| experiment_id | model_name | model_type | metric_value |
| --- | --- | --- | --- |
| EXP-FC-001 | bilstm | model | 645.0 |
| EXP-FC-001 | ets | model | 645.0 |
| EXP-FC-001 | lightgbm | model | 645.0 |
| EXP-FC-001 | lstm | model | 645.0 |
| EXP-FC-001 | moving_average_rule | baseline | 645.0 |
| EXP-FC-001 | persistence | baseline | 645.0 |
| EXP-FC-001 | sarimax | model | 645.0 |
| EXP-FC-001 | xgboost | model | 645.0 |
| EXP-FC-001 | zero_return | baseline | 645.0 |
| EXP-FC-002 | ets | model | 645.0 |
| EXP-FC-002 | lightgbm | model | 645.0 |
| EXP-FC-002 | persistence | baseline | 645.0 |
| EXP-FC-002 | sarimax | model | 645.0 |
| EXP-FC-002 | stacking | model | 645.0 |
| EXP-FC-002 | xgboost | model | 645.0 |
| EXP-FC-002 | zero_return | baseline | 645.0 |
| EXP-FC-003 | ets | model | 1905.0 |
| EXP-FC-003 | lightgbm | model | 1905.0 |
| EXP-FC-003 | moving_average_rule | baseline | 1905.0 |
| EXP-FC-003 | persistence | baseline | 1905.0 |
| EXP-FC-003 | sarimax | model | 1905.0 |
| EXP-FC-003 | xgboost | model | 1905.0 |
| EXP-FC-003 | zero_return | baseline | 1905.0 |

## Error Distribution Discussion

Residual summaries were computed from `predictions/predictions.csv` where `y_true` and `y_pred` were available.

| experiment_id | model_name | model_type | sample_size | mean_absolute_error | median_absolute_error | error_std |
| --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | persistence | baseline | 645 | 0.6532403100775194 | 0.4760000000000019 | 0.9142572029294544 |
| EXP-FC-001 | zero_return | baseline | 645 | 0.6532403100775194 | 0.4760000000000019 | 0.9142572029294544 |
| EXP-FC-001 | moving_average_rule | baseline | 645 | 1.052186046511628 | 0.7496000000000024 | 1.417779879545329 |
| EXP-FC-001 | lightgbm | model | 645 | 5.619846902827109 | 5.471564501953289 | 2.5459225505645895 |
| EXP-FC-001 | xgboost | model | 645 | 5.855268431316051 | 5.657169174194339 | 2.4617888025591945 |
| EXP-FC-001 | ets | model | 645 | 6.496177713254281 | 6.436545867814584 | 3.784439670793289 |
| EXP-FC-001 | lstm | model | 645 | 65.47051519943254 | 65.19301142054796 | 3.1842510123106695 |
| EXP-FC-001 | bilstm | model | 645 | 65.84176523230578 | 65.57522004818915 | 3.2043847455935497 |
| EXP-FC-001 | sarimax | model | 645 | 2.2495078535529055e+90 | 4.456454527169619e+45 | 2.104280177966503e+91 |
| EXP-FC-002 | persistence | baseline | 645 | 0.6532403100775194 | 0.4760000000000019 | 0.9142572029294544 |
| EXP-FC-002 | zero_return | baseline | 645 | 0.6532403100775194 | 0.4760000000000019 | 0.9142572029294544 |
| EXP-FC-002 | lightgbm | model | 645 | 5.619846902827109 | 5.471564501953289 | 2.5459225505645895 |
| EXP-FC-002 | xgboost | model | 645 | 5.855268431316051 | 5.657169174194339 | 2.4617888025591945 |
| EXP-FC-002 | ets | model | 645 | 6.496177713254281 | 6.436545867814584 | 3.784439670793289 |
| EXP-FC-002 | stacking | model | 645 | 9.67889172472195e+54 | 19174656712.92977 | 9.054024847634264e+55 |
| EXP-FC-002 | sarimax | model | 645 | 2.2495078535529055e+90 | 4.456454527169619e+45 | 2.104280177966503e+91 |
| EXP-FC-003 | persistence | baseline | 1905 | 1.166844974750249 | 0.8653333333333341 | 1.5715745041862839 |
| EXP-FC-003 | zero_return | baseline | 1905 | 1.166844974750249 | 0.8653333333333341 | 1.5715745041862839 |
| EXP-FC-003 | moving_average_rule | baseline | 1905 | 1.406332671671855 | 1.0393333333333343 | 1.8791718471685235 |
| EXP-FC-003 | lightgbm | model | 1905 | 5.30203517251397 | 5.068312730546109 | 2.7313318006226948 |
| EXP-FC-003 | xgboost | model | 1905 | 5.601719091307094 | 5.371804916381837 | 2.7300624446738424 |
| EXP-FC-003 | ets | model | 1905 | 8.527978843248423 | 8.417882292607509 | 4.123402069902147 |
| EXP-FC-003 | sarimax | model | 1905 | 7.498359511843019e+89 | 1.485484842389873e+45 | 7.014267259888345e+90 |

## Missing Artifacts And Limitations

- EXP-FC-001 warnings: expected_artifact_not_generated:forecast_summary.csv, expected_artifact_not_generated:model_consensus_summary.csv, expected_artifact_not_generated:model_health_summary.csv, expected_artifact_not_generated:risk_summary.csv, expected_artifact_not_generated:strategy_metrics.csv, expected_artifact_not_generated:decision_lane_candidates.csv, expected_artifact_not_generated:analysis_packets.jsonl.
- EXP-FC-002 warnings: expected_artifact_not_generated:forecast_summary.csv, expected_artifact_not_generated:model_consensus_summary.csv, expected_artifact_not_generated:model_health_summary.csv, expected_artifact_not_generated:risk_summary.csv, expected_artifact_not_generated:strategy_metrics.csv, expected_artifact_not_generated:decision_lane_candidates.csv, expected_artifact_not_generated:analysis_packets.jsonl.
- EXP-FC-003 warnings: expected_artifact_not_generated:forecast_summary.csv, expected_artifact_not_generated:model_consensus_summary.csv, expected_artifact_not_generated:model_health_summary.csv, expected_artifact_not_generated:risk_summary.csv, expected_artifact_not_generated:strategy_metrics.csv, expected_artifact_not_generated:decision_lane_candidates.csv, expected_artifact_not_generated:analysis_packets.jsonl.
- Generated 3 chart artifact(s) from actual metric rows.
- Raw experiment outputs under `outputs/experiments/` are local run evidence and should not be committed.

## Generated Report Artifacts

- `reports/forecasting_core/forecast_metrics.csv`
- `reports/forecasting_core/model_ranking.csv`
- `reports/forecasting_core/stability_metrics.csv`
- `reports/forecasting_core/horizon_comparison.csv`
- `reports/forecasting_core/error_distribution_summary.csv`
- `reports/forecasting_core/EXP-FC-001_BASELINE_COMPARISON.md`
- `reports/forecasting_core/EXP-FC-002_ENSEMBLE_COMPARISON.md`
- `reports/forecasting_core/EXP-FC-003_MULTI_HORIZON.md`

## Acceptance Criteria

| Criterion | Met | Evidence |
| --- | --- | --- |
| Universe config exists | yes | `configs/universe/ticker_universe.yaml` |
| Phase 2 experiment configs exist | yes | `EXP-FC-001`, `EXP-FC-002`, `EXP-FC-003` |
| At least one run produced manifest, metrics, predictions, summary | yes | loaded artifact set |
| Baselines and models compared in same metric table | yes | `forecast_metrics.csv` |
| Stacking evaluated or disclosed | yes | `EXP-FC-002` |
| Multi-horizon T+1/T+3/T+5 covered or failures disclosed | yes | `horizon_comparison.csv` and manifests |
| Report generated from actual artifacts | yes | this report and generated CSVs |
| No fake artifacts created | yes | missing artifacts remain warnings |
| Diagnostic-only disclaimer present | yes | report disclaimer |

## Diagnostic-Only Disclaimer

All Phase 2 outputs are experiment validation evidence only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instruction, or proof of guaranteed profitable trading.
