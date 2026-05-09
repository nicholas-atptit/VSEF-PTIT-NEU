# EXP-FC-001 Report

- Status: `completed`
- Provider: `vnstock_data`
- Frequency: `daily`
- Universe: FPT, ACB, HPG, MWG, DGC
- Horizons: 1
- Models: sarimax, ets, xgboost, lightgbm, lstm, bilstm
- Baselines: persistence, zero_return, moving_average_rule
- Metric rows: 540
- Prediction rows: 5805
- Error count: 0
- Warning count: 7

## Evidence

- Manifest: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\manifests\run_manifest.json`
- Metrics: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\metrics\metrics.csv`
- Predictions: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\predictions\predictions.csv`
- Summary: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-001\reports\summary.md`

## Metric Winners

| ticker | horizon | metric_name | model_name | model_type | metric_value |
| --- | --- | --- | --- | --- | --- |
| ACB | 1 | coverage_count | bilstm | model | 129.0 |
| ACB | 1 | directional_accuracy | ets | model | 0.5271317829457365 |
| ACB | 1 | mae | persistence | baseline | 0.1475968992248061 |
| ACB | 1 | mape | persistence | baseline | 0.7121662491841998 |
| ACB | 1 | missing_prediction_rate | bilstm | model | 0.0 |
| ACB | 1 | prediction_count | bilstm | model | 129.0 |
| ACB | 1 | rmse | persistence | baseline | 0.2066829200910524 |
| DGC | 1 | coverage_count | bilstm | model | 129.0 |
| DGC | 1 | directional_accuracy | lightgbm | model | 0.4961240310077519 |
| DGC | 1 | mae | persistence | baseline | 1.0748837209302324 |
| DGC | 1 | mape | persistence | baseline | 1.0231099288143952 |
| DGC | 1 | missing_prediction_rate | bilstm | model | 0.0 |
| DGC | 1 | prediction_count | bilstm | model | 129.0 |
| DGC | 1 | rmse | persistence | baseline | 1.6217016615486857 |
| FPT | 1 | coverage_count | bilstm | model | 129.0 |
| FPT | 1 | directional_accuracy | ets | model | 0.5658914728682171 |
| FPT | 1 | mae | persistence | baseline | 1.2013178294573643 |
| FPT | 1 | mape | persistence | baseline | 1.0520509448195785 |
| FPT | 1 | missing_prediction_rate | bilstm | model | 0.0 |
| FPT | 1 | prediction_count | bilstm | model | 129.0 |
| FPT | 1 | rmse | persistence | baseline | 1.5943589997723344 |
| HPG | 1 | coverage_count | bilstm | model | 129.0 |
| HPG | 1 | directional_accuracy | bilstm | model | 0.5271317829457365 |
| HPG | 1 | mae | persistence | baseline | 0.2034883720930232 |
| HPG | 1 | mape | persistence | baseline | 0.9170040900525624 |
| HPG | 1 | missing_prediction_rate | bilstm | model | 0.0 |
| HPG | 1 | prediction_count | bilstm | model | 129.0 |
| HPG | 1 | rmse | persistence | baseline | 0.2645062681515749 |
| MWG | 1 | coverage_count | bilstm | model | 129.0 |
| MWG | 1 | directional_accuracy | bilstm | model | 0.4961240310077519 |
| MWG | 1 | mae | persistence | baseline | 0.6389147286821705 |
| MWG | 1 | mape | persistence | baseline | 1.0166184746406497 |
| MWG | 1 | missing_prediction_rate | bilstm | model | 0.0 |
| MWG | 1 | prediction_count | bilstm | model | 129.0 |
| MWG | 1 | rmse | persistence | baseline | 0.8939607885544358 |

## Horizon Comparison

| experiment_id | horizon | metric_name | model_name | model_type | mean_metric_value | sample_size | context_count | rank_within_horizon | metric_direction | best_baseline_value | model_vs_best_baseline_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | 1 | directional_accuracy | ets | model | 0.4837209302325582 | 645 | 5 | 1 | higher | 0.44341085271317826 | 0.04031007751937993 |
| EXP-FC-001 | 1 | directional_accuracy | lightgbm | model | 0.4775193798449612 | 645 | 5 | 2 | higher | 0.44341085271317826 | 0.03410852713178292 |
| EXP-FC-001 | 1 | directional_accuracy | bilstm | model | 0.47441860465116276 | 645 | 5 | 3 | higher | 0.44341085271317826 | 0.031007751937984496 |
| EXP-FC-001 | 1 | directional_accuracy | lstm | model | 0.47441860465116276 | 645 | 5 | 4 | higher | 0.44341085271317826 | 0.031007751937984496 |
| EXP-FC-001 | 1 | directional_accuracy | xgboost | model | 0.47441860465116276 | 645 | 5 | 5 | higher | 0.44341085271317826 | 0.031007751937984496 |
| EXP-FC-001 | 1 | directional_accuracy | moving_average_rule | baseline | 0.44341085271317826 | 645 | 5 | 6 | higher | 0.44341085271317826 | 0.0 |
| EXP-FC-001 | 1 | directional_accuracy | sarimax | model | 0.4387596899224806 | 645 | 5 | 7 | higher | 0.44341085271317826 | -0.0046511627906976605 |
| EXP-FC-001 | 1 | directional_accuracy | persistence | baseline | 0.0728682170542635 | 645 | 5 | 8 | higher | 0.44341085271317826 | -0.37054263565891477 |
| EXP-FC-001 | 1 | directional_accuracy | zero_return | baseline | 0.0728682170542635 | 645 | 5 | 9 | higher | 0.44341085271317826 | -0.37054263565891477 |
| EXP-FC-001 | 1 | mae | persistence | baseline | 0.6532403100775193 | 645 | 5 | 1 | lower | 0.6532403100775193 | 0.0 |
| EXP-FC-001 | 1 | mae | zero_return | baseline | 0.6532403100775193 | 645 | 5 | 2 | lower | 0.6532403100775193 | 0.0 |
| EXP-FC-001 | 1 | mae | moving_average_rule | baseline | 1.0521860465116277 | 645 | 5 | 3 | lower | 0.6532403100775193 | 0.39894573643410847 |
| EXP-FC-001 | 1 | mae | lightgbm | model | 5.619846902827106 | 645 | 5 | 4 | lower | 0.6532403100775193 | 4.966606592749587 |
| EXP-FC-001 | 1 | mae | xgboost | model | 5.855268431316051 | 645 | 5 | 5 | lower | 0.6532403100775193 | 5.2020281212385315 |
| EXP-FC-001 | 1 | mae | ets | model | 6.496177713254281 | 645 | 5 | 6 | lower | 0.6532403100775193 | 5.842937403176761 |
| EXP-FC-001 | 1 | mae | lstm | model | 65.47051519943254 | 645 | 5 | 7 | lower | 0.6532403100775193 | 64.81727488935502 |
| EXP-FC-001 | 1 | mae | bilstm | model | 65.84176523230579 | 645 | 5 | 8 | lower | 0.6532403100775193 | 65.18852492222827 |
| EXP-FC-001 | 1 | mae | sarimax | model | 2.2495078535529055e+90 | 645 | 5 | 9 | lower | 0.6532403100775193 | 2.2495078535529055e+90 |
| EXP-FC-001 | 1 | mape | persistence | baseline | 0.9441899375022771 | 645 | 5 | 1 | lower | 0.9441899375022771 | 0.0 |
| EXP-FC-001 | 1 | mape | zero_return | baseline | 0.9441899375022771 | 645 | 5 | 2 | lower | 0.9441899375022771 | 0.0 |
| EXP-FC-001 | 1 | mape | moving_average_rule | baseline | 1.4992316056908985 | 645 | 5 | 3 | lower | 0.9441899375022771 | 0.5550416681886214 |
| EXP-FC-001 | 1 | mape | lightgbm | model | 6.949793443463294 | 645 | 5 | 4 | lower | 0.9441899375022771 | 6.005603505961017 |
| EXP-FC-001 | 1 | mape | xgboost | model | 7.317216178562821 | 645 | 5 | 5 | lower | 0.9441899375022771 | 6.373026241060543 |
| EXP-FC-001 | 1 | mape | ets | model | 9.439600828986267 | 645 | 5 | 6 | lower | 0.9441899375022771 | 8.49541089148399 |
| EXP-FC-001 | 1 | mape | lstm | model | 99.56315656631389 | 645 | 5 | 7 | lower | 0.9441899375022771 | 98.61896662881162 |
| EXP-FC-001 | 1 | mape | bilstm | model | 100.4550659353774 | 645 | 5 | 8 | lower | 0.9441899375022771 | 99.51087599787513 |
| EXP-FC-001 | 1 | mape | sarimax | model | 1.0462694961799166e+91 | 645 | 5 | 9 | lower | 0.9441899375022771 | 1.0462694961799166e+91 |
| EXP-FC-001 | 1 | missing_prediction_rate | bilstm | model | 0.0 | 645 | 5 | 1 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | ets | model | 0.0 | 645 | 5 | 2 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | lightgbm | model | 0.0 | 645 | 5 | 3 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | lstm | model | 0.0 | 645 | 5 | 4 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | moving_average_rule | baseline | 0.0 | 645 | 5 | 5 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | persistence | baseline | 0.0 | 645 | 5 | 6 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | sarimax | model | 0.0 | 645 | 5 | 7 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | xgboost | model | 0.0 | 645 | 5 | 8 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | missing_prediction_rate | zero_return | baseline | 0.0 | 645 | 5 | 9 | lower | 0.0 | 0.0 |
| EXP-FC-001 | 1 | rmse | persistence | baseline | 0.9162421276236167 | 645 | 5 | 1 | lower | 0.9162421276236167 | 0.0 |
| EXP-FC-001 | 1 | rmse | zero_return | baseline | 0.9162421276236167 | 645 | 5 | 2 | lower | 0.9162421276236167 | 0.0 |
| EXP-FC-001 | 1 | rmse | moving_average_rule | baseline | 1.425975313630508 | 645 | 5 | 3 | lower | 0.9162421276236167 | 0.5097331860068913 |
| EXP-FC-001 | 1 | rmse | lightgbm | model | 6.146893980487432 | 645 | 5 | 4 | lower | 0.9162421276236167 | 5.230651852863815 |
| EXP-FC-001 | 1 | rmse | xgboost | model | 6.337455716520394 | 645 | 5 | 5 | lower | 0.9162421276236167 | 5.421213588896777 |
| EXP-FC-001 | 1 | rmse | ets | model | 7.335861131492662 | 645 | 5 | 6 | lower | 0.9162421276236167 | 6.419619003869045 |
| EXP-FC-001 | 1 | rmse | lstm | model | 65.54982745048983 | 645 | 5 | 7 | lower | 0.9162421276236167 | 64.63358532286621 |
| EXP-FC-001 | 1 | rmse | bilstm | model | 65.9215260132597 | 645 | 5 | 8 | lower | 0.9162421276236167 | 65.00528388563608 |
| EXP-FC-001 | 1 | rmse | sarimax | model | 2.1162698134251936e+91 | 645 | 5 | 9 | lower | 0.9162421276236167 | 2.1162698134251936e+91 |
