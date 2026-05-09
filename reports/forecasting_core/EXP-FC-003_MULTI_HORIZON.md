# EXP-FC-003 Report

- Status: `completed`
- Provider: `vnstock_data`
- Frequency: `daily`
- Universe: FPT, ACB, HPG, MWG, DGC
- Horizons: 1, 3, 5
- Models: sarimax, ets, xgboost, lightgbm
- Baselines: persistence, zero_return, moving_average_rule
- Metric rows: 1260
- Prediction rows: 13335
- Error count: 0
- Warning count: 7

## Evidence

- Manifest: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\manifests\run_manifest.json`
- Metrics: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\metrics\metrics.csv`
- Predictions: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\predictions\predictions.csv`
- Summary: `K:\Repos\VSEF-PTIT-NEU\outputs\experiments\EXP-FC-003\reports\summary.md`

## Metric Winners

| ticker | horizon | metric_name | model_name | model_type | metric_value |
| --- | --- | --- | --- | --- | --- |
| ACB | 1 | coverage_count | ets | model | 129.0 |
| ACB | 1 | directional_accuracy | ets | model | 0.5271317829457365 |
| ACB | 1 | mae | persistence | baseline | 0.1475968992248061 |
| ACB | 1 | mape | persistence | baseline | 0.7121662491841998 |
| ACB | 1 | missing_prediction_rate | ets | model | 0.0 |
| ACB | 1 | prediction_count | ets | model | 129.0 |
| ACB | 1 | rmse | persistence | baseline | 0.2066829200910524 |
| ACB | 3 | coverage_count | ets | model | 127.0 |
| ACB | 3 | directional_accuracy | ets | model | 0.5984251968503937 |
| ACB | 3 | mae | sarimax | model | 0.2762968530820263 |
| ACB | 3 | mape | sarimax | model | 1.328049516823783 |
| ACB | 3 | missing_prediction_rate | ets | model | 0.0 |
| ACB | 3 | prediction_count | ets | model | 127.0 |
| ACB | 3 | rmse | sarimax | model | 0.3519031924555273 |
| ACB | 5 | coverage_count | ets | model | 125.0 |
| ACB | 5 | directional_accuracy | ets | model | 0.584 |
| ACB | 5 | mae | persistence | baseline | 0.3535200000000001 |
| ACB | 5 | mape | persistence | baseline | 1.7001725305804851 |
| ACB | 5 | missing_prediction_rate | ets | model | 0.0 |
| ACB | 5 | prediction_count | ets | model | 125.0 |
| ACB | 5 | rmse | persistence | baseline | 0.4446220867208465 |
| DGC | 1 | coverage_count | ets | model | 129.0 |
| DGC | 1 | directional_accuracy | lightgbm | model | 0.4961240310077519 |
| DGC | 1 | mae | persistence | baseline | 1.0748837209302324 |
| DGC | 1 | mape | persistence | baseline | 1.0231099288143952 |
| DGC | 1 | missing_prediction_rate | ets | model | 0.0 |
| DGC | 1 | prediction_count | ets | model | 129.0 |
| DGC | 1 | rmse | persistence | baseline | 1.6217016615486857 |
| DGC | 3 | coverage_count | ets | model | 127.0 |
| DGC | 3 | directional_accuracy | ets | model | 0.5511811023622047 |
| DGC | 3 | mae | persistence | baseline | 2.1466141732283464 |
| DGC | 3 | mape | persistence | baseline | 2.0341676209126867 |
| DGC | 3 | missing_prediction_rate | ets | model | 0.0 |
| DGC | 3 | prediction_count | ets | model | 127.0 |
| DGC | 3 | rmse | persistence | baseline | 3.0561369780077685 |
| DGC | 5 | coverage_count | ets | model | 125.0 |
| DGC | 5 | directional_accuracy | ets | model | 0.576 |
| DGC | 5 | mae | persistence | baseline | 2.797599999999999 |
| DGC | 5 | mape | persistence | baseline | 2.6568713285980325 |
| DGC | 5 | missing_prediction_rate | ets | model | 0.0 |
| DGC | 5 | prediction_count | ets | model | 125.0 |
| DGC | 5 | rmse | persistence | baseline | 3.9149160910548257 |
| FPT | 1 | coverage_count | ets | model | 129.0 |
| FPT | 1 | directional_accuracy | ets | model | 0.5658914728682171 |
| FPT | 1 | mae | persistence | baseline | 1.2013178294573643 |
| FPT | 1 | mape | persistence | baseline | 1.0520509448195785 |
| FPT | 1 | missing_prediction_rate | ets | model | 0.0 |
| FPT | 1 | prediction_count | ets | model | 129.0 |
| FPT | 1 | rmse | persistence | baseline | 1.5943589997723344 |
| FPT | 3 | coverage_count | ets | model | 127.0 |
| FPT | 3 | directional_accuracy | ets | model | 0.5669291338582677 |
| FPT | 3 | mae | persistence | baseline | 2.2508661417322826 |
| FPT | 3 | mape | persistence | baseline | 1.974927544051582 |
| FPT | 3 | missing_prediction_rate | ets | model | 0.0 |
| FPT | 3 | prediction_count | ets | model | 127.0 |
| FPT | 3 | rmse | persistence | baseline | 3.000623557243304 |
| FPT | 5 | coverage_count | ets | model | 125.0 |
| FPT | 5 | directional_accuracy | ets | model | 0.544 |
| FPT | 5 | mae | persistence | baseline | 2.8957599999999992 |
| FPT | 5 | mape | persistence | baseline | 2.538639566759473 |
| FPT | 5 | missing_prediction_rate | ets | model | 0.0 |
| FPT | 5 | prediction_count | ets | model | 125.0 |
| FPT | 5 | rmse | persistence | baseline | 3.7751745919890896 |
| HPG | 1 | coverage_count | ets | model | 129.0 |
| HPG | 1 | directional_accuracy | lightgbm | model | 0.5271317829457365 |
| HPG | 1 | mae | persistence | baseline | 0.2034883720930232 |
| HPG | 1 | mape | persistence | baseline | 0.9170040900525624 |
| HPG | 1 | missing_prediction_rate | ets | model | 0.0 |
| HPG | 1 | prediction_count | ets | model | 129.0 |
| HPG | 1 | rmse | persistence | baseline | 0.2645062681515749 |
| HPG | 3 | coverage_count | ets | model | 127.0 |
| HPG | 3 | directional_accuracy | lightgbm | model | 0.5118110236220472 |
| HPG | 3 | mae | persistence | baseline | 0.3515748031496064 |
| HPG | 3 | mape | persistence | baseline | 1.585637303292463 |
| HPG | 3 | missing_prediction_rate | ets | model | 0.0 |
| HPG | 3 | prediction_count | ets | model | 127.0 |
| HPG | 3 | rmse | persistence | baseline | 0.4360605679233539 |
| HPG | 5 | coverage_count | ets | model | 125.0 |
| HPG | 5 | directional_accuracy | lightgbm | model | 0.512 |
| HPG | 5 | mae | persistence | baseline | 0.4529600000000001 |
| HPG | 5 | mape | persistence | baseline | 2.0463793862493667 |
| HPG | 5 | missing_prediction_rate | ets | model | 0.0 |
| HPG | 5 | prediction_count | ets | model | 125.0 |
| HPG | 5 | rmse | persistence | baseline | 0.5719874124489106 |
| MWG | 1 | coverage_count | ets | model | 129.0 |
| MWG | 1 | directional_accuracy | lightgbm | model | 0.4961240310077519 |
| MWG | 1 | mae | persistence | baseline | 0.6389147286821705 |
| MWG | 1 | mape | persistence | baseline | 1.0166184746406497 |
| MWG | 1 | missing_prediction_rate | ets | model | 0.0 |
| MWG | 1 | prediction_count | ets | model | 129.0 |
| MWG | 1 | rmse | persistence | baseline | 0.8939607885544358 |
| MWG | 3 | coverage_count | ets | model | 127.0 |
| MWG | 3 | directional_accuracy | lightgbm | model | 0.5433070866141733 |
| MWG | 3 | mae | persistence | baseline | 1.2022834645669298 |
| MWG | 3 | mape | persistence | baseline | 1.9162868642677156 |
| MWG | 3 | missing_prediction_rate | ets | model | 0.0 |
| MWG | 3 | prediction_count | ets | model | 127.0 |
| MWG | 3 | rmse | persistence | baseline | 1.586175264119302 |
| MWG | 5 | coverage_count | ets | model | 125.0 |
| MWG | 5 | directional_accuracy | lightgbm | model | 0.528 |
| MWG | 5 | mae | persistence | baseline | 1.5012 |
| MWG | 5 | mape | persistence | baseline | 2.389940985958444 |
| MWG | 5 | missing_prediction_rate | ets | model | 0.0 |
| MWG | 5 | prediction_count | ets | model | 125.0 |
| MWG | 5 | rmse | persistence | baseline | 1.971559585708736 |

## Horizon Comparison

| experiment_id | horizon | metric_name | model_name | model_type | mean_metric_value | sample_size | context_count | rank_within_horizon | metric_direction | best_baseline_value | model_vs_best_baseline_gap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-003 | 1 | directional_accuracy | ets | model | 0.4837209302325582 | 645 | 5 | 1 | higher | 0.44341085271317826 | 0.04031007751937993 |
| EXP-FC-003 | 1 | directional_accuracy | lightgbm | model | 0.4775193798449612 | 645 | 5 | 2 | higher | 0.44341085271317826 | 0.03410852713178292 |
| EXP-FC-003 | 1 | directional_accuracy | xgboost | model | 0.47441860465116276 | 645 | 5 | 3 | higher | 0.44341085271317826 | 0.031007751937984496 |
| EXP-FC-003 | 1 | directional_accuracy | moving_average_rule | baseline | 0.44341085271317826 | 645 | 5 | 4 | higher | 0.44341085271317826 | 0.0 |
| EXP-FC-003 | 1 | directional_accuracy | sarimax | model | 0.4387596899224806 | 645 | 5 | 5 | higher | 0.44341085271317826 | -0.0046511627906976605 |
| EXP-FC-003 | 1 | directional_accuracy | persistence | baseline | 0.0728682170542635 | 645 | 5 | 6 | higher | 0.44341085271317826 | -0.37054263565891477 |
| EXP-FC-003 | 1 | directional_accuracy | zero_return | baseline | 0.0728682170542635 | 645 | 5 | 7 | higher | 0.44341085271317826 | -0.37054263565891477 |
| EXP-FC-003 | 1 | mae | persistence | baseline | 0.6532403100775193 | 645 | 5 | 1 | lower | 0.6532403100775193 | 0.0 |
| EXP-FC-003 | 1 | mae | zero_return | baseline | 0.6532403100775193 | 645 | 5 | 2 | lower | 0.6532403100775193 | 0.0 |
| EXP-FC-003 | 1 | mae | moving_average_rule | baseline | 1.0521860465116277 | 645 | 5 | 3 | lower | 0.6532403100775193 | 0.39894573643410847 |
| EXP-FC-003 | 1 | mae | lightgbm | model | 5.619846902827106 | 645 | 5 | 4 | lower | 0.6532403100775193 | 4.966606592749587 |
| EXP-FC-003 | 1 | mae | xgboost | model | 5.855268431316051 | 645 | 5 | 5 | lower | 0.6532403100775193 | 5.2020281212385315 |
| EXP-FC-003 | 1 | mae | ets | model | 6.496177713254281 | 645 | 5 | 6 | lower | 0.6532403100775193 | 5.842937403176761 |
| EXP-FC-003 | 1 | mae | sarimax | model | 2.2495078535529055e+90 | 645 | 5 | 7 | lower | 0.6532403100775193 | 2.2495078535529055e+90 |
| EXP-FC-003 | 1 | mape | persistence | baseline | 0.9441899375022771 | 645 | 5 | 1 | lower | 0.9441899375022771 | 0.0 |
| EXP-FC-003 | 1 | mape | zero_return | baseline | 0.9441899375022771 | 645 | 5 | 2 | lower | 0.9441899375022771 | 0.0 |
| EXP-FC-003 | 1 | mape | moving_average_rule | baseline | 1.4992316056908985 | 645 | 5 | 3 | lower | 0.9441899375022771 | 0.5550416681886214 |
| EXP-FC-003 | 1 | mape | lightgbm | model | 6.949793443463294 | 645 | 5 | 4 | lower | 0.9441899375022771 | 6.005603505961017 |
| EXP-FC-003 | 1 | mape | xgboost | model | 7.317216178562821 | 645 | 5 | 5 | lower | 0.9441899375022771 | 6.373026241060543 |
| EXP-FC-003 | 1 | mape | ets | model | 9.439600828986267 | 645 | 5 | 6 | lower | 0.9441899375022771 | 8.49541089148399 |
| EXP-FC-003 | 1 | mape | sarimax | model | 1.0462694961799166e+91 | 645 | 5 | 7 | lower | 0.9441899375022771 | 1.0462694961799166e+91 |
| EXP-FC-003 | 1 | missing_prediction_rate | ets | model | 0.0 | 645 | 5 | 1 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | missing_prediction_rate | lightgbm | model | 0.0 | 645 | 5 | 2 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | missing_prediction_rate | moving_average_rule | baseline | 0.0 | 645 | 5 | 3 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | missing_prediction_rate | persistence | baseline | 0.0 | 645 | 5 | 4 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | missing_prediction_rate | sarimax | model | 0.0 | 645 | 5 | 5 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | missing_prediction_rate | xgboost | model | 0.0 | 645 | 5 | 6 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | missing_prediction_rate | zero_return | baseline | 0.0 | 645 | 5 | 7 | lower | 0.0 | 0.0 |
| EXP-FC-003 | 1 | rmse | persistence | baseline | 0.9162421276236167 | 645 | 5 | 1 | lower | 0.9162421276236167 | 0.0 |
| EXP-FC-003 | 1 | rmse | zero_return | baseline | 0.9162421276236167 | 645 | 5 | 2 | lower | 0.9162421276236167 | 0.0 |
| EXP-FC-003 | 1 | rmse | moving_average_rule | baseline | 1.425975313630508 | 645 | 5 | 3 | lower | 0.9162421276236167 | 0.5097331860068913 |
| EXP-FC-003 | 1 | rmse | lightgbm | model | 6.146893980487432 | 645 | 5 | 4 | lower | 0.9162421276236167 | 5.230651852863815 |
| EXP-FC-003 | 1 | rmse | xgboost | model | 6.337455716520394 | 645 | 5 | 5 | lower | 0.9162421276236167 | 5.421213588896777 |
| EXP-FC-003 | 1 | rmse | ets | model | 7.335861131492662 | 645 | 5 | 6 | lower | 0.9162421276236167 | 6.419619003869045 |
| EXP-FC-003 | 1 | rmse | sarimax | model | 2.1162698134251936e+91 | 645 | 5 | 7 | lower | 0.9162421276236167 | 2.1162698134251936e+91 |
| EXP-FC-003 | 3 | directional_accuracy | ets | model | 0.5354330708661418 | 635 | 5 | 1 | higher | 0.5007874015748032 | 0.03464566929133861 |
| EXP-FC-003 | 3 | directional_accuracy | sarimax | model | 0.5039370078740157 | 635 | 5 | 2 | higher | 0.5007874015748032 | 0.0031496062992125706 |
| EXP-FC-003 | 3 | directional_accuracy | moving_average_rule | baseline | 0.5007874015748032 | 635 | 5 | 3 | higher | 0.5007874015748032 | 0.0 |
| EXP-FC-003 | 3 | directional_accuracy | lightgbm | model | 0.4866141732283465 | 635 | 5 | 4 | higher | 0.5007874015748032 | -0.014173228346456679 |
| EXP-FC-003 | 3 | directional_accuracy | xgboost | model | 0.4818897637795276 | 635 | 5 | 5 | higher | 0.5007874015748032 | -0.01889763779527559 |
| EXP-FC-003 | 3 | directional_accuracy | persistence | baseline | 0.029921259842519605 | 635 | 5 | 6 | higher | 0.5007874015748032 | -0.4708661417322836 |
| EXP-FC-003 | 3 | directional_accuracy | zero_return | baseline | 0.029921259842519605 | 635 | 5 | 7 | higher | 0.5007874015748032 | -0.4708661417322836 |
| EXP-FC-003 | 3 | mae | persistence | baseline | 1.2470866141732282 | 635 | 5 | 1 | lower | 1.2470866141732282 | 0.0 |
| EXP-FC-003 | 3 | mae | zero_return | baseline | 1.2470866141732282 | 635 | 5 | 2 | lower | 1.2470866141732282 | 0.0 |
| EXP-FC-003 | 3 | mae | sarimax | model | 1.31313331518028 | 635 | 5 | 3 | lower | 1.2470866141732282 | 0.06604670100705179 |
| EXP-FC-003 | 3 | mae | moving_average_rule | baseline | 1.4494519685039369 | 635 | 5 | 4 | lower | 1.2470866141732282 | 0.20236535433070868 |
| EXP-FC-003 | 3 | mae | lightgbm | model | 5.253772830247582 | 635 | 5 | 5 | lower | 1.2470866141732282 | 4.006686216074354 |
| EXP-FC-003 | 3 | mae | xgboost | model | 5.627706011794684 | 635 | 5 | 6 | lower | 1.2470866141732282 | 4.380619397621456 |
| EXP-FC-003 | 3 | mae | ets | model | 7.493465512998097 | 635 | 5 | 7 | lower | 1.2470866141732282 | 6.246378898824869 |
| EXP-FC-003 | 3 | mape | persistence | baseline | 1.7754280934280153 | 635 | 5 | 1 | lower | 1.7754280934280153 | 0.0 |
