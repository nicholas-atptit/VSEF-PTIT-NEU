# Phase 4 Regime-aware Analysis Report

## 1. Executive summary

Persistence is the stable MAE winner in several observed ranking contexts; this supports baseline competitiveness and does not prove a universal ML model across future regimes.
Risk-aware diagnostics show mixed context-specific improvements in selected metric/regime rows, but this does not support a universal risk-layer improvement claim.
Horizon ranking is relatively stable in the observed rows, but this remains exploratory under the rule-based regime definition.

## 2. Phase 4 objective

Phase 4 tests whether forecasting quality and risk-aware utility depend on market regime rather than trying to prove one universal best model.

## 3. Relation to prior phases

- Phase 0 froze VSEF v1 governance: vnstock_data, daily OHLCV, diagnostic-only outputs.
- Phase 1 standardized config-driven experiment execution.
- Phase 2 validated forecasting core against baselines but did not prove consistent model superiority on MAE/RMSE.
- Phase 3 evaluated risk-aware diagnostic candidates but did not prove aggregate risk-aware improvement over forecast-only ranking.

## 4. Regime policy definition

- Return window: 20
- Volatility window: 20
- Minimum periods: 10
- Rolling return rule: bull >= 0.03, bear <= -0.03, sideway within {'lower': -0.03, 'upper': 0.03}
- Realized volatility rule: expanding ticker-level quantiles, high_vol >= q0.7, low_vol <= q0.3, fallback=median_split
- Robustness alternatives: [{"id": "conservative", "bull_threshold": 0.05, "bear_threshold": -0.05, "high_vol_quantile": 0.75, "low_vol_quantile": 0.25}, {"id": "loose", "bull_threshold": 0.02, "bear_threshold": -0.02, "high_vol_quantile": 0.65, "low_vol_quantile": 0.35}]
- Limitation: rule-based labels are transparent diagnostics but can be sensitive to threshold choice and ticker-specific volatility history.

## 5. Regime label dataset

- Tickers: ACB, DGC, FPT, HPG, MWG
- Date range: 2023-01-03 to 2024-12-31
- Total observations: 2495
- Insufficient-history observations: 50

Trend distribution:

| trend_regime | rows |
| --- | --- |
| bull | 1057 |
| sideway | 949 |
| bear | 439 |
| insufficient_history | 50 |

Volatility distribution:

| volatility_regime | rows |
| --- | --- |
| low_vol | 1276 |
| high_vol | 1124 |
| insufficient_history | 95 |

Summary sample:

| ticker | trend_regime | volatility_regime | combined_regime | observation_count | start_date | end_date | mean_return | mean_realized_volatility |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ACB | bear | high_vol | bear_high_vol | 23 | 2023-02-24 | 2024-08-16 | -0.00027885382544495677 | 0.014811110125867482 |
| ACB | bear | low_vol | bear_low_vol | 32 | 2023-02-28 | 2024-11-20 | -0.0012998587313778655 | 0.010227168231744754 |
| ACB | bull | high_vol | bull_high_vol | 71 | 2023-02-06 | 2024-10-17 | 0.0020312131884906645 | 0.014710192155174828 |
| ACB | bull | insufficient_history | insufficient_history | 9 | 2023-01-17 | 2023-02-03 | -0.0020572647268595114 | 0.01637968952136677 |
| ACB | bull | low_vol | bull_low_vol | 94 | 2023-04-03 | 2024-12-31 | 0.0028554798184038327 | 0.008401907106338171 |
| ACB | insufficient_history | insufficient_history | insufficient_history | 10 | 2023-01-03 | 2023-01-16 | 0.01090344835000361 |  |
| ACB | sideway | high_vol | sideway_high_vol | 114 | 2023-02-14 | 2024-11-05 | -0.000321423715026714 | 0.013953380247822868 |
| ACB | sideway | low_vol | sideway_low_vol | 146 | 2023-02-16 | 2024-12-30 | 0.0007652632881612301 | 0.009189820410856111 |
| DGC | bear | high_vol | bear_high_vol | 60 | 2023-02-07 | 2024-08-16 | -0.0029675423001613503 | 0.02484351400104794 |
| DGC | bear | low_vol | bear_low_vol | 49 | 2023-02-10 | 2024-11-08 | -0.002939671770269686 | 0.013211388427415392 |
| DGC | bull | high_vol | bull_high_vol | 118 | 2023-06-05 | 2024-08-30 | 0.007697885714319466 | 0.0257586669759675 |
| DGC | bull | low_vol | bull_low_vol | 97 | 2023-03-27 | 2024-12-31 | 0.0044135048867614225 | 0.014936856843587325 |
| DGC | insufficient_history | insufficient_history | insufficient_history | 10 | 2023-01-03 | 2023-01-16 | -0.006159948129041011 |  |
| DGC | sideway | high_vol | sideway_high_vol | 35 | 2023-08-22 | 2024-08-28 | -0.0038926278316532147 | 0.024521801044311536 |
| DGC | sideway | insufficient_history | insufficient_history | 9 | 2023-01-17 | 2023-02-03 | 0.004838601996574178 | 0.0196919545220497 |
| DGC | sideway | low_vol | sideway_low_vol | 121 | 2023-02-06 | 2024-12-12 | -0.00012730672063690414 | 0.012050995260062326 |
| FPT | bear | high_vol | bear_high_vol | 45 | 2023-03-17 | 2024-08-13 | -0.004728493022682684 | 0.01603120169746505 |
| FPT | bear | low_vol | bear_low_vol | 4 | 2023-02-24 | 2024-11-07 | -0.001848968978918225 | 0.008472402903948075 |
| FPT | bull | high_vol | bull_high_vol | 217 | 2023-04-03 | 2024-12-26 | 0.0044414421733708735 | 0.014986474177322793 |
| FPT | bull | insufficient_history | insufficient_history | 5 | 2023-01-18 | 2023-01-31 | 0.0036492425585033794 | 0.00726853468545148 |

## 6. Model health / eligibility gate

- Minimum prediction count: 10
- Maximum missing prediction rate: 0.2
- Maximum error std: None
- Extreme prediction z-score threshold: 5.0
- Baseline gap ratio: 2.0
- The health gate is diagnostic and is not used to polish results.

| status | rows |
| --- | --- |
| eligible | 742 |
| flagged | 608 |
| excluded | 300 |

| reason | rows |
| --- | --- |
| none | 742 |
| baseline_gap_flag | 554 |
| minimum_prediction_count_not_met | 300 |
| extreme_prediction_flag;baseline_gap_flag | 53 |
| extreme_prediction_flag | 1 |

## 7. Model performance by regime

| experiment_id | horizon | regime_column | regime | model_name | model_type | mae | rmse | mape | directional_accuracy | prediction_count | missing_prediction_rate | error_std | rank | is_best | is_baseline | best_baseline_mae | model_vs_best_baseline_mae_gap | baseline_competitive | small_sample_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | 1 | trend_regime | bear | persistence | baseline | 0.861372549019608 | 1.4774883740404987 | 1.2793502814680573 | 0.058823529411764705 | 153 | 0.0 | 1.4772147956474766 | 1 | True | True | 0.861372549019608 | 0.0 | True | False |
| EXP-FC-001 | 1 | trend_regime | bull | persistence | baseline | 0.6517127071823207 | 1.0367298779233696 | 0.8037149955793037 | 0.06629834254143646 | 181 | 0.0 | 1.0361229397683533 | 1 | True | True | 0.6517127071823207 | 0.0 | True | False |
| EXP-FC-001 | 1 | trend_regime | sideway | persistence | baseline | 0.5517363344051445 | 0.9077513647430809 | 0.8610594933263729 | 0.08360128617363344 | 311 | 0.0 | 0.9076678237086784 | 1 | True | True | 0.5517363344051445 | 0.0 | True | False |
| EXP-FC-003 | 1 | trend_regime | bear | persistence | baseline | 0.861372549019608 | 1.4774883740404987 | 1.2793502814680573 | 0.058823529411764705 | 153 | 0.0 | 1.4772147956474766 | 1 | True | True | 0.861372549019608 | 0.0 | True | False |
| EXP-FC-003 | 1 | trend_regime | bull | persistence | baseline | 0.6517127071823207 | 1.0367298779233696 | 0.8037149955793037 | 0.06629834254143646 | 181 | 0.0 | 1.0361229397683533 | 1 | True | True | 0.6517127071823207 | 0.0 | True | False |
| EXP-FC-003 | 1 | trend_regime | sideway | persistence | baseline | 0.5517363344051445 | 0.9077513647430809 | 0.8610594933263729 | 0.08360128617363344 | 311 | 0.0 | 0.9076678237086784 | 1 | True | True | 0.5517363344051445 | 0.0 | True | False |
| EXP-FC-003 | 3 | trend_regime | bear | persistence | baseline | 1.4936601307189545 | 2.5952509418128322 | 2.1233357057337927 | 0.032679738562091505 | 153 | 0.0 | 2.592374423547777 | 1 | True | True | 1.4936601307189545 | 0.0 | True | False |
| EXP-FC-003 | 3 | trend_regime | bull | persistence | baseline | 1.1751412429378527 | 1.8617863968701185 | 1.5086272819965207 | 0.05649717514124294 | 177 | 0.0 | 1.8615986350766676 | 1 | True | True | 1.1751412429378527 | 0.0 | True | False |
| EXP-FC-003 | 3 | trend_regime | sideway | persistence | baseline | 1.1651475409836065 | 1.8515975379717242 | 1.75573589323323 | 0.013114754098360656 | 305 | 0.0 | 1.8505304674656244 | 1 | True | True | 1.1651475409836065 | 0.0 | True | False |
| EXP-FC-003 | 5 | trend_regime | bear | persistence | baseline | 1.8626143790849667 | 3.1922618858283567 | 2.6435523346680903 | 0.026143790849673203 | 153 | 0.0 | 3.1817733408509588 | 1 | True | True | 1.8626143790849667 | 0.0 | True | False |
| EXP-FC-003 | 5 | trend_regime | bull | persistence | baseline | 1.4188888888888886 | 2.2410371189297713 | 1.85970245842409 | 0.04678362573099415 | 171 | 0.0 | 2.2319844605629577 | 1 | True | True | 1.4188888888888886 | 0.0 | True | False |
| EXP-FC-003 | 5 | trend_regime | sideway | persistence | baseline | 1.569833887043189 | 2.466281449073992 | 2.305740356058101 | 0.026578073089700997 | 301 | 0.0 | 2.463051551340148 | 1 | True | True | 1.569833887043189 | 0.0 | True | False |

Baseline competitiveness by regime:

| experiment_id | horizon | regime_column | regime | model_name | model_type | mae | rmse | mape | directional_accuracy | prediction_count | missing_prediction_rate | error_std | rank | is_best | is_baseline | best_baseline_mae | model_vs_best_baseline_mae_gap | baseline_competitive | small_sample_flag |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | 1 | combined_regime | bear_high_vol | persistence | baseline | 0.987222222222222 | 1.6607868505509567 | 1.2448773535662878 | 0.07407407407407407 | 54 | 0.0 | 1.6536072822456214 | 1 | True | True | 0.987222222222222 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bear_high_vol | zero_return | baseline | 0.987222222222222 | 1.6607868505509567 | 1.2448773535662878 | 0.07407407407407407 | 54 | 0.0 | 1.6536072822456214 | 2 | False | True | 0.987222222222222 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bear_high_vol | moving_average_rule | baseline | 1.6728518518518516 | 2.585158079556508 | 2.0492870548696507 | 0.5370370370370371 | 54 | 0.0 | 2.5716693251601805 | 3 | False | True | 0.987222222222222 | 0.6856296296296296 | True | False |
| EXP-FC-001 | 1 | combined_regime | bear_low_vol | persistence | baseline | 0.7927272727272731 | 1.3671883704903431 | 1.298153696687204 | 0.050505050505050504 | 99 | 0.0 | 1.366597174728426 | 1 | True | True | 0.7927272727272731 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bear_low_vol | zero_return | baseline | 0.7927272727272731 | 1.3671883704903431 | 1.298153696687204 | 0.050505050505050504 | 99 | 0.0 | 1.366597174728426 | 2 | False | True | 0.7927272727272731 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bear_low_vol | moving_average_rule | baseline | 1.257393939393939 | 2.224782754120119 | 2.0287641208022364 | 0.40404040404040403 | 99 | 0.0 | 2.139638069645094 | 3 | False | True | 0.7927272727272731 | 0.464666666666666 | True | False |
| EXP-FC-001 | 1 | combined_regime | bull_high_vol | persistence | baseline | 0.9847058823529418 | 1.3511030352140818 | 0.9077441336813515 | 0.04411764705882353 | 68 | 0.0 | 1.3478618849495128 | 1 | True | True | 0.9847058823529418 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bull_high_vol | zero_return | baseline | 0.9847058823529418 | 1.3511030352140818 | 0.9077441336813515 | 0.04411764705882353 | 68 | 0.0 | 1.3478618849495128 | 2 | False | True | 0.9847058823529418 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bull_high_vol | moving_average_rule | baseline | 1.6721470588235279 | 2.2501507531196006 | 1.611830560890462 | 0.4411764705882353 | 68 | 0.0 | 2.0021866668624924 | 3 | False | True | 0.9847058823529418 | 0.6874411764705861 | True | False |
| EXP-FC-001 | 1 | combined_regime | bull_low_vol | persistence | baseline | 0.4513274336283185 | 0.7893528210541817 | 0.7411133903497531 | 0.07964601769911504 | 113 | 0.0 | 0.7893526424693775 | 1 | True | True | 0.4513274336283185 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bull_low_vol | zero_return | baseline | 0.4513274336283185 | 0.7893528210541817 | 0.7411133903497531 | 0.07964601769911504 | 113 | 0.0 | 0.7893526424693775 | 2 | False | True | 0.4513274336283185 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | bull_low_vol | moving_average_rule | baseline | 0.7451150442477887 | 1.2416334516750847 | 1.2187221375318538 | 0.4690265486725664 | 113 | 0.0 | 1.2170485906433006 | 3 | False | True | 0.4513274336283185 | 0.29378761061947023 | True | False |
| EXP-FC-001 | 1 | combined_regime | sideway_high_vol | persistence | baseline | 0.8064705882352946 | 1.2174007991957103 | 0.8957923338730306 | 0.11764705882352941 | 68 | 0.0 | 1.209694226285001 | 1 | True | True | 0.8064705882352946 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | sideway_high_vol | zero_return | baseline | 0.8064705882352946 | 1.2174007991957103 | 0.8957923338730306 | 0.11764705882352941 | 68 | 0.0 | 1.209694226285001 | 2 | False | True | 0.8064705882352946 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | sideway_high_vol | moving_average_rule | baseline | 1.411499999999999 | 1.976031629301515 | 1.6078136821805484 | 0.39705882352941174 | 68 | 0.0 | 1.9639468406518985 | 3 | False | True | 0.8064705882352946 | 0.6050294117647045 | True | False |
| EXP-FC-001 | 1 | combined_regime | sideway_low_vol | persistence | baseline | 0.48045267489711885 | 0.7999164050974531 | 0.8513400153133164 | 0.07407407407407407 | 243 | 0.0 | 0.7995996136037244 | 1 | True | True | 0.48045267489711885 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | sideway_low_vol | zero_return | baseline | 0.48045267489711885 | 0.7999164050974531 | 0.8513400153133164 | 0.07407407407407407 | 243 | 0.0 | 0.7995996136037244 | 2 | False | True | 0.48045267489711885 | 0.0 | True | False |
| EXP-FC-001 | 1 | combined_regime | sideway_low_vol | moving_average_rule | baseline | 0.6994156378600828 | 1.133097702447667 | 1.2298099863305296 | 0.4403292181069959 | 243 | 0.0 | 1.1330839210812713 | 3 | False | True | 0.48045267489711885 | 0.21896296296296391 | True | False |
| EXP-FC-001 | 1 | trend_regime | bear | persistence | baseline | 0.861372549019608 | 1.4774883740404987 | 1.2793502814680573 | 0.058823529411764705 | 153 | 0.0 | 1.4772147956474766 | 1 | True | True | 0.861372549019608 | 0.0 | True | False |
| EXP-FC-001 | 1 | trend_regime | bear | zero_return | baseline | 0.861372549019608 | 1.4774883740404987 | 1.2793502814680573 | 0.058823529411764705 | 153 | 0.0 | 1.4772147956474766 | 2 | False | True | 0.861372549019608 | 0.0 | True | False |

Ranking consistency:

| experiment_id | horizon | regime_column | model_name | model_type | regime_count | mean_rank | rank_std | best_regime_count | rank_min | rank_max | stable_best_across_observed_regimes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | 1 | combined_regime | bilstm | model | 6 | 8.0 | 0.0 | 0 | 8 | 8 | False |
| EXP-FC-001 | 1 | combined_regime | ets | model | 6 | 5.333333333333333 | 0.9428090415820634 | 0 | 4 | 6 | False |
| EXP-FC-001 | 1 | combined_regime | lightgbm | model | 6 | 4.333333333333333 | 0.4714045207910317 | 0 | 4 | 5 | False |
| EXP-FC-001 | 1 | combined_regime | lstm | model | 6 | 7.0 | 0.0 | 0 | 7 | 7 | False |
| EXP-FC-001 | 1 | combined_regime | moving_average_rule | baseline | 6 | 3.0 | 0.0 | 0 | 3 | 3 | False |
| EXP-FC-001 | 1 | combined_regime | persistence | baseline | 6 | 1.0 | 0.0 | 6 | 1 | 1 | True |
| EXP-FC-001 | 1 | combined_regime | sarimax | model | 6 | 9.0 | 0.0 | 0 | 9 | 9 | False |
| EXP-FC-001 | 1 | combined_regime | xgboost | model | 6 | 5.333333333333333 | 0.4714045207910317 | 0 | 5 | 6 | False |
| EXP-FC-001 | 1 | combined_regime | zero_return | baseline | 6 | 2.0 | 0.0 | 0 | 2 | 2 | False |
| EXP-FC-001 | 1 | trend_regime | bilstm | model | 3 | 8.0 | 0.0 | 0 | 8 | 8 | False |
| EXP-FC-001 | 1 | trend_regime | ets | model | 3 | 5.333333333333333 | 0.9428090415820634 | 0 | 4 | 6 | False |
| EXP-FC-001 | 1 | trend_regime | lightgbm | model | 3 | 4.333333333333333 | 0.4714045207910317 | 0 | 4 | 5 | False |
| EXP-FC-001 | 1 | trend_regime | lstm | model | 3 | 7.0 | 0.0 | 0 | 7 | 7 | False |
| EXP-FC-001 | 1 | trend_regime | moving_average_rule | baseline | 3 | 3.0 | 0.0 | 0 | 3 | 3 | False |
| EXP-FC-001 | 1 | trend_regime | persistence | baseline | 3 | 1.0 | 0.0 | 3 | 1 | 1 | True |
| EXP-FC-001 | 1 | trend_regime | sarimax | model | 3 | 9.0 | 0.0 | 0 | 9 | 9 | False |
| EXP-FC-001 | 1 | trend_regime | xgboost | model | 3 | 5.333333333333333 | 0.4714045207910317 | 0 | 5 | 6 | False |
| EXP-FC-001 | 1 | trend_regime | zero_return | baseline | 3 | 2.0 | 0.0 | 0 | 2 | 2 | False |
| EXP-FC-001 | 1 | volatility_regime | bilstm | model | 2 | 8.0 | 0.0 | 0 | 8 | 8 | False |
| EXP-FC-001 | 1 | volatility_regime | ets | model | 2 | 5.0 | 1.0 | 0 | 4 | 6 | False |

## 8. Risk layer by regime

Risk-aware diagnostics show mixed context-specific improvements in selected metric/regime rows, but this does not support a universal risk-layer improvement claim.

| regime_column | regime | horizon | top_n | forecast_only_average_realized_return | risk_aware_average_realized_return | average_realized_return_delta | forecast_only_hit_ratio | risk_aware_hit_ratio | hit_ratio_delta | forecast_only_max_drawdown | risk_aware_max_drawdown | max_drawdown_improvement | forecast_only_var_95 | risk_aware_var_95 | var_95_improvement | forecast_only_cvar_95 | risk_aware_cvar_95 | cvar_95_improvement | candidate_count_forecast_only | candidate_count_risk_aware |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_regime | bear_high_vol | 1 | 1 | -0.0017803970807459498 | -0.0012556315543105 | 0.0005247655264354499 | 0.42857142857142855 | 0.3333333333333333 | -0.09523809523809523 | -0.05506056662328562 | -0.026908261856826377 | 0.028152304766459246 | -0.025942182774108417 | -0.014266802724840657 | 0.01167538004926776 | -0.0550605666232856 | -0.0171773444753945 | 0.0378832221478911 | 14 | 9 |
| combined_regime | bear_high_vol | 1 | 3 | 0.00029188703211937666 | 0.00029188703211937666 | 0.0 | 0.375 | 0.375 | 0.0 | -0.05506056662328562 | -0.05506056662328562 | 0.0 | -0.01834835058835793 | -0.01834835058835793 | 0.0 | -0.0369871458584214 | -0.0369871458584214 | 0.0 | 30 | 30 |
| combined_regime | bear_high_vol | 1 | 5 | 0.00029188703211937666 | 0.00029188703211937666 | 0.0 | 0.375 | 0.375 | 0.0 | -0.05506056662328562 | -0.05506056662328562 | 0.0 | -0.01834835058835793 | -0.01834835058835793 | 0.0 | -0.0369871458584214 | -0.0369871458584214 | 0.0 | 30 | 30 |
| combined_regime | bear_high_vol | 3 | 1 | 0.007296612973440278 | 0.0033439631367566287 | -0.003952649836683649 | 0.5789473684210527 | 0.47058823529411764 | -0.10835913312693501 | -0.07153306018023997 | -0.04788564961690889 | 0.023647410563331084 | -0.023610788007380284 | -0.03244660214701384 | -0.008835814139633552 | -0.0619544592030361 | -0.0619544592030361 | 0.0 | 19 | 17 |
| combined_regime | bear_high_vol | 3 | 3 | 0.0026602698472585266 | -0.0004697640849525862 | -0.003130033932211113 | 0.4074074074074074 | 0.4074074074074074 | 0.0 | -0.09685379714886388 | -0.09685379714886388 | 0.0 | -0.020926385339902594 | -0.020926385339902594 | 0.0 | -0.0420822340964581 | -0.0420822340964581 | 0.0 | 41 | 41 |
| combined_regime | bear_high_vol | 3 | 5 | 0.0012097970760589502 | 0.0012097970760589502 | 0.0 | 0.4074074074074074 | 0.4074074074074074 | 0.0 | -0.096853797148864 | -0.096853797148864 | 0.0 | -0.020926385339902594 | -0.020926385339902594 | 0.0 | -0.0420822340964581 | -0.0420822340964581 | 0.0 | 45 | 45 |
| combined_regime | bear_high_vol | 5 | 1 | 0.010166728316714165 | 0.011110003794752058 | 0.0009432754780378926 | 0.65 | 0.47619047619047616 | -0.17380952380952386 | -0.17340770423960883 | -0.08398632250707172 | 0.08942138173253711 | -0.056421103739603275 | -0.0226958206476611 | 0.03372528309194217 | -0.1044592030360531 | -0.02588413551995425 | 0.07857506751609886 | 20 | 21 |
| combined_regime | bear_high_vol | 5 | 3 | 0.004634079675345746 | 0.004634079675345746 | 0.0 | 0.5625 | 0.5625 | 0.0 | -0.1722712828945947 | -0.1722712828945947 | 0.0 | -0.04024159994153635 | -0.04024159994153635 | 0.0 | -0.0791759928800269 | -0.0791759928800269 | 0.0 | 47 | 47 |
| combined_regime | bear_high_vol | 5 | 5 | 0.003332084955108128 | 0.003332084955108128 | 0.0 | 0.5625 | 0.5625 | 0.0 | -0.1722712828945947 | -0.1722712828945947 | 0.0 | -0.04024159994153635 | -0.04024159994153635 | 0.0 | -0.0791759928800269 | -0.0791759928800269 | 0.0 | 51 | 51 |
| combined_regime | bear_low_vol | 1 | 1 | -0.00041887995904353836 | -0.0005027860344199442 | -8.390607537640582e-05 | 0.5161290322580645 | 0.52 | 0.003870967741935516 | -0.07704376709649252 | -0.0605409846999001 | 0.01650278239659242 | -0.022632090370647985 | -0.01817444040268518 | 0.004457649967962805 | -0.04100601713209185 | -0.03326158228731995 | 0.0077444348447719 | 31 | 25 |
| combined_regime | bear_low_vol | 1 | 3 | -0.0008869656352713594 | -0.0008869656352713594 | 0.0 | 0.4883720930232558 | 0.4883720930232558 | 0.0 | -0.07166197122410589 | -0.07166197122410589 | 0.0 | -0.035267869974847155 | -0.035267869974847155 | 0.0 | -0.044369640750744455 | -0.044369640750744455 | 0.0 | 53 | 53 |
| combined_regime | bear_low_vol | 1 | 5 | -0.0008869656352713594 | -0.0008869656352713594 | 0.0 | 0.4883720930232558 | 0.4883720930232558 | 0.0 | -0.07166197122410589 | -0.07166197122410589 | 0.0 | -0.035267869974847155 | -0.035267869974847155 | 0.0 | -0.044369640750744455 | -0.044369640750744455 | 0.0 | 53 | 53 |
| combined_regime | bear_low_vol | 3 | 1 | -0.003103024610332449 | -0.00487353641361793 | -0.0017705118032854813 | 0.45454545454545453 | 0.34782608695652173 | -0.1067193675889328 | -0.1811794417162076 | -0.20606435142020618 | -0.024884909703998592 | -0.04647358888205031 | -0.04608839286009108 | 0.0003851960219592304 | -0.07180766980024 | -0.0467657608198987 | 0.025041908980341303 | 22 | 23 |
| combined_regime | bear_low_vol | 3 | 3 | -0.0016849705557112835 | -0.0010403560568429066 | 0.0006446144988683769 | 0.45454545454545453 | 0.45454545454545453 | 0.0 | -0.2711133506867869 | -0.25013268558960633 | 0.02098066509718055 | -0.04722167766821755 | -0.04722167766821755 | 0.0 | -0.05569673451393257 | -0.05569673451393257 | 0.0 | 51 | 51 |
| combined_regime | bear_low_vol | 3 | 5 | -0.0014051664807766988 | -0.0014051664807766988 | 0.0 | 0.45454545454545453 | 0.45454545454545453 | 0.0 | -0.2621741489275047 | -0.2621741489275047 | 0.0 | -0.04722167766821755 | -0.04722167766821755 | 0.0 | -0.05569673451393257 | -0.05569673451393257 | 0.0 | 53 | 53 |
| combined_regime | bear_low_vol | 5 | 1 | 0.0010717862632151104 | 0.003661375299724789 | 0.0025895890365096787 | 0.65 | 0.6111111111111112 | -0.03888888888888886 | -0.1592193684561758 | -0.1075874949940483 | 0.0516318734621275 | -0.05368483692919588 | -0.025928239208867998 | 0.02775659772032788 | -0.0569439145364364 | -0.0423467137185707 | 0.014597200817865703 | 20 | 18 |
| combined_regime | bear_low_vol | 5 | 3 | -0.003040022771426157 | -0.0001317114715464331 | 0.002908311299879724 | 0.5476190476190477 | 0.5555555555555556 | 0.007936507936507908 | -0.29264773097353414 | -0.27178682364545514 | 0.020860907328078993 | -0.054064835507852946 | -0.04500760160622038 | 0.009057233901632565 | -0.0645878750341532 | -0.052436962330064577 | 0.01215091270408862 | 46 | 38 |
| combined_regime | bear_low_vol | 5 | 5 | -0.0013027225677217221 | -0.0013027225677217221 | 0.0 | 0.5555555555555556 | 0.5555555555555556 | 0.0 | -0.2689905105862931 | -0.2689905105862931 | 0.0 | -0.053096178868248696 | -0.053096178868248696 | 0.0 | -0.0645878750341532 | -0.0645878750341532 | 0.0 | 55 | 55 |
| combined_regime | bull_high_vol | 1 | 1 | 0.0028414255257067002 | 0.0028414255257067002 | 0.0 | 0.5 | 0.5 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0010846787402047995 | -0.0010846787402047995 | 0.0 | -0.0015209125475283 | -0.0015209125475283 | 0.0 | 2 | 2 |
| combined_regime | bull_high_vol | 1 | 3 | 0.006432335229103966 | 0.006432335229103966 | 0.0 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | -0.009047824213700872 | -0.009047824213700872 | 0.0 | -0.007166096297157748 | -0.007166096297157748 | 0.0 | -0.0090478242137009 | -0.0090478242137009 | 0.0 | 6 | 6 |
| combined_regime | bull_high_vol | 1 | 5 | 0.006432335229103966 | 0.006432335229103966 | 0.0 | 0.6666666666666666 | 0.6666666666666666 | 0.0 | -0.009047824213700872 | -0.009047824213700872 | 0.0 | -0.007166096297157748 | -0.007166096297157748 | 0.0 | -0.0090478242137009 | -0.0090478242137009 | 0.0 | 6 | 6 |
| combined_regime | bull_high_vol | 3 | 3 | 0.00849572433194525 | 0.00849572433194525 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0018609254290357202 | 0.0018609254290357202 | 0.0 | 0.0017233950883239 | 0.0017233950883239 | 0.0 | 4 | 4 |
| combined_regime | bull_high_vol | 3 | 5 | 0.00849572433194525 | 0.00849572433194525 | 0.0 | 1.0 | 1.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0018609254290357202 | 0.0018609254290357202 | 0.0 | 0.0017233950883239 | 0.0017233950883239 | 0.0 | 4 | 4 |
| combined_regime | bull_high_vol | 5 | 1 | 0.0242045999616622 | 0.04698084212131765 | 0.02277624215965545 | 0.6666666666666666 | 1.0 | 0.33333333333333337 | -0.021347884357648672 | 0.0 | 0.021347884357648672 | -0.017238334338221376 | 0.022470938465093817 | 0.039709272803315196 | -0.0213478843576487 | 0.0197476158366245 | 0.0410955001942732 | 3 | 2 |
| combined_regime | bull_high_vol | 5 | 3 | 0.005810384882655069 | 0.007962613140404187 | 0.002152228257749118 | 0.5 | 0.5384615384615384 | 0.038461538461538436 | -0.056735716844073525 | -0.04480087207464878 | 0.011934844769424746 | -0.021224963753158438 | -0.020562696343982733 | 0.0006622674091757048 | -0.0285099052540912 | -0.0285099052540912 | 0.0 | 13 | 14 |
| combined_regime | bull_high_vol | 5 | 5 | 0.007962613140404187 | 0.007962613140404187 | 0.0 | 0.5384615384615384 | 0.5384615384615384 | 0.0 | -0.04480087207464878 | -0.04480087207464878 | 0.0 | -0.020562696343982733 | -0.020562696343982733 | 0.0 | -0.0285099052540912 | -0.0285099052540912 | 0.0 | 14 | 14 |
| combined_regime | bull_low_vol | 1 | 1 | -0.004609204681207842 | -0.00424302989072521 | 0.00036617479048263197 | 0.23076923076923078 | 0.32142857142857145 | 0.09065934065934067 | -0.12149096456894481 | -0.12049039397568151 | 0.0010005705932633058 | -0.018152444760388423 | -0.018041768430230552 | 0.00011067633015787032 | -0.030653126127519498 | -0.030653126127519498 | 0.0 | 26 | 28 |
| combined_regime | bull_low_vol | 1 | 3 | -0.001412610772318854 | -0.001412610772318854 | 0.0 | 0.4107142857142857 | 0.4107142857142857 | 0.0 | -0.0917495130135364 | -0.0917495130135364 | 0.0 | -0.013360120704211763 | -0.013360120704211763 | 0.0 | -0.025599620289705766 | -0.025599620289705766 | 0.0 | 68 | 68 |
| combined_regime | bull_low_vol | 1 | 5 | -0.001412610772318854 | -0.001412610772318854 | 0.0 | 0.4107142857142857 | 0.4107142857142857 | 0.0 | -0.0917495130135364 | -0.0917495130135364 | 0.0 | -0.013360120704211763 | -0.013360120704211763 | 0.0 | -0.025599620289705766 | -0.025599620289705766 | 0.0 | 68 | 68 |
| combined_regime | bull_low_vol | 3 | 1 | 0.006284265074613594 | -0.007891550899619564 | -0.014175815974233157 | 0.6153846153846154 | 0.25 | -0.3653846153846154 | -0.02139831017443694 | -0.04199854079036358 | -0.02060023061592664 | -0.017008138440356417 | -0.018520187830300617 | -0.0015120493899441999 | -0.0206370569762225 | -0.0206370569762225 | 0.0 | 13 | 8 |

## 9. Horizon by regime

Horizon ranking is relatively stable in the observed rows, but this remains exploratory under the rule-based regime definition.

| experiment_id | regime_column | regime | horizon | mae | rmse | directional_accuracy | prediction_count | rank | is_best_horizon |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-003 | combined_regime | bear_high_vol | 3 | 3.1230400767182953 | 5.379526154960469 | 0.3941798941798942 | 378 | 1 | True |
| EXP-FC-003 | combined_regime | bear_high_vol | 5 | 3.8573162155863745 | 6.4748021883079 | 0.3888888888888889 | 378 | 2 | False |
| EXP-FC-003 | combined_regime | bear_high_vol | 1 | 1.5050850615054326e+23 | 1.7512729882308843e+24 | 0.36507936507936506 | 378 | 3 | False |
| EXP-FC-003 | combined_regime | bear_low_vol | 3 | 2.925901240131548 | 5.377211929487251 | 0.33766233766233766 | 693 | 1 | True |
| EXP-FC-003 | combined_regime | bear_low_vol | 5 | 3.054096211983072 | 5.688199945856024 | 0.3477633477633478 | 693 | 2 | False |
| EXP-FC-003 | combined_regime | bear_low_vol | 1 | 7.56954115887848e+69 | 1.6505354041743336e+71 | 0.34054834054834054 | 693 | 3 | False |
| EXP-FC-003 | combined_regime | bull_high_vol | 3 | 6.024472416227016 | 9.766145170329962 | 0.36764705882352944 | 476 | 1 | True |
| EXP-FC-003 | combined_regime | bull_high_vol | 5 | 7.9077577144087865 | 12.558493048600862 | 0.3658008658008658 | 462 | 2 | False |
| EXP-FC-003 | combined_regime | bull_high_vol | 1 | 3.315977147151693e+52 | 5.9924861801197e+53 | 0.3739495798319328 | 476 | 3 | False |
| EXP-FC-003 | combined_regime | bull_low_vol | 3 | 2.7023958408377964 | 5.394167415464191 | 0.37876802096985585 | 763 | 1 | True |
| EXP-FC-003 | combined_regime | bull_low_vol | 5 | 2.819384533295397 | 5.590179307232647 | 0.37142857142857144 | 735 | 2 | False |
| EXP-FC-003 | combined_regime | bull_low_vol | 1 | 4.1774251343307836e+83 | 1.1291822054755586e+85 | 0.37168141592920356 | 791 | 3 | False |
| EXP-FC-003 | combined_regime | sideway_high_vol | 3 | 4.367474162133698 | 6.81242724443398 | 0.35714285714285715 | 476 | 1 | True |
| EXP-FC-003 | combined_regime | sideway_high_vol | 5 | 6.094264199499031 | 10.349597198448949 | 0.35084033613445376 | 476 | 2 | False |
| EXP-FC-003 | combined_regime | sideway_high_vol | 1 | 1.020139099014447e+62 | 1.855471729853334e+63 | 0.3235294117647059 | 476 | 3 | False |
| EXP-FC-003 | combined_regime | sideway_low_vol | 3 | 2.8869668561304946 | 5.802942823543246 | 0.37010247136829416 | 1659 | 1 | True |
| EXP-FC-003 | combined_regime | sideway_low_vol | 5 | 3.4563449009565965 | 7.241413696074729 | 0.3678724708767627 | 1631 | 2 | False |
| EXP-FC-003 | combined_regime | sideway_low_vol | 1 | 8.529877925380928e+89 | 2.9139610330335344e+91 | 0.3462669018224574 | 1701 | 3 | False |
| EXP-FC-003 | trend_regime | bear | 3 | 2.995479653044518 | 5.378028828659867 | 0.357609710550887 | 1071 | 1 | True |
| EXP-FC-003 | trend_regime | bear | 5 | 3.3375856250195324 | 5.977655379094529 | 0.3622782446311858 | 1071 | 2 | False |
| EXP-FC-003 | trend_regime | bear | 1 | 4.897938396921369e+69 | 1.32769012873004e+71 | 0.3492063492063492 | 1071 | 3 | False |
| EXP-FC-003 | trend_regime | bull | 3 | 3.9786738472020167 | 7.386525299008232 | 0.3744955609362389 | 1239 | 1 | True |
| EXP-FC-003 | trend_regime | bull | 5 | 4.783318041795302 | 8.947703922259727 | 0.3692564745196324 | 1197 | 2 | False |
| EXP-FC-003 | trend_regime | bull | 1 | 2.608005746847395e+83 | 8.922035766237523e+84 | 0.372533543804262 | 1267 | 3 | False |
| EXP-FC-003 | trend_regime | sideway | 3 | 3.2170471735344877 | 6.0426342318515545 | 0.36721311475409835 | 2135 | 1 | True |
| EXP-FC-003 | trend_regime | sideway | 5 | 4.052286802288443 | 8.04923262011577 | 0.3640246796392976 | 2107 | 2 | False |
| EXP-FC-003 | trend_regime | sideway | 1 | 6.664824231085419e+89 | 2.575767636458376e+91 | 0.3412953605879651 | 2177 | 3 | False |
| EXP-FC-003 | volatility_regime | high_vol | 3 | 4.606823849849034 | 7.67916584275856 | 0.37142857142857144 | 1330 | 1 | True |
| EXP-FC-003 | volatility_regime | high_vol | 5 | 6.088388565737119 | 10.30307942888877 | 0.3670212765957447 | 1316 | 2 | False |
| EXP-FC-003 | volatility_regime | high_vol | 1 | 3.65102414502795e+61 | 1.11002305130018e+63 | 0.3533834586466165 | 1330 | 3 | False |

## 10. Research discussion

- No universal-best-model conclusion is claimed from these diagnostics.
- The evidence is evaluated by trend, volatility, and combined regimes, with small samples explicitly flagged.
- Risk-aware layer utility is treated as context-specific decision-support evidence, not a general improvement claim.
- Next roadmap work should focus on robustness checks, better risk features, and whether eligibility rules improve interpretability without filtering results opportunistically.

## 11. Missing artifacts and limitations

Loaded source table status:

| artifact | rows | loaded |
| --- | --- | --- |
| forecast_metrics | 1295 | True |
| model_ranking | 1295 | True |
| horizon_comparison | 185 | True |
| candidate_comparison | 1780 | True |
| topn_basket_metrics | 18 | True |

Missing source paths:

_No rows available._

- Basket period return rows read from EXP-RK-002: 2070
- Chart generation notes: generated:regime_distribution_trend_regime.png, generated:regime_distribution_volatility_regime.png, generated:regime_distribution_combined_regime.png, generated:model_mae_by_regime_trend_regime.png, generated:risk_policy_by_regime_trend_regime.png, generated:horizon_by_regime_trend_regime.png, generated:model_health_by_regime_status.png

## 12. Acceptance criteria table

| criterion | status |
| --- | --- |
| regime_labels.csv exists and has rows | True |
| regime_summary.csv exists and has rows | True |
| model_health_by_regime.csv exists | True |
| regime_model_metrics.csv has rows | True |
| regime_risk_metrics.csv has rows where source artifacts allow | True |
| regime_horizon_metrics.csv covers available horizons | True |
| small-sample rows are flagged | True |
| diagnostic-only disclaimer included | True |

## 13. Diagnostic-only disclaimer

All Phase 4 outputs are regime-analysis research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of guaranteed profitable trading.
