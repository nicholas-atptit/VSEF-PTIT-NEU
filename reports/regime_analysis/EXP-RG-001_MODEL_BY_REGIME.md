# EXP-RG-001 Model Performance By Regime

A stable MAE winner appears in observed context rows, mainly the persistence baseline; this supports baseline competitiveness, not a universal ML-model claim.

## Best Model Rows By Trend Regime

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

## Baseline Competitiveness

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

## Ranking Consistency

| experiment_id | horizon | regime_column | model_name | model_type | regime_count | mean_rank | rank_std | best_regime_count | rank_min | rank_max | stable_best_across_observed_regimes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | 1 | combined_regime | persistence | baseline | 6 | 1.0 | 0.0 | 6 | 1 | 1 | True |
| EXP-FC-003 | 1 | combined_regime | persistence | baseline | 6 | 1.0 | 0.0 | 6 | 1 | 1 | True |
| EXP-FC-003 | 3 | combined_regime | persistence | baseline | 6 | 1.0 | 0.0 | 6 | 1 | 1 | True |
| EXP-FC-001 | 1 | trend_regime | persistence | baseline | 3 | 1.0 | 0.0 | 3 | 1 | 1 | True |
| EXP-FC-003 | 1 | trend_regime | persistence | baseline | 3 | 1.0 | 0.0 | 3 | 1 | 1 | True |
| EXP-FC-003 | 3 | trend_regime | persistence | baseline | 3 | 1.0 | 0.0 | 3 | 1 | 1 | True |
| EXP-FC-003 | 5 | trend_regime | persistence | baseline | 3 | 1.0 | 0.0 | 3 | 1 | 1 | True |
| EXP-FC-001 | 1 | volatility_regime | persistence | baseline | 2 | 1.0 | 0.0 | 2 | 1 | 1 | True |
| EXP-FC-003 | 1 | volatility_regime | persistence | baseline | 2 | 1.0 | 0.0 | 2 | 1 | 1 | True |
| EXP-FC-003 | 3 | volatility_regime | persistence | baseline | 2 | 1.0 | 0.0 | 2 | 1 | 1 | True |
| EXP-FC-003 | 5 | volatility_regime | persistence | baseline | 2 | 1.0 | 0.0 | 2 | 1 | 1 | True |
| EXP-FC-003 | 5 | combined_regime | persistence | baseline | 6 | 1.1666666666666667 | 0.3726779962499649 | 5 | 1 | 2 | False |
| EXP-FC-003 | 5 | combined_regime | sarimax | model | 6 | 3.0 | 1.0 | 1 | 1 | 4 | False |
| EXP-FC-001 | 1 | combined_regime | bilstm | model | 6 | 8.0 | 0.0 | 0 | 8 | 8 | False |
| EXP-FC-001 | 1 | combined_regime | ets | model | 6 | 5.333333333333333 | 0.9428090415820634 | 0 | 4 | 6 | False |
| EXP-FC-001 | 1 | combined_regime | lightgbm | model | 6 | 4.333333333333333 | 0.4714045207910317 | 0 | 4 | 5 | False |
| EXP-FC-001 | 1 | combined_regime | lstm | model | 6 | 7.0 | 0.0 | 0 | 7 | 7 | False |
| EXP-FC-001 | 1 | combined_regime | moving_average_rule | baseline | 6 | 3.0 | 0.0 | 0 | 3 | 3 | False |
| EXP-FC-001 | 1 | combined_regime | sarimax | model | 6 | 9.0 | 0.0 | 0 | 9 | 9 | False |
| EXP-FC-001 | 1 | combined_regime | xgboost | model | 6 | 5.333333333333333 | 0.4714045207910317 | 0 | 5 | 6 | False |

## Small Samples

| experiment_id | ticker | horizon | model_name | model_type | regime_column | regime | mae | rmse | mape | directional_accuracy | prediction_count | missing_prediction_rate | error_std | small_sample_flag | regime_label_missing_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| EXP-FC-001 | ACB | 1 | bilstm | model | combined_regime | bear_high_vol | 19.906171248214584 | 19.907428171024154 | 100.90103366995132 | 0.42857142857142855 | 7 | 0.0 | 0.2237020814412391 | True | 0 |
| EXP-FC-001 | ACB | 1 | bilstm | model | combined_regime | bull_high_vol | 21.93505129239389 | 21.935314007353107 | 100.91801722534659 | 0.8571428571428571 | 7 | 0.0 | 0.10735642146854413 | True | 0 |
| EXP-FC-001 | ACB | 1 | ets | model | combined_regime | bear_high_vol | 0.6956393711986212 | 0.7203251714779498 | 3.5373660115173005 | 0.5714285714285714 | 7 | 0.0 | 0.18696047149899012 | True | 0 |
| EXP-FC-001 | ACB | 1 | ets | model | combined_regime | bull_high_vol | 0.5279691888508974 | 0.5467379957027403 | 2.4252981604480772 | 0.8571428571428571 | 7 | 0.0 | 0.1420245456573443 | True | 0 |
| EXP-FC-001 | ACB | 1 | lightgbm | model | combined_regime | bear_high_vol | 0.7464623841050125 | 0.7712376997228021 | 3.775572963095203 | 0.42857142857142855 | 7 | 0.0 | 0.19391105845201237 | True | 0 |
| EXP-FC-001 | ACB | 1 | lightgbm | model | combined_regime | bull_high_vol | 2.097481737144515 | 2.101576921271767 | 9.646633338868925 | 0.8571428571428571 | 7 | 0.0 | 0.13113320848414717 | True | 0 |
| EXP-FC-001 | ACB | 1 | lstm | model | combined_regime | bear_high_vol | 19.553646821847984 | 19.555354695277966 | 99.11184593578585 | 0.42857142857142855 | 7 | 0.0 | 0.2584438518598933 | True | 0 |
| EXP-FC-001 | ACB | 1 | lstm | model | combined_regime | bull_high_vol | 21.516979742475918 | 21.517351084879124 | 98.99381096371773 | 0.8571428571428571 | 7 | 0.0 | 0.12641389096798408 | True | 0 |
| EXP-FC-001 | ACB | 1 | moving_average_rule | baseline | combined_regime | bear_high_vol | 0.2614285714285717 | 0.2858121261048448 | 1.324685880130331 | 0.42857142857142855 | 7 | 0.0 | 0.2815055732220984 | True | 0 |
| EXP-FC-001 | ACB | 1 | moving_average_rule | baseline | combined_regime | bull_high_vol | 0.2068571428571424 | 0.24369536017624308 | 0.950618219441291 | 0.7142857142857143 | 7 | 0.0 | 0.2249556872463779 | True | 0 |
| EXP-FC-001 | ACB | 1 | persistence | baseline | combined_regime | bear_high_vol | 0.17857142857142858 | 0.22293496809607982 | 0.9027910312764277 | 0.0 | 7 | 0.0 | 0.1955004045092118 | True | 0 |
| EXP-FC-001 | ACB | 1 | persistence | baseline | combined_regime | bull_high_vol | 0.1142857142857144 | 0.1495708145709869 | 0.5253448071634211 | 0.0 | 7 | 0.0 | 0.14858516420022602 | True | 0 |
| EXP-FC-001 | ACB | 1 | sarimax | model | combined_regime | bear_high_vol | 4.870387227630462e+23 | 1.0742016158505003e+24 | 2.426636953474245e+24 | 0.42857142857142855 | 7 | 0.0 | 9.574457655790046e+23 | True | 0 |
| EXP-FC-001 | ACB | 1 | sarimax | model | combined_regime | bull_high_vol | 2.254864460063151e+54 | 4.941530696137524e+54 | 1.0340459630953485e+55 | 0.8571428571428571 | 7 | 0.0 | 4.3970799273624215e+54 | True | 0 |
| EXP-FC-001 | ACB | 1 | xgboost | model | combined_regime | bear_high_vol | 0.8233440399169919 | 0.846811016950271 | 4.163846558822614 | 0.42857142857142855 | 7 | 0.0 | 0.1979734587299996 | True | 0 |
| EXP-FC-001 | ACB | 1 | xgboost | model | combined_regime | bull_high_vol | 2.192335455758235 | 2.196253780632378 | 10.083044955641503 | 0.8571428571428571 | 7 | 0.0 | 0.13113320848414717 | True | 0 |
| EXP-FC-001 | ACB | 1 | zero_return | baseline | combined_regime | bear_high_vol | 0.17857142857142858 | 0.22293496809607982 | 0.9027910312764277 | 0.0 | 7 | 0.0 | 0.1955004045092118 | True | 0 |
| EXP-FC-001 | ACB | 1 | zero_return | baseline | combined_regime | bull_high_vol | 0.1142857142857144 | 0.1495708145709869 | 0.5253448071634211 | 0.0 | 7 | 0.0 | 0.14858516420022602 | True | 0 |
| EXP-FC-001 | DGC | 1 | bilstm | model | combined_regime | bull_high_vol | 113.4788264658302 | 113.61735231967569 | 100.13470275409155 | 0.25 | 8 | 0.0 | 5.608804870140439 | True | 0 |
| EXP-FC-001 | DGC | 1 | ets | model | combined_regime | bull_high_vol | 7.000839311760762 | 9.142865478689341 | 6.443973244548936 | 0.625 | 8 | 0.0 | 8.287597861510989 | True | 0 |

All Phase 4 outputs are regime-analysis research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, or proof of guaranteed profitable trading.
