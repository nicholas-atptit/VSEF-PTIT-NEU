# Feature Interpretation Report

## 1. Executive summary

Groups with context-specific positive evidence: rolling_mean, spread_range. Groups where removal improved more metrics than it worsened: momentum_indicators. Groups with mixed evidence: lag_returns, rolling_volatility, volume.
Tree importance rows generated: 1188.
SHAP rows generated: 0.
No causal or investment claims are made from these diagnostics.

## 2. Phase 5 objective

Phase 5 investigates which governed feature groups contribute to forecasting performance and diagnostic decision quality, and whether feature value varies by model, horizon, and regime.

## 3. Relation to Phase 0-4

- Phase 0 froze VSEF v1 governance: `vnstock_data`, daily OHLCV, frozen model scope, and diagnostic-only outputs.
- Phase 1 implemented standardized config-driven experiment execution.
- Phase 2 showed weak consistent model-vs-baseline superiority.
- Phase 3 showed weak aggregate risk-aware improvement.
- Phase 4 showed regime-dependent behavior.
- Phase 5 investigates which feature groups explain or improve performance without treating importance as causal proof.

## 4. Feature group registry

| feature_group | description | expected_patterns | hypothesis | risk |
| --- | --- | --- | --- | --- |
| lag_returns | Lagged close returns and price-change features. | lag,return_lag,close_return_lag,pct_change_lag,return_1 | Captures short-term persistence and reversal behavior. | May overfit short-term noise. |
| rolling_mean | Rolling mean and moving-average level features. | rolling_mean,ma_,moving_average,sma | Captures smoothed trend and price level context. | May lag turning points. |
| rolling_volatility | Rolling standard deviation and realized volatility features. | rolling_std,volatility,realized_vol,vol_ | Captures uncertainty and regime-sensitive variation. | May react late to volatility shifts. |
| volume | Volume and volume shock features. | volume,volume_mean,volume_shock,liquidity | Captures participation, attention, and liquidity pressure. | Volume spikes may be noisy or event-driven. |
| momentum_indicators | RSI, MACD, and momentum-style indicators. | rsi,macd,momentum,signal | Captures technical momentum and overbought/oversold behavior. | Indicator performance may be regime-dependent. |
| spread_range | Intraday range, high-low spread, open-close spread, ATR-like proxies. | high_low,range,spread,atr,open_close,close_open,true_range | Captures intraday uncertainty and price pressure. | Range features may overlap with volatility features. |
| calendar_optional | Calendar/time features if present. | day,month,quarter,weekday | Captures seasonal or calendar effects. | Weak evidence without sufficient cycles. |

Guarded/excluded fields:

| description | risk |
| --- | --- |
| date | Excluded from model features unless explicitly safe. |
| ticker | Excluded from model features unless explicitly safe. |
| y_true | Excluded from model features unless explicitly safe. |
| y_pred | Excluded from model features unless explicitly safe. |
| target | Excluded from model features unless explicitly safe. |
| future_return | Excluded from model features unless explicitly safe. |
| realized_future_return | Excluded from model features unless explicitly safe. |
| actual_direction | Excluded from model features unless explicitly safe. |
| diagnostics | Excluded from model features unless explicitly safe. |
| model_name | Excluded from model features unless explicitly safe. |
| model_type | Excluded from model features unless explicitly safe. |

## 5. Ablation study design

- `EXP-FA-000`: full feature reference.
- `EXP-FA-001`: remove lag_returns.
- `EXP-FA-002`: remove rolling_volatility.
- `EXP-FA-003`: remove momentum_indicators.
- `EXP-FA-004`: remove volume.
- `EXP-FA-005`: remove spread_range.
- `EXP-FA-006`: remove rolling_mean as the reduced/core comparison.
- Models in the local ablation evidence run: XGBoost, LightGBM, ETS. SARIMAX was attempted in the initial full default set but exceeded the 10-minute local timeout before metrics were finalized, so it is disclosed as missing runtime evidence rather than forced or faked.
- Tickers: FPT, ACB, HPG. Horizons: T+1, T+3, T+5.

Experiment artifact status:

| experiment_id | status | metric_rows | prediction_rows | errors | warnings |
| --- | --- | --- | --- | --- | --- |
| EXP-FA-000 | completed | 648 | 6858 | 0 | 7 |
| EXP-FA-001 | completed | 648 | 6858 | 0 | 10 |
| EXP-FA-002 | completed | 648 | 6858 | 0 | 10 |
| EXP-FA-003 | completed | 648 | 6858 | 0 | 10 |
| EXP-FA-004 | completed | 648 | 6858 | 0 | 10 |
| EXP-FA-005 | completed | 648 | 6858 | 0 | 10 |
| EXP-FA-006 | completed | 648 | 6858 | 0 | 10 |
| EXP-FA-007 | completed | 216 | 2286 | 0 | 11 |
| EXP-FA-008 | completed | 216 | 2286 | 0 | 12 |

## 6. Ablation results

| removed_group | context_count | mae_worsened_when_removed | rmse_worsened_when_removed | directional_accuracy_worsened_when_removed | mae_improved_when_removed | rmse_improved_when_removed | directional_accuracy_improved_when_removed | mean_delta_mae | mean_delta_rmse | mean_delta_directional_accuracy | evidence_label |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | 27 | 8 | 8 | 1 | 10 | 10 | 0 | 0.008142735336239305 | 0.009592015205380148 | -0.00029629629629629656 | mixed |
| momentum_indicators | 27 | 6 | 6 | 1 | 12 | 12 | 1 | -0.024742180541768487 | -0.023647013739118174 | 4.666083406237556e-06 | removal_improved_metrics |
| rolling_mean | 27 | 13 | 14 | 3 | 5 | 4 | 0 | 0.5810485089990705 | 0.5655187187982368 | -0.003249927092446783 | positive_context_specific |
| rolling_volatility | 27 | 8 | 8 | 1 | 10 | 10 | 0 | -0.03338109283584042 | -0.03087077280209054 | -0.00029629629629629656 | mixed |
| spread_range | 27 | 10 | 11 | 2 | 8 | 7 | 0 | 0.00876522349333917 | 0.007610566495007094 | -0.0005879265091863555 | positive_context_specific |
| volume | 27 | 7 | 7 | 1 | 11 | 11 | 0 | -0.08506580701609424 | -0.08338609592091878 | -0.00029629629629629656 | mixed |

- lag_returns has mixed evidence across model/horizon/ticker contexts.
- The current evidence does not show a consistent positive contribution from momentum_indicators; removal improved more metric contexts than it worsened.
- Removing rolling_mean worsened enough MAE/RMSE/directional-accuracy contexts to suggest this group contributes useful predictive information under those tested contexts.
- rolling_volatility has mixed evidence across model/horizon/ticker contexts.
- Removing spread_range worsened enough MAE/RMSE/directional-accuracy contexts to suggest this group contributes useful predictive information under those tested contexts.
- volume has mixed evidence across model/horizon/ticker contexts.

Ablation delta rows are written to `ablation_delta_metrics.csv` and group-specific files. Positive delta MAE/RMSE means removing a feature group worsened error. Negative delta directional accuracy means removing a feature group worsened direction.

## 7. Tree feature importance

Tree feature importance was extracted from `feature_importances_` where available.

| model_name | feature_group | mean_normalized_importance |
| --- | --- | --- |
| lightgbm | default_ohlcv | 0.5457382641254945 |
| lightgbm | rolling_mean | 0.29093296453029976 |
| lightgbm | momentum_indicators | 0.07526591174394766 |
| lightgbm | volume | 0.05876755328289298 |
| lightgbm | rolling_volatility | 0.019901717270767328 |
| lightgbm | spread_range | 0.008802572498133862 |
| lightgbm | lag_returns | 0.0005910165484633556 |
| xgboost | default_ohlcv | 0.7571454296252094 |
| xgboost | rolling_mean | 0.223881434746278 |
| xgboost | momentum_indicators | 0.008357796849924194 |
| xgboost | rolling_volatility | 0.004418235268543017 |
| xgboost | volume | 0.0030939483337390442 |
| xgboost | lag_returns | 0.0016636053527880538 |
| xgboost | spread_range | 0.0014395498235175261 |

| model_name | feature_group | feature_name | context_count | mean_importance_value | mean_normalized_importance | mean_rank | top_5_count | top_10_count | sample_size | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lightgbm | default_ohlcv | close | 18 | 42.5 | 0.26904397025456855 | 1.5 | 18 | 18 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_mean | sma_20 | 18 | 22.444444444444443 | 0.13802218431514862 | 3.0555555555555554 | 17 | 18 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | default_ohlcv | high | 18 | 21.833333333333332 | 0.13769173202428203 | 3.5 | 16 | 18 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | default_ohlcv | low | 18 | 17.11111111111111 | 0.1067951826386928 | 5.388888888888889 | 11 | 17 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_mean | rolling_mean_10 | 18 | 11.666666666666666 | 0.07110563688116528 | 7.722222222222222 | 11 | 14 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_mean | ma_3 | 18 | 7.666666666666667 | 0.04555975547084085 | 6.722222222222222 | 6 | 18 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | momentum_indicators | rsi_14 | 18 | 5.888888888888889 | 0.036377563292456366 | 12.944444444444445 | 4 | 11 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | volume | volume_mean_10 | 18 | 5.666666666666667 | 0.03569127159408562 | 11.61111111111111 | 5 | 7 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | default_ohlcv | open | 18 | 5.222222222222222 | 0.0322073792079511 | 8.777777777777779 | 1 | 12 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_mean | ma_5 | 18 | 4.666666666666667 | 0.02886222023339518 | 9.555555555555555 | 0 | 13 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | volume | volume_mean_5 | 18 | 2.8333333333333335 | 0.017555232500995603 | 15.5 | 0 | 4 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | momentum_indicators | momentum_5 | 18 | 2.5 | 0.0150609549720273 | 21.72222222222222 | 0 | 5 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_volatility | rolling_std_10 | 18 | 2.3333333333333335 | 0.014206273919821467 | 15.777777777777779 | 0 | 6 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | momentum_indicators | macd_signal_9 | 18 | 2.1666666666666665 | 0.012434420599641385 | 19.333333333333332 | 0 | 3 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | momentum_indicators | macd_12_26 | 18 | 1.8333333333333333 | 0.011392972879822605 | 17.77777777777778 | 0 | 2 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | spread_range | atr_proxy_5 | 18 | 1.3333333333333333 | 0.008528899810668074 | 23.22222222222222 | 1 | 4 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_mean | rolling_mean_5 | 18 | 1.2777777777777777 | 0.0073831676297498395 | 16.944444444444443 | 0 | 1 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | volume | volume | 18 | 0.9444444444444444 | 0.005521049187811762 | 13.555555555555555 | 0 | 5 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_volatility | rolling_std_5 | 18 | 0.5555555555555556 | 0.0032980474935738776 | 20.61111111111111 | 0 | 1 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |
| lightgbm | rolling_volatility | realized_vol_10 | 18 | 0.3888888888888889 | 0.0023973958573719834 | 21.72222222222222 | 0 | 0 | 6642 | High feature importance indicates strong model reliance, not causal market influence. |

High feature importance indicates strong model reliance, not causal market influence.

## 8. SHAP explanation if available

SHAP was not generated because shap is unavailable in the active environment: No module named 'shap' No SHAP values were fabricated.

## 9. Feature value by horizon

T+1:

| removed_group | horizon | context_count | mean_delta_mae | mean_delta_rmse | mean_delta_mape | mean_delta_directional_accuracy | mae_worsened_when_removed_count | rmse_worsened_when_removed_count | directional_accuracy_worsened_when_removed_count | small_sample_flag | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | 1 | 9 | 0.008530014717073264 | 0.010716289467948052 | 0.008740987020250786 | 0.0 | 3 | 3 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| momentum_indicators | 1 | 9 | -0.06228815220136494 | -0.05879114848653599 | -0.05893384739164942 | 0.0 | 2 | 2 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| rolling_mean | 1 | 9 | 0.528399141852351 | 0.5119515422384714 | 0.466369995953115 | 0.0 | 4 | 5 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| rolling_volatility | 1 | 9 | -0.027327595751189964 | -0.024766349524070863 | -0.02027593734879396 | 0.0 | 3 | 3 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| spread_range | 1 | 9 | 0.06057133592639346 | 0.05846241255803301 | 0.053269783027912326 | 0.0 | 4 | 4 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| volume | 1 | 9 | -0.004531721498004872 | -0.0056802187553942 | -0.002590012590338543 | 0.0 | 3 | 3 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |

T+3:

| removed_group | horizon | context_count | mean_delta_mae | mean_delta_rmse | mean_delta_mape | mean_delta_directional_accuracy | mae_worsened_when_removed_count | rmse_worsened_when_removed_count | directional_accuracy_worsened_when_removed_count | small_sample_flag | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | 3 | 9 | -0.011743971958313929 | -0.009131805367942657 | -0.002412741952942178 | 0.0 | 3 | 3 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| momentum_indicators | 3 | 9 | -0.052123887351800985 | -0.04992334276067223 | -0.04174919568766934 | -0.000874890638670177 | 2 | 2 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| rolling_mean | 3 | 9 | 0.6786196451540024 | 0.6541517765534796 | 0.6449477043912837 | -0.0017497812773403416 | 5 | 5 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| rolling_volatility | 3 | 9 | -0.05613149156512657 | -0.05169402549180687 | -0.030913568954061957 | 0.0 | 2 | 2 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| spread_range | 3 | 9 | -0.013930156275864025 | -0.013164026894930101 | -0.0067831469737044 | -0.000874890638670177 | 4 | 5 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| volume | 3 | 9 | -0.1077780003047913 | -0.10265129873458569 | -0.08995123620335911 | 0.0 | 2 | 2 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |

T+5:

| removed_group | horizon | context_count | mean_delta_mae | mean_delta_rmse | mean_delta_mape | mean_delta_directional_accuracy | mae_worsened_when_removed_count | rmse_worsened_when_removed_count | directional_accuracy_worsened_when_removed_count | small_sample_flag | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | 5 | 9 | 0.027642163249958582 | 0.027191561516135047 | 0.018729785455394103 | -0.0008888888888888897 | 2 | 2 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| momentum_indicators | 5 | 9 | 0.04018549792786046 | 0.037773450029853696 | 0.0275630766621229 | 0.0008888888888888897 | 2 | 2 | 0 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| rolling_mean | 5 | 9 | 0.5361267399908578 | 0.5304528376027587 | 0.5486943045318181 | -0.008000000000000007 | 4 | 4 | 2 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| rolling_volatility | 5 | 9 | -0.016684191191204716 | -0.016151943390393893 | -0.007198451037550384 | -0.0008888888888888897 | 3 | 3 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| spread_range | 5 | 9 | -0.020345509170511922 | -0.022466686178081625 | -0.022402656340153617 | -0.0008888888888888897 | 2 | 2 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |
| volume | 5 | 9 | -0.14288769924548655 | -0.14182677027277643 | -0.13608489687663397 | -0.0008888888888888897 | 2 | 2 | 1 | False | Aggregated over model rows only; ablation deltas are diagnostic and not causal. |

## 10. Feature value by regime

Bull, bear, and sideway rows:

| removed_group | regime_column | regime | context_count | mean_delta_mae | mean_delta_rmse | mean_delta_directional_accuracy | small_sample_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | trend_regime | bear | 27 | -0.01977525926054771 | -0.019134250022419266 | -0.0009746588693957105 | 0 |
| lag_returns | trend_regime | bull | 27 | 0.011032335709607299 | 0.011024293582516541 | 0.0 | 0 |
| lag_returns | trend_regime | sideway | 27 | 0.011741426559400647 | 0.012328412792220186 | 0.0 | 0 |
| momentum_indicators | trend_regime | bear | 27 | -0.029042753685885086 | -0.02999314301424769 | 0.0 | 0 |
| momentum_indicators | trend_regime | bull | 27 | -0.01258142925811518 | -0.012232914428768553 | 0.0 | 0 |
| momentum_indicators | trend_regime | sideway | 27 | -0.030295428698725438 | -0.031109609188478243 | 0.0 | 0 |
| rolling_mean | trend_regime | bear | 27 | 0.37738893500408227 | 0.3760210999367739 | -0.008771929824561401 | 0 |
| rolling_mean | trend_regime | bull | 27 | 0.4980904877169201 | 0.47292013314355735 | 0.0 | 0 |
| rolling_mean | trend_regime | sideway | 27 | 0.6865833262315739 | 0.6965579739375863 | -0.0011574074074074073 | 0 |
| rolling_volatility | trend_regime | bear | 27 | -0.027558295369413878 | -0.02505937034910201 | -0.0009746588693957105 | 0 |
| rolling_volatility | trend_regime | bull | 27 | -0.02810316972565475 | -0.025734758880609596 | 0.0 | 0 |
| rolling_volatility | trend_regime | sideway | 27 | -0.04039821689709177 | -0.04068722509675723 | 0.0 | 0 |
| spread_range | trend_regime | bear | 27 | 0.012219674436713011 | 0.01094218263686848 | -0.001949317738791421 | 0 |
| spread_range | trend_regime | bull | 27 | 0.005575762524750881 | 0.005808688179163929 | 0.0 | 0 |
| spread_range | trend_regime | sideway | 27 | 0.008600601645476137 | 0.007888111675145606 | 0.0 | 0 |
| volume | trend_regime | bear | 27 | -0.021227442247382062 | -0.022318330392487602 | 0.0 | 0 |
| volume | trend_regime | bull | 27 | -0.0788056854332788 | -0.07264153360953818 | 0.0 | 0 |
| volume | trend_regime | sideway | 27 | -0.10708022642817819 | -0.10911678041399206 | -0.0005787037037037037 | 0 |

High-volatility and low-volatility rows:

| removed_group | regime_column | regime | context_count | mean_delta_mae | mean_delta_rmse | mean_delta_directional_accuracy | small_sample_rows |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | volatility_regime | high_vol | 27 | 0.005304082195963854 | 0.006849269185819591 | 0.0 | 0 |
| lag_returns | volatility_regime | low_vol | 27 | 0.019946618085906496 | 0.01988667188158764 | -0.00033978933061501666 | 0 |
| momentum_indicators | volatility_regime | high_vol | 27 | -0.019250370474554843 | -0.016231177291376575 | 0.0 | 0 |
| momentum_indicators | volatility_regime | low_vol | 27 | -0.03268119161069949 | -0.033008857519895005 | 0.0 | 0 |
| rolling_mean | volatility_regime | high_vol | 27 | 0.5354905902613224 | 0.5195511979611701 | -0.011574074074074073 | 0 |
| rolling_mean | volatility_regime | low_vol | 27 | 0.8127981512275874 | 0.8038300005654754 | -0.002038735983690108 | 0 |
| rolling_volatility | volatility_regime | high_vol | 27 | -0.026953481461358456 | -0.025010439222542265 | 0.0 | 0 |
| rolling_volatility | volatility_regime | low_vol | 27 | -0.04946076839266388 | -0.04878429013329725 | -0.00033978933061501666 | 0 |
| spread_range | volatility_regime | high_vol | 27 | 0.01284640669894242 | 0.010769194307924957 | -0.0023148148148148147 | 0 |
| spread_range | volatility_regime | low_vol | 27 | 0.007081686998166285 | 0.006360434714664466 | -0.00033978933061501666 | 0 |
| volume | volatility_regime | high_vol | 27 | -0.06498519052029349 | -0.06486462302042122 | 0.0023148148148148147 | 0 |
| volume | volatility_regime | low_vol | 27 | -0.13586440059823116 | -0.13504386275768362 | -0.0006795786612300333 | 0 |

## 11. Feature value for decision quality

Top-N source metrics were available from Phase 3, but they are not feature-ablation-specific. Delta top-N hit ratio, realized return, and return/volatility proxy fields are intentionally blank.

| removed_group | horizon | top_n | source_candidate_type | source_hit_ratio | source_average_realized_return | source_return_volatility_proxy | delta_topn_hit_ratio | delta_average_realized_return | delta_return_volatility_proxy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lag_returns | 1 | 1 | forecast_only | 0.425 | -0.0007247382211796 | -0.0508878736432739 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 1 | 3 | forecast_only | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 1 | 5 | forecast_only | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 3 | 1 | forecast_only | 0.5892857142857143 | 0.0049634108964371 | 0.2024637110447544 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 3 | 3 | forecast_only | 0.5357142857142857 | 0.0014482649262396 | 0.0696683079299817 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 3 | 5 | forecast_only | 0.5357142857142857 | 0.0014409071949368 | 0.0700913930049204 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 5 | 1 | forecast_only | 0.6637168141592921 | 0.00862069894258 | 0.2773977504412089 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 5 | 3 | forecast_only | 0.5663716814159292 | 0.0042074774448724 | 0.1672105480849101 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 5 | 5 | forecast_only | 0.5752212389380531 | 0.0045809878505931 | 0.1810839838143172 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 1 | 1 | risk_aware | 0.4 | -0.0011528064930302 | -0.0832651406650748 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 1 | 3 | risk_aware | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 1 | 5 | risk_aware | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 3 | 1 | risk_aware | 0.5 | 0.0006583087375213 | 0.0285561343780321 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 3 | 3 | risk_aware | 0.5267857142857143 | 0.0011140513654093 | 0.0551854163039771 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 3 | 5 | risk_aware | 0.5357142857142857 | 0.0014409071949368 | 0.0700913930049204 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 5 | 1 | risk_aware | 0.5575221238938053 | 0.0053884263634947 | 0.1732298521024511 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 5 | 3 | risk_aware | 0.5752212389380531 | 0.0052623290218489 | 0.2075733780824754 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| lag_returns | 5 | 5 | risk_aware | 0.5752212389380531 | 0.0045809878505931 | 0.1810839838143173 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 1 | 1 | forecast_only | 0.425 | -0.0007247382211796 | -0.0508878736432739 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 1 | 3 | forecast_only | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 1 | 5 | forecast_only | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 3 | 1 | forecast_only | 0.5892857142857143 | 0.0049634108964371 | 0.2024637110447544 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 3 | 3 | forecast_only | 0.5357142857142857 | 0.0014482649262396 | 0.0696683079299817 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 3 | 5 | forecast_only | 0.5357142857142857 | 0.0014409071949368 | 0.0700913930049204 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 5 | 1 | forecast_only | 0.6637168141592921 | 0.00862069894258 | 0.2773977504412089 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 5 | 3 | forecast_only | 0.5663716814159292 | 0.0042074774448724 | 0.1672105480849101 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 5 | 5 | forecast_only | 0.5752212389380531 | 0.0045809878505931 | 0.1810839838143172 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 1 | 1 | risk_aware | 0.4 | -0.0011528064930302 | -0.0832651406650748 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 1 | 3 | risk_aware | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |
| momentum_indicators | 1 | 5 | risk_aware | 0.5 | -0.0001996538833849 | -0.0165111685014708 |  |  |  | Source top-N metrics are available, but they are not feature-ablation-specific; deltas are left blank. |

## 12. Interpretability caveats

- Feature importance is not causality.
- Ablation can be affected by feature correlation.
- SHAP explains model behavior, not market truth.
- Regime labels are rule-based diagnostics.
- The feature appears important to the model but does not necessarily improve out-of-sample performance when ablation and importance disagree; this may indicate redundancy, overfitting, or correlated features.

## 13. Acceptance criteria table

| criterion | status | evidence |
| --- | --- | --- |
| feature_group_registry.yaml exists | True | configs/features/feature_group_registry.yaml |
| Full feature reference config exists | True | EXP-FA-000 |
| Ablation configs EXP-FA-001 to EXP-FA-006 exist | True | configs/experiments |
| Tree importance config EXP-FA-007 exists | True | EXP-FA-007 |
| SHAP config EXP-FA-008 exists or SHAP_NOT_AVAILABLE.md explains missing evidence | True | EXP-FA-008 and report output |
| At least one ablation experiment produced metrics and manifest | True | {'EXP-FA-000': 'completed', 'EXP-FA-001': 'completed', 'EXP-FA-002': 'completed', 'EXP-FA-003': 'completed', 'EXP-FA-004': 'completed', 'EXP-FA-005': 'completed', 'EXP-FA-006': 'completed', 'EXP-FA-007': 'completed', 'EXP-FA-008': 'completed'} |
| Ablation delta metrics generated from actual artifacts | True | ablation_delta_metrics.csv |
| Tree feature importance generated or honestly marked unavailable | True | rows=1188 |
| Feature report generated | True | FEATURE_INTERPRETATION_REPORT.md |
| Feature contribution discussed by horizon/regime where evidence allows | True | feature_value_by_horizon.csv and feature_value_by_regime.csv |
| Interpretability caveats included | True | report caveats |
| No causal claims made from importance alone | True | diagnostic-only language |
| No fake metrics, SHAP, charts, or importance values created | True | missing evidence disclosed |
| Diagnostic outputs not presented as investment advice | True | disclaimer |

## 14. Diagnostic-only disclaimer

All Phase 5 outputs are feature-analysis and interpretability research artifacts only. They are not BUY / SELL / HOLD advice, capital allocation guidance, broker execution instructions, portfolio recommendations, causal proof, or proof of guaranteed profitable trading.

## Generated artifacts

- `feature_group_registry_summary.csv`
- `ablation_delta_metrics.csv`
- `delta_metrics_lag.csv`
- `delta_metrics_volatility.csv`
- `delta_metrics_momentum.csv`
- `delta_metrics_volume.csv`
- `delta_metrics_spread_range.csv`
- `delta_metrics_all_groups.csv`
- `tree_feature_importance.csv`
- `feature_importance_summary.csv`
- `feature_value_by_horizon.csv`
- `feature_value_by_regime.csv`
- `feature_decision_quality.csv`
- `feature_analysis_limitations.md`
- Chart notes: Generated 7 chart artifact(s) from actual Phase 5 tables.
