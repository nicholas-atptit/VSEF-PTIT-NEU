# Risk Tuning Report

Best validation score: `0.029250`

## Best Parameters
- `risk_enabled`: `True`
- `enable_covar`: `True`
- `enable_risk_engine`: `True`
- `enable_regime_detection`: `True`
- `enable_regime_switching`: `True`
- `enable_risk_allocation`: `True`
- `covar_quantile`: `0.05`
- `covar_window`: `60`
- `risk_penalty_strength`: `1.0`
- `high_vol_threshold`: `0.03`
- `crisis_drawdown_threshold`: `-0.12`
- `crisis_delta_covar_threshold`: `0.015`
- `high_vol_exposure_cut`: `0.6`
- `crisis_exposure_cut`: `0.25`
- `regime_method`: `threshold`
- `random_seed`: `42`
- `simulations`: `10000`
- `confidence_levels`: `[0.95, 0.99]`

## Trial Leaderboard
 trial_number   score  risk_enabled  enable_covar  enable_risk_engine  enable_regime_detection  enable_regime_switching  enable_risk_allocation  covar_quantile  covar_window  risk_penalty_strength  high_vol_threshold  crisis_drawdown_threshold  crisis_delta_covar_threshold  high_vol_exposure_cut  crisis_exposure_cut regime_method  random_seed  simulations confidence_levels
            0 0.02925          True          True                True                     True                     True                    True            0.05            60                    1.0                0.03                      -0.12                         0.015                    0.6                 0.25     threshold           42        10000      [0.95, 0.99]