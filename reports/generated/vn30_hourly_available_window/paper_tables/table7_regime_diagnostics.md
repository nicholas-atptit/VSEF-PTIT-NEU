# Table 7: Regime diagnostics

| frequency | regime | model | horizon | n_obs | accuracy | passed_60pct | reliable |
| --- | --- | --- | --- | --- | --- | --- | --- |
| hourly | bull | random_forest | 1 | 1149 | 57.70% | False | True |
| hourly | high_volatility | xgboost | 1 | 816 | 57.23% | False | True |
| hourly | high_volatility | stacking | 1 | 816 | 57.11% | False | True |
| hourly | high_volatility | xgboost | 4 | 864 | 56.48% | False | True |
| hourly | bull | xgboost | 1 | 1149 | 56.40% | False | True |
| hourly | high_volatility | stacking | 20 | 782 | 56.39% | False | True |
| hourly | sideways | stacking | 1 | 2746 | 56.37% | False | True |
| hourly | high_volatility | random_forest | 1 | 816 | 56.37% | False | True |
| hourly | bear | xgboost | 4 | 718 | 56.27% | False | True |
| hourly | sideways | random_forest | 1 | 2746 | 56.23% | False | True |
| hourly | bull | lightgbm | 1 | 1149 | 56.14% | False | True |
| hourly | bull | stacking | 4 | 1199 | 56.05% | False | True |
| hourly | high_volatility | xgboost | 8 | 871 | 56.03% | False | True |
| hourly | high_volatility | lightgbm | 1 | 816 | 55.76% | False | True |
| hourly | bull | stacking | 1 | 1149 | 55.35% | False | True |
| hourly | high_volatility | random_forest | 4 | 864 | 55.32% | False | True |
| hourly | bear | xgboost | 8 | 721 | 55.20% | False | True |
| hourly | low_volatility | random_forest | 1 | 1887 | 55.01% | False | True |
| hourly | bull | stacking | 20 | 1060 | 54.81% | False | True |
| hourly | bull | xgboost | 8 | 1169 | 54.58% | False | True |

## Note

Post-hoc regime diagnostics from benchmark artifacts.
