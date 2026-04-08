# Algo Trading AI

This repository's active technical ML path is the manifest-driven training and inference stack built around `DualModelTrainer`.

The current supported algorithms are:

| Algorithm | Framework | Sequence history required |
| --- | --- | --- |
| `cart` | scikit-learn | No |
| `lstm` | PyTorch | Yes |
| `bilstm` | PyTorch | Yes |

## Current Training CLI

The public training entrypoint is `scripts/train_ml_tickers.py`. Its active arguments are:

- `--daily`: source directory for per-ticker OHLCV CSVs. Default: `data/daily_market_split_data`
- `--output`: artifact root. Default: `models`
- `--report`: benchmark CSV path. Default: `reports/ml_benchmark.csv`
- `--prepared-output`: directory for rebuilt feature datasets when `--prepare-only` is used
- `--tickers`: comma-separated ticker list
- `--all`: train every CSV in the daily directory
- `--vn100`: train the dynamic VN100 universe
- `--max-tickers`: optional cap for batch runs
- `--algorithms`: comma-separated algorithm list from `cart,lstm,bilstm`
- `--primary-algorithm`: default algorithm stored in each manifest for inference
- `--sequence-length`: rolling window for `lstm` and `bilstm`
- `--hidden-size`, `--num-layers`, `--dropout`, `--learning-rate`, `--batch-size`, `--epochs`, `--patience`: sequence-model hyperparameters
- `--max-depth`, `--min-samples-split`, `--min-samples-leaf`, `--criterion`: CART hyperparameters
- `--prepare-only`: rebuild the 5-year feature dataset without training models
- `--no-clean-output`: keep existing ticker artifacts instead of replacing them

You must provide one of `--tickers`, `--vn100`, or `--all`.

Example commands:

```powershell
python scripts/train_ml_tickers.py --tickers SSI --algorithms cart --primary-algorithm cart
python scripts/train_ml_tickers.py --tickers SSI --algorithms lstm --primary-algorithm lstm --sequence-length 20 --epochs 30 --batch-size 32
python scripts/train_ml_tickers.py --tickers SSI --algorithms cart,lstm,bilstm --primary-algorithm bilstm --sequence-length 20
python scripts/train_ml_tickers.py --vn100 --algorithms cart --max-tickers 20
python scripts/train_ml_tickers.py --tickers SSI --prepare-only --sequence-length 20
```

## Artifact Contract

Training writes per-ticker artifacts under `models/<TICKER>/`.

Expected files:

```text
models/SSI/
|-- manifest.json
|-- trend_classifier_cart_short.joblib
|-- return_regressor_cart_short.joblib
|-- trend_classifier_lstm_short.pt
|-- trend_classifier_lstm_short.meta.joblib
|-- trend_classifier_lstm_short.scaler.joblib
|-- return_regressor_lstm_short.pt
|-- return_regressor_lstm_short.meta.joblib
|-- return_regressor_lstm_short.scaler.joblib
`-- ...
```

`manifest.json` is the primary inference contract. It records:

- `schema_version`
- `ticker`
- `primary_algorithm`
- `feature_columns`
- `data_window.start` and `data_window.end`
- `raw_stats`
- `horizons.<horizon>.algorithms.<algorithm>` with artifact filenames, sequence length, calibration, and metrics

Inference should load artifacts from the manifest rather than from a single hard-coded `model_path`.

## Inference Requirements

- `cart` can score the latest feature row once the required feature columns exist.
- `lstm` and `bilstm` require full feature history for the configured `sequence_length`.
- The inference engine and inference pipeline should receive full per-ticker OHLCV history, rebuild features through `DualModelTrainer`, and then predict from the manifest-selected artifacts.
- If artifacts are missing, inference must fail with a clear manifest or artifact error.
- If a sequence model does not have enough feature history, inference must fail with an explicit insufficient-history error.

## Verification

Import smoke test:

```powershell
python -c "from src.ml.inference.engine import InferenceEngine; from src.ml.pipelines.inference_pipeline import InferencePipeline; print('Imports OK')"
```

CLI help:

```powershell
python scripts/train_ml_tickers.py --help
```
