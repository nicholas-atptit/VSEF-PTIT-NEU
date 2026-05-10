"""Experimental placeholder module.
Not part of canonical governed runtime.

Implements multi-quantile forecasting [0.1, 0.5, 0.9] for OHLCV data
using PyTorch Forecasting and PyTorch Lightning.
"""

import os
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

# ── Data & Hyperparameters ───────────────────────────────────────────
max_prediction_length = 5
max_encoder_length = 60
training_cutoff = "2024-01-01" # Placeholder for time-based split

def train_tft(data):
    """
    Main training loop for TFT.
    Assumes data is a pandas DataFrame in Long format (ticker, time_idx, value).
    """
    # 1. Define Dataset
    training = TimeSeriesDataSet(
        data[lambda x: x.time < training_cutoff],
        time_idx="time_idx",
        target="close",
        group_ids=["ticker"],
        min_encoder_length=max_encoder_length // 2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["sector"],
        time_varying_known_reals=["day_of_week", "week_of_year"],
        time_varying_unknown_reals=["close", "volume", "rsi", "macd"],
        target_normalizer="scaling", # Good for price data
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
    
    train_dataloader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    # 2. Define Model
    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.01,
        hidden_size=16,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=8,
        output_size=3, # [q10, q50, q90]
        loss=QuantileLoss(),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )

    # 3. Training
    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        enable_model_summary=True,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=10),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=TensorBoardLogger("lightning_logs", name="tft_stock_forecast"),
    )

    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    # 4. Save best model
    best_model_path = trainer.checkpoint_callback.best_model_path
    print(f"✅ TFT Training Complete. Best model saved at: {best_model_path}")
    return tft

if __name__ == "__main__":
    # Placeholder for data loading
    print("🚀 TFT Training Script Loaded. Ensure data is prepared in Long Format.")
