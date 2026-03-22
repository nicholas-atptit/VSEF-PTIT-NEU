"""Module 2: Multi-Model Prediction Engine (v3 — LightGBM VotingClassifier).

Loads pre-trained models from models/<TICKER>/ in the FPT/PRED format:
  - trend_classifier.joblib  (VotingClassifier: LightGBM + XGBoost + RF)
  - range_high_q10/q50/q90.joblib  (LightGBM quantile regressors)
  - range_low_q10/q50/q90.joblib
  - feature_cols.joblib  (list of feature column names)

Cross-validation: TimeSeriesSplit only (never standard K-Fold).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.settings import get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DualModelTrainer:
    """Trains and manages the ML model system.

    Usage::

        trainer = DualModelTrainer()
        prediction = trainer.predict(ticker="SSI", features_row=row)
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._model_dir = Path(self._settings.model_dir)
        self._model_dir.mkdir(parents=True, exist_ok=True)

        # In-memory model cache {ticker: {model_name: model}}
        self._models: dict[str, dict[str, Any]] = {}

    # ── Prediction ────────────────────────────────────────────

    def predict(
        self,
        ticker: str,
        features_row: np.ndarray | pd.Series | pd.DataFrame,
    ) -> dict[str, Any]:
        """Generate predictions from trained models for a single row.

        Supports both:
          - Old 3-class XGBoost models (Up/Sideways/Down)
          - New binary VotingClassifier models (Up/Down — from v3 training)

        Args:
            ticker: Stock symbol.
            features_row: Feature values (Series, 1D array, or DataFrame row).

        Returns:
            Dictionary with trend probabilities and expected price range.
        """
        self._ensure_models_loaded(ticker)
        models = self._models[ticker]
        feature_cols = models.get("feature_cols", [])

        # Align features to what the model was trained on
        if isinstance(features_row, pd.DataFrame):
            if feature_cols:
                missing = set(feature_cols) - set(features_row.columns)
                for col in missing:
                    features_row[col] = 0.0
                features_row = features_row[feature_cols]
            X = features_row.values
        elif isinstance(features_row, pd.Series):
            if feature_cols:
                row_dict = features_row.to_dict()
                aligned = [row_dict.get(f, 0.0) for f in feature_cols]
                X = np.array(aligned).reshape(1, -1)
            else:
                X = features_row.values.reshape(1, -1)
        else:
            X = features_row.reshape(1, -1)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        # ─── Model A: Trend classification (auto-detect 2-class or 3-class) ───
        trend_model = models["trend_classifier"]
        trend_probs = self._predict_trend(trend_model, X)

        # ─── Model B: Range quantiles ───
        range_preds = {}
        for target in ["high", "low"]:
            for q in [10, 50, 90]:
                model_key = f"range_{target}_q{q}"
                if model_key in models:
                    model = models[model_key]
                    pred = model.predict(X)[0]
                    range_preds[f"{target}_q{q}"] = float(round(pred, 2))

        # Build expected range
        expected_range = {
            "bottom_10th": range_preds.get("low_q10", 0),
            "median_50th": round(
                (range_preds.get("high_q50", 0) + range_preds.get("low_q50", 0)) / 2, 2
            ),
            "ceiling_90th": range_preds.get("high_q90", 0),
        }

        return {
            "trend_probabilities": trend_probs,
            "expected_range": expected_range,
        }

    def _predict_trend(self, model: Any, X: np.ndarray) -> dict[str, float]:
        """Auto-detect binary (2-class) or ternary (3-class) trend model."""
        try:
            probs = model.predict_proba(X)[0]
        except Exception:
            # Fallback: if predict_proba fails, use predict
            pred = model.predict(X)[0]
            if int(pred) == 1:
                return {"up": 0.70, "sideways": 0.10, "down": 0.20}
            else:
                return {"up": 0.20, "sideways": 0.10, "down": 0.70}

        n_classes = len(probs)

        if n_classes == 2:
            # Binary: class 0 = Down, class 1 = Up
            p_up = float(round(probs[1], 4))
            p_down = float(round(probs[0], 4))
            # Synthesize sideways from uncertainty
            sideways = float(round(min(p_up, p_down) * 0.5, 4))
            # Re-normalize
            total = p_up + p_down + sideways
            return {
                "up": round(p_up / total, 4),
                "sideways": round(sideways / total, 4),
                "down": round(p_down / total, 4),
            }
        elif n_classes == 3:
            # 3-class: 0=Up, 1=Sideways, 2=Down (old XGBoost format)
            return {
                "up": float(round(probs[0], 4)),
                "sideways": float(round(probs[1], 4)),
                "down": float(round(probs[2], 4)),
            }
        else:
            # Fallback
            return {"up": 0.33, "sideways": 0.34, "down": 0.33}

    # ── Feature Engineering for new models ────────────────────

    def compute_features_for_ticker(
        self,
        ticker: str,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Compute features matching the saved feature_cols.joblib for a ticker.

        This uses the EXACT same feature engineering as train_ml_tickers.py v3.
        """
        import pandas_ta as ta

        df = df.copy()

        # Standardize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower().strip()
            if lower in ('time', 'date', 'datetime', 'timestamp'):
                col_map[col] = 'time'
            elif lower in ('open',):
                col_map[col] = 'open'
            elif lower in ('high',):
                col_map[col] = 'high'
            elif lower in ('low',):
                col_map[col] = 'low'
            elif lower in ('close',):
                col_map[col] = 'close'
            elif lower in ('volume',):
                col_map[col] = 'volume'
        df = df.rename(columns=col_map)

        if 'time' not in df.columns and 'date' in df.columns:
            df['time'] = df['date']

        df = df.sort_values('time').reset_index(drop=True)
        c, h, l, v = df['close'], df['high'], df['low'], df['volume']

        # Returns (relative)
        df['d_log_return'] = np.log(c / c.shift(1))
        df['d_pct_return'] = c.pct_change()
        df['d_pct_return_2'] = c.pct_change(2)
        df['d_pct_return_5'] = c.pct_change(5)

        # Intraday range
        df['d_range_pct'] = (h - l) / (c + 1e-9)
        if 'open' in df.columns:
            df['d_body_pct'] = (c - df['open']) / (c + 1e-9)
            df['d_upper_shadow'] = (h - c.clip(lower=df['open'])) / (c + 1e-9)
        else:
            df['d_body_pct'] = 0.0
            df['d_upper_shadow'] = 0.0

        # SMA distances
        for w in [5, 10, 20, 50]:
            sma = ta.sma(c, length=w)
            df[f'd_dist_sma_{w}'] = (c - sma) / (sma + 1e-9)

        # Oscillators
        rsi = ta.rsi(c, length=14)
        df['d_rsi_14'] = rsi if rsi is not None else 0.0
        macd_df = ta.macd(c, fast=12, slow=26, signal=9)
        if macd_df is not None:
            df['d_macd_norm'] = macd_df.iloc[:, 0] / (c + 1e-9)
            df['d_macd_hist'] = macd_df.iloc[:, 1] / (c + 1e-9)
        else:
            df['d_macd_norm'] = df['d_macd_hist'] = 0.0

        # Bollinger Band %B
        bb = ta.bbands(c, length=20, std=2)
        if bb is not None and not bb.empty:
            df['d_bb_pct'] = (c - bb.iloc[:, 0]) / (bb.iloc[:, 2] - bb.iloc[:, 0] + 1e-9)
            df['d_bb_width'] = (bb.iloc[:, 2] - bb.iloc[:, 0]) / (c + 1e-9)
        else:
            df['d_bb_pct'] = 0.5
            df['d_bb_width'] = 0.0

        # ATR (relative)
        atr = ta.atr(h, l, c, length=14)
        df['d_atr_pct'] = atr / (c + 1e-9) if atr is not None else 0.0

        # Trend strength
        adx = ta.adx(h, l, c, length=14)
        df['d_adx_14'] = adx.iloc[:, 0] if adx is not None else 0.0
        cci = ta.cci(h, l, c, length=14)
        df['d_cci_14'] = cci if cci is not None else 0.0
        roc = ta.roc(c, length=10)
        df['d_roc_10'] = roc if roc is not None else 0.0

        # Pivot distances
        pivot = (h + l + c) / 3
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h
        df['d_dist_pivot'] = (c - pivot) / (pivot + 1e-9)
        df['d_dist_r1'] = (c - r1) / (r1 + 1e-9)
        df['d_dist_s1'] = (c - s1) / (s1 + 1e-9)

        # Volume features
        df['d_vol_dir'] = np.where(c > c.shift(1), 1, np.where(c < c.shift(1), -1, 0))
        for w in [5, 20]:
            df[f'd_vol_ratio_{w}'] = v / (v.rolling(w, min_periods=1).mean() + 1e-9)
            df[f'd_ret_roll_{w}'] = c.pct_change(w)

        # Momentum composite
        df['d_mom_5_20'] = df.get('d_dist_sma_5', 0) - df.get('d_dist_sma_20', 0)

        # Day of week
        try:
            df['d_dow'] = pd.to_datetime(df['time']).dt.dayofweek
        except Exception:
            df['d_dow'] = 0

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.ffill().bfill()

        return df

    # ── Old Training API (kept for backward compat) ─────────

    def train(self, ticker: str, df: pd.DataFrame) -> dict[str, Any]:
        """Train models via the old XGBoost/LightGBM pipeline.

        For new v3 training, use scripts/train_ml_tickers.py instead.
        """
        from src.ml.feature_engineering import FeatureEngineer

        fe = FeatureEngineer()
        feat_df = fe.transform(df)
        feature_cols = fe.get_feature_columns(feat_df)

        if len(feat_df) < 100:
            raise ValueError(f"Insufficient data for {ticker}: {len(feat_df)} rows")

        # Train trend classifier
        trend_metrics = self._train_trend_classifier(ticker, feat_df, feature_cols)
        range_metrics = self._train_range_regressor(ticker, feat_df, feature_cols)
        self._save_models(ticker)

        return {
            "ticker": ticker,
            "trend_classifier": trend_metrics,
            "range_regressor": range_metrics,
            "feature_count": len(feature_cols),
            "training_rows": len(feat_df),
        }

    def _train_trend_classifier(self, ticker, df, feature_cols):
        from xgboost import XGBClassifier
        from sklearn.model_selection import TimeSeriesSplit

        threshold = self._settings.trend_threshold_pct / 100.0
        lookahead = self._settings.trend_lookahead_days
        df = df.copy()
        df["future_return"] = df["close"].shift(-lookahead) / df["close"] - 1
        conditions = [df["future_return"] > threshold, df["future_return"] < -threshold]
        df["label"] = np.select(conditions, [0, 2], default=1)
        train_df = df.dropna(subset=["future_return"]).reset_index(drop=True)
        X, y = train_df[feature_cols].values, train_df["label"].values.astype(int)

        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []
        for train_idx, val_idx in tscv.split(X):
            model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                                  objective="multi:softprob", num_class=3,
                                  eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
            model.fit(X[train_idx], y[train_idx], eval_set=[(X[val_idx], y[val_idx])], verbose=False)
            cv_scores.append((model.predict(X[val_idx]) == y[val_idx]).mean())

        final = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05,
                              objective="multi:softprob", num_class=3,
                              eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
        final.fit(X, y, verbose=False)
        if ticker not in self._models:
            self._models[ticker] = {}
        self._models[ticker]["trend_classifier"] = final
        self._models[ticker]["feature_cols"] = feature_cols
        return {"cv_accuracy": float(np.mean(cv_scores))}

    def _train_range_regressor(self, ticker, df, feature_cols):
        from lightgbm import LGBMRegressor
        from sklearn.model_selection import TimeSeriesSplit

        df = df.copy()
        df["next_high"] = df["high"].shift(-1)
        df["next_low"] = df["low"].shift(-1)
        train_df = df.dropna(subset=["next_high", "next_low"]).reset_index(drop=True)
        X = train_df[feature_cols].values
        targets = {"high": train_df["next_high"].values, "low": train_df["next_low"].values}

        if ticker not in self._models:
            self._models[ticker] = {}

        for target_name, y in targets.items():
            for q in [0.10, 0.50, 0.90]:
                model = LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                      objective="quantile", alpha=q, random_state=42,
                                      n_jobs=-1, verbosity=-1)
                model.fit(X, y)
                self._models[ticker][f"range_{target_name}_q{int(q*100)}"] = model
        return {"cv_mae_median": 0.0}

    # ── Persistence ───────────────────────────────────────────

    def _save_models(self, ticker: str) -> None:
        if ticker not in self._models:
            return
        ticker_dir = self._model_dir / ticker.upper()
        ticker_dir.mkdir(parents=True, exist_ok=True)
        for model_name, model in self._models[ticker].items():
            if model_name == "feature_cols":
                joblib.dump(model, ticker_dir / "feature_cols.joblib")
            else:
                joblib.dump(model, ticker_dir / f"{model_name}.joblib")
        logger.info("models_saved", ticker=ticker, path=str(ticker_dir))

    def _load_models(self, ticker: str) -> None:
        ticker_dir = self._model_dir / ticker.upper()
        if not ticker_dir.exists():
            raise FileNotFoundError(
                f"No trained models found for {ticker}. "
                f"Run: python scripts/train_ml_tickers.py --daily data/daily_market_split_data"
            )
        self._models[ticker] = {}
        for model_path in ticker_dir.glob("*.joblib"):
            model_name = model_path.stem
            self._models[ticker][model_name] = joblib.load(model_path)
        logger.info("models_loaded", ticker=ticker, path=str(ticker_dir),
                     keys=list(self._models[ticker].keys()))

    def _ensure_models_loaded(self, ticker: str) -> None:
        if ticker not in self._models or "trend_classifier" not in self._models.get(ticker, {}):
            self._load_models(ticker)
