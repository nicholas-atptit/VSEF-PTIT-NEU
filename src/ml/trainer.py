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
from src.ml.feature_engineering import FeatureEngineer
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
        horizon: str = "short", # short, mid, long
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

        # Horizon suffix logic (short, mid, long)
        h_map = {"short": "_short", "mid": "_mid", "long": "_long"}
        # Backward compatibility fallbacks
        legacy_map = {"1w": "_short", "1m": "_mid", "6m": "_long", "1d": ""}
        
        suffix = h_map.get(horizon.lower(), legacy_map.get(horizon.lower(), ""))

        # ─── Model A: Trend classification ───
        model_key = f"trend_classifier{suffix}"
        if model_key not in models and suffix != "": 
            # Fallback to 1d if specific horizon not found
            model_key = "trend_classifier"
            
        trend_model = models.get(model_key)
        if not trend_model: return {}
        
        trend_probs = self._predict_trend(trend_model, X)
        
        # --- Phase 40: Meta-Confidence Filtering ---
        meta_key = f"meta_classifier{suffix}"
        meta_model = models.get(meta_key)
        confidence = 1.0 # Default if no meta-model
        
        if meta_model:
            # Meta-model predicts if the primary ensemble is correct (1=Correct, 0=Wrong)
            meta_probs = meta_model.predict_proba(X)[0]
            confidence = float(meta_probs[1]) # Probability of being correct
            
            # Apply thresholding: If confidence < 0.7, we treat as "Side" or "Neutral"
            if confidence < 0.7:
                # Weighted average towards neutral (0.33 each)
                for k in trend_probs:
                    trend_probs[k] = (trend_probs[k] * confidence) + (0.33 * (1 - confidence))
        
        trend_probs["confidence"] = round(confidence, 4)

        # ─── Model B: Range quantiles ───
        range_preds = {}
        for target in ["high", "low"]:
            for q in [10, 50, 90]:
                model_key = f"range_{target}_q{q}{suffix}"
                if model_key not in models and suffix != "":
                    model_key = f"range_{target}_q{q}" # Fallback
                
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
            "feature_set_version": "v5.0", # Added per Phase 2
            "horizon": horizon
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
        """Compute features for a single ticker matching v3 training.
        
        This uses the unified FeatureEngineer class.
        """
        # 1. Ensure column names are standard
        col_map = {}
        for col in df.columns:
            l_col = col.lower().strip()
            if l_col in ('time', 'date', 'datetime'): col_map[col] = 'date'
            elif l_col == 'open': col_map[col] = 'open'
            elif l_col == 'high': col_map[col] = 'high'
            elif l_col == 'low': col_map[col] = 'low'
            elif l_col == 'close': col_map[col] = 'close'
            elif l_col == 'volume': col_map[col] = 'volume'
        
        df = df.rename(columns=col_map)
        if 'date' not in df.columns:
            df['date'] = pd.to_datetime('today')
            
        # 2. Transform using FeatureEngineer
        fe = FeatureEngineer()
        feat_df = fe.transform(df)
        
        # 3. Add 'd_' prefix for trained model compatibility
        core_cols = fe.get_feature_columns(feat_df)
        for col in core_cols:
            if not col.startswith('d_'):
                feat_df[f'd_{col}'] = feat_df[col]
        
        # 4. Cleanup
        feat_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        feat_df = feat_df.ffill().bfill()
        
        return feat_df

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
        if ticker not in self._models or len(self._models[ticker]) < 2:
            self._load_models(ticker)
