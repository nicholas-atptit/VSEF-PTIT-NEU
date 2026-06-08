"""Bounded local-data-only VN forecast engine."""

from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.metrics import direction_metrics, interval_metrics, ranking_metrics, return_price_metrics
from src.features.builders import (
    build_market_context_features,
    build_momentum_features,
    build_range_features,
    build_relative_strength_features,
    build_volume_volatility_features,
)
from src.forecasting.panels import build_forecast_panel
from src.governance.split_policy import assign_split
from src.utils.paths import REPO_ROOT
from src.utils.research_io import json_safe, write_json, write_markdown

CLAIM_LABEL = "offline_diagnostic_forecast_only"
OHLCV = ("open", "high", "low", "close", "volume")
FEATURE_COLUMNS = (
    "return_1_lag_1",
    "return_1_lag_2",
    "return_1_lag_3",
    "momentum_3",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "volume_ratio_5",
    "volume_ratio_10",
    "range_pct",
    "range_mean_5",
    "range_mean_10",
    "range_mean_20",
    "VNINDEX_lag_return_1",
    "VN30_lag_return_1",
    "relative_strength_vs_VNINDEX_lag_return_1",
    "relative_strength_vs_VN30_lag_return_1",
)


@dataclass(frozen=True)
class EngineConfig:
    frequency: str
    horizons: tuple[int, ...]
    index_codes: tuple[str, ...]
    timeout_seconds: int = 14400
    enable_qml_features: bool = False
    asof: pd.Timestamp | None = None


@dataclass
class ThresholdedClassifier:
    model: Any
    positive_ratio: float

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        probability = self.predict_proba(frame)[:, 1]
        count = min(len(probability) - 1, max(1, int(round(len(probability) * self.positive_ratio))))
        prediction = np.zeros(len(probability), dtype=int)
        prediction[np.argsort(probability)[-count:]] = 1
        return prediction


@dataclass
class ConstantReturnModel:
    value: float = 0.0

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.value, dtype=float)


class OfflineVNForecastEngine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self.started = time.monotonic()
        self.output_dir = REPO_ROOT / "reports" / "generated" / "vn_forecast_engine_v1"
        self.result_dir = REPO_ROOT / "reports" / "results"
        self.claim_dir = REPO_ROOT / "reports" / "claims"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.selected_models: dict[int, dict[str, str]] = {}
        self.fitted_models: dict[tuple[int, str], Any] = {}
        self.range_parameters: dict[tuple[int, str], tuple[float, float]] = {}
        self.audit_rows: list[dict[str, Any]] = []

    def _check_timeout(self) -> None:
        if time.monotonic() - self.started > self.config.timeout_seconds:
            raise TimeoutError("forecast engine timeout exceeded")

    @staticmethod
    def _configured_stocks() -> list[str]:
        path = REPO_ROOT / "configs" / "universes" / "vn30_jan2025_joint_panel_universe.csv"
        frame = pd.read_csv(path)
        active = frame["active_for_joint_panel"].astype(str).str.lower().isin({"true", "1", "yes"})
        return frame.loc[active & frame["instrument_type"].eq("stock"), "instrument_code"].astype(str).str.upper().tolist()

    def _candidate_paths(self, code: str, asset_type: str) -> list[Path]:
        if asset_type == "stock":
            return [
                REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "vn30" / "hourly_2015" / f"{code}.csv",
                REPO_ROOT / "data" / "hourly_market_split_data" / f"{code}.csv",
            ]
        return [
            REPO_ROOT / "archive" / "generated_data_snapshots" / "vn30_hourly_pre_benchmark_20260514_062528" / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly" / f"{code}.csv",
            REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015" / f"{code}.csv",
        ]

    @staticmethod
    def _standardize(path: Path, code: str, asset_type: str) -> pd.DataFrame:
        raw = pd.read_csv(path, low_memory=False)
        if "time" in raw and "datetime" not in raw:
            raw = raw.rename(columns={"time": "datetime"})
        raw["feature_timestamp"] = pd.to_datetime(raw.get("datetime"), errors="coerce")
        raw["asset_code"] = code
        raw["asset_type"] = asset_type
        for column in OHLCV:
            if column not in raw:
                raw[column] = 0.0 if column == "volume" else np.nan
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        out = raw[["asset_code", "asset_type", "feature_timestamp", *OHLCV]].dropna(
            subset=["feature_timestamp", "open", "high", "low", "close"]
        )
        out["volume"] = out["volume"].fillna(0.0)
        out["data_source_path"] = path.relative_to(REPO_ROOT).as_posix()
        return out.sort_values("feature_timestamp").drop_duplicates("feature_timestamp", keep="last")

    def load_local_panel(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        coverage: list[dict[str, Any]] = []
        source_audit: list[dict[str, Any]] = []
        requested = [(code, "stock") for code in self._configured_stocks()]
        requested.extend((code, "index") for code in self.config.index_codes)
        for code, asset_type in requested:
            candidates = self._candidate_paths(code, asset_type)
            loaded: list[tuple[pd.DataFrame, Path]] = []
            for path in candidates:
                if path.exists():
                    frame = self._standardize(path, code, asset_type)
                    source_audit.append(
                        {
                            "asset_code": code,
                            "asset_type": asset_type,
                            "source_path": path.relative_to(REPO_ROOT).as_posix(),
                            "exists": True,
                            "rows": len(frame),
                            "min_timestamp": frame["feature_timestamp"].min(),
                            "max_timestamp": frame["feature_timestamp"].max(),
                        }
                    )
                    if not frame.empty:
                        loaded.append((frame, path))
                else:
                    source_audit.append({"asset_code": code, "asset_type": asset_type, "source_path": path.relative_to(REPO_ROOT).as_posix(), "exists": False, "rows": 0})
            chosen = max(loaded, key=lambda item: len(item[0])) if loaded else None
            coverage.append(
                {
                    "asset_code": code,
                    "asset_type": asset_type,
                    "status": "loaded" if chosen else "missing",
                    "rows": len(chosen[0]) if chosen else 0,
                    "source_path": chosen[1].relative_to(REPO_ROOT).as_posix() if chosen else "",
                    "min_timestamp": chosen[0]["feature_timestamp"].min() if chosen else "",
                    "max_timestamp": chosen[0]["feature_timestamp"].max() if chosen else "",
                }
            )
            if chosen:
                frames.append(chosen[0])
        pd.DataFrame(source_audit).to_csv(self.output_dir / "local_data_source_audit.csv", index=False)
        coverage_frame = pd.DataFrame(coverage)
        coverage_frame.to_csv(self.output_dir / "asset_coverage_audit.csv", index=False)
        coverage_frame[coverage_frame["asset_type"].eq("index")].to_csv(self.output_dir / "index_coverage_audit.csv", index=False)
        loaded_stocks = coverage_frame.query("asset_type == 'stock' and status == 'loaded'")
        if len(loaded_stocks) < 20:
            raise RuntimeError(f"insufficient VN30 stock coverage: {len(loaded_stocks)}/30")
        return pd.concat(frames, ignore_index=True).sort_values(["asset_code", "feature_timestamp"]).reset_index(drop=True)

    def build_features(self, panel: pd.DataFrame) -> pd.DataFrame:
        frame = panel.rename(columns={"feature_timestamp": "asof_timestamp"}).copy()
        grouped = frame.groupby("asset_code", group_keys=False)
        frame["return_1"] = grouped["close"].pct_change(fill_method=None)
        for lag in (1, 2, 3):
            frame[f"return_1_lag_{lag}"] = frame.groupby("asset_code")["return_1"].shift(lag)
        frame = build_momentum_features(frame)
        frame = build_volume_volatility_features(frame)
        frame = build_range_features(frame)
        for market_code in ("VNINDEX", "VN30"):
            if frame["asset_code"].eq(market_code).any():
                frame = build_market_context_features(frame, market_code=market_code)
        benchmark_columns = [column for column in ("VNINDEX_lag_return_1", "VN30_lag_return_1") if column in frame]
        frame = build_relative_strength_features(frame, benchmark_return_columns=benchmark_columns)
        frame = frame.rename(columns={"asof_timestamp": "feature_timestamp"})
        available = [column for column in FEATURE_COLUMNS if column in frame]
        audit = pd.DataFrame(
            [
                {"feature_group": "momentum_features", "feature": column, "point_in_time_safe": True}
                for column in available if column.startswith("momentum") or column.startswith("return_1_lag")
            ]
            + [{"feature_group": "market_context_features", "feature": column, "point_in_time_safe": True} for column in available if "_lag_return_" in column and column.startswith(("VN", "HNX", "UPCOM"))]
            + [{"feature_group": "relative_strength_features", "feature": column, "point_in_time_safe": True} for column in available if column.startswith("relative_strength")]
            + [{"feature_group": "volume_volatility_features", "feature": column, "point_in_time_safe": True} for column in available if column.startswith(("volume_ratio", "rolling_volatility"))]
            + [{"feature_group": "range_features", "feature": column, "point_in_time_safe": True} for column in available if column.startswith("range")]
        )
        audit.to_csv(self.output_dir / "feature_audit.csv", index=False)
        audit.assign(column_order=range(1, len(audit) + 1)).to_csv(self.output_dir / "feature_column_registry.csv", index=False)
        return frame

    @staticmethod
    def _market_future_return(frame: pd.DataFrame, code: str, horizon: int) -> pd.Series:
        market = frame[frame["asset_code"].eq(code)][["feature_timestamp", "close"]].copy()
        market[f"{code}_future_return"] = market["close"].shift(-horizon) / market["close"] - 1.0
        mapping = market.set_index("feature_timestamp")[f"{code}_future_return"]
        return frame["feature_timestamp"].map(mapping)

    def build_dataset(self, features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        datasets: list[pd.DataFrame] = []
        feature_columns = [column for column in FEATURE_COLUMNS if column in features]
        for horizon in self.config.horizons:
            self._check_timeout()
            frame = features.copy()
            grouped = frame.groupby("asset_code", group_keys=False)
            frame["horizon"] = horizon
            frame["target_timestamp"] = grouped["feature_timestamp"].shift(-horizon)
            frame["future_close_h"] = grouped["close"].shift(-horizon)
            frame["future_high_h"] = grouped["high"].shift(-horizon)
            frame["future_low_h"] = grouped["low"].shift(-horizon)
            frame["forward_simple_return_h"] = frame["future_close_h"] / frame["close"] - 1.0
            frame["forward_log_return_h"] = np.log(frame["future_close_h"] / frame["close"])
            frame["absolute_direction_h"] = (frame["forward_simple_return_h"] > 0).astype(float)
            frame.loc[frame["forward_simple_return_h"].isna(), "absolute_direction_h"] = np.nan
            vn30 = self._market_future_return(frame, "VN30", horizon)
            vnindex = self._market_future_return(frame, "VNINDEX", horizon)
            frame["market_relative_vn30_h"] = (frame["forward_simple_return_h"] > vn30).astype(float)
            frame["market_relative_vnindex_h"] = (frame["forward_simple_return_h"] > vnindex).astype(float)
            frame["market_excess_return_h"] = frame["forward_simple_return_h"] - vnindex
            volatility = frame.get("rolling_volatility_20", pd.Series(np.nan, index=frame.index)).replace(0.0, np.nan)
            frame["volatility_adjusted_return_h"] = frame["forward_simple_return_h"] / volatility
            frame["future_high_return_h"] = frame["future_high_h"] / frame["close"] - 1.0
            frame["future_low_return_h"] = frame["future_low_h"] / frame["close"] - 1.0
            frame["future_range_pct_h"] = (frame["future_high_h"] - frame["future_low_h"]) / frame["close"]
            frame["cross_sectional_forward_return_rank_h"] = frame.groupby("feature_timestamp")["forward_simple_return_h"].rank(pct=True)
            frame["market_excess_return_rank_h"] = frame.groupby("feature_timestamp")["market_excess_return_h"].rank(pct=True)
            frame["top_20pct_forward_return_h"] = (frame["cross_sectional_forward_return_rank_h"] >= 0.8).astype(float)
            frame["top_30pct_forward_return_h"] = (frame["cross_sectional_forward_return_rank_h"] >= 0.7).astype(float)
            frame = assign_split(frame)
            datasets.append(frame)
        dataset = pd.concat(datasets, ignore_index=True)
        target_columns = [
            "absolute_direction_h", "market_relative_vn30_h", "market_relative_vnindex_h",
            "forward_simple_return_h", "forward_log_return_h", "future_close_h", "market_excess_return_h",
            "volatility_adjusted_return_h", "future_high_h", "future_low_h", "future_high_return_h",
            "future_low_return_h", "future_range_pct_h", "cross_sectional_forward_return_rank_h",
            "market_excess_return_rank_h", "top_20pct_forward_return_h", "top_30pct_forward_return_h",
        ]
        pd.DataFrame(
            [{"target": target, "non_null_rows": int(dataset[target].notna().sum()), "task": self._target_task(target)} for target in target_columns]
        ).to_csv(self.output_dir / "target_audit.csv", index=False)
        balances = dataset.groupby(["horizon", "split"], dropna=False)["absolute_direction_h"].agg(rows="count", positive_ratio="mean").reset_index()
        balances.to_csv(self.output_dir / "class_balance_audit.csv", index=False)
        dataset.groupby(["horizon", "split"], dropna=False).agg(rows=("asset_code", "size"), assets=("asset_code", "nunique"), min_feature_timestamp=("feature_timestamp", "min"), max_feature_timestamp=("feature_timestamp", "max"), min_target_timestamp=("target_timestamp", "min"), max_target_timestamp=("target_timestamp", "max")).reset_index().to_csv(self.output_dir / "split_guard_audit.csv", index=False)
        dataset.groupby(["horizon", "asset_type"]).agg(rows=("asset_code", "size"), assets=("asset_code", "nunique"), min_timestamp=("feature_timestamp", "min"), max_timestamp=("feature_timestamp", "max")).reset_index().to_csv(self.output_dir / "dataset_audit.csv", index=False)
        return dataset, feature_columns

    @staticmethod
    def _target_task(target: str) -> str:
        if "direction" in target:
            return "direction"
        if "rank" in target or "top_" in target:
            return "ranking"
        if "high" in target or "low" in target or "range" in target:
            return "range_interval"
        return "return_price"

    @staticmethod
    def _direction_candidates() -> dict[str, Any]:
        return {
            "logistic_regression": Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", LogisticRegression(max_iter=500, C=0.5))]),
            "hist_gradient_boosting_classifier": Pipeline([("imputer", SimpleImputer()), ("model", HistGradientBoostingClassifier(max_iter=80, max_depth=4, random_state=42))]),
            "random_forest_classifier": Pipeline([("imputer", SimpleImputer()), ("model", RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=10, random_state=42, n_jobs=4))]),
            "extra_trees_classifier": Pipeline([("imputer", SimpleImputer()), ("model", ExtraTreesClassifier(n_estimators=80, max_depth=7, min_samples_leaf=8, random_state=42, n_jobs=4))]),
        }

    @staticmethod
    def _return_candidates() -> dict[str, Any]:
        return {
            "ridge": Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
            "lasso": Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", Lasso(alpha=0.0005, max_iter=1000))]),
            "elasticnet": Pipeline([("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", ElasticNet(alpha=0.0005, l1_ratio=0.25, max_iter=1000))]),
            "hist_gradient_boosting_regressor": Pipeline([("imputer", SimpleImputer()), ("model", HistGradientBoostingRegressor(max_iter=80, max_depth=4, random_state=42))]),
            "gradient_boosting_regressor": Pipeline([("imputer", SimpleImputer()), ("model", GradientBoostingRegressor(n_estimators=80, max_depth=3, random_state=42))]),
        }

    @staticmethod
    def _stability(frame: pd.DataFrame, truth: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
        work = frame[["asset_code", "feature_timestamp"]].copy()
        work["correct"] = truth == pred
        asset = work.groupby("asset_code")["correct"].mean()
        quarter = work.groupby(work["feature_timestamp"].dt.to_period("Q"))["correct"].mean()
        return float(asset.mean()), float(quarter.mean())

    def evaluate(self, dataset: pd.DataFrame, feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        direction_rows: list[dict[str, Any]] = []
        return_rows: list[dict[str, Any]] = []
        range_rows: list[dict[str, Any]] = []
        ranking_rows: list[dict[str, Any]] = []
        baseline_rows: list[dict[str, Any]] = []
        for horizon in self.config.horizons:
            self._check_timeout()
            scoped = dataset[dataset["horizon"].eq(horizon)].dropna(subset=feature_columns + ["absolute_direction_h", "forward_simple_return_h"]).copy()
            train = scoped[scoped["split"].eq("train")]
            validation = scoped[scoped["split"].eq("validation")]
            final = scoped[scoped["split"].eq("final")]
            if train.empty or validation.empty:
                continue
            strongest_baseline = 0.0
            for name, prediction in {
                "always_up": np.ones(len(validation), dtype=int),
                "always_down": np.zeros(len(validation), dtype=int),
                "lag1_direction": (validation["return_1_lag_1"].to_numpy() > 0).astype(int),
                "majority_class": np.full(len(validation), int(train["absolute_direction_h"].mean() >= 0.5)),
            }.items():
                metrics = direction_metrics(validation["absolute_direction_h"], prediction)
                strongest_baseline = max(strongest_baseline, float(metrics["balanced_accuracy"]))
                baseline_rows.append({"horizon": horizon, "task": "direction", "model_id": name, "split": "validation", **metrics})
            direction_candidates = self._direction_candidates()
            fitted_direction: dict[str, Any] = {}
            for model_id, model in direction_candidates.items():
                model.fit(train[feature_columns], train["absolute_direction_h"].astype(int))
                positive_ratio = float(np.clip(train["absolute_direction_h"].mean(), 0.1, 0.9))
                locked_model = ThresholdedClassifier(model, positive_ratio)
                fitted_direction[model_id] = locked_model
                for split_name, split_frame in (("validation", validation), ("final", final)):
                    if split_frame.empty:
                        continue
                    pred = locked_model.predict(split_frame[feature_columns]).astype(int)
                    probability = locked_model.predict_proba(split_frame[feature_columns])[:, 1]
                    metrics = direction_metrics(split_frame["absolute_direction_h"], pred, probability)
                    asset_stability, quarter_stability = self._stability(split_frame, split_frame["absolute_direction_h"].to_numpy().astype(int), pred)
                    direction_rows.append({"horizon": horizon, "model_id": model_id, "target": "absolute_direction_h", "split": split_name, "locked_train_positive_ratio": positive_ratio, "decision_rule": "probability_rank_at_train_positive_ratio", "lift_over_strongest_simple_baseline": float(metrics["balanced_accuracy"]) - strongest_baseline, "prediction_up_ratio": metrics["prediction_balance"], "asset_level_stability": asset_stability, "quarter_stability": quarter_stability, **metrics})
            validation_direction = pd.DataFrame(direction_rows).query("horizon == @horizon and split == 'validation' and prediction_up_ratio >= 0.05 and prediction_up_ratio <= 0.95").sort_values(["balanced_accuracy", "mcc", "lift_over_strongest_simple_baseline"], ascending=False)
            selected_direction = str(validation_direction.iloc[0]["model_id"])
            self.fitted_models[(horizon, "direction")] = fitted_direction[selected_direction]

            return_candidates = self._return_candidates()
            fitted_return: dict[str, Any] = {}
            validation_baselines = {
                "random_walk": np.zeros(len(validation)),
                "last_close": np.zeros(len(validation)),
                "historical_mean_return": np.full(len(validation), train["forward_simple_return_h"].mean()),
                "rolling_mean_return": validation["return_1_lag_1"].fillna(0.0).to_numpy() * horizon,
            }
            baseline_rmse: dict[str, float] = {}
            for name, prediction in validation_baselines.items():
                metrics = return_price_metrics(validation["forward_simple_return_h"], prediction)
                baseline_rmse[name] = float(metrics["rmse"])
                baseline_rows.append({"horizon": horizon, "task": "return_price", "model_id": name, "split": "validation", **metrics})
            for split_name, split_frame in (("validation", validation), ("final", final)):
                if split_frame.empty:
                    continue
                metrics = return_price_metrics(split_frame["forward_simple_return_h"], np.zeros(len(split_frame)))
                return_rows.append({"horizon": horizon, "model_id": "random_walk_return_baseline", "target": "forward_simple_return_h", "split": split_name, "beats_random_walk": False, "beats_last_close": False, "candidate_confirmed": False, "improvement_vs_random_walk": 0.0, "improvement_vs_last_close": 0.0, "improvement_vs_historical_mean_return": baseline_rmse["historical_mean_return"] - float(metrics["rmse"]), "improvement_vs_rolling_mean_return": baseline_rmse["rolling_mean_return"] - float(metrics["rmse"]), **metrics})
            for model_id, model in return_candidates.items():
                model.fit(train[feature_columns], train["forward_simple_return_h"])
                fitted_return[model_id] = model
                for split_name, split_frame in (("validation", validation), ("final", final)):
                    if split_frame.empty:
                        continue
                    pred = model.predict(split_frame[feature_columns])
                    metrics = return_price_metrics(split_frame["forward_simple_return_h"], pred)
                    beats_random_walk = float(metrics["rmse"]) < baseline_rmse["random_walk"]
                    beats_last_close = float(metrics["rmse"]) < baseline_rmse["last_close"]
                    return_rows.append({"horizon": horizon, "model_id": model_id, "target": "forward_simple_return_h", "split": split_name, "beats_random_walk": beats_random_walk, "beats_last_close": beats_last_close, "candidate_confirmed": beats_random_walk or beats_last_close, "improvement_vs_random_walk": baseline_rmse["random_walk"] - float(metrics["rmse"]), "improvement_vs_last_close": baseline_rmse["last_close"] - float(metrics["rmse"]), "improvement_vs_historical_mean_return": baseline_rmse["historical_mean_return"] - float(metrics["rmse"]), "improvement_vs_rolling_mean_return": baseline_rmse["rolling_mean_return"] - float(metrics["rmse"]), **metrics})
            validation_return = pd.DataFrame(return_rows).query("horizon == @horizon and split == 'validation'").sort_values(["rmse", "mae"])
            eligible_return = validation_return[validation_return["candidate_confirmed"].astype(bool)]
            selected_return = str((eligible_return if not eligible_return.empty else validation_return.query("model_id == 'random_walk_return_baseline'")).iloc[0]["model_id"])
            self.fitted_models[(horizon, "return")] = fitted_return[selected_return] if selected_return in fitted_return else ConstantReturnModel()

            train_quantiles = tuple(train["forward_simple_return_h"].quantile([0.1, 0.9]).tolist())
            self.range_parameters[(horizon, "historical_quantile_return_band")] = train_quantiles
            for range_id in ("historical_quantile_return_band", "rolling_high_low_range_band", "rolling_volatility_band"):
                for split_name, split_frame in (("validation", validation), ("final", final)):
                    if split_frame.empty:
                        continue
                    low_return, high_return = self._range_returns(split_frame, range_id, train_quantiles)
                    low = split_frame["close"].to_numpy() * (1 + low_return)
                    high = split_frame["close"].to_numpy() * (1 + high_return)
                    metrics = interval_metrics(split_frame["future_close_h"], low, high, alpha=0.2)
                    actual_hl = float(np.mean((split_frame["future_low_h"].to_numpy() >= low) & (split_frame["future_high_h"].to_numpy() <= high)))
                    range_rows.append({"horizon": horizon, "model_id": range_id, "split": split_name, "actual_close_inside_interval": metrics["interval_coverage"], "actual_high_low_coverage": actual_hl, **metrics})
            validation_range = pd.DataFrame(range_rows).query("horizon == @horizon and split == 'validation'").copy()
            in_band = validation_range[validation_range["interval_coverage"].between(0.7, 0.9)]
            selected_range = str((in_band if not in_band.empty else validation_range).sort_values(["winkler_score", "average_interval_width"]).iloc[0]["model_id"])

            selected_return_model = self.fitted_models[(horizon, "return")]
            for split_name, split_frame in (("validation", validation), ("final", final)):
                if split_frame.empty:
                    continue
                candidates = {
                    "relative_strength_rank_baseline": split_frame.get("relative_strength_vs_VNINDEX_lag_return_1", split_frame["momentum_20"]).fillna(0.0).to_numpy(),
                    "momentum_rank_baseline": split_frame["momentum_20"].fillna(0.0).to_numpy(),
                    "return_model_rank": selected_return_model.predict(split_frame[feature_columns]),
                }
                for model_id, scores in candidates.items():
                    metric_groups = [
                        ranking_metrics(group["forward_simple_return_h"], pd.Series(scores, index=split_frame.index).loc[group.index])
                        for _, group in split_frame.groupby("feature_timestamp") if len(group) >= 5
                    ]
                    metrics = pd.DataFrame(metric_groups).mean(numeric_only=True).to_dict() if metric_groups else ranking_metrics([], [])
                    top_count = max(1, int(len(split_frame) * 0.1))
                    top_return = float(split_frame.assign(score=scores).nlargest(top_count, "score")["forward_simple_return_h"].mean())
                    ranking_rows.append({"horizon": horizon, "model_id": model_id, "split": split_name, "top_decile_realized_return": top_return, **metrics})
            validation_rank = pd.DataFrame(ranking_rows).query("horizon == @horizon and split == 'validation'").sort_values(["spearman_ic", "ndcg_at_10"], ascending=False)
            selected_ranking = str(validation_rank.iloc[0]["model_id"])
            self.selected_models[horizon] = {"direction": selected_direction, "return_price": selected_return, "range_interval": selected_range, "ranking": selected_ranking}
        frames = tuple(pd.DataFrame(rows) for rows in (direction_rows, return_rows, range_rows, ranking_rows, baseline_rows))
        names = ("direction_evaluation.csv", "return_price_evaluation.csv", "range_interval_evaluation.csv", "ranking_evaluation.csv", "baseline_comparison.csv")
        for name, frame in zip(names, frames):
            frame.to_csv(self.output_dir / name, index=False)
        self._write_selection(frames)
        return frames

    @staticmethod
    def _range_returns(frame: pd.DataFrame, model_id: str, quantiles: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        if model_id == "historical_quantile_return_band":
            return np.full(len(frame), quantiles[0]), np.full(len(frame), quantiles[1])
        if model_id == "rolling_high_low_range_band":
            width = frame["range_mean_20"].fillna(frame["range_pct"]).fillna(0.02).to_numpy()
        else:
            width = frame["rolling_volatility_20"].fillna(0.01).to_numpy() * 1.645 * np.sqrt(frame["horizon"].to_numpy())
        return -width, width

    def _write_selection(self, frames: tuple[pd.DataFrame, ...]) -> None:
        direction, returns, ranges, ranking, _ = frames
        rows: list[dict[str, Any]] = []
        for horizon, selected in self.selected_models.items():
            for task, model_id in selected.items():
                source = {"direction": direction, "return_price": returns, "range_interval": ranges, "ranking": ranking}[task]
                metrics = source.query("horizon == @horizon and split == 'validation' and model_id == @model_id").iloc[0].to_dict()
                rows.append({"horizon": horizon, "task": task, "model_id": model_id, "selection_split": "validation", **metrics})
        pd.DataFrame(rows).to_csv(self.output_dir / "model_selection_summary.csv", index=False)
        write_json(self.output_dir / "selected_model_registry.json", {"selection_policy": "validation_only", "final_rows": "scoring_only", "selected_models": self.selected_models, "claim_label": CLAIM_LABEL})

    def forecast_panel(self, dataset: pd.DataFrame, feature_columns: list[str], asof: pd.Timestamp | None = None) -> pd.DataFrame:
        run_timestamp = pd.Timestamp(datetime.now())
        source = dataset.copy()
        if asof is not None:
            source = source[source["feature_timestamp"] <= asof]
        rows: list[dict[str, Any]] = []
        for horizon, selected in self.selected_models.items():
            scoped = source[source["horizon"].eq(horizon)].sort_values("feature_timestamp").groupby("asset_code", as_index=False).tail(1)
            if scoped.empty:
                continue
            direction_model = self.fitted_models[(horizon, "direction")]
            return_model = self.fitted_models[(horizon, "return")]
            probability = direction_model.predict_proba(scoped[feature_columns])[:, 1]
            predicted_direction = direction_model.predict(scoped[feature_columns])
            predicted_return = return_model.predict(scoped[feature_columns])
            quantiles = self.range_parameters[(horizon, "historical_quantile_return_band")]
            low_return, high_return = self._range_returns(scoped, selected["range_interval"], quantiles)
            rank_score = self._rank_scores(scoped, selected["ranking"], predicted_return)
            rank = pd.Series(rank_score, index=scoped.index).rank(pct=True)
            for position, (index, row) in enumerate(scoped.iterrows()):
                p = float(probability[position])
                predicted = float(predicted_return[position])
                low_r, high_r = float(low_return[position]), float(high_return[position])
                target_timestamp = row.get("target_timestamp")
                if pd.isna(target_timestamp):
                    target_timestamp = pd.Timestamp(row["feature_timestamp"]) + pd.Timedelta(hours=horizon)
                actual_return = row.get("forward_simple_return_h")
                low_price = float(row["close"] * (1 + low_r))
                high_price = float(row["close"] * (1 + high_r))
                rows.append(
                    {
                        "forecast_id": f"v1-{uuid.uuid4().hex[:12]}",
                        "run_timestamp": run_timestamp,
                        "asof_timestamp": row["feature_timestamp"],
                        "asset_code": row["asset_code"],
                        "asset_type": row["asset_type"],
                        "horizon": horizon,
                        "target_timestamp": target_timestamp,
                        "direction_model_id": selected["direction"],
                        "direction_target": "absolute_direction_h",
                        "direction_probability": p,
                        "predicted_direction": int(predicted_direction[position]),
                        "direction_confidence_label": "high" if abs(p - 0.5) >= 0.2 else "moderate" if abs(p - 0.5) >= 0.1 else "low",
                        "return_model_id": selected["return_price"],
                        "predicted_return": predicted,
                        "predicted_log_return": math.log1p(max(predicted, -0.999999)),
                        "predicted_close_mid": float(row["close"] * (1 + predicted)),
                        "range_model_id": selected["range_interval"],
                        "predicted_return_p10": low_r,
                        "predicted_return_p50": predicted,
                        "predicted_return_p90": high_r,
                        "predicted_close_low": low_price,
                        "predicted_close_high": high_price,
                        "predicted_low_price": low_price,
                        "predicted_high_price": high_price,
                        "predicted_range_pct": high_r - low_r,
                        "ranking_model_id": selected["ranking"],
                        "rank_score": float(rank_score[position]),
                        "cross_sectional_rank": float(rank.loc[index]),
                        "actual_return": actual_return,
                        "actual_close": row.get("future_close_h"),
                        "actual_high": row.get("future_high_h"),
                        "actual_low": row.get("future_low_h"),
                        "correct_direction": bool((actual_return > 0) == bool(predicted_direction[position])) if pd.notna(actual_return) else pd.NA,
                        "interval_hit": bool(low_price <= row.get("future_close_h") <= high_price) if pd.notna(row.get("future_close_h")) else pd.NA,
                        "claim_label": CLAIM_LABEL,
                    }
                )
        return build_forecast_panel(rows)

    @staticmethod
    def _rank_scores(frame: pd.DataFrame, model_id: str, predicted_return: np.ndarray) -> np.ndarray:
        if model_id == "return_model_rank":
            return predicted_return
        if model_id == "relative_strength_rank_baseline":
            return frame.get("relative_strength_vs_VNINDEX_lag_return_1", frame["momentum_20"]).fillna(0.0).to_numpy()
        return frame["momentum_20"].fillna(0.0).to_numpy()

    def write_panels_and_reports(self, dataset: pd.DataFrame, feature_columns: list[str], asof: pd.Timestamp | None) -> pd.DataFrame:
        latest = self.forecast_panel(dataset, feature_columns, asof)
        latest.to_csv(self.output_dir / "forecast_panel_latest.csv", index=False)
        validation = self.forecast_panel(dataset[dataset["split"].eq("validation")], feature_columns)
        validation.to_csv(self.output_dir / "forecast_panel_validation_scoring.csv", index=False)
        final = self.forecast_panel(dataset[dataset["split"].eq("final")], feature_columns)
        final.to_csv(self.output_dir / "forecast_panel_final_scoring.csv", index=False)
        self._write_reports(latest)
        return latest

    def _write_reports(self, latest: pd.DataFrame) -> None:
        selection = pd.read_csv(self.output_dir / "model_selection_summary.csv")
        direction = pd.read_csv(self.output_dir / "direction_evaluation.csv")
        returns = pd.read_csv(self.output_dir / "return_price_evaluation.csv")
        ranges = pd.read_csv(self.output_dir / "range_interval_evaluation.csv")
        ranking = pd.read_csv(self.output_dir / "ranking_evaluation.csv")
        selected_direction = selection.query("task == 'direction'")
        selected_return = selection.query("task == 'return_price'")
        selected_range = selection.query("task == 'range_interval'")
        selected_ranking = selection.query("task == 'ranking'")
        direction_confirmed = bool((selected_direction["lift_over_strongest_simple_baseline"] > 0).any())
        return_confirmed = bool((selected_return["beats_random_walk"] | selected_return["beats_last_close"]).any())
        range_confirmed = bool(selected_range["interval_coverage"].between(0.7, 0.9).any())
        ranking_confirmed = bool((selected_ranking["spearman_ic"] > 0).any())
        table_columns = ["asset_code", "horizon", "predicted_direction", "direction_probability", "predicted_return", "predicted_close_low", "predicted_close_mid", "predicted_close_high", "predicted_range_pct", "rank_score"]
        top_rows = latest.sort_values(["cross_sectional_rank", "direction_probability"], ascending=False).head(30)
        top_up = latest[latest["predicted_direction"].eq(1)].nlargest(10, "direction_probability")
        top_down = latest[latest["predicted_direction"].eq(0)].nsmallest(10, "direction_probability")
        confidence = latest.assign(confidence=(latest["direction_probability"] - 0.5).abs()).nlargest(10, "confidence")
        widest = latest.nlargest(10, "predicted_range_pct")
        narrowest = latest.nsmallest(10, "predicted_range_pct")
        best_ranked = latest.nlargest(10, "cross_sectional_rank")
        compact = ["asset_code", "horizon", "direction_probability", "predicted_return", "predicted_range_pct", "rank_score"]
        latest_report = "\n".join(
            [
                "# VN Forecast Engine V1 Latest Forecast Report",
                "",
                f"- Data snapshot/asof timestamp: `{latest['asof_timestamp'].max() if not latest.empty else 'unavailable'}`",
                f"- Assets forecasted: {latest['asset_code'].nunique()}",
                f"- Index assets forecasted: {latest[latest['asset_type'].eq('index')]['asset_code'].nunique()}",
                f"- Index coverage: {', '.join(sorted(latest[latest['asset_type'].eq('index')]['asset_code'].unique()))}",
                f"- Claim label: `{CLAIM_LABEL}`",
                "",
                "## Top Forecasted UP Assets",
                "",
                top_up[compact].to_markdown(index=False),
                "",
                "## Top Forecasted DOWN Assets",
                "",
                top_down[compact].to_markdown(index=False),
                "",
                "## Highest Confidence Direction Forecasts",
                "",
                confidence[compact].to_markdown(index=False),
                "",
                "## Widest Predicted Ranges",
                "",
                widest[compact].to_markdown(index=False),
                "",
                "## Narrowest Predicted Ranges",
                "",
                narrowest[compact].to_markdown(index=False),
                "",
                "## Best Ranking Candidates",
                "",
                best_ranked[compact].to_markdown(index=False),
                "",
                "## Forecast Table",
                "",
                top_rows[table_columns].to_markdown(index=False),
                "",
                "## Claim Boundary",
                "",
                "Offline diagnostic forecast only. No trading, profitability, BUY/SELL, recommendation, live deployment, production, investment advice, or daily T+1 claim. Final rows are scoring-only.",
            ]
        )
        write_markdown(self.result_dir / "VN_FORECAST_ENGINE_V1_LATEST_FORECAST_REPORT.md", latest_report)
        evaluation_report = "\n".join(
            [
                "# VN Forecast Engine V1 Evaluation Summary",
                "",
                "Selection uses validation rows only; final rows are scoring-only.",
                "",
                "## Selected Models",
                "",
                selection[["horizon", "task", "model_id", "selection_split"]].to_markdown(index=False),
                "",
                f"- Direction beats strongest simple baseline on any selected validation row: {direction_confirmed}",
                f"- Return/price beats random walk or last close on any selected validation row: {return_confirmed}",
                f"- Range interval has selected validation coverage between 70%-90%: {range_confirmed}",
                f"- Ranking has positive selected validation Spearman IC: {ranking_confirmed}",
                "- Results claimable: offline diagnostic evidence only.",
                "",
                "## Limitations and Next Steps",
                "",
                "- Local cache coverage and timestamp granularity vary by asset.",
                "- Pooled bounded models are research baselines, not deployed predictors.",
                "- No live data fetch, provider call, or production workflow was used.",
            ]
        )
        write_markdown(self.result_dir / "VN_FORECAST_ENGINE_V1_EVALUATION_SUMMARY.md", evaluation_report)
        write_markdown(
            self.claim_dir / "VN_FORECAST_ENGINE_V1_CLAIM_BOUNDARY.md",
            "# VN Forecast Engine V1 Claim Boundary\n\nOffline diagnostic forecast only. No trading, profitability, BUY/SELL, recommendation, live deployment, production, investment advice, or daily T+1 operation. Final rows are scoring-only. Future-blind validation-only selection is required.",
        )
        decision = {
            "any_direction_candidate_confirmed": direction_confirmed,
            "any_return_price_candidate_confirmed": return_confirmed,
            "any_range_candidate_confirmed": range_confirmed,
            "any_ranking_candidate_confirmed": ranking_confirmed,
            "selected_models": self.selected_models,
            "limitations": ["local cache coverage varies", "bounded pooled diagnostic models", "no live data"],
            "claim_label": CLAIM_LABEL,
            "future_blind_required": True,
        }
        write_json(self.output_dir / "forecast_engine_decision.json", decision)

    def run(self, *, forecast: bool = True, asof: pd.Timestamp | None = None) -> pd.DataFrame:
        panel = self.load_local_panel()
        features = self.build_features(panel)
        dataset, feature_columns = self.build_dataset(features)
        self.evaluate(dataset, feature_columns)
        return self.write_panels_and_reports(dataset, feature_columns, asof) if forecast else pd.DataFrame()
