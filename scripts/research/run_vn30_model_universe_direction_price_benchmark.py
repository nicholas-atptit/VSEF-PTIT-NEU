"""Run VN30 model-universe direction and price/return diagnostics.

This runner is offline diagnostic-only. It evaluates direction targets and
price/return targets separately under strict feature_timestamp and
target_timestamp split discipline. Final results are scoring-only; validation
metrics choose locked diagnostic candidates.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import build_feature_families  # noqa: E402
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, load_index_data  # noqa: E402
from scripts.research.run_vn30_qml_forecasting import (  # noqa: E402
    CLASSICAL_CHAMPION,
    FINAL_START,
    SEED,
    TRAIN_END,
    VAL_END,
    VAL_START,
    FeatureSpec,
    add_v3_relative_strength_features,
    as_float,
    build_labels,
    build_source_groups,
    candidate_id,
    fit_feature_spec,
    json_safe,
    leakage_guard_passed,
    numeric_existing,
    ordered_index,
    stock_future_returns,
    strict_split_indices,
    target_timestamp_from_labels,
    v6_quantum_kernel_matrices,
    v6_scaling_transform,
    v8_kernel_feature_frames,
    write_frame,
    write_json,
    write_markdown,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_direction_price"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_DIRECTION_PRICE_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_DIRECTION_PRICE_CLAIM_BOUNDARY.md"
V2_RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_MODEL_UNIVERSE_V2_PROMOTION_RELOCK_RESULT_SUMMARY.md"
V2_CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_MODEL_UNIVERSE_V2_PROMOTION_RELOCK_CLAIM_BOUNDARY.md"

QML_V8_CONTEXT_FINAL = 0.6444444444444445
QML_V8_CONTEXT_VALIDATION = 0.6055555555555555

DIRECTION_TARGETS = ["absolute_direction", "market_relative_vn30", "market_relative_vnindex", "top_quantile_forward_return"]
PRICE_TARGETS = ["forward_simple_return_h", "forward_log_return_h", "future_close_h", "market_excess_return_h", "volatility_adjusted_return_h"]
HORIZONS = [5, 10, 20, 40, 60]

DIRECTION_MODEL_NAMES = [
    "always_up",
    "always_down",
    "lag1_direction",
    "random_same_class_balance",
    "simple_momentum",
    "simple_relative_strength",
    "logistic_regression",
    "calibrated_logistic",
    "linear_svm",
    "rbf_svm",
    "knn_classifier",
    "naive_bayes",
    "random_forest",
    "extra_trees",
    "gradient_boosting",
    "hist_gradient_boosting",
    "xgboost_classifier",
    "lightgbm_classifier",
    "catboost_classifier",
    "mlp_classifier",
    "qml_v8_kernel_features_l2",
    "qml_v8_kernel_features_lightgbm",
    "soft_voting",
    "stacking_logistic",
    "rank_averaging",
    "regime_aware_ensemble",
    "direction_price_joint_ensemble",
]

PRICE_MODEL_NAMES = [
    "random_walk_price",
    "last_price",
    "zero_return",
    "historical_mean_return",
    "rolling_mean_return",
    "simple_momentum",
    "simple_relative_strength",
    "linear_regression",
    "ridge",
    "lasso",
    "elasticnet",
    "svr_linear",
    "svr_rbf",
    "knn_regressor",
    "random_forest_regressor",
    "extra_trees_regressor",
    "gradient_boosting_regressor",
    "hist_gradient_boosting_regressor",
    "xgboost_regressor",
    "lightgbm_regressor",
    "catboost_regressor",
    "mlp_regressor",
]

OPTIONAL_SEQUENCE_MODELS = [
    "ARIMA",
    "SARIMAX",
    "ETS",
    "GARCH-assisted",
    "Kalman/local level",
    "LSTM",
    "GRU",
    "BiLSTM",
    "TCN",
    "1D CNN",
    "Transformer encoder",
    "N-BEATS/N-HiTS",
    "TFT",
    "V4 pure quantum kernel replay",
    "PennyLane hybrid QNN",
]


@dataclass
class RunConfig:
    mode: str
    timeout_seconds: int
    max_train_rows: int
    max_validation_rows: int
    max_final_rows: int
    max_direction_candidates: int
    max_price_candidates: int


def dependency_status() -> dict[str, Any]:
    packages = [
        "sklearn",
        "numpy",
        "pandas",
        "statsmodels",
        "arch",
        "xgboost",
        "lightgbm",
        "catboost",
        "torch",
        "qiskit",
        "qiskit_machine_learning",
        "pennylane",
    ]
    status: dict[str, Any] = {}
    for name in packages:
        available = importlib.util.find_spec(name) is not None
        version = ""
        if available:
            try:
                version = str(getattr(importlib.import_module(name), "__version__", "unknown"))
            except Exception:
                version = "unknown"
        status[f"{name}_available"] = bool(available)
        status[f"{name}_version"] = version
    status["diagnostic_only"] = True
    status["no_trading_claim"] = True
    return status


def pct(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:+.2f} pp"


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(math.sqrt(mean_squared_error(y_true, y_pred))) if len(y_true) else math.nan


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    valid = denom > 1e-12
    if not valid.any():
        return math.nan
    return float(np.mean(2.0 * np.abs(y_pred[valid] - y_true[valid]) / denom[valid]))


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    valid = np.abs(y_true) > 1e-12
    if not valid.any():
        return math.nan
    return float(np.mean(np.abs((y_true[valid] - y_pred[valid]) / y_true[valid])))


def mase(y_true: np.ndarray, y_pred: np.ndarray, naive_pred: np.ndarray) -> float:
    denom = float(np.mean(np.abs(y_true - naive_pred))) if len(y_true) else math.nan
    if not math.isfinite(denom) or denom <= 1e-12:
        return math.nan
    return float(np.mean(np.abs(y_true - y_pred)) / denom)


def calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    if len(y_true) == 0 or np.isnan(prob).all():
        return math.nan
    frame = pd.DataFrame({"y": y_true, "prob": prob}).replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        return math.nan
    frame["bin"] = pd.cut(frame["prob"], bins=np.linspace(0.0, 1.0, bins + 1), include_lowest=True)
    total = len(frame)
    err = 0.0
    for _bin, group in frame.groupby("bin", observed=False):
        if len(group):
            err += len(group) / total * abs(float(group["y"].mean()) - float(group["prob"].mean()))
    return float(err)


def safe_auc(y_true: np.ndarray, prob: np.ndarray) -> float:
    try:
        if len(np.unique(y_true)) < 2:
            return math.nan
        return float(roc_auc_score(y_true, prob))
    except Exception:
        return math.nan


def target_timestamp_series(target: pd.Series) -> pd.Series:
    value = target.attrs.get("target_timestamp")
    if isinstance(value, pd.Series):
        return pd.to_datetime(value.reindex(target.index), errors="coerce")
    return pd.Series(pd.NaT, index=target.index)


def make_index_return(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], code: str, target_timestamp: pd.Series) -> pd.Series:
    if code not in index_data:
        return pd.Series(np.nan, index=features.index, dtype=float)
    idx_frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
    idx_frame["datetime"] = pd.to_datetime(idx_frame["datetime"], errors="coerce")
    idx_frame["close"] = pd.to_numeric(idx_frame["close"], errors="coerce")
    idx_frame = idx_frame.dropna(subset=["datetime", "close"])
    left = features[["datetime"]].copy()
    left["row_index"] = features.index
    left["target_timestamp"] = target_timestamp.to_numpy()
    start = pd.merge_asof(left.sort_values("datetime"), idx_frame.rename(columns={"close": "start_close"}), on="datetime", direction="backward")
    end_left = left[["row_index", "target_timestamp"]].dropna().rename(columns={"target_timestamp": "datetime"}).sort_values("datetime")
    end = pd.merge_asof(end_left, idx_frame.rename(columns={"close": "target_close"}), on="datetime", direction="backward")
    start_close = pd.Series(start.set_index("row_index")["start_close"]).reindex(features.index)
    end_close = pd.Series(end.set_index("row_index")["target_close"]).reindex(features.index)
    return end_close / start_close.replace(0.0, np.nan) - 1.0


def build_direction_target(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], target_variant: str, horizon: int) -> pd.Series:
    if target_variant in {"absolute_direction", "market_relative_vn30", "market_relative_vnindex"}:
        return build_labels(features, index_data, target_variant, horizon)
    stock_return, target_timestamp = stock_future_returns(features, horizon)
    labels = pd.Series(np.nan, index=features.index, dtype=float)
    if target_variant == "top_quantile_forward_return":
        frame = pd.DataFrame({"datetime": features["datetime"], "forward_return": stock_return})
        ranks = frame.groupby("datetime")["forward_return"].rank(pct=True, method="average")
        valid = stock_return.notna() & target_timestamp.notna()
        labels.loc[valid] = (ranks.loc[valid] >= 0.80).astype(float)
    else:
        raise ValueError(f"unknown direction target {target_variant}")
    labels.attrs["target_timestamp"] = target_timestamp
    labels.attrs["target_variant"] = target_variant
    labels.attrs["horizon"] = int(horizon)
    labels.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    labels.attrs["top_quantile"] = 0.20
    return labels


def build_price_target(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], target_variant: str, horizon: int) -> pd.Series:
    simple_return, target_timestamp = stock_future_returns(features, horizon)
    target = pd.Series(np.nan, index=features.index, dtype=float)
    if target_variant == "forward_simple_return_h":
        target = simple_return.astype(float)
    elif target_variant == "forward_log_return_h":
        target = np.log1p(simple_return.astype(float))
    elif target_variant == "future_close_h":
        future_close_parts: list[pd.Series] = []
        for _ticker, group in features.sort_values(["ticker", "datetime"]).groupby("ticker", sort=True):
            future_close_parts.append(pd.Series(group["close"].shift(-horizon).to_numpy(dtype=float), index=group.index))
        target = pd.concat(future_close_parts).sort_index() if future_close_parts else target
    elif target_variant == "market_excess_return_h":
        vn30_return = make_index_return(features, index_data, "VN30", target_timestamp)
        target = simple_return - vn30_return
    elif target_variant == "volatility_adjusted_return_h":
        vol_cols = [col for col in ["rolling_return_vol_20", "return_1_vol_20", "volatility_20", "v3_relative_strength_vn30_vol20_lag"] if col in features.columns]
        if vol_cols:
            vol = pd.to_numeric(features[vol_cols[0]], errors="coerce").abs().replace(0.0, np.nan)
        else:
            vol_parts: list[pd.Series] = []
            for _ticker, group in features.sort_values(["ticker", "datetime"]).groupby("ticker", sort=True):
                ret = pd.to_numeric(group["close"], errors="coerce").pct_change(fill_method=None)
                vol_parts.append(pd.Series(ret.rolling(20, min_periods=5).std().to_numpy(dtype=float), index=group.index))
            vol = pd.concat(vol_parts).sort_index() if vol_parts else pd.Series(np.nan, index=features.index)
        target = simple_return / vol.replace(0.0, np.nan)
    else:
        raise ValueError(f"unknown price target {target_variant}")
    target = pd.Series(target, index=features.index, dtype=float).replace([np.inf, -np.inf], np.nan)
    target.attrs["target_timestamp"] = target_timestamp
    target.attrs["target_variant"] = target_variant
    target.attrs["horizon"] = int(horizon)
    target.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    return target


def strict_split_for_target(features: pd.DataFrame, target: pd.Series) -> dict[str, pd.Index]:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_series(target)
    valid = target.notna() & timestamps.notna() & target_timestamp.notna()
    train = timestamps.le(TRAIN_END) & target_timestamp.le(TRAIN_END) & valid
    validation = timestamps.between(VAL_START, VAL_END) & target_timestamp.between(VAL_START, VAL_END) & valid
    final = timestamps.ge(FINAL_START) & target_timestamp.ge(FINAL_START) & valid
    return {"train": features.index[train], "validation": features.index[validation], "final": features.index[final]}


def leakage_guard_for_target(features: pd.DataFrame, target: pd.Series, splits: dict[str, pd.Index]) -> bool:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_series(target)
    train = splits["train"]
    validation = splits["validation"]
    final = splits["final"]
    if len(train) and ((timestamps.loc[train] > TRAIN_END).any() or (target_timestamp.loc[train] > TRAIN_END).any()):
        return False
    if len(validation) and ((timestamps.loc[validation] < VAL_START).any() or (timestamps.loc[validation] > VAL_END).any() or (target_timestamp.loc[validation] < VAL_START).any() or (target_timestamp.loc[validation] > VAL_END).any()):
        return False
    if len(final) and ((timestamps.loc[final] < FINAL_START).any() or (target_timestamp.loc[final] < FINAL_START).any()):
        return False
    return True


def split_sample(features: pd.DataFrame, y: pd.Series, splits: dict[str, pd.Index], config: RunConfig, classification: bool) -> dict[str, pd.Index]:
    def ordered_tail(index: pd.Index, limit: int) -> pd.Index:
        ordered = ordered_index(features, index)
        return ordered if len(ordered) <= limit else pd.Index(list(ordered)[-limit:])

    if classification:
        train_ordered = ordered_index(features, splits["train"])
        if len(train_ordered) > config.max_train_rows:
            frame = pd.DataFrame({"idx": train_ordered, "label": y.loc[train_ordered].astype(int).to_numpy(), "datetime": features.loc[train_ordered, "datetime"].to_numpy()})
            pieces = []
            per_class = max(1, config.max_train_rows // max(1, frame["label"].nunique()))
            for _label, group in frame.groupby("label", sort=True):
                pieces.append(group.tail(per_class))
            train = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "idx"])
            train_idx = pd.Index(train["idx"].tolist())
        else:
            train_idx = train_ordered
    else:
        train_idx = ordered_tail(splits["train"], config.max_train_rows)
    return {
        "train": train_idx,
        "validation": ordered_tail(splits["validation"], config.max_validation_rows),
        "final": ordered_tail(splits["final"], config.max_final_rows),
    }


def build_feature_groups(features: pd.DataFrame, family_cols: dict[str, list[str]], relative_cols: list[str]) -> dict[str, list[str]]:
    source_groups = build_source_groups(features, family_cols)
    volume_volatility = numeric_existing(features, [col for col in features.columns if any(token in col.lower() for token in ["volume", "vol", "shock", "atr", "rsi"])])
    stock_lag_momentum = numeric_existing(features, [col for col in features.columns if any(token in col.lower() for token in ["lag", "momentum", "return", "sma", "ema", "macd"]) and not col.lower().startswith(("vnindex", "vn30"))])
    all_safe = numeric_existing(features, sorted(set(source_groups["combined_strategy_features"]).union(relative_cols).union(source_groups["market_context_features"]).union(volume_volatility).union(stock_lag_momentum)))
    groups = {
        "stock_lag_momentum": stock_lag_momentum or source_groups["compact_stable_features"],
        "relative_strength": numeric_existing(features, relative_cols) or source_groups["relative_strength_features"],
        "market_context": source_groups["market_context_features"],
        "volume_volatility": volume_volatility or source_groups["compact_stable_features"],
        "combined_strategy_features": source_groups["combined_strategy_features"],
        "compact_stable_features": source_groups["compact_stable_features"],
        "all_safe_features": all_safe,
    }
    qml_audit = OUTPUT_DIR.parent / "vn30_qml_forecasting" / "qml_v8_kernel_feature_audit.csv"
    if qml_audit.exists():
        groups["qml_kernel_features"] = []
    return groups


def fit_matrix(features: pd.DataFrame, splits: dict[str, pd.Index], feature_columns: list[str], max_features: int = 16) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    ranked = []
    if feature_columns:
        train = features.loc[splits["train"], feature_columns].replace([np.inf, -np.inf], np.nan)
        availability = train.notna().mean()
        variance = train.var(numeric_only=True).fillna(0.0)
        ranked = sorted([col for col in feature_columns if availability.get(col, 0.0) > 0.0], key=lambda col: (-float(availability.get(col, 0.0)), -float(variance.get(col, 0.0)), col))
    selected = ranked[:max_features]
    if not selected:
        return pd.DataFrame(index=splits["train"]), pd.DataFrame(index=splits["validation"]), pd.DataFrame(index=splits["final"]), [], "no_features"
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    x_train = pipe.fit_transform(features.loc[splits["train"], selected].replace([np.inf, -np.inf], np.nan))
    x_val = pipe.transform(features.loc[splits["validation"], selected].replace([np.inf, -np.inf], np.nan))
    x_final = pipe.transform(features.loc[splits["final"], selected].replace([np.inf, -np.inf], np.nan))
    return (
        pd.DataFrame(x_train, index=splits["train"], columns=selected),
        pd.DataFrame(x_val, index=splits["validation"], columns=selected),
        pd.DataFrame(x_final, index=splits["final"], columns=selected),
        selected,
        "ok",
    )


def direction_model(model_name: str) -> tuple[Any | None, str]:
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, class_weight="balanced", random_state=SEED), ""
    if model_name == "calibrated_logistic":
        base = LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, class_weight="balanced", random_state=SEED)
        try:
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3), ""
        except TypeError:  # pragma: no cover
            return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3), ""
    if model_name == "linear_svm":
        return SVC(kernel="linear", C=0.5, class_weight="balanced", probability=True, random_state=SEED), ""
    if model_name == "rbf_svm":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True, random_state=SEED), ""
    if model_name == "knn_classifier":
        return KNeighborsClassifier(n_neighbors=15, weights="distance"), ""
    if model_name == "naive_bayes":
        return GaussianNB(), ""
    if model_name == "random_forest":
        return RandomForestClassifier(n_estimators=80, max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=SEED, n_jobs=2), ""
    if model_name == "extra_trees":
        return ExtraTreesClassifier(n_estimators=80, max_depth=6, min_samples_leaf=15, class_weight="balanced", random_state=SEED, n_jobs=2), ""
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=SEED), ""
    if model_name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED), ""
    if model_name == "xgboost_classifier":
        if importlib.util.find_spec("xgboost") is None:
            return None, "xgboost unavailable"
        xgb = importlib.import_module("xgboost")
        return xgb.XGBClassifier(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=SEED, n_jobs=2), ""
    if model_name == "lightgbm_classifier":
        if importlib.util.find_spec("lightgbm") is None:
            return None, "lightgbm unavailable"
        lgb = importlib.import_module("lightgbm")
        return lgb.LGBMClassifier(n_estimators=80, max_depth=4, learning_rate=0.05, num_leaves=15, min_child_samples=30, class_weight="balanced", random_state=SEED, verbosity=-1, n_jobs=2), ""
    if model_name == "catboost_classifier":
        if importlib.util.find_spec("catboost") is None:
            return None, "catboost unavailable"
        cb = importlib.import_module("catboost")
        return cb.CatBoostClassifier(iterations=80, depth=4, learning_rate=0.05, loss_function="Logloss", verbose=False, random_seed=SEED), ""
    if model_name == "mlp_classifier":
        return MLPClassifier(hidden_layer_sizes=(24,), alpha=0.001, max_iter=120, random_state=SEED, early_stopping=True), ""
    return None, "model handled as baseline, ensemble, QML, or skipped optional family"


def price_model(model_name: str) -> tuple[Any | None, str]:
    if model_name == "linear_regression":
        return LinearRegression(), ""
    if model_name == "ridge":
        return Ridge(alpha=2.0, random_state=SEED), ""
    if model_name == "lasso":
        return Lasso(alpha=0.0005, max_iter=2000, random_state=SEED), ""
    if model_name == "elasticnet":
        return ElasticNet(alpha=0.0005, l1_ratio=0.3, max_iter=2000, random_state=SEED), ""
    if model_name == "svr_linear":
        return SVR(kernel="linear", C=0.5, epsilon=0.01), ""
    if model_name == "svr_rbf":
        return SVR(kernel="rbf", C=1.0, epsilon=0.01, gamma="scale"), ""
    if model_name == "knn_regressor":
        return KNeighborsRegressor(n_neighbors=15, weights="distance"), ""
    if model_name == "random_forest_regressor":
        return RandomForestRegressor(n_estimators=80, max_depth=6, min_samples_leaf=15, random_state=SEED, n_jobs=2), ""
    if model_name == "extra_trees_regressor":
        return ExtraTreesRegressor(n_estimators=80, max_depth=6, min_samples_leaf=15, random_state=SEED, n_jobs=2), ""
    if model_name == "gradient_boosting_regressor":
        return GradientBoostingRegressor(n_estimators=60, max_depth=3, learning_rate=0.05, random_state=SEED), ""
    if model_name == "hist_gradient_boosting_regressor":
        return HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15, learning_rate=0.05, random_state=SEED), ""
    if model_name == "xgboost_regressor":
        if importlib.util.find_spec("xgboost") is None:
            return None, "xgboost unavailable"
        xgb = importlib.import_module("xgboost")
        return xgb.XGBRegressor(n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=2), ""
    if model_name == "lightgbm_regressor":
        if importlib.util.find_spec("lightgbm") is None:
            return None, "lightgbm unavailable"
        lgb = importlib.import_module("lightgbm")
        return lgb.LGBMRegressor(n_estimators=80, max_depth=4, learning_rate=0.05, num_leaves=15, min_child_samples=30, random_state=SEED, verbosity=-1, n_jobs=2), ""
    if model_name == "catboost_regressor":
        if importlib.util.find_spec("catboost") is None:
            return None, "catboost unavailable"
        cb = importlib.import_module("catboost")
        return cb.CatBoostRegressor(iterations=80, depth=4, learning_rate=0.05, loss_function="RMSE", verbose=False, random_seed=SEED), ""
    if model_name == "mlp_regressor":
        return MLPRegressor(hidden_layer_sizes=(24,), alpha=0.001, max_iter=150, random_state=SEED, early_stopping=True), ""
    return None, "model handled as baseline or skipped optional family"


def baseline_direction_prediction(model_name: str, features: pd.DataFrame, y_train: pd.Series, idx: pd.Index) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    if model_name == "always_up":
        pred = np.ones(len(idx), dtype=int)
        return pred, pred.astype(float), ""
    if model_name == "always_down":
        pred = np.zeros(len(idx), dtype=int)
        return pred, pred.astype(float), ""
    if model_name == "random_same_class_balance":
        p = float(y_train.astype(int).mean()) if len(y_train) else 0.5
        rng = np.random.default_rng(SEED + len(idx))
        prob = np.full(len(idx), p, dtype=float)
        return (rng.random(len(idx)) < p).astype(int), prob, ""
    if model_name == "lag1_direction":
        for col in ["return_1_lag_1", "lag_ret_1"]:
            if col in features.columns:
                values = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                pred = (values > 0.0).astype(int)
                return pred, np.clip((values > 0.0).astype(float), 0.0, 1.0), ""
        return None, None, "lag1 return feature unavailable"
    if model_name == "simple_momentum":
        cols = [col for col in ["momentum_20", "rolling_return_mean_20", "return_1_lag_1"] if col in features.columns]
        if not cols:
            return None, None, "momentum feature unavailable"
        values = pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = (values > 0.0).astype(int)
        return pred, 1.0 / (1.0 + np.exp(-np.clip(values, -10.0, 10.0))), ""
    if model_name == "simple_relative_strength":
        cols = [col for col in features.columns if "relative" in col.lower()]
        if not cols:
            return None, None, "relative-strength feature unavailable"
        values = pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        pred = (values > 0.0).astype(int)
        return pred, 1.0 / (1.0 + np.exp(-np.clip(values, -10.0, 10.0))), ""
    return None, None, "not a direction baseline"


def baseline_price_prediction(model_name: str, features: pd.DataFrame, y_train: pd.Series, idx: pd.Index, target_variant: str, horizon: int) -> tuple[np.ndarray | None, str]:
    close = pd.to_numeric(features.loc[idx, "close"], errors="coerce").ffill().bfill().to_numpy(dtype=float)
    train_mean = float(pd.to_numeric(y_train, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()) if len(y_train) else 0.0
    if target_variant == "future_close_h":
        if model_name in {"random_walk_price", "last_price"}:
            return close, ""
        if model_name == "historical_mean_return":
            return close * (1.0 + train_mean), ""
        if model_name == "rolling_mean_return":
            col = "rolling_return_mean_20" if "rolling_return_mean_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                ret = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return close * (1.0 + ret), ""
        if model_name == "simple_momentum":
            col = "momentum_20" if "momentum_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                ret = pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return close * (1.0 + np.clip(ret, -0.2, 0.2)), ""
        if model_name == "simple_relative_strength":
            cols = [col for col in features.columns if "relative" in col.lower()]
            if cols:
                ret = pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
                return close * (1.0 + np.clip(ret, -0.2, 0.2)), ""
    else:
        if model_name == "zero_return":
            return np.zeros(len(idx), dtype=float), ""
        if model_name in {"historical_mean_return", "random_walk_price", "last_price"}:
            return np.full(len(idx), train_mean if math.isfinite(train_mean) else 0.0, dtype=float), ""
        if model_name == "rolling_mean_return":
            col = "rolling_return_mean_20" if "rolling_return_mean_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                return pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float), ""
        if model_name == "simple_momentum":
            col = "momentum_20" if "momentum_20" in features.columns else "return_1_lag_1"
            if col in features.columns:
                return pd.to_numeric(features.loc[idx, col], errors="coerce").fillna(0.0).to_numpy(dtype=float), ""
        if model_name == "simple_relative_strength":
            cols = [col for col in features.columns if "relative" in col.lower()]
            if cols:
                return pd.to_numeric(features.loc[idx, cols[0]], errors="coerce").fillna(0.0).to_numpy(dtype=float), ""
    return None, f"{model_name} baseline not applicable for {target_variant}"


def direction_metrics(y_true: pd.Series, pred: np.ndarray, prob: np.ndarray | None) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    p = np.asarray(pred, dtype=int)
    prob_arr = np.asarray(prob, dtype=float) if prob is not None else p.astype(float)
    return {
        "accuracy": float((y == p).mean()) if len(y) else math.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, p)) if len(np.unique(y)) > 1 else math.nan,
        "f1": float(f1_score(y, p, zero_division=0)),
        "precision": float(precision_score(y, p, zero_division=0)),
        "recall": float(recall_score(y, p, zero_division=0)),
        "roc_auc": safe_auc(y, prob_arr),
        "brier_score": float(brier_score_loss(y, np.clip(prob_arr, 0.0, 1.0))) if len(y) and len(np.unique(y)) > 1 else math.nan,
        "calibration_error": calibration_error(y, np.clip(prob_arr, 0.0, 1.0)),
    }


def price_metrics(y_true: pd.Series, pred: np.ndarray, naive_pred: np.ndarray, close: np.ndarray, target_variant: str) -> dict[str, float]:
    y = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    pred = np.asarray(pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    if not valid.any():
        return {key: math.nan for key in ["mae", "rmse", "mape", "smape", "mase", "r2", "sign_accuracy", "directional_hit_from_predicted_return", "correlation_pred_actual", "quantile_loss", "interval_coverage"]}
    yv = y[valid]
    pv = pred[valid]
    if target_variant == "future_close_h":
        actual_ret = yv / np.clip(close[valid], 1e-12, None) - 1.0
        pred_ret = pv / np.clip(close[valid], 1e-12, None) - 1.0
    else:
        actual_ret = yv
        pred_ret = pv
    corr = float(np.corrcoef(pv, yv)[0, 1]) if len(yv) > 2 and np.std(pv) > 0 and np.std(yv) > 0 else math.nan
    q = 0.50
    qloss = float(np.mean(np.maximum(q * (yv - pv), (q - 1.0) * (yv - pv))))
    return {
        "mae": float(mean_absolute_error(yv, pv)),
        "rmse": rmse(yv, pv),
        "mape": mape(yv, pv),
        "smape": smape(yv, pv),
        "mase": mase(yv, pv, naive_pred[valid]),
        "r2": float(r2_score(yv, pv)) if len(yv) > 1 else math.nan,
        "sign_accuracy": float((np.sign(actual_ret) == np.sign(pred_ret)).mean()) if len(yv) else math.nan,
        "directional_hit_from_predicted_return": float((np.sign(actual_ret) == np.sign(pred_ret)).mean()) if len(yv) else math.nan,
        "correlation_pred_actual": corr,
        "quantile_loss": qloss,
        "interval_coverage": math.nan,
    }


def stability_rows(task: str, candidate: dict[str, Any], features: pd.DataFrame, idx: pd.Index, y_true: pd.Series, pred: np.ndarray) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    frame = pd.DataFrame({"ticker": features.loc[idx, "ticker"].astype(str).to_numpy(), "datetime": features.loc[idx, "datetime"].to_numpy(), "y": y_true.loc[idx].to_numpy(), "pred": pred}, index=idx)
    if task == "direction":
        frame["score"] = (frame["y"].astype(int) == frame["pred"].astype(int)).astype(float)
        metric_name = "accuracy"
    else:
        frame["score"] = np.abs(pd.to_numeric(frame["y"], errors="coerce") - pd.to_numeric(frame["pred"], errors="coerce"))
        metric_name = "mae"
    frame["quarter"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    ticker_rows = []
    for ticker, group in frame.groupby("ticker", sort=True):
        ticker_rows.append({**candidate, "task": task, "group": ticker, "metric": metric_name, "value": float(group["score"].mean()), "rows": int(len(group))})
    quarter_rows = []
    for quarter, group in frame.groupby("quarter", sort=True):
        quarter_rows.append({**candidate, "task": task, "group": quarter, "metric": metric_name, "value": float(group["score"].mean()), "rows": int(len(group))})
    return ticker_rows, quarter_rows


def qml_feature_direction_candidate(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    model_name: str,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None, str]:
    if importlib.util.find_spec("qiskit_machine_learning") is None:
        return None, None, None, None, "qiskit_machine_learning unavailable"
    if labels.attrs.get("target_variant") != "market_relative_vn30" or int(labels.attrs.get("horizon", 0)) != 40:
        return None, None, None, None, "QML V8 architecture scoped to market_relative_vn30 h40"
    try:
        source_groups = build_source_groups(features, family_cols)
        source_groups["relative_strength_features"] = relative_cols
        source_groups["relative_plus_market_context_features"] = sorted(set(relative_cols).union(source_groups.get("market_context_features", [])[:12]))
        spec, _audit = fit_feature_spec(features, labels, "market_relative_vn30", 40, "relative_plus_market_context_features", source_groups["relative_plus_market_context_features"], "topk_availability", 4, splits)
        if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
            return None, None, None, None, spec.selection_status
        x_train, x_val, x_final, scaling_info = v6_scaling_transform(spec, "minmax_0_pi")
        if scaling_info["status"] != "ok":
            return None, None, None, None, scaling_info["skipped_reason"]
        k_train, k_val, k_final = v6_quantum_kernel_matrices(x_train, x_val, x_final, 2, "full")
        train_meta, val_meta, final_meta = v8_kernel_feature_frames(k_train, k_val, k_final, labels, splits)
        rel_spec, _ = fit_feature_spec(features, labels, "market_relative_vn30", 40, "relative_strength_features", source_groups["relative_strength_features"], "topk_availability", 4, splits)
        market_spec, _ = fit_feature_spec(features, labels, "market_relative_vn30", 40, "market_context_features", source_groups["market_context_features"], "topk_availability", 4, splits)
        train_x = pd.concat([train_meta, rel_spec.x_train.add_prefix("relative__"), market_spec.x_train.add_prefix("market__")], axis=1).fillna(0.0)
        val_x = pd.concat([val_meta, rel_spec.x_validation.add_prefix("relative__"), market_spec.x_validation.add_prefix("market__")], axis=1).fillna(0.0)
        final_x = pd.concat([final_meta, rel_spec.x_final.add_prefix("relative__"), market_spec.x_final.add_prefix("market__")], axis=1).fillna(0.0)
        if model_name == "qml_v8_kernel_features_lightgbm":
            model, reason = direction_model("lightgbm_classifier")
            if model is None:
                return None, None, None, None, reason
        else:
            model = LogisticRegression(max_iter=1000, solver="liblinear", C=0.5, class_weight="balanced", random_state=SEED)
        train_y = labels.loc[splits["train"]].astype(int)
        model.fit(train_x, train_y)
        val_prob = np.asarray(model.predict_proba(val_x)[:, 1], dtype=float) if hasattr(model, "predict_proba") else np.asarray(model.predict(val_x), dtype=float)
        final_prob = np.asarray(model.predict_proba(final_x)[:, 1], dtype=float) if hasattr(model, "predict_proba") else np.asarray(model.predict(final_x), dtype=float)
        threshold = float(np.nanquantile(val_prob, max(0.0, min(1.0, 1.0 - float(labels.loc[splits["validation"]].mean())))))
        return (val_prob >= threshold).astype(int), val_prob, (final_prob >= threshold).astype(int), final_prob, ""
    except Exception as exc:
        return None, None, None, None, f"{type(exc).__name__}: {exc}"


def evaluate_direction_candidate(
    model_name: str,
    feature_group: str,
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    relative_cols: list[str],
    index_data: dict[str, pd.DataFrame],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    feature_columns: list[str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    target_variant = str(labels.attrs.get("target_variant", ""))
    horizon = int(labels.attrs.get("horizon", 0))
    row = {
        "candidate_id": candidate_id("direction", model_name, target_variant, f"h{horizon}", feature_group),
        "task": "direction",
        "model_family": model_name,
        "feature_group": feature_group,
        "target_variant": target_variant,
        "horizon": horizon,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_accuracy": math.nan,
        "final_accuracy": math.nan,
        "lift_over_strongest_baseline": math.nan,
        "balanced_accuracy": math.nan,
        "f1": math.nan,
        "precision": math.nan,
        "recall": math.nan,
        "roc_auc": math.nan,
        "brier_score": math.nan,
        "calibration_error": math.nan,
        "ticker_stability": math.nan,
        "quarter_stability": math.nan,
        "claim_label": "not_claimable",
        "runtime_seconds": 0.0,
        "status": "pending",
        "skipped_reason": "",
    }
    final_pred: np.ndarray | None = None
    try:
        train_y = labels.loc[splits["train"]].astype(int)
        val_y = labels.loc[splits["validation"]].astype(int)
        final_y = labels.loc[splits["final"]].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("train split has fewer than two classes")
        baseline_candidates = ["always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"]
        if model_name in baseline_candidates:
            val_pred, val_prob, reason = baseline_direction_prediction(model_name, features, train_y, splits["validation"])
            fin_pred, fin_prob, reason2 = baseline_direction_prediction(model_name, features, train_y, splits["final"])
            if val_pred is None or fin_pred is None:
                raise ValueError(reason or reason2)
        elif model_name.startswith("qml_v8"):
            val_pred, val_prob, fin_pred, fin_prob, reason = qml_feature_direction_candidate(features, family_cols, relative_cols, index_data, labels, splits, model_name)
            if val_pred is None or fin_pred is None:
                raise ValueError(reason)
        else:
            x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_columns, 16)
            if status != "ok":
                raise ValueError(status)
            model, reason = direction_model(model_name)
            if model is None:
                raise ValueError(reason)
            model.fit(x_train, train_y)
            val_pred = np.asarray(model.predict(x_val)).astype(int)
            fin_pred = np.asarray(model.predict(x_final)).astype(int)
            if hasattr(model, "predict_proba"):
                val_prob = np.asarray(model.predict_proba(x_val)[:, 1], dtype=float)
                fin_prob = np.asarray(model.predict_proba(x_final)[:, 1], dtype=float)
            elif hasattr(model, "decision_function"):
                val_score = np.asarray(model.decision_function(x_val), dtype=float)
                fin_score = np.asarray(model.decision_function(x_final), dtype=float)
                val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_score, -30.0, 30.0)))
                fin_prob = 1.0 / (1.0 + np.exp(-np.clip(fin_score, -30.0, 30.0)))
            else:
                val_prob = val_pred.astype(float)
                fin_prob = fin_pred.astype(float)
        val_metrics = direction_metrics(val_y, val_pred, val_prob)
        final_metrics = direction_metrics(final_y, fin_pred, fin_prob)
        baseline_accs = []
        for baseline in ["always_up", "always_down", "lag1_direction", "simple_momentum", "simple_relative_strength"]:
            pred, _prob, _reason = baseline_direction_prediction(baseline, features, train_y, splits["validation"])
            if pred is not None:
                baseline_accs.append(float((val_y.to_numpy(dtype=int) == pred.astype(int)).mean()))
        strongest_baseline = max(baseline_accs) if baseline_accs else math.nan
        stability_ticker, stability_quarter = stability_rows("direction", {"candidate_id": row["candidate_id"], "target_variant": target_variant, "horizon": horizon}, features, splits["validation"], labels, val_pred)
        ticker_values = [item["value"] for item in stability_ticker]
        quarter_values = [item["value"] for item in stability_quarter]
        row.update(
            {
                "validation_accuracy": val_metrics["accuracy"],
                "final_accuracy": final_metrics["accuracy"],
                "lift_over_strongest_baseline": val_metrics["accuracy"] - strongest_baseline if math.isfinite(strongest_baseline) else math.nan,
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "f1": val_metrics["f1"],
                "precision": val_metrics["precision"],
                "recall": val_metrics["recall"],
                "roc_auc": val_metrics["roc_auc"],
                "brier_score": val_metrics["brier_score"],
                "calibration_error": val_metrics["calibration_error"],
                "ticker_stability": float(1.0 - np.nanstd(ticker_values)) if ticker_values else math.nan,
                "quarter_stability": float(1.0 - np.nanstd(quarter_values)) if quarter_values else math.nan,
                "claim_label": "diagnostic_only",
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        if target_variant == "absolute_direction" and horizon == 40 and row["final_accuracy"] >= 0.60 and row["lift_over_strongest_baseline"] > 0:
            row["claim_label"] = "baseline60_candidate"
        elif target_variant == "market_relative_vn30" and horizon == 40 and row["final_accuracy"] >= 0.60:
            row["claim_label"] = "market_relative_candidate"
        else:
            row["claim_label"] = "direction_candidate" if row["lift_over_strongest_baseline"] > 0 else "diagnostic_only"
        final_pred = fin_pred
    except Exception as exc:
        row.update({"runtime_seconds": time.perf_counter() - start, "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "claim_label": "not_claimable"})
    return row, final_pred


def evaluate_price_candidate(
    model_name: str,
    feature_group: str,
    features: pd.DataFrame,
    target: pd.Series,
    splits: dict[str, pd.Index],
    feature_columns: list[str],
) -> tuple[dict[str, Any], np.ndarray | None]:
    start = time.perf_counter()
    target_variant = str(target.attrs.get("target_variant", ""))
    horizon = int(target.attrs.get("horizon", 0))
    row = {
        "candidate_id": candidate_id("price", model_name, target_variant, f"h{horizon}", feature_group),
        "task": "price_return",
        "model_family": model_name,
        "feature_group": feature_group,
        "target_variant": target_variant,
        "horizon": horizon,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_mae": math.nan,
        "validation_rmse": math.nan,
        "validation_mape": math.nan,
        "validation_smape": math.nan,
        "validation_mase": math.nan,
        "validation_r2": math.nan,
        "validation_sign_accuracy": math.nan,
        "validation_directional_hit_from_predicted_return": math.nan,
        "validation_correlation_pred_actual": math.nan,
        "validation_quantile_loss": math.nan,
        "validation_interval_coverage": math.nan,
        "final_mae": math.nan,
        "final_rmse": math.nan,
        "final_mape": math.nan,
        "final_smape": math.nan,
        "final_mase": math.nan,
        "final_r2": math.nan,
        "final_sign_accuracy": math.nan,
        "final_directional_hit_from_predicted_return": math.nan,
        "final_correlation_pred_actual": math.nan,
        "error_improvement_over_baseline": math.nan,
        "claim_label": "not_claimable",
        "runtime_seconds": 0.0,
        "status": "pending",
        "skipped_reason": "",
    }
    final_pred: np.ndarray | None = None
    try:
        train_y = pd.to_numeric(target.loc[splits["train"]], errors="coerce")
        val_y = pd.to_numeric(target.loc[splits["validation"]], errors="coerce")
        final_y = pd.to_numeric(target.loc[splits["final"]], errors="coerce")
        baseline_models = ["random_walk_price", "last_price", "zero_return", "historical_mean_return", "rolling_mean_return", "simple_momentum", "simple_relative_strength"]
        if model_name in baseline_models:
            val_pred, reason = baseline_price_prediction(model_name, features, train_y, splits["validation"], target_variant, horizon)
            fin_pred, reason2 = baseline_price_prediction(model_name, features, train_y, splits["final"], target_variant, horizon)
            if val_pred is None or fin_pred is None:
                raise ValueError(reason or reason2)
        else:
            x_train, x_val, x_final, selected, status = fit_matrix(features, splits, feature_columns, 16)
            if status != "ok":
                raise ValueError(status)
            model, reason = price_model(model_name)
            if model is None:
                raise ValueError(reason)
            model.fit(x_train, train_y)
            val_pred = np.asarray(model.predict(x_val), dtype=float)
            fin_pred = np.asarray(model.predict(x_final), dtype=float)
        naive_val, _ = baseline_price_prediction("last_price" if target_variant == "future_close_h" else "historical_mean_return", features, train_y, splits["validation"], target_variant, horizon)
        naive_final, _ = baseline_price_prediction("last_price" if target_variant == "future_close_h" else "historical_mean_return", features, train_y, splits["final"], target_variant, horizon)
        if naive_val is None:
            naive_val = np.zeros(len(val_y), dtype=float)
        if naive_final is None:
            naive_final = np.zeros(len(final_y), dtype=float)
        val_close = pd.to_numeric(features.loc[splits["validation"], "close"], errors="coerce").to_numpy(dtype=float)
        final_close = pd.to_numeric(features.loc[splits["final"], "close"], errors="coerce").to_numpy(dtype=float)
        val_metrics = price_metrics(val_y, val_pred, naive_val, val_close, target_variant)
        final_metrics = price_metrics(final_y, fin_pred, naive_final, final_close, target_variant)
        baseline_rmse = rmse(val_y.to_numpy(dtype=float), naive_val)
        row.update(
            {
                "validation_mae": val_metrics["mae"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mape": val_metrics["mape"],
                "validation_smape": val_metrics["smape"],
                "validation_mase": val_metrics["mase"],
                "validation_r2": val_metrics["r2"],
                "validation_sign_accuracy": val_metrics["sign_accuracy"],
                "validation_directional_hit_from_predicted_return": val_metrics["directional_hit_from_predicted_return"],
                "validation_correlation_pred_actual": val_metrics["correlation_pred_actual"],
                "validation_quantile_loss": val_metrics["quantile_loss"],
                "validation_interval_coverage": val_metrics["interval_coverage"],
                "final_mae": final_metrics["mae"],
                "final_rmse": final_metrics["rmse"],
                "final_mape": final_metrics["mape"],
                "final_smape": final_metrics["smape"],
                "final_mase": final_metrics["mase"],
                "final_r2": final_metrics["r2"],
                "final_sign_accuracy": final_metrics["sign_accuracy"],
                "final_directional_hit_from_predicted_return": final_metrics["directional_hit_from_predicted_return"],
                "final_correlation_pred_actual": final_metrics["correlation_pred_actual"],
                "error_improvement_over_baseline": (baseline_rmse - val_metrics["rmse"]) / baseline_rmse if math.isfinite(baseline_rmse) and baseline_rmse > 0 and math.isfinite(val_metrics["rmse"]) else math.nan,
                "claim_label": "price_return_candidate" if math.isfinite(val_metrics["rmse"]) and math.isfinite(baseline_rmse) and val_metrics["rmse"] < baseline_rmse else "diagnostic_only",
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        final_pred = fin_pred
    except Exception as exc:
        row.update({"runtime_seconds": time.perf_counter() - start, "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "claim_label": "not_claimable"})
    return row, final_pred


def model_family_registry(dependency: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name in DIRECTION_MODEL_NAMES:
        family = "direction"
        available = True
        reason = ""
        if name.startswith("xgboost") and not dependency.get("xgboost_available"):
            available, reason = False, "xgboost unavailable"
        if name.startswith("lightgbm") and not dependency.get("lightgbm_available"):
            available, reason = False, "lightgbm unavailable"
        if name.startswith("catboost") and not dependency.get("catboost_available"):
            available, reason = False, "catboost unavailable"
        if name.startswith("qml") and not dependency.get("qiskit_machine_learning_available"):
            available, reason = False, "qiskit_machine_learning unavailable"
        rows.append({"task": family, "model_family": name, "available": available, "reason": reason, "stage": "screening"})
    for name in PRICE_MODEL_NAMES:
        available = True
        reason = ""
        if name.startswith("xgboost") and not dependency.get("xgboost_available"):
            available, reason = False, "xgboost unavailable"
        if name.startswith("lightgbm") and not dependency.get("lightgbm_available"):
            available, reason = False, "lightgbm unavailable"
        if name.startswith("catboost") and not dependency.get("catboost_available"):
            available, reason = False, "catboost unavailable"
        rows.append({"task": "price_return", "model_family": name, "available": available, "reason": reason, "stage": "screening"})
    for name in OPTIONAL_SEQUENCE_MODELS:
        rows.append({"task": "optional_sequence_or_qml", "model_family": name, "available": False, "reason": "not enabled in bounded model-universe screening; requires separate sequence adapter or focused QML replay", "stage": "skipped"})
    return rows


def selected_feature_groups(mode: str, target_variant: str, horizon: int, task: str) -> list[str]:
    if mode == "smoke":
        return ["compact_stable_features", "relative_strength"]
    base = ["compact_stable_features", "relative_strength"]
    if horizon == 40:
        base.extend(["market_context", "volume_volatility", "combined_strategy_features", "all_safe_features"])
    if task == "direction" and target_variant == "market_relative_vn30" and horizon == 40:
        base.append("qml_kernel_features")
    return list(dict.fromkeys(base))


def selected_direction_models(mode: str, target_variant: str, horizon: int, feature_group: str) -> list[str]:
    if mode == "smoke":
        return ["always_up", "always_down", "lag1_direction", "logistic_regression", "random_forest", "lightgbm_classifier"]
    models = [
        "always_up",
        "always_down",
        "lag1_direction",
        "random_same_class_balance",
        "simple_momentum",
        "simple_relative_strength",
        "logistic_regression",
        "calibrated_logistic",
        "linear_svm",
        "knn_classifier",
        "naive_bayes",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "xgboost_classifier",
        "lightgbm_classifier",
        "catboost_classifier",
        "mlp_classifier",
    ]
    if horizon == 40 and feature_group in {"compact_stable_features", "relative_strength"}:
        models.extend(["rbf_svm", "gradient_boosting"])
    if target_variant == "market_relative_vn30" and horizon == 40 and feature_group == "qml_kernel_features":
        models = ["qml_v8_kernel_features_l2", "qml_v8_kernel_features_lightgbm"]
    return models


def selected_price_models(mode: str, target_variant: str, horizon: int, feature_group: str) -> list[str]:
    if mode == "smoke":
        return ["last_price", "historical_mean_return", "ridge", "random_forest_regressor", "lightgbm_regressor"]
    models = [
        "random_walk_price",
        "last_price",
        "zero_return",
        "historical_mean_return",
        "rolling_mean_return",
        "simple_momentum",
        "simple_relative_strength",
        "linear_regression",
        "ridge",
        "lasso",
        "elasticnet",
        "knn_regressor",
        "random_forest_regressor",
        "extra_trees_regressor",
        "hist_gradient_boosting_regressor",
        "xgboost_regressor",
        "lightgbm_regressor",
        "catboost_regressor",
        "mlp_regressor",
    ]
    if horizon == 40 and feature_group in {"compact_stable_features", "relative_strength"}:
        models.extend(["svr_linear", "svr_rbf", "gradient_boosting_regressor"])
    return models


def build_candidate_grid(mode: str, feature_groups: dict[str, list[str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in HORIZONS if mode != "smoke" else [40]:
        for target in DIRECTION_TARGETS if mode != "smoke" else ["absolute_direction", "market_relative_vn30"]:
            for group in selected_feature_groups(mode, target, horizon, "direction"):
                for model in selected_direction_models(mode, target, horizon, group):
                    rows.append({"task": "direction", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": model, "candidate_id": candidate_id("direction", model, target, f"h{horizon}", group), "planned_stage": mode})
        for target in PRICE_TARGETS if mode != "smoke" else ["forward_simple_return_h", "future_close_h"]:
            for group in selected_feature_groups(mode, target, horizon, "price_return"):
                for model in selected_price_models(mode, target, horizon, group):
                    rows.append({"task": "price_return", "target_variant": target, "horizon": horizon, "feature_group": group, "model_family": model, "candidate_id": candidate_id("price", model, target, f"h{horizon}", group), "planned_stage": mode})
    return rows


def select_locked_direction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_accuracy")))]
    if not valid:
        return {}
    non_trivial = [
        row for row in valid
        if as_float(row.get("lift_over_strongest_baseline")) > 0.0
        and row.get("model_family") not in {"always_up", "always_down", "random_same_class_balance"}
        and as_float(row.get("balanced_accuracy")) > 0.5
    ]
    pool = non_trivial if non_trivial else valid
    return max(pool, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("lift_over_strongest_baseline")), as_float(row.get("balanced_accuracy")), as_float(row.get("ticker_stability")), -as_float(row.get("runtime_seconds"))))


def select_locked_price(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_rmse")))]
    if not valid:
        return {}
    improved = [row for row in valid if as_float(row.get("error_improvement_over_baseline")) > 0.0]
    pool = improved if improved else valid
    return max(pool, key=lambda row: (as_float(row.get("error_improvement_over_baseline")), -as_float(row.get("validation_rmse")), as_float(row.get("validation_correlation_pred_actual"))))


def write_reports(direction_rows: list[dict[str, Any]], price_rows: list[dict[str, Any]], skipped_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    locked_direction = manifest.get("locked_direction", {})
    locked_price = manifest.get("locked_price", {})
    abs_h40 = [
        row for row in direction_rows
        if row.get("status") == "ok" and row.get("target_variant") == "absolute_direction" and int(row.get("horizon", 0)) == 40
    ]
    abs_h40_nontrivial = [row for row in abs_h40 if as_float(row.get("lift_over_strongest_baseline")) > 0.0 and row.get("model_family") not in {"always_up", "always_down", "random_same_class_balance"}]
    best_abs = max(abs_h40_nontrivial or abs_h40, key=lambda row: as_float(row.get("final_accuracy")), default={})
    mr_h40 = [
        row for row in direction_rows
        if row.get("status") == "ok" and row.get("target_variant") == "market_relative_vn30" and int(row.get("horizon", 0)) == 40
    ]
    mr_h40_nontrivial = [row for row in mr_h40 if as_float(row.get("lift_over_strongest_baseline")) > 0.0 and row.get("model_family") not in {"always_up", "always_down", "random_same_class_balance"}]
    best_mr = max(mr_h40_nontrivial or mr_h40, key=lambda row: as_float(row.get("final_accuracy")), default={})
    price_beats_baseline = as_float(locked_price.get("error_improvement_over_baseline")) > 0.0
    best_abs_beats = as_float(best_abs.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"]
    best_mr_beats = as_float(best_mr.get("final_accuracy")) > QML_V8_CONTEXT_FINAL
    summary = f"""# VN30 Model Universe Direction + Price Forecasting Result Summary

## Required Answers

1. Which direction target performed best: `{locked_direction.get("target_variant", "")}` h{locked_direction.get("horizon", "")} with `{locked_direction.get("model_family", "")}` on `{locked_direction.get("feature_group", "")}`.
2. Which price/return target performed best: `{locked_price.get("target_variant", "")}` h{locked_price.get("horizon", "")} with `{locked_price.get("model_family", "")}` on `{locked_price.get("feature_group", "")}`.
3. Which model family performed best for direction: `{locked_direction.get("model_family", "")}` with validation accuracy {pct(locked_direction.get("validation_accuracy"))} and final accuracy {pct(locked_direction.get("final_accuracy"))}.
4. Which model family performed best for price/return: `{locked_price.get("model_family", "")}` with validation RMSE {as_float(locked_price.get("validation_rmse")):.6g} and final RMSE {as_float(locked_price.get("final_rmse")):.6g}.
5. Did any model beat the 61.61% absolute-direction classical champion on comparable absolute-direction scope: exploratory final-ranked rows did ({str(best_abs_beats).lower()}, best comparable final accuracy {pct(best_abs.get("final_accuracy"))}), but no claimable replacement is made because final-ranked rows are `exploratory_not_claimable`.
6. Did any model beat the 64.44% QML V8 market-relative result on comparable market_relative_vn30 scope: exploratory final-ranked rows did ({str(best_mr_beats).lower()}, best comparable final accuracy {pct(best_mr.get("final_accuracy"))}), but no claimable replacement is made because final-ranked rows are `exploratory_not_claimable`.
7. Did any model forecast price/return better than random walk / last price baseline: validation-screening yes ({str(price_beats_baseline).lower()}, locked validation error improvement {pp(locked_price.get("error_improvement_over_baseline"))}); final transfer is reported separately and is not a trading or production claim.
8. Which models failed or were skipped: {len(skipped_rows)} candidate/model rows were skipped; see `skipped_models.csv`.
9. Is there evidence that stock direction can be forecast: diagnostic evidence exists when validation-selected direction candidates beat simple validation baselines, but this is not a trading claim.
10. Is there evidence that stock price/return can be forecast: diagnostic evidence exists if validation RMSE improves over random-walk/last-price or historical-return baselines; this is reported separately from direction accuracy.
11. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, VN100, index-as-stock, merge, tag, DOCX, or production claim is made.

## Locked Direction Candidate

- Candidate: `{locked_direction.get("candidate_id", "")}`.
- Validation accuracy: {pct(locked_direction.get("validation_accuracy"))}.
- Final accuracy: {pct(locked_direction.get("final_accuracy"))}.
- Lift over strongest validation baseline: {pp(locked_direction.get("lift_over_strongest_baseline"))}.
- Claim label: `{locked_direction.get("claim_label", "")}`.

## Locked Price/Return Candidate

- Candidate: `{locked_price.get("candidate_id", "")}`.
- Validation RMSE: {as_float(locked_price.get("validation_rmse")):.6g}.
- Final RMSE: {as_float(locked_price.get("final_rmse")):.6g}.
- Validation sign accuracy: {pct(locked_price.get("validation_sign_accuracy"))}.
- Final sign accuracy: {pct(locked_price.get("final_sign_accuracy"))}.
- Claim label: `{locked_price.get("claim_label", "")}`.
"""
    write_markdown(RESULT_PATH, summary)
    claim = """# VN30 Model Universe Direction + Price Forecasting Claim Boundary

- This benchmark is offline diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Direction and price/return targets are separate; directional accuracy is not mixed with price or return errors.
- Price-level forecasts and return forecasts are reported separately.
- Index data may be used only as lagged market-context features or market-relative target context.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Final-ranked rows are exploratory_not_claimable.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, DOCX, tag, merge, push --mirror, or main-branch claim is made.
- Comparisons against the 61.61% absolute-direction classical champion are only valid on comparable absolute_direction h40 scope.
- Comparisons against the 64.44% QML V8 result are only valid on comparable market_relative_vn30 h40 scope.
- Stronger claims require future-blind confirmation.
"""
    write_markdown(CLAIM_PATH, claim)


def read_artifact(name: str) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def v2_baseline_maps(baseline: pd.DataFrame) -> tuple[dict[tuple[str, int, str], float], dict[tuple[str, int], float], dict[tuple[str, int, str], float], dict[tuple[str, int], float]]:
    if baseline.empty:
        return {}, {}, {}, {}
    frame = baseline.copy()
    frame["horizon"] = numeric_series(frame, "horizon").astype("Int64")
    frame["final_value_num"] = numeric_series(frame, "final_value")
    direction = frame[(frame["task"] == "direction") & frame["final_value_num"].notna()]
    price = frame[(frame["task"] == "price_return") & frame["final_value_num"].notna()]
    dir_feature = direction.groupby(["target_variant", "horizon", "feature_group"], dropna=False)["final_value_num"].max().to_dict()
    dir_target = direction.groupby(["target_variant", "horizon"], dropna=False)["final_value_num"].max().to_dict()
    price_feature = price.groupby(["target_variant", "horizon", "feature_group"], dropna=False)["final_value_num"].min().to_dict()
    price_target = price.groupby(["target_variant", "horizon"], dropna=False)["final_value_num"].min().to_dict()
    return dir_feature, dir_target, price_feature, price_target


def v2_lookup_baseline(row: pd.Series, exact: dict[tuple[str, int, str], float], fallback: dict[tuple[str, int], float]) -> float:
    target = str(row.get("target_variant", ""))
    horizon = int(as_float(row.get("horizon"))) if math.isfinite(as_float(row.get("horizon"))) else 0
    feature_group = str(row.get("feature_group", ""))
    value = exact.get((target, horizon, feature_group))
    if value is None:
        value = fallback.get((target, horizon))
    return float(value) if value is not None and math.isfinite(float(value)) else math.nan


def annotate_v2_direction(direction: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if direction.empty:
        return direction
    dir_feature, dir_target, _price_feature, _price_target = v2_baseline_maps(baseline)
    frame = direction.copy()
    frame = frame[frame.get("status", "ok") == "ok"].copy()
    for column in ["horizon", "validation_accuracy", "final_accuracy", "lift_over_strongest_baseline", "balanced_accuracy", "ticker_stability", "quarter_stability", "validation_rows", "final_rows"]:
        frame[column] = numeric_series(frame, column)
    frame["strongest_final_baseline_accuracy"] = frame.apply(lambda row: v2_lookup_baseline(row, dir_feature, dir_target), axis=1)
    frame["final_lift_over_strongest_baseline"] = frame["final_accuracy"] - frame["strongest_final_baseline_accuracy"]
    frame["is_trivial_baseline"] = frame["model_family"].isin(["always_up", "always_down", "random_same_class_balance"])
    frame["comparable_absolute_h40_scope"] = (frame["target_variant"] == "absolute_direction") & (frame["horizon"] == 40)
    frame["comparable_market_relative_h40_scope"] = (frame["target_variant"] == "market_relative_vn30") & (frame["horizon"] == 40)
    frame["v2_claim_label"] = "exploratory_not_claimable"
    return frame


def annotate_v2_price(price: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if price.empty:
        return price
    _dir_feature, _dir_target, price_feature, price_target = v2_baseline_maps(baseline)
    frame = price.copy()
    frame = frame[frame.get("status", "ok") == "ok"].copy()
    for column in [
        "horizon",
        "validation_rmse",
        "final_rmse",
        "validation_sign_accuracy",
        "final_sign_accuracy",
        "validation_correlation_pred_actual",
        "final_correlation_pred_actual",
        "error_improvement_over_baseline",
        "validation_rows",
        "final_rows",
    ]:
        frame[column] = numeric_series(frame, column)
    frame["strongest_final_baseline_rmse"] = frame.apply(lambda row: v2_lookup_baseline(row, price_feature, price_target), axis=1)
    frame["final_rmse_improvement_over_baseline"] = np.where(
        (frame["strongest_final_baseline_rmse"] > 0.0) & frame["strongest_final_baseline_rmse"].notna() & frame["final_rmse"].notna(),
        (frame["strongest_final_baseline_rmse"] - frame["final_rmse"]) / frame["strongest_final_baseline_rmse"],
        np.nan,
    )
    frame["is_price_baseline"] = frame["model_family"].isin(["random_walk_price", "last_price", "zero_return", "historical_mean_return", "rolling_mean_return"])
    frame["v2_claim_label"] = "exploratory_not_claimable"
    return frame


def source_bucket_frame(frame: pd.DataFrame, bucket: str, sort_column: str, ascending: bool, n: int = 50) -> pd.DataFrame:
    if frame.empty or sort_column not in frame.columns:
        return pd.DataFrame()
    subset = frame[frame[sort_column].notna()].sort_values(sort_column, ascending=ascending).head(n).copy()
    subset["v2_selection_bucket"] = bucket
    return subset


def v2_top_direction_candidates(direction: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        source_bucket_frame(direction[direction["target_variant"] == "absolute_direction"], "top50_absolute_direction_final_accuracy", "final_accuracy", False),
        source_bucket_frame(direction[direction["target_variant"] == "market_relative_vn30"], "top50_market_relative_vn30_final_accuracy", "final_accuracy", False),
        source_bucket_frame(direction, "top50_final_lift_over_strongest_baseline", "final_lift_over_strongest_baseline", False),
        source_bucket_frame(direction[(direction["target_variant"].isin(["absolute_direction", "market_relative_vn30"])) & (direction["horizon"] == 40)], "included_comparable_h40_rows", "final_accuracy", False),
    ]
    combined = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    if combined.empty:
        return combined
    bucket_map = combined.groupby("candidate_id")["v2_selection_bucket"].apply(lambda values: "|".join(sorted(set(values)))).to_dict()
    top = combined.sort_values(["final_accuracy", "final_lift_over_strongest_baseline"], ascending=False).drop_duplicates("candidate_id").copy()
    top["v2_selection_bucket"] = top["candidate_id"].map(bucket_map)
    return top.sort_values(["final_accuracy", "final_lift_over_strongest_baseline"], ascending=False)


def v2_top_price_candidates(price: pd.DataFrame) -> pd.DataFrame:
    pieces = [
        source_bucket_frame(price, "top50_final_rmse_improvement", "final_rmse_improvement_over_baseline", False),
        source_bucket_frame(price, "top50_final_sign_accuracy", "final_sign_accuracy", False),
        source_bucket_frame(price, "top50_final_correlation_pred_actual", "final_correlation_pred_actual", False),
        source_bucket_frame(price[price["target_variant"].isin(["forward_log_return_h", "market_excess_return_h"])], "included_forward_log_and_market_excess", "final_rmse_improvement_over_baseline", False),
    ]
    combined = pd.concat([piece for piece in pieces if not piece.empty], ignore_index=True)
    if combined.empty:
        return combined
    bucket_map = combined.groupby("candidate_id")["v2_selection_bucket"].apply(lambda values: "|".join(sorted(set(values)))).to_dict()
    top = combined.sort_values(["final_rmse_improvement_over_baseline", "final_sign_accuracy", "final_correlation_pred_actual"], ascending=False).drop_duplicates("candidate_id").copy()
    top["v2_selection_bucket"] = top["candidate_id"].map(bucket_map)
    return top.sort_values(["final_rmse_improvement_over_baseline", "final_sign_accuracy"], ascending=False)


def candidate_coverage_maps(ticker_stability: pd.DataFrame, quarter_stability: pd.DataFrame, task: str) -> tuple[dict[str, int], dict[str, int]]:
    ticker_counts: dict[str, int] = {}
    quarter_counts: dict[str, int] = {}
    if not ticker_stability.empty and "candidate_id" in ticker_stability.columns:
        ticker_counts = ticker_stability[ticker_stability.get("task") == task].groupby("candidate_id")["group"].nunique().to_dict()
    if not quarter_stability.empty and "candidate_id" in quarter_stability.columns:
        quarter_counts = quarter_stability[quarter_stability.get("task") == task].groupby("candidate_id")["group"].nunique().to_dict()
    return ticker_counts, quarter_counts


def v2_cluster_summary(top: pd.DataFrame, task: str, ticker_stability: pd.DataFrame, quarter_stability: pd.DataFrame) -> pd.DataFrame:
    if top.empty:
        return pd.DataFrame()
    ticker_counts, quarter_counts = candidate_coverage_maps(ticker_stability, quarter_stability, task)
    frame = top.copy()
    frame["ticker_coverage"] = frame["candidate_id"].map(ticker_counts).fillna(0).astype(int)
    frame["quarter_coverage"] = frame["candidate_id"].map(quarter_counts).fillna(0).astype(int)
    support_key = ["model_family", "target_variant", "horizon"]
    support_counts = frame.groupby(support_key, dropna=False)["candidate_id"].nunique().to_dict()
    rows: list[dict[str, Any]] = []
    group_cols = ["model_family", "target_variant", "horizon", "feature_group"]
    for key, group in frame.groupby(group_cols, dropna=False):
        model_family, target_variant, horizon, feature_group = key
        support_count = int(support_counts.get((model_family, target_variant, horizon), len(group)))
        if task == "direction":
            final_values = numeric_series(group, "final_accuracy")
            validation_values = numeric_series(group, "validation_accuracy")
            max_final = float(final_values.max())
            median_final = float(final_values.median())
            median_validation = float(validation_values.median())
            gap = median_final - median_validation
            metric_name = "accuracy"
        else:
            final_values = numeric_series(group, "final_rmse_improvement_over_baseline")
            validation_values = numeric_series(group, "error_improvement_over_baseline")
            max_final = float(final_values.max())
            median_final = float(final_values.median())
            median_validation = float(validation_values.median())
            gap = median_final - median_validation
            metric_name = "rmse_improvement_over_baseline"
        cluster_count = int(len(group))
        verdict = "cluster_supported" if cluster_count >= 2 or support_count >= 2 else "isolated_one_off"
        rows.append(
            {
                "task": task,
                "model_family": model_family,
                "target_variant": target_variant,
                "horizon": int(horizon) if math.isfinite(as_float(horizon)) else horizon,
                "feature_group": feature_group,
                "cluster_count": cluster_count,
                "family_target_horizon_support_count": support_count,
                "metric_name": metric_name,
                "max_final_metric": max_final,
                "median_final_metric": median_final,
                "validation_median_metric": median_validation,
                "validation_final_gap": gap,
                "median_row_count": float(numeric_series(group, "final_rows").median()) if "final_rows" in group.columns else math.nan,
                "median_ticker_coverage": float(group["ticker_coverage"].median()),
                "median_quarter_coverage": float(group["quarter_coverage"].median()),
                "median_ticker_stability": float(numeric_series(group, "ticker_stability").median()) if "ticker_stability" in group.columns else math.nan,
                "median_quarter_stability": float(numeric_series(group, "quarter_stability").median()) if "quarter_stability" in group.columns else math.nan,
                "cluster_verdict": verdict,
                "candidate_ids": "|".join(group["candidate_id"].astype(str).tolist()[:12]),
            }
        )
    return pd.DataFrame(rows).sort_values(["max_final_metric", "family_target_horizon_support_count"], ascending=False)


def v2_relock_direction_candidates(direction: pd.DataFrame, clusters: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    locked: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if direction.empty or clusters.empty:
        return locked, audit
    interesting = clusters.head(24).copy()
    required = clusters[((clusters["target_variant"].isin(["absolute_direction", "market_relative_vn30"])) & (clusters["horizon"] == 40))]
    interesting = pd.concat([interesting, required], ignore_index=True).drop_duplicates(["model_family", "target_variant", "horizon", "feature_group"])
    for _, cluster in interesting.iterrows():
        subset = direction[
            (direction["model_family"] == cluster["model_family"])
            & (direction["target_variant"] == cluster["target_variant"])
            & (direction["horizon"] == cluster["horizon"])
            & (direction["feature_group"] == cluster["feature_group"])
        ].copy()
        if subset.empty:
            continue
        ranked = subset.sort_values(["validation_accuracy", "lift_over_strongest_baseline", "balanced_accuracy"], ascending=False)
        selected = ranked.iloc[0].to_dict()
        is_trivial = bool(selected.get("is_trivial_baseline"))
        validation_lift = as_float(selected.get("lift_over_strongest_baseline"))
        final_accuracy = as_float(selected.get("final_accuracy"))
        target = str(selected.get("target_variant", ""))
        horizon = int(as_float(selected.get("horizon"))) if math.isfinite(as_float(selected.get("horizon"))) else 0
        if is_trivial:
            label = "not_claimable"
            reason = "trivial_baseline_family"
        elif validation_lift <= 0:
            label = "exploratory_not_claimable"
            reason = "validation_lift_not_positive"
        elif target == "absolute_direction" and horizon == 40 and final_accuracy > CLASSICAL_CHAMPION["final_accuracy"]:
            label = "future_blind_required"
            reason = "validation_relocked_final_beats_context_but_hypothesis_was_final_discovered"
        elif target == "market_relative_vn30" and horizon == 40 and final_accuracy > QML_V8_CONTEXT_FINAL:
            label = "future_blind_required"
            reason = "validation_relocked_final_beats_qml_v8_context_but_hypothesis_was_final_discovered"
        else:
            label = "future_blind_required"
            reason = "validation_relocked_diagnostic_requires_future_blind_confirmation"
        selected.update(
            {
                "relock_source": "final_discovered_cluster_hypothesis",
                "selection_rule": "within_cluster_validation_accuracy_then_validation_lift",
                "relock_claim_label": label,
                "relock_reason": reason,
                "cluster_verdict": cluster.get("cluster_verdict", ""),
                "family_target_horizon_support_count": int(cluster.get("family_target_horizon_support_count", 0)),
                "final_selection_used": False,
                "future_blind_required": True,
            }
        )
        locked.append(json_safe(selected))
        audit.append(
            {
                "task": "direction",
                "source_model_family": cluster["model_family"],
                "target_variant": cluster["target_variant"],
                "horizon": cluster["horizon"],
                "feature_group": cluster["feature_group"],
                "source_cluster_verdict": cluster.get("cluster_verdict", ""),
                "cluster_count": cluster.get("cluster_count", 0),
                "family_target_horizon_support_count": cluster.get("family_target_horizon_support_count", 0),
                "selection_freeze": "model_family/target/horizon/feature_group",
                "selection_metric": "validation_accuracy_then_validation_lift",
                "selected_candidate_id": selected.get("candidate_id", ""),
                "selected_validation_accuracy": selected.get("validation_accuracy", math.nan),
                "selected_final_accuracy": selected.get("final_accuracy", math.nan),
                "final_selection_used": False,
                "claim_label": label,
                "audit_note": reason,
            }
        )
    return locked, audit


def v2_relock_price_candidates(price: pd.DataFrame, clusters: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    locked: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    if price.empty or clusters.empty:
        return locked, audit
    interesting = clusters.head(24).copy()
    required = clusters[clusters["target_variant"].isin(["forward_log_return_h", "market_excess_return_h"])]
    interesting = pd.concat([interesting, required], ignore_index=True).drop_duplicates(["model_family", "target_variant", "horizon", "feature_group"])
    for _, cluster in interesting.iterrows():
        subset = price[
            (price["model_family"] == cluster["model_family"])
            & (price["target_variant"] == cluster["target_variant"])
            & (price["horizon"] == cluster["horizon"])
            & (price["feature_group"] == cluster["feature_group"])
        ].copy()
        if subset.empty:
            continue
        ranked = subset.sort_values(["error_improvement_over_baseline", "validation_sign_accuracy", "validation_correlation_pred_actual", "validation_rmse"], ascending=[False, False, False, True])
        selected = ranked.iloc[0].to_dict()
        final_improvement = as_float(selected.get("final_rmse_improvement_over_baseline"))
        validation_improvement = as_float(selected.get("error_improvement_over_baseline"))
        if bool(selected.get("is_price_baseline")):
            label = "not_claimable"
            reason = "baseline_family"
        elif validation_improvement <= 0:
            label = "exploratory_not_claimable"
            reason = "validation_error_improvement_not_positive"
        elif final_improvement > 0:
            label = "future_blind_required"
            reason = "validation_relocked_final_beats_price_baseline_but_hypothesis_was_final_discovered"
        else:
            label = "future_blind_required"
            reason = "validation_relocked_price_diagnostic_requires_future_blind_confirmation"
        selected.update(
            {
                "relock_source": "final_discovered_cluster_hypothesis",
                "selection_rule": "within_cluster_validation_error_improvement_then_sign_accuracy",
                "relock_claim_label": label,
                "relock_reason": reason,
                "cluster_verdict": cluster.get("cluster_verdict", ""),
                "family_target_horizon_support_count": int(cluster.get("family_target_horizon_support_count", 0)),
                "final_selection_used": False,
                "future_blind_required": True,
            }
        )
        locked.append(json_safe(selected))
        audit.append(
            {
                "task": "price_return",
                "source_model_family": cluster["model_family"],
                "target_variant": cluster["target_variant"],
                "horizon": cluster["horizon"],
                "feature_group": cluster["feature_group"],
                "source_cluster_verdict": cluster.get("cluster_verdict", ""),
                "cluster_count": cluster.get("cluster_count", 0),
                "family_target_horizon_support_count": cluster.get("family_target_horizon_support_count", 0),
                "selection_freeze": "model_family/target/horizon/feature_group",
                "selection_metric": "validation_error_improvement_then_sign_accuracy",
                "selected_candidate_id": selected.get("candidate_id", ""),
                "selected_validation_rmse": selected.get("validation_rmse", math.nan),
                "selected_final_rmse": selected.get("final_rmse", math.nan),
                "selected_final_rmse_improvement": selected.get("final_rmse_improvement_over_baseline", math.nan),
                "final_selection_used": False,
                "claim_label": label,
                "audit_note": reason,
            }
        )
    return locked, audit


def v2_rolling_rows_from_artifacts(locked: list[dict[str, Any]], quarter_stability: pd.DataFrame, task: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    quarter_frame = quarter_stability[quarter_stability.get("task") == task].copy() if not quarter_stability.empty and "task" in quarter_stability.columns else pd.DataFrame()
    for item in locked:
        candidate = str(item.get("candidate_id", ""))
        if task == "direction":
            rows.extend(
                [
                    {"candidate_id": candidate, "task": task, "window": "validation_2024_full", "metric": "accuracy", "value": item.get("validation_accuracy", math.nan), "rows": item.get("validation_rows", math.nan), "status": "aggregate_available"},
                    {"candidate_id": candidate, "task": task, "window": "validation_early_2024", "metric": "accuracy", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "validation_late_2024", "metric": "accuracy", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "final_2025plus_diagnostic", "metric": "accuracy", "value": item.get("final_accuracy", math.nan), "rows": item.get("final_rows", math.nan), "status": "aggregate_available"},
                ]
            )
        else:
            rows.extend(
                [
                    {"candidate_id": candidate, "task": task, "window": "validation_2024_full", "metric": "rmse", "value": item.get("validation_rmse", math.nan), "rows": item.get("validation_rows", math.nan), "status": "aggregate_available"},
                    {"candidate_id": candidate, "task": task, "window": "validation_early_2024", "metric": "rmse", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "validation_late_2024", "metric": "rmse", "value": math.nan, "rows": math.nan, "status": "not_available_from_v1_aggregate_artifacts"},
                    {"candidate_id": candidate, "task": task, "window": "final_2025plus_diagnostic", "metric": "rmse", "value": item.get("final_rmse", math.nan), "rows": item.get("final_rows", math.nan), "status": "aggregate_available"},
                ]
            )
        if not quarter_frame.empty:
            for _, qrow in quarter_frame[quarter_frame["candidate_id"] == candidate].iterrows():
                rows.append(
                    {
                        "candidate_id": candidate,
                        "task": task,
                        "window": f"final_quarter_{qrow.get('group', '')}",
                        "metric": qrow.get("metric", ""),
                        "value": as_float(qrow.get("value")),
                        "rows": as_float(qrow.get("rows")),
                        "status": "quarter_artifact_available",
                    }
                )
    return rows


def best_relocked_direction(locked: list[dict[str, Any]], target: str | None = None, horizon: int | None = None) -> dict[str, Any]:
    rows = [
        row for row in locked
        if (target is None or row.get("target_variant") == target)
        and (horizon is None or int(as_float(row.get("horizon"))) == horizon)
        and row.get("relock_claim_label") != "not_claimable"
    ]
    if not rows:
        rows = [row for row in locked if target is None or row.get("target_variant") == target]
    return max(rows, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("lift_over_strongest_baseline")), as_float(row.get("final_accuracy"))), default={})


def best_relocked_price(locked: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in locked if row.get("relock_claim_label") != "not_claimable"]
    if not rows:
        rows = locked
    return max(rows, key=lambda row: (as_float(row.get("error_improvement_over_baseline")), as_float(row.get("final_rmse_improvement_over_baseline")), as_float(row.get("validation_sign_accuracy"))), default={})


def v2_comparison_rows(locked_direction: list[dict[str, Any]], locked_price: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    v1_direction = manifest.get("locked_direction", {}) if isinstance(manifest, dict) else {}
    v1_price = manifest.get("locked_price", {}) if isinstance(manifest, dict) else {}
    for item in locked_direction:
        target = str(item.get("target_variant", ""))
        horizon = int(as_float(item.get("horizon"))) if math.isfinite(as_float(item.get("horizon"))) else 0
        if target == "absolute_direction" and horizon == 40:
            rows.append({"task": "direction", "candidate_id": item.get("candidate_id", ""), "comparison": "61.61_absolute_direction_classical_champion", "candidate_metric": item.get("final_accuracy", math.nan), "reference_metric": CLASSICAL_CHAMPION["final_accuracy"], "delta": as_float(item.get("final_accuracy")) - CLASSICAL_CHAMPION["final_accuracy"], "claim_status": item.get("relock_claim_label", "")})
        if target == "market_relative_vn30" and horizon == 40:
            rows.append({"task": "direction", "candidate_id": item.get("candidate_id", ""), "comparison": "64.44_qml_v8_market_relative_context", "candidate_metric": item.get("final_accuracy", math.nan), "reference_metric": QML_V8_CONTEXT_FINAL, "delta": as_float(item.get("final_accuracy")) - QML_V8_CONTEXT_FINAL, "claim_status": item.get("relock_claim_label", "")})
        if v1_direction:
            rows.append({"task": "direction", "candidate_id": item.get("candidate_id", ""), "comparison": "v1_locked_direction_final_accuracy", "candidate_metric": item.get("final_accuracy", math.nan), "reference_metric": v1_direction.get("final_accuracy", math.nan), "delta": as_float(item.get("final_accuracy")) - as_float(v1_direction.get("final_accuracy")), "claim_status": item.get("relock_claim_label", "")})
    for item in locked_price:
        if v1_price:
            rows.append({"task": "price_return", "candidate_id": item.get("candidate_id", ""), "comparison": "v1_locked_price_final_rmse", "candidate_metric": item.get("final_rmse", math.nan), "reference_metric": v1_price.get("final_rmse", math.nan), "delta": as_float(v1_price.get("final_rmse")) - as_float(item.get("final_rmse")), "claim_status": item.get("relock_claim_label", "")})
        rows.append({"task": "price_return", "candidate_id": item.get("candidate_id", ""), "comparison": "strongest_price_baseline_final_rmse", "candidate_metric": item.get("final_rmse", math.nan), "reference_metric": item.get("strongest_final_baseline_rmse", math.nan), "delta": as_float(item.get("strongest_final_baseline_rmse")) - as_float(item.get("final_rmse")), "claim_status": item.get("relock_claim_label", "")})
    return rows


def write_v2_reports(
    top_direction: pd.DataFrame,
    direction_clusters: pd.DataFrame,
    top_price: pd.DataFrame,
    locked_direction: list[dict[str, Any]],
    locked_price: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
) -> None:
    abs_source_all = top_direction[(top_direction["target_variant"] == "absolute_direction") & (top_direction["horizon"] == 40)].sort_values("final_accuracy", ascending=False).head(1)
    abs_source_promotable = top_direction[
        (top_direction["target_variant"] == "absolute_direction")
        & (top_direction["horizon"] == 40)
        & (~top_direction["is_trivial_baseline"])
        & (top_direction["lift_over_strongest_baseline"] > 0)
    ].sort_values("final_accuracy", ascending=False).head(1)
    mr_source = top_direction[(top_direction["target_variant"] == "market_relative_vn30") & (top_direction["horizon"] == 40)].sort_values("final_accuracy", ascending=False).head(1)
    abs_row = abs_source_promotable.iloc[0].to_dict() if not abs_source_promotable.empty else (abs_source_all.iloc[0].to_dict() if not abs_source_all.empty else {})
    abs_all_row = abs_source_all.iloc[0].to_dict() if not abs_source_all.empty else {}
    mr_row = mr_source.iloc[0].to_dict() if not mr_source.empty else {}
    locked_abs = best_relocked_direction(locked_direction, "absolute_direction", 40)
    locked_mr = best_relocked_direction(locked_direction, "market_relative_vn30", 40)
    locked_price_best = best_relocked_price(locked_price)
    abs_cluster = direction_clusters[
        (direction_clusters["model_family"] == abs_row.get("model_family"))
        & (direction_clusters["target_variant"] == abs_row.get("target_variant"))
        & (direction_clusters["horizon"] == abs_row.get("horizon"))
        & (direction_clusters["feature_group"] == abs_row.get("feature_group"))
    ]
    mr_cluster = direction_clusters[
        (direction_clusters["model_family"] == mr_row.get("model_family"))
        & (direction_clusters["target_variant"] == mr_row.get("target_variant"))
        & (direction_clusters["horizon"] == mr_row.get("horizon"))
        & (direction_clusters["feature_group"] == mr_row.get("feature_group"))
    ]
    abs_cluster_verdict = str(abs_cluster.iloc[0].get("cluster_verdict", "")) if not abs_cluster.empty else ""
    mr_cluster_verdict = str(mr_cluster.iloc[0].get("cluster_verdict", "")) if not mr_cluster.empty else ""
    abs_final_relocks = [
        row for row in locked_direction
        if row.get("target_variant") == "absolute_direction"
        and int(as_float(row.get("horizon"))) == 40
        and row.get("relock_claim_label") == "future_blind_required"
        and as_float(row.get("final_accuracy")) > CLASSICAL_CHAMPION["final_accuracy"]
    ]
    abs_final_best = max(abs_final_relocks, key=lambda row: as_float(row.get("final_accuracy")), default={})
    mr_trivial_final_relocks = [
        row for row in locked_direction
        if row.get("target_variant") == "market_relative_vn30"
        and int(as_float(row.get("horizon"))) == 40
        and as_float(row.get("final_accuracy")) > QML_V8_CONTEXT_FINAL
    ]
    mr_nontrivial_final_relocks = [row for row in mr_trivial_final_relocks if row.get("relock_claim_label") == "future_blind_required"]
    mr_final_best = max(mr_nontrivial_final_relocks or mr_trivial_final_relocks, key=lambda row: as_float(row.get("final_accuracy")), default={})
    price_final_relocks = [
        row for row in locked_price
        if row.get("relock_claim_label") == "future_blind_required"
        and as_float(row.get("final_rmse_improvement_over_baseline")) > 0
    ]
    price_final_best = max(price_final_relocks, key=lambda row: as_float(row.get("final_rmse_improvement_over_baseline")), default={})
    price_final_beats = bool(price_final_relocks)
    summary = f"""# VN30 Model Universe V2 Promotion Relock Result Summary

## Required Answers

1. What produced the exploratory 69.86% absolute-direction final row: the strongest non-trivial comparable h40 source was `{abs_row.get("candidate_id", "")}` with final accuracy {pct(abs_row.get("final_accuracy"))}, validation accuracy {pct(abs_row.get("validation_accuracy"))}, and validation lift {pp(abs_row.get("lift_over_strongest_baseline"))}. The absolute highest h40 final row was `{abs_all_row.get("candidate_id", "")}` at {pct(abs_all_row.get("final_accuracy"))}, but it did not pass validation-lift promotion screening.
2. What produced the exploratory 74.57% market-relative final row: `{mr_row.get("candidate_id", "")}` with final accuracy {pct(mr_row.get("final_accuracy"))}; this is a trivial/simple baseline pattern with validation lift {pp(mr_row.get("lift_over_strongest_baseline"))}, so it is not promotable.
3. Were those results isolated one-offs or cluster-supported: the absolute-direction source is `{abs_cluster_verdict}` and the market-relative source is `{mr_cluster_verdict}` under the model/target/horizon/feature cluster audit.
4. Could any family be re-locked by validation: yes, V2 froze final-discovered hypothesis families and selected exact candidates by validation only, but all such relocks remain future-blind-required because the families were discovered from final exploratory rows.
5. Did any relocked direction candidate beat 61.61% on comparable scope: {str(bool(abs_final_relocks)).lower()} for `{abs_final_best.get("candidate_id", "")}` with final accuracy {pct(abs_final_best.get("final_accuracy"))}; this is future-blind-required and not a replacement claim. The validation-best absolute h40 relock was `{locked_abs.get("candidate_id", "")}` with final accuracy {pct(locked_abs.get("final_accuracy"))}.
6. Did any relocked market-relative candidate beat 64.44% on comparable scope: {str(bool(mr_nontrivial_final_relocks)).lower()} for non-trivial relocks. The strongest row over 64.44% was `{mr_final_best.get("candidate_id", "")}` at {pct(mr_final_best.get("final_accuracy"))}, but its label is `{mr_final_best.get("relock_claim_label", "")}`.
7. Did any relocked price/return candidate beat random walk/last price on final: {str(price_final_beats).lower()} for validation-supported relocks. Best validation-supported final improvement was `{price_final_best.get("candidate_id", locked_price_best.get("candidate_id", ""))}` with final RMSE improvement {pp(price_final_best.get("final_rmse_improvement_over_baseline", locked_price_best.get("final_rmse_improvement_over_baseline")))}.
8. Which results remain exploratory: all source final-ranked rows and all relocked rows remain diagnostic/future-blind-required unless confirmed by a new pre-registered future-blind run.
9. Exact claim boundary: offline diagnostic-only; no trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, VN100, index-as-stock, tag, merge, push --mirror, DOCX, or production claim is made.

## Relocked Direction Candidates

- Best absolute h40 relock: `{locked_abs.get("candidate_id", "")}`; validation {pct(locked_abs.get("validation_accuracy"))}, final {pct(locked_abs.get("final_accuracy"))}, label `{locked_abs.get("relock_claim_label", "")}`.
- Strongest absolute h40 future-blind-required final diagnostic: `{abs_final_best.get("candidate_id", "")}`; validation {pct(abs_final_best.get("validation_accuracy"))}, final {pct(abs_final_best.get("final_accuracy"))}, label `{abs_final_best.get("relock_claim_label", "")}`.
- Best market-relative h40 relock: `{locked_mr.get("candidate_id", "")}`; validation {pct(locked_mr.get("validation_accuracy"))}, final {pct(locked_mr.get("final_accuracy"))}, label `{locked_mr.get("relock_claim_label", "")}`.

## Relocked Price/Return Candidate

- Best price/return relock: `{locked_price_best.get("candidate_id", "")}`; validation RMSE {as_float(locked_price_best.get("validation_rmse")):.6g}, final RMSE {as_float(locked_price_best.get("final_rmse")):.6g}, final baseline improvement {pp(locked_price_best.get("final_rmse_improvement_over_baseline"))}, label `{locked_price_best.get("relock_claim_label", "")}`.

## Audit Note

V2 uses V1 aggregate artifacts for promotion relock. Early/late validation window rows are marked unavailable when row-level validation predictions were not preserved by V1; final quarter diagnostics are included where V1 stability artifacts exist.
"""
    write_markdown(V2_RESULT_PATH, summary)
    claim = """# VN30 Model Universe V2 Promotion Relock Claim Boundary

- V2 is an offline diagnostic promotion-relock audit only.
- Exploratory final rows are used only to define hypothesis families, not to select claimable rows.
- Exact relocked candidates are selected within each frozen hypothesis family by validation metrics only.
- Because hypothesis families were discovered from final exploratory rows, all stronger interpretation requires a new future-blind protocol.
- Final-ranked rows remain exploratory_not_claimable unless re-run under a future-blind design.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, production, VN100, index-as-stock, DOCX, tag, merge, push --mirror, or main-branch claim is made.
- Comparisons against the 61.61% absolute-direction champion are contextual and only valid on comparable absolute_direction h40 scope.
- Comparisons against the 64.44% QML V8 market-relative result are contextual and only valid on comparable market_relative_vn30 h40 scope.
"""
    write_markdown(V2_CLAIM_PATH, claim)


def run_promotion_relock(timeout_seconds: int) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    direction_raw = read_artifact("direction_validation_results.csv")
    price_raw = read_artifact("price_validation_results.csv")
    baseline = read_artifact("baseline_comparison.csv")
    ticker_stability = read_artifact("ticker_stability.csv")
    quarter_stability = read_artifact("quarter_stability.csv")
    manifest_path = OUTPUT_DIR / "model_universe_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    direction = annotate_v2_direction(direction_raw, baseline)
    price = annotate_v2_price(price_raw, baseline)
    top_direction = v2_top_direction_candidates(direction)
    top_price = v2_top_price_candidates(price)
    direction_clusters = v2_cluster_summary(top_direction, "direction", ticker_stability, quarter_stability)
    price_clusters = v2_cluster_summary(top_price, "price_return", ticker_stability, quarter_stability)
    locked_direction, direction_audit = v2_relock_direction_candidates(direction, direction_clusters)
    locked_price, price_audit = v2_relock_price_candidates(price, price_clusters)
    protocol_audit = direction_audit + price_audit
    rolling_direction = v2_rolling_rows_from_artifacts(locked_direction, quarter_stability, "direction")
    rolling_price = v2_rolling_rows_from_artifacts(locked_price, quarter_stability, "price_return")
    comparison_rows = v2_comparison_rows(locked_direction, locked_price, manifest)

    write_frame(OUTPUT_DIR / "v2_top_exploratory_direction_candidates.csv", top_direction.to_dict("records"), list(top_direction.columns) if not top_direction.empty else [])
    write_frame(OUTPUT_DIR / "v2_top_exploratory_price_candidates.csv", top_price.to_dict("records"), list(top_price.columns) if not top_price.empty else [])
    write_frame(OUTPUT_DIR / "v2_direction_candidate_cluster_summary.csv", direction_clusters.to_dict("records"), list(direction_clusters.columns) if not direction_clusters.empty else [])
    write_frame(OUTPUT_DIR / "v2_price_candidate_cluster_summary.csv", price_clusters.to_dict("records"), list(price_clusters.columns) if not price_clusters.empty else [])
    write_frame(OUTPUT_DIR / "v2_relock_protocol_audit.csv", protocol_audit, list(protocol_audit[0].keys()) if protocol_audit else [])
    write_json(OUTPUT_DIR / "v2_locked_direction_candidates.json", {"locked_candidates": locked_direction, "selection": "validation_only_within_final_discovered_hypothesis_family", "future_blind_required": True})
    write_json(OUTPUT_DIR / "v2_locked_price_candidates.json", {"locked_candidates": locked_price, "selection": "validation_only_within_final_discovered_hypothesis_family", "future_blind_required": True})
    write_frame(OUTPUT_DIR / "v2_rolling_origin_direction.csv", rolling_direction, list(rolling_direction[0].keys()) if rolling_direction else [])
    write_frame(OUTPUT_DIR / "v2_rolling_origin_price.csv", rolling_price, list(rolling_price[0].keys()) if rolling_price else [])
    write_frame(OUTPUT_DIR / "v2_relocked_comparison_summary.csv", comparison_rows, list(comparison_rows[0].keys()) if comparison_rows else [])
    write_v2_reports(top_direction, direction_clusters, top_price, locked_direction, locked_price, comparison_rows)

    result = {
        "status": "ok",
        "mode": "promotion_relock",
        "runtime_seconds": time.perf_counter() - started,
        "timeout_seconds": timeout_seconds,
        "top_direction_candidates": int(len(top_direction)),
        "top_price_candidates": int(len(top_price)),
        "direction_clusters": int(len(direction_clusters)),
        "price_clusters": int(len(price_clusters)),
        "locked_direction_candidates": int(len(locked_direction)),
        "locked_price_candidates": int(len(locked_price)),
        "diagnostic_only": True,
        "future_blind_required": True,
        "no_trading_claim": True,
    }
    print(json.dumps(json_safe(result), indent=2))
    return result


def run_benchmark(config: RunConfig) -> dict[str, Any]:
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    write_json(OUTPUT_DIR / "dependency_status.json", dependency)
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    feature_groups = build_feature_groups(features, family_cols, relative_cols)
    run_config = {
        "mode": config.mode,
        "timeout_seconds": config.timeout_seconds,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "max_final_rows": config.max_final_rows,
        "horizons": HORIZONS if config.mode != "smoke" else [40],
        "direction_targets": DIRECTION_TARGETS if config.mode != "smoke" else ["absolute_direction", "market_relative_vn30"],
        "price_targets": PRICE_TARGETS if config.mode != "smoke" else ["forward_simple_return_h", "future_close_h"],
        "diagnostic_only": True,
        "no_trading_claim": True,
        "no_vn100": True,
        "no_index_as_stock": True,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)

    dataset_rows = [
        {
            "rows": int(len(features)),
            "tickers": int(features["ticker"].nunique()),
            "timestamp_start": str(pd.to_datetime(features["datetime"]).min()),
            "timestamp_end": str(pd.to_datetime(features["datetime"]).max()),
            "feature_timestamp_present": "feature_timestamp" in features.columns,
            "index_context_allowed_lagged_only": True,
        }
    ]
    write_frame(OUTPUT_DIR / "dataset_audit.csv", dataset_rows, list(dataset_rows[0].keys()))
    feature_group_rows = [{"feature_group": name, "feature_count": int(len(cols)), "sample_features": "|".join(cols[:20]), "train_only_transform_required": True} for name, cols in feature_groups.items()]
    write_frame(OUTPUT_DIR / "feature_group_audit.csv", feature_group_rows, list(feature_group_rows[0].keys()))
    registry_rows = model_family_registry(dependency)
    write_frame(OUTPUT_DIR / "model_family_registry.csv", registry_rows, list(registry_rows[0].keys()))
    candidate_grid = build_candidate_grid(config.mode, feature_groups)
    write_frame(OUTPUT_DIR / "candidate_grid.csv", candidate_grid, list(candidate_grid[0].keys()) if candidate_grid else [])

    direction_rows: list[dict[str, Any]] = []
    price_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    split_guard_rows: list[dict[str, Any]] = []
    target_audit_rows: list[dict[str, Any]] = []
    ticker_rows: list[dict[str, Any]] = []
    quarter_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []

    evaluated_direction = 0
    evaluated_price = 0
    horizons = [40] if config.mode == "smoke" else HORIZONS
    direction_targets = ["absolute_direction", "market_relative_vn30"] if config.mode == "smoke" else DIRECTION_TARGETS
    price_targets = ["forward_simple_return_h", "future_close_h"] if config.mode == "smoke" else PRICE_TARGETS

    for horizon in horizons:
        for target_variant in direction_targets:
            labels = build_direction_target(features, index_data, target_variant, horizon)
            full_splits = strict_split_indices(features, labels)
            splits = split_sample(features, labels, full_splits, config, classification=True)
            guard = leakage_guard_passed(features, labels, splits)
            target_ts = target_timestamp_from_labels(labels, features.index)
            target_audit_rows.append({"task": "direction", "target_variant": target_variant, "horizon": horizon, "valid_rows": int(labels.notna().sum()), "positive_ratio": float(labels.dropna().astype(int).mean()) if labels.notna().any() else math.nan, "target_timestamp_min": str(target_ts.dropna().min()) if target_ts.notna().any() else "", "target_timestamp_max": str(target_ts.dropna().max()) if target_ts.notna().any() else ""})
            for split_name, idx in splits.items():
                split_guard_rows.append({"task": "direction", "target_variant": target_variant, "horizon": horizon, "split": split_name, "rows": int(len(idx)), "feature_timestamp_min": str(pd.to_datetime(features.loc[idx, "datetime"]).min()) if len(idx) else "", "feature_timestamp_max": str(pd.to_datetime(features.loc[idx, "datetime"]).max()) if len(idx) else "", "target_timestamp_min": str(target_ts.loc[idx].min()) if len(idx) else "", "target_timestamp_max": str(target_ts.loc[idx].max()) if len(idx) else "", "guard_passed": bool(guard)})
            for group in selected_feature_groups(config.mode, target_variant, horizon, "direction"):
                cols = feature_groups.get(group, [])
                for model_name in selected_direction_models(config.mode, target_variant, horizon, group):
                    if time.perf_counter() - started >= config.timeout_seconds or evaluated_direction >= config.max_direction_candidates:
                        skipped_rows.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": "runtime_or_candidate_budget_exhausted"})
                        continue
                    row, final_pred = evaluate_direction_candidate(model_name, group, features, family_cols, relative_cols, index_data, labels, splits, cols)
                    direction_rows.append(row)
                    evaluated_direction += 1
                    if row["status"] == "skipped":
                        skipped_rows.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": row["skipped_reason"]})
                    else:
                        if model_name in {"always_up", "always_down", "lag1_direction", "random_same_class_balance", "simple_momentum", "simple_relative_strength"}:
                            baseline_rows.append({"task": "direction", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "validation_metric": "accuracy", "validation_value": row["validation_accuracy"], "final_metric": "accuracy", "final_value": row["final_accuracy"]})
                        if final_pred is not None:
                            tr, qr = stability_rows("direction", {"candidate_id": row["candidate_id"], "target_variant": target_variant, "horizon": horizon}, features, splits["final"], labels, final_pred)
                            ticker_rows.extend(tr)
                            quarter_rows.extend(qr)

        for target_variant in price_targets:
            target = build_price_target(features, index_data, target_variant, horizon)
            full_splits = strict_split_for_target(features, target)
            splits = split_sample(features, target, full_splits, config, classification=False)
            guard = leakage_guard_for_target(features, target, splits)
            target_ts = target_timestamp_series(target)
            target_audit_rows.append({"task": "price_return", "target_variant": target_variant, "horizon": horizon, "valid_rows": int(target.notna().sum()), "positive_ratio": float((target.dropna() > 0).mean()) if target.notna().any() and target_variant != "future_close_h" else math.nan, "target_timestamp_min": str(target_ts.dropna().min()) if target_ts.notna().any() else "", "target_timestamp_max": str(target_ts.dropna().max()) if target_ts.notna().any() else ""})
            for split_name, idx in splits.items():
                split_guard_rows.append({"task": "price_return", "target_variant": target_variant, "horizon": horizon, "split": split_name, "rows": int(len(idx)), "feature_timestamp_min": str(pd.to_datetime(features.loc[idx, "datetime"]).min()) if len(idx) else "", "feature_timestamp_max": str(pd.to_datetime(features.loc[idx, "datetime"]).max()) if len(idx) else "", "target_timestamp_min": str(target_ts.loc[idx].min()) if len(idx) else "", "target_timestamp_max": str(target_ts.loc[idx].max()) if len(idx) else "", "guard_passed": bool(guard)})
            for group in selected_feature_groups(config.mode, target_variant, horizon, "price_return"):
                if group == "qml_kernel_features":
                    continue
                cols = feature_groups.get(group, [])
                for model_name in selected_price_models(config.mode, target_variant, horizon, group):
                    if time.perf_counter() - started >= config.timeout_seconds or evaluated_price >= config.max_price_candidates:
                        skipped_rows.append({"task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": "runtime_or_candidate_budget_exhausted"})
                        continue
                    row, final_pred = evaluate_price_candidate(model_name, group, features, target, splits, cols)
                    price_rows.append(row)
                    evaluated_price += 1
                    if row["status"] == "skipped":
                        skipped_rows.append({"task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "skipped_reason": row["skipped_reason"]})
                    else:
                        if model_name in {"random_walk_price", "last_price", "zero_return", "historical_mean_return", "rolling_mean_return", "simple_momentum", "simple_relative_strength"}:
                            baseline_rows.append({"task": "price_return", "model_family": model_name, "target_variant": target_variant, "horizon": horizon, "feature_group": group, "validation_metric": "rmse", "validation_value": row["validation_rmse"], "final_metric": "rmse", "final_value": row["final_rmse"]})
                        if final_pred is not None:
                            tr, qr = stability_rows("price_return", {"candidate_id": row["candidate_id"], "target_variant": target_variant, "horizon": horizon}, features, splits["final"], target, final_pred)
                            ticker_rows.extend(tr)
                            quarter_rows.extend(qr)

    for registry in registry_rows:
        if not registry["available"]:
            skipped_rows.append({"task": registry["task"], "model_family": registry["model_family"], "target_variant": "", "horizon": "", "feature_group": "", "skipped_reason": registry["reason"]})

    locked_direction = select_locked_direction(direction_rows)
    locked_price = select_locked_price(price_rows)
    direction_leaderboard = sorted([row for row in direction_rows if row["status"] == "ok"], key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("lift_over_strongest_baseline"))), reverse=True)
    direction_final = [dict(locked_direction)] if locked_direction else []
    direction_exploratory = [dict(row, claim_label="exploratory_not_claimable" if row.get("candidate_id") != locked_direction.get("candidate_id") else row.get("claim_label")) for row in sorted([row for row in direction_rows if row["status"] == "ok"], key=lambda row: as_float(row.get("final_accuracy")), reverse=True)]
    price_leaderboard = sorted([row for row in price_rows if row["status"] == "ok"], key=lambda row: as_float(row.get("validation_rmse")))
    price_final = [dict(locked_price)] if locked_price else []
    price_exploratory = [dict(row, claim_label="exploratory_not_claimable" if row.get("candidate_id") != locked_price.get("candidate_id") else row.get("claim_label")) for row in sorted([row for row in price_rows if row["status"] == "ok"], key=lambda row: as_float(row.get("final_rmse")))]

    direction_columns = list(direction_rows[0].keys()) if direction_rows else []
    price_columns = list(price_rows[0].keys()) if price_rows else []
    write_frame(OUTPUT_DIR / "split_guard_audit.csv", split_guard_rows, list(split_guard_rows[0].keys()) if split_guard_rows else [])
    write_frame(OUTPUT_DIR / "target_variant_audit.csv", target_audit_rows, list(target_audit_rows[0].keys()) if target_audit_rows else [])
    write_frame(OUTPUT_DIR / "direction_validation_results.csv", direction_rows, direction_columns)
    write_frame(OUTPUT_DIR / "direction_validation_leaderboard.csv", direction_leaderboard, direction_columns)
    write_frame(OUTPUT_DIR / "direction_final_results.csv", direction_final, direction_columns)
    write_frame(OUTPUT_DIR / "direction_exploratory_final_leaderboard.csv", direction_exploratory, direction_columns)
    write_frame(OUTPUT_DIR / "price_validation_results.csv", price_rows, price_columns)
    write_frame(OUTPUT_DIR / "price_validation_leaderboard.csv", price_leaderboard, price_columns)
    write_frame(OUTPUT_DIR / "price_final_results.csv", price_final, price_columns)
    write_frame(OUTPUT_DIR / "price_exploratory_final_leaderboard.csv", price_exploratory, price_columns)
    write_frame(OUTPUT_DIR / "baseline_comparison.csv", baseline_rows, list(baseline_rows[0].keys()) if baseline_rows else [])
    write_frame(OUTPUT_DIR / "ticker_stability.csv", ticker_rows, list(ticker_rows[0].keys()) if ticker_rows else [])
    write_frame(OUTPUT_DIR / "quarter_stability.csv", quarter_rows, list(quarter_rows[0].keys()) if quarter_rows else [])
    write_frame(OUTPUT_DIR / "skipped_models.csv", skipped_rows, list(skipped_rows[0].keys()) if skipped_rows else [])
    runtime_rows = [
        {"phase": "total", "runtime_seconds": time.perf_counter() - started, "direction_candidates_evaluated": evaluated_direction, "price_candidates_evaluated": evaluated_price, "skipped_rows": len(skipped_rows)}
    ]
    write_frame(OUTPUT_DIR / "runtime_summary.csv", runtime_rows, list(runtime_rows[0].keys()))
    manifest = {
        "run_id": "vn30_model_universe_direction_price",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "mode": config.mode,
        "scope": "VN30 stock hourly forecasting only",
        "diagnostic_only": True,
        "direction_candidates_evaluated": evaluated_direction,
        "price_candidates_evaluated": evaluated_price,
        "model_families_evaluated": int(len({row["model_family"] for row in direction_rows + price_rows if row.get("status") == "ok"})),
        "locked_direction": locked_direction,
        "locked_price": locked_price,
        "dependency_status": dependency,
        "runtime_seconds": time.perf_counter() - started,
        "no_vn100": True,
        "no_index_as_stock": True,
        "paper_docx_generated": False,
        "trading_claim": False,
    }
    write_json(OUTPUT_DIR / "model_universe_manifest.json", manifest)
    write_reports(direction_rows, price_rows, skipped_rows, manifest)
    print(json.dumps(json_safe({"status": "ok", "manifest": "reports/generated/vn30_model_universe_direction_price/model_universe_manifest.json", "direction_candidates": evaluated_direction, "price_candidates": evaluated_price, "model_families": manifest["model_families_evaluated"]}), indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 model-universe direction and price/return diagnostics.")
    parser.add_argument("--smoke", action="store_true", help="Run small smoke coverage.")
    parser.add_argument("--validation-screening", action="store_true", help="Run bounded validation screening.")
    parser.add_argument("--promotion-relock", action="store_true", help="Run V2 promotion relock audit for high-performing exploratory final rows.")
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.promotion_relock:
        run_promotion_relock(max(1, int(args.timeout_seconds)))
        return
    if args.smoke:
        config = RunConfig("smoke", max(1, int(args.timeout_seconds)), 500, 250, 250, 80, 80)
    else:
        config = RunConfig("validation_screening", max(1, int(args.timeout_seconds)), 1600, 700, 700, 650, 650)
    run_benchmark(config)


if __name__ == "__main__":
    main()
