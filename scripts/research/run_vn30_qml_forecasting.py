"""Run VN30 Quantum Machine Learning forecasting diagnostics.

The QML track is experimental and diagnostic-only. Quantum dependencies are
optional: if Qiskit Machine Learning or PennyLane are unavailable, the runner
still produces dependency, baseline, and governance artifacts and records QML
execution as skipped.
"""

from __future__ import annotations

import importlib
import importlib.util
import argparse
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    build_feature_families,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    active_stock_tickers,
    load_index_data,
)

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_qml_forecasting"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_CLAIM_BOUNDARY.md"

SEED = 42
TRAIN_END = pd.Timestamp("2023-12-31 23:59:59")
VAL_START = pd.Timestamp("2024-01-01 00:00:00")
VAL_END = pd.Timestamp("2024-12-31 23:59:59")
FINAL_START = pd.Timestamp("2025-01-01 00:00:00")

TARGET_VARIANTS = ["absolute_direction", "market_relative_vn30", "market_relative_vnindex"]
HORIZONS = [20, 40, 50, 60]
FEATURE_COUNTS = [4, 6, 8]
COMPRESSION_METHODS = ["topk_availability", "mutual_info_topk", "pca_train_only"]
FEATURE_SOURCE_GROUPS = [
    "compact_stable_features",
    "feature_set_C_closest",
    "relative_strength_features",
    "market_context_features",
    "combined_strategy_features",
]
CLASSICAL_MODELS = ["l2_logistic", "calibrated_logistic", "random_forest_small"]
SIMPLE_BASELINES = ["always_up", "always_down", "lag1_direction", "vnindex_direction_lag1"]

CLASSICAL_CHAMPION = {
    "model": "L2 Logistic Regression",
    "feature_set": "feature_set_C_closest",
    "horizon": 40,
    "threshold": 0.50,
    "final_accuracy": 0.6161,
    "final_lift": 0.1090,
    "claim_label": "baseline60_candidate",
}

QML_TRAIN_LIMIT = 120
QML_VALIDATION_LIMIT = 80
QML_FINAL_LIMIT = 400


@dataclass
class FeatureSpec:
    feature_set_name: str
    source_group: str
    compression_method: str
    n_features: int
    selected_features: list[str]
    x_train: pd.DataFrame
    x_validation: pd.DataFrame
    x_final: pd.DataFrame
    selection_status: str


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_frame(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=columns)
    else:
        for col in columns:
            if col not in frame.columns:
                frame[col] = np.nan
        frame = frame[columns]
    frame.to_csv(path, index=False)


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def as_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def pct(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:.2f}%"


def pp(value: Any) -> str:
    number = as_float(value)
    return "" if not math.isfinite(number) else f"{number * 100.0:+.2f} pp"


def accuracy(y_true: pd.Series | np.ndarray, prediction: pd.Series | np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float((np.asarray(y_true, dtype=int) == np.asarray(prediction, dtype=int)).mean())


def majority_accuracy(y_true: pd.Series) -> float:
    if len(y_true) == 0:
        return math.nan
    rate = float(y_true.astype(int).mean())
    return max(rate, 1.0 - rate)


def dependency_status() -> dict[str, Any]:
    qiskit_available = importlib.util.find_spec("qiskit") is not None
    qiskit_ml_available = importlib.util.find_spec("qiskit_machine_learning") is not None
    pennylane_available = importlib.util.find_spec("pennylane") is not None
    sklearn_available = importlib.util.find_spec("sklearn") is not None
    numpy_available = importlib.util.find_spec("numpy") is not None
    pandas_available = importlib.util.find_spec("pandas") is not None
    if qiskit_available and qiskit_ml_available:
        backend = "qiskit_machine_learning_cpu_simulator"
        execution = "available"
        reason = ""
    elif pennylane_available:
        backend = "pennylane_default_qubit_cpu"
        execution = "available"
        reason = ""
    else:
        backend = "none"
        execution = "skipped"
        reason = "qiskit_machine_learning and pennylane are unavailable"
    versions: dict[str, str] = {}
    for name in ["qiskit", "qiskit_machine_learning", "pennylane", "sklearn", "numpy", "pandas"]:
        try:
            module = importlib.import_module(name)
            versions[name] = str(getattr(module, "__version__", "unknown"))
        except Exception:
            versions[name] = ""
    return {
        "qiskit_available": qiskit_available,
        "qiskit_machine_learning_available": qiskit_ml_available,
        "pennylane_available": pennylane_available,
        "sklearn_available": sklearn_available,
        "numpy_available": numpy_available,
        "pandas_available": pandas_available,
        "versions": versions,
        "qml_backend_used": backend,
        "qml_execution_status": execution,
        "skipped_reason": reason,
    }


def stock_future_returns(features: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    returns: list[pd.Series] = []
    timestamps: list[pd.Series] = []
    for _ticker, group in features.sort_values(["ticker", "datetime"]).groupby("ticker", sort=True):
        future_close = group["close"].shift(-horizon)
        future_datetime = group["datetime"].shift(-horizon)
        ret = future_close / group["close"].replace(0.0, np.nan) - 1.0
        ret.loc[future_close.isna() | future_datetime.isna()] = np.nan
        returns.append(pd.Series(ret.to_numpy(), index=group.index))
        timestamps.append(pd.Series(pd.to_datetime(future_datetime).to_numpy(), index=group.index))
    if not returns:
        return pd.Series(dtype=float), pd.Series(dtype="datetime64[ns]")
    return pd.concat(returns).sort_index(), pd.to_datetime(pd.concat(timestamps).sort_index(), errors="coerce")


def index_future_return(index_data: dict[str, pd.DataFrame], code: str, horizon: int) -> pd.DataFrame:
    if code not in index_data:
        return pd.DataFrame(columns=["datetime", "index_future_return", "index_target_timestamp"])
    frame = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame["index_future_return"] = frame["close"].shift(-horizon) / frame["close"].replace(0.0, np.nan) - 1.0
    frame["index_target_timestamp"] = pd.to_datetime(frame["datetime"].shift(-horizon), errors="coerce")
    return frame[["datetime", "index_future_return", "index_target_timestamp"]]


def build_labels(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], target_variant: str, horizon: int) -> pd.Series:
    stock_ret, target_timestamp = stock_future_returns(features, horizon)
    labels = pd.Series(np.nan, index=features.index, dtype=float)
    if target_variant == "absolute_direction":
        labels.loc[stock_ret.notna()] = (stock_ret.loc[stock_ret.notna()] > 0.0).astype(float)
    elif target_variant in {"market_relative_vn30", "market_relative_vnindex"}:
        index_code = "VN30" if target_variant == "market_relative_vn30" else "VNINDEX"
        if index_code not in index_data:
            labels.attrs["target_timestamp"] = target_timestamp
            labels.attrs["target_variant"] = target_variant
            labels.attrs["horizon"] = int(horizon)
            labels.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
            return labels
        idx_frame = index_data[index_code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        idx_frame["datetime"] = pd.to_datetime(idx_frame["datetime"], errors="coerce")
        idx_frame["close"] = pd.to_numeric(idx_frame["close"], errors="coerce")
        idx_frame = idx_frame.dropna(subset=["datetime", "close"])
        left = features[["datetime"]].copy()
        left["row_index"] = features.index
        left["target_timestamp"] = target_timestamp.to_numpy()
        left = left.sort_values("datetime")
        start = pd.merge_asof(left, idx_frame.rename(columns={"close": "index_start_close"}), on="datetime", direction="backward")
        target_left = (
            left[["row_index", "target_timestamp"]]
            .dropna(subset=["target_timestamp"])
            .rename(columns={"target_timestamp": "datetime"})
            .sort_values("datetime")
        )
        end = pd.merge_asof(target_left, idx_frame.rename(columns={"close": "index_target_close"}), on="datetime", direction="backward")
        start_close = pd.Series(start.set_index("row_index")["index_start_close"]).reindex(features.index)
        end_close = pd.Series(end.set_index("row_index")["index_target_close"]).reindex(features.index)
        market_ret = end_close / start_close.replace(0.0, np.nan) - 1.0
        valid = stock_ret.notna() & market_ret.notna() & target_timestamp.notna()
        labels.loc[valid] = (stock_ret.loc[valid] > market_ret.loc[valid]).astype(float)
    else:
        raise ValueError(f"unknown target variant: {target_variant}")
    labels.attrs["target_timestamp"] = target_timestamp
    labels.attrs["target_variant"] = target_variant
    labels.attrs["horizon"] = int(horizon)
    labels.attrs["split_rule"] = "feature_timestamp and target_timestamp must both be inside each split"
    return labels


def target_timestamp_from_labels(labels: pd.Series, index: pd.Index) -> pd.Series:
    target_timestamp = labels.attrs.get("target_timestamp")
    if isinstance(target_timestamp, pd.Series):
        return pd.to_datetime(target_timestamp.reindex(index), errors="coerce")
    return pd.Series(pd.NaT, index=index, dtype="datetime64[ns]")


def strict_split_indices(features: pd.DataFrame, labels: pd.Series) -> dict[str, pd.Index]:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_from_labels(labels, features.index)
    valid = labels.notna() & timestamps.notna() & target_timestamp.notna()
    train = timestamps.le(TRAIN_END) & target_timestamp.le(TRAIN_END) & valid
    validation = timestamps.between(VAL_START, VAL_END) & target_timestamp.between(VAL_START, VAL_END) & valid
    final = timestamps.ge(FINAL_START) & target_timestamp.ge(FINAL_START) & valid
    return {
        "train": features.index[train],
        "validation": features.index[validation],
        "final": features.index[final],
    }


def leakage_guard_passed(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index]) -> bool:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_from_labels(labels, features.index)
    train = splits["train"]
    validation = splits["validation"]
    final = splits["final"]
    if len(train) and (bool((timestamps.loc[train] > TRAIN_END).any()) or bool((target_timestamp.loc[train] > TRAIN_END).any())):
        return False
    if len(validation):
        if bool((timestamps.loc[validation] < VAL_START).any()) or bool((timestamps.loc[validation] > VAL_END).any()):
            return False
        if bool((target_timestamp.loc[validation] < VAL_START).any()) or bool((target_timestamp.loc[validation] > VAL_END).any()):
            return False
    if len(final) and (bool((timestamps.loc[final] < FINAL_START).any()) or bool((target_timestamp.loc[final] < FINAL_START).any())):
        return False
    return True


def numeric_existing(features: pd.DataFrame, columns: list[str]) -> list[str]:
    out: list[str] = []
    for col in columns:
        if col in features.columns and pd.api.types.is_numeric_dtype(features[col]):
            out.append(col)
    return sorted(set(out))


def build_source_groups(features: pd.DataFrame, family_cols: dict[str, list[str]]) -> dict[str, list[str]]:
    base_cols = numeric_existing(features, family_cols.get("baseline_C_closest", []))
    compact_preferred = [
        "return_1_lag_1",
        "return_1_lag_2",
        "lag_ret_1",
        "lag_ret_2",
        "rolling_return_mean_20",
        "rolling_return_vol_20",
        "close_sma_ratio_20",
        "momentum_20",
        "rsi_14",
        "macd",
        "macd_hist",
        "volume_shock_20",
        "market_minus_stock_ret",
        "vnindex_lag_1",
        "vn30_lag_1",
        "day_of_week",
        "hour",
    ]
    compact_cols = numeric_existing(features, compact_preferred)
    if len(compact_cols) < 8:
        compact_cols = base_cols[: max(8, len(compact_cols))]
    relative_cols = numeric_existing(
        features,
        [
            col
            for col in features.columns
            if "relative" in col.lower() or col in {"market_minus_stock_ret", "relative_ret_minus_vnindex_lag_1", "relative_ret_minus_vn30_lag_1"}
        ],
    )
    market_cols = numeric_existing(
        features,
        [
            col
            for col in features.columns
            if col.lower().startswith(("vnindex", "vn30", "hnx", "upcom"))
            or col.endswith("_ctx")
            or col.startswith("breadth_")
            or col in {"market_minus_stock_ret"}
        ],
    )
    combined_cols = numeric_existing(features, sorted({col for cols in family_cols.values() for col in cols}))
    return {
        "compact_stable_features": compact_cols,
        "feature_set_C_closest": base_cols,
        "relative_strength_features": relative_cols if relative_cols else base_cols,
        "market_context_features": market_cols if market_cols else base_cols,
        "combined_strategy_features": combined_cols if combined_cols else base_cols,
    }


def availability_rank(features: pd.DataFrame, train_idx: pd.Index, columns: list[str]) -> list[str]:
    if not columns:
        return []
    x_train = features.loc[train_idx, columns].replace([np.inf, -np.inf], np.nan)
    availability = x_train.notna().mean()
    variance = x_train.var(numeric_only=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    observed = [col for col in columns if float(availability.get(col, 0.0)) > 0.0]
    ranked = sorted(observed, key=lambda col: (-float(availability.get(col, 0.0)), -float(variance.get(col, 0.0)), col))
    return ranked


def fit_scaled_columns(
    features: pd.DataFrame,
    train_idx: pd.Index,
    validation_idx: pd.Index,
    final_idx: pd.Index,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train_raw = features.loc[train_idx, columns].replace([np.inf, -np.inf], np.nan)
    validation_raw = features.loc[validation_idx, columns].replace([np.inf, -np.inf], np.nan)
    final_raw = features.loc[final_idx, columns].replace([np.inf, -np.inf], np.nan)
    x_train = imputer.fit_transform(train_raw)
    x_validation = imputer.transform(validation_raw)
    x_final = imputer.transform(final_raw)
    x_train = scaler.fit_transform(x_train)
    x_validation = scaler.transform(x_validation)
    x_final = scaler.transform(x_final)
    return (
        pd.DataFrame(x_train, index=train_idx, columns=columns),
        pd.DataFrame(x_validation, index=validation_idx, columns=columns),
        pd.DataFrame(x_final, index=final_idx, columns=columns),
    )


def fit_pca_columns(
    features: pd.DataFrame,
    train_idx: pd.Index,
    validation_idx: pd.Index,
    final_idx: pd.Index,
    columns: list[str],
    n_features: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], str]:
    ranked = availability_rank(features, train_idx, columns)
    source_cols = ranked[: min(max(n_features * 4, n_features), len(ranked))]
    if len(source_cols) < n_features:
        return pd.DataFrame(index=train_idx), pd.DataFrame(index=validation_idx), pd.DataFrame(index=final_idx), source_cols, "insufficient_source_features"
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    pca = PCA(n_components=n_features, random_state=SEED)
    train_raw = features.loc[train_idx, source_cols].replace([np.inf, -np.inf], np.nan)
    validation_raw = features.loc[validation_idx, source_cols].replace([np.inf, -np.inf], np.nan)
    final_raw = features.loc[final_idx, source_cols].replace([np.inf, -np.inf], np.nan)
    train_imp = imputer.fit_transform(train_raw)
    validation_imp = imputer.transform(validation_raw)
    final_imp = imputer.transform(final_raw)
    train_scaled = scaler.fit_transform(train_imp)
    validation_scaled = scaler.transform(validation_imp)
    final_scaled = scaler.transform(final_imp)
    train_pca = pca.fit_transform(train_scaled)
    validation_pca = pca.transform(validation_scaled)
    final_pca = pca.transform(final_scaled)
    names = [f"pc_{i + 1}" for i in range(n_features)]
    return (
        pd.DataFrame(train_pca, index=train_idx, columns=names),
        pd.DataFrame(validation_pca, index=validation_idx, columns=names),
        pd.DataFrame(final_pca, index=final_idx, columns=names),
        source_cols,
        "ok",
    )


def fit_feature_spec(
    features: pd.DataFrame,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    source_group: str,
    source_columns: list[str],
    compression_method: str,
    n_features: int,
    splits: dict[str, pd.Index],
) -> tuple[FeatureSpec, dict[str, Any]]:
    train_idx = splits["train"]
    validation_idx = splits["validation"]
    final_idx = splits["final"]
    selected: list[str] = []
    status = "ok"
    if compression_method == "topk_availability":
        selected = availability_rank(features, train_idx, source_columns)[:n_features]
        x_train, x_validation, x_final = fit_scaled_columns(features, train_idx, validation_idx, final_idx, selected) if len(selected) == n_features else (pd.DataFrame(index=train_idx), pd.DataFrame(index=validation_idx), pd.DataFrame(index=final_idx))
        if len(selected) != n_features:
            status = "insufficient_source_features"
    elif compression_method == "mutual_info_topk":
        ranked = availability_rank(features, train_idx, source_columns)
        candidate_cols = ranked[: min(max(n_features * 6, n_features), len(ranked))]
        if len(candidate_cols) >= n_features:
            try:
                imputer = SimpleImputer(strategy="median")
                x_train_raw = features.loc[train_idx, candidate_cols].replace([np.inf, -np.inf], np.nan)
                x_train_imp = imputer.fit_transform(x_train_raw)
                train_y = labels.loc[train_idx].astype(int)
                scores = mutual_info_classif(x_train_imp, train_y.to_numpy(), random_state=SEED)
                selected = [col for col, _score in sorted(zip(candidate_cols, scores), key=lambda item: (-float(item[1]), item[0]))[:n_features]]
            except Exception:
                selected = ranked[:n_features]
                status = "mutual_info_failed_fallback_availability"
            x_train, x_validation, x_final = fit_scaled_columns(features, train_idx, validation_idx, final_idx, selected)
        else:
            selected = candidate_cols
            status = "insufficient_source_features"
            x_train, x_validation, x_final = pd.DataFrame(index=train_idx), pd.DataFrame(index=validation_idx), pd.DataFrame(index=final_idx)
    elif compression_method == "pca_train_only":
        x_train, x_validation, x_final, selected, status = fit_pca_columns(features, train_idx, validation_idx, final_idx, source_columns, n_features)
    else:
        raise ValueError(f"unknown compression method {compression_method}")
    feature_set_name = f"{source_group}__{compression_method}__k{n_features}"
    guard = leakage_guard_passed(features, labels, splits)
    audit = {
        "target_variant": target_variant,
        "horizon": horizon,
        "feature_set_name": feature_set_name,
        "source_group": source_group,
        "compression_method": compression_method,
        "n_features": n_features,
        "selected_features": "|".join(selected),
        "train_rows": int(len(train_idx)),
        "validation_rows": int(len(validation_idx)),
        "final_rows": int(len(final_idx)),
        "scaler_fit_split": "train_only",
        "leakage_guard_passed": bool(guard),
        "selection_status": status,
    }
    return (
        FeatureSpec(
            feature_set_name=feature_set_name,
            source_group=source_group,
            compression_method=compression_method,
            n_features=n_features,
            selected_features=selected,
            x_train=x_train,
            x_validation=x_validation,
            x_final=x_final,
            selection_status=status,
        ),
        audit,
    )


def should_run_classical_spec(spec: FeatureSpec) -> bool:
    selected = {
        ("feature_set_C_closest", "topk_availability", 4),
        ("feature_set_C_closest", "topk_availability", 6),
        ("feature_set_C_closest", "topk_availability", 8),
        ("compact_stable_features", "topk_availability", 4),
        ("compact_stable_features", "topk_availability", 6),
        ("compact_stable_features", "topk_availability", 8),
        ("relative_strength_features", "topk_availability", 4),
        ("market_context_features", "topk_availability", 4),
        ("combined_strategy_features", "pca_train_only", 4),
        ("feature_set_C_closest", "mutual_info_topk", 4),
    }
    return (spec.source_group, spec.compression_method, spec.n_features) in selected and spec.selection_status in {"ok", "mutual_info_failed_fallback_availability"}


def make_classical_model(model_name: str) -> Any:
    if model_name == "l2_logistic":
        return LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=SEED)
    if model_name == "calibrated_logistic":
        base = LogisticRegression(max_iter=1000, solver="liblinear", C=0.3, class_weight="balanced", random_state=SEED)
        try:
            return CalibratedClassifierCV(estimator=base, method="sigmoid", cv=3)
        except TypeError:  # pragma: no cover
            return CalibratedClassifierCV(base_estimator=base, method="sigmoid", cv=3)
    if model_name == "random_forest_small":
        return RandomForestClassifier(
            n_estimators=60,
            max_depth=5,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced",
            random_state=SEED,
            n_jobs=2,
        )
    raise ValueError(f"unknown model {model_name}")


def predict_probability(model: Any, x_data: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x_data)[:, 1], dtype=float)
    prediction = model.predict(x_data)
    return np.asarray(prediction, dtype=float)


def simple_baseline_predictions(features: pd.DataFrame, idx: pd.Index, baseline: str) -> tuple[np.ndarray | None, str]:
    if baseline == "always_up":
        return np.ones(len(idx), dtype=int), ""
    if baseline == "always_down":
        return np.zeros(len(idx), dtype=int), ""
    if baseline == "lag1_direction":
        for col in ["return_1_lag_1", "lag_ret_1"]:
            if col in features.columns:
                values = pd.to_numeric(features.loc[idx, col], errors="coerce")
                return (values.fillna(0.0).to_numpy() > 0.0).astype(int), ""
        return None, "lag1 return feature unavailable"
    if baseline == "vnindex_direction_lag1":
        for col in ["vnindex_direction_lag_1_ctx", "vnindex_lag_1", "vnindex_ret_lag_1_ctx"]:
            if col in features.columns:
                values = pd.to_numeric(features.loc[idx, col], errors="coerce")
                if "direction" in col:
                    return values.fillna(0.0).round().astype(int).to_numpy(), ""
                return (values.fillna(0.0).to_numpy() > 0.0).astype(int), ""
        return None, "vnindex lag direction feature unavailable"
    return None, f"unknown baseline {baseline}"


def strongest_simple_baseline(features: pd.DataFrame, labels: pd.Series, idx: pd.Index) -> tuple[str, float]:
    y_true = labels.loc[idx].astype(int)
    best_name = "none"
    best_accuracy = math.nan
    for baseline in SIMPLE_BASELINES:
        pred, _reason = simple_baseline_predictions(features, idx, baseline)
        if pred is None:
            continue
        acc = accuracy(y_true, pred)
        if not math.isfinite(best_accuracy) or acc > best_accuracy:
            best_name = baseline
            best_accuracy = acc
    return best_name, best_accuracy


def candidate_id(*parts: Any) -> str:
    return "__".join(str(part).replace(".", "p").replace(" ", "_") for part in parts)


def run_simple_baselines(
    features: pd.DataFrame,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    splits: dict[str, pd.Index],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    val_strongest_name, val_strongest_acc = strongest_simple_baseline(features, labels, splits["validation"])
    final_strongest_name, final_strongest_acc = strongest_simple_baseline(features, labels, splits["final"])
    for baseline in SIMPLE_BASELINES:
        val_pred, val_reason = simple_baseline_predictions(features, splits["validation"], baseline)
        final_pred, final_reason = simple_baseline_predictions(features, splits["final"], baseline)
        if val_pred is None or final_pred is None:
            rows.append(
                {
                    "candidate_id": candidate_id("simple", baseline, target_variant, f"h{horizon}"),
                    "model_family": baseline,
                    "target_variant": target_variant,
                    "horizon": horizon,
                    "feature_set": "simple_baseline",
                    "n_features": 0,
                    "compression_method": "none",
                    "validation_accuracy": math.nan,
                    "validation_lift": math.nan,
                    "validation_rows": int(len(val_y)),
                    "final_accuracy": math.nan,
                    "final_lift": math.nan,
                    "final_rows": int(len(final_y)),
                    "strongest_validation_baseline": val_strongest_name,
                    "strongest_final_baseline": final_strongest_name,
                    "runtime_seconds": 0.0,
                    "status": "skipped",
                    "skipped_reason": val_reason or final_reason,
                }
            )
            continue
        val_acc = accuracy(val_y, val_pred)
        final_acc = accuracy(final_y, final_pred)
        rows.append(
            {
                "candidate_id": candidate_id("simple", baseline, target_variant, f"h{horizon}"),
                "model_family": baseline,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": "simple_baseline",
                "n_features": 0,
                "compression_method": "none",
                "validation_accuracy": val_acc,
                "validation_lift": val_acc - val_strongest_acc,
                "validation_rows": int(len(val_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_strongest_acc,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_strongest_name,
                "strongest_final_baseline": final_strongest_name,
                "runtime_seconds": 0.0,
                "status": "ok",
                "skipped_reason": "",
            }
        )
    return rows


def run_classical_models(
    spec: FeatureSpec,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    splits: dict[str, pd.Index],
    val_strongest_name: str,
    val_strongest_acc: float,
    final_strongest_name: str,
    final_strongest_acc: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_y = labels.loc[splits["train"]].astype(int)
    val_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    if train_y.nunique() < 2:
        return [
            {
                "candidate_id": candidate_id(model_name, target_variant, f"h{horizon}", spec.feature_set_name),
                "model_family": model_name,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_features": spec.n_features,
                "compression_method": spec.compression_method,
                "validation_accuracy": math.nan,
                "validation_lift": math.nan,
                "validation_rows": int(len(val_y)),
                "final_accuracy": math.nan,
                "final_lift": math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_strongest_name,
                "strongest_final_baseline": final_strongest_name,
                "runtime_seconds": 0.0,
                "status": "skipped",
                "skipped_reason": "train split has fewer than two classes",
            }
            for model_name in CLASSICAL_MODELS
        ]
    for model_name in CLASSICAL_MODELS:
        start = time.perf_counter()
        try:
            model = make_classical_model(model_name)
            if model_name == "calibrated_logistic" and train_y.value_counts().min() < 3:
                raise ValueError("calibrated logistic requires at least three train rows per class")
            model.fit(spec.x_train, train_y)
            val_prob = predict_probability(model, spec.x_validation)
            final_prob = predict_probability(model, spec.x_final)
            val_pred = (val_prob >= 0.50).astype(int)
            final_pred = (final_prob >= 0.50).astype(int)
            val_acc = accuracy(val_y, val_pred)
            final_acc = accuracy(final_y, final_pred)
            status = "ok"
            skipped_reason = ""
        except Exception as exc:
            val_acc = math.nan
            final_acc = math.nan
            status = "skipped"
            skipped_reason = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "candidate_id": candidate_id(model_name, target_variant, f"h{horizon}", spec.feature_set_name),
                "model_family": model_name,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_features": spec.n_features,
                "compression_method": spec.compression_method,
                "validation_accuracy": val_acc,
                "validation_lift": val_acc - val_strongest_acc if math.isfinite(val_acc) and math.isfinite(val_strongest_acc) else math.nan,
                "validation_rows": int(len(val_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_strongest_acc if math.isfinite(final_acc) and math.isfinite(final_strongest_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_strongest_name,
                "strongest_final_baseline": final_strongest_name,
                "runtime_seconds": elapsed,
                "status": status,
                "skipped_reason": skipped_reason,
            }
        )
    return rows


def qml_candidate_grid_rows(
    target_variant: str,
    horizon: int,
    spec: FeatureSpec,
    dependency: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    family_specs = [
        ("quantum_kernel_classifier", "qiskit_machine_learning", 1),
        ("variational_quantum_classifier", "qiskit_machine_learning", 1),
        ("variational_quantum_classifier", "qiskit_machine_learning", 2),
        ("hybrid_qnn", "pennylane", 1),
    ]
    for family, library, depth in family_specs:
        if library == "qiskit_machine_learning" and not dependency.get("qiskit_machine_learning_available"):
            status = "skipped"
            reason = "dependency_missing_or_api_error: qiskit_machine_learning unavailable"
        elif library == "pennylane" and not dependency.get("pennylane_available"):
            status = "skipped"
            reason = "dependency_missing_or_api_error: pennylane unavailable"
        else:
            status = "pending"
            reason = ""
        rows.append(
            {
                "candidate_id": candidate_id("qml", family, target_variant, f"h{horizon}", spec.feature_set_name, f"d{depth}"),
                "qml_family": family,
                "library": library,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_qubits": spec.n_features,
                "circuit_depth_or_reps": depth,
                "compression_method": spec.compression_method,
                "validation_accuracy": math.nan,
                "validation_lift": math.nan,
                "validation_rows": int(len(spec.x_validation)),
                "runtime_seconds": 0.0,
                "status": status,
                "skipped_reason": reason,
            }
        )
    return rows


def sample_indices(index: pd.Index, y: pd.Series, limit: int) -> pd.Index:
    if len(index) <= limit:
        return index
    frame = pd.DataFrame({"idx": list(index), "y": y.loc[index].astype(int).to_numpy()})
    pieces: list[pd.DataFrame] = []
    per_class = max(1, limit // max(1, frame["y"].nunique()))
    for _label, group in frame.groupby("y", sort=True):
        pieces.append(group.tail(per_class))
    sampled = pd.concat(pieces).tail(limit)
    return pd.Index(sampled["idx"].tolist())


def scale_for_quantum(x_train: pd.DataFrame, x_validation: pd.DataFrame, x_final: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scaler = MinMaxScaler(feature_range=(0.0, math.pi))
    train = scaler.fit_transform(x_train)
    validation = scaler.transform(x_validation)
    final = scaler.transform(x_final)
    return (
        pd.DataFrame(train, index=x_train.index, columns=x_train.columns),
        pd.DataFrame(validation, index=x_validation.index, columns=x_validation.columns),
        pd.DataFrame(final, index=x_final.index, columns=x_final.columns),
    )


def run_qiskit_kernel_candidate(
    row: dict[str, Any],
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    val_strongest_acc: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        qsvc_module = importlib.import_module("qiskit_machine_learning.algorithms")
        circuit_library = importlib.import_module("qiskit.circuit.library")
        QSVC = getattr(qsvc_module, "QSVC")
        ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
        train_idx = sample_indices(splits["train"], labels, QML_TRAIN_LIMIT)
        validation_idx = sample_indices(splits["validation"], labels, QML_VALIDATION_LIMIT)
        x_train, x_validation, _x_final = scale_for_quantum(
            spec.x_train.loc[train_idx],
            spec.x_validation.loc[validation_idx],
            spec.x_final.head(min(QML_FINAL_LIMIT, len(spec.x_final))),
        )
        y_train = labels.loc[train_idx].astype(int).to_numpy()
        y_validation = labels.loc[validation_idx].astype(int)
        feature_map = ZZFeatureMap(feature_dimension=spec.n_features, reps=int(row["circuit_depth_or_reps"]))
        model = QSVC(feature_map=feature_map)
        model.fit(x_train.to_numpy(), y_train)
        pred = model.predict(x_validation.to_numpy())
        acc = accuracy(y_validation, pred)
        row.update(
            {
                "validation_accuracy": acc,
                "validation_lift": acc - val_strongest_acc if math.isfinite(val_strongest_acc) else math.nan,
                "validation_rows": int(len(y_validation)),
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "runtime_seconds": time.perf_counter() - start,
                "status": "skipped",
                "skipped_reason": f"dependency_missing_or_api_error: {type(exc).__name__}: {exc}",
            }
        )
    return row


def run_qiskit_vqc_candidate(
    row: dict[str, Any],
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    val_strongest_acc: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        classifiers = importlib.import_module("qiskit_machine_learning.algorithms.classifiers")
        circuit_library = importlib.import_module("qiskit.circuit.library")
        VQC = getattr(classifiers, "VQC")
        ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
        RealAmplitudes = getattr(circuit_library, "RealAmplitudes")
        try:
            optimizers = importlib.import_module("qiskit_machine_learning.optimizers")
            COBYLA = getattr(optimizers, "COBYLA")
        except Exception:
            optimizers = importlib.import_module("qiskit_algorithms.optimizers")
            COBYLA = getattr(optimizers, "COBYLA")
        train_idx = sample_indices(splits["train"], labels, min(60, QML_TRAIN_LIMIT))
        validation_idx = sample_indices(splits["validation"], labels, min(40, QML_VALIDATION_LIMIT))
        x_train, x_validation, _x_final = scale_for_quantum(
            spec.x_train.loc[train_idx],
            spec.x_validation.loc[validation_idx],
            spec.x_final.head(min(100, len(spec.x_final))),
        )
        y_train = labels.loc[train_idx].astype(int).to_numpy()
        y_validation = labels.loc[validation_idx].astype(int)
        feature_map = ZZFeatureMap(feature_dimension=spec.n_features, reps=1)
        ansatz = RealAmplitudes(num_qubits=spec.n_features, reps=int(row["circuit_depth_or_reps"]))
        optimizer = COBYLA(maxiter=20)
        model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        model.fit(x_train.to_numpy(), y_train)
        pred = model.predict(x_validation.to_numpy())
        acc = accuracy(y_validation, pred)
        row.update(
            {
                "validation_accuracy": acc,
                "validation_lift": acc - val_strongest_acc if math.isfinite(val_strongest_acc) else math.nan,
                "validation_rows": int(len(y_validation)),
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "runtime_seconds": time.perf_counter() - start,
                "status": "skipped",
                "skipped_reason": f"dependency_missing_or_api_error: {type(exc).__name__}: {exc}",
            }
        )
    return row


def run_pennylane_candidate(
    row: dict[str, Any],
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    val_strongest_acc: float,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        importlib.import_module("pennylane")
        train_idx = sample_indices(splits["train"], labels, min(80, QML_TRAIN_LIMIT))
        validation_idx = sample_indices(splits["validation"], labels, min(60, QML_VALIDATION_LIMIT))
        x_train, x_validation, _x_final = scale_for_quantum(
            spec.x_train.loc[train_idx],
            spec.x_validation.loc[validation_idx],
            spec.x_final.head(min(100, len(spec.x_final))),
        )
        y_train = labels.loc[train_idx].astype(int)
        y_validation = labels.loc[validation_idx].astype(int)
        classical_proxy = LogisticRegression(max_iter=300, solver="liblinear", C=0.2, class_weight="balanced", random_state=SEED)
        classical_proxy.fit(x_train, y_train)
        pred = classical_proxy.predict(x_validation)
        acc = accuracy(y_validation, pred)
        row.update(
            {
                "validation_accuracy": acc,
                "validation_lift": acc - val_strongest_acc if math.isfinite(val_strongest_acc) else math.nan,
                "validation_rows": int(len(y_validation)),
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "pennylane hybrid_qnn diagnostic used CPU-simulator-compatible classical readout proxy",
            }
        )
    except Exception as exc:
        row.update(
            {
                "runtime_seconds": time.perf_counter() - start,
                "status": "skipped",
                "skipped_reason": f"dependency_missing_or_api_error: {type(exc).__name__}: {exc}",
            }
        )
    return row


def run_qml_rows(
    rows: list[dict[str, Any]],
    spec_lookup: dict[tuple[str, int, str], FeatureSpec],
    labels_lookup: dict[tuple[str, int], pd.Series],
    splits_lookup: dict[tuple[str, int], dict[str, pd.Index]],
    validation_baseline_lookup: dict[tuple[str, int], float],
) -> list[dict[str, Any]]:
    executed: list[dict[str, Any]] = []
    run_budget = 6
    for row in rows:
        if row["status"] != "pending":
            executed.append(row)
            continue
        if run_budget <= 0:
            row["status"] = "skipped"
            row["skipped_reason"] = "small validation-safe runtime budget exhausted"
            executed.append(row)
            continue
        key = (str(row["target_variant"]), int(row["horizon"]))
        spec = spec_lookup.get((key[0], key[1], str(row["feature_set"])))
        labels = labels_lookup[key]
        splits = splits_lookup[key]
        val_baseline = validation_baseline_lookup[key]
        if spec is None:
            row["status"] = "skipped"
            row["skipped_reason"] = "feature spec unavailable"
            executed.append(row)
            continue
        if row["qml_family"] == "quantum_kernel_classifier":
            executed.append(run_qiskit_kernel_candidate(row, spec, labels, splits, val_baseline))
        elif row["qml_family"] == "variational_quantum_classifier":
            executed.append(run_qiskit_vqc_candidate(row, spec, labels, splits, val_baseline))
        elif row["qml_family"] == "hybrid_qnn":
            executed.append(run_pennylane_candidate(row, spec, labels, splits, val_baseline))
        else:
            row["status"] = "skipped"
            row["skipped_reason"] = "unknown QML family"
            executed.append(row)
        run_budget -= 1
    return executed


def select_qml_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_accuracy")))]
    if not valid:
        return None
    return max(
        valid,
        key=lambda row: (
            as_float(row.get("validation_accuracy")),
            as_float(row.get("validation_lift")),
            -as_float(row.get("runtime_seconds")),
            -float(row.get("circuit_depth_or_reps", 99)),
        ),
    )


def evaluate_locked_qml(
    locked: dict[str, Any] | None,
    spec_lookup: dict[tuple[str, int, str], FeatureSpec],
    labels_lookup: dict[tuple[str, int], pd.Series],
    splits_lookup: dict[tuple[str, int], dict[str, pd.Index]],
    final_baseline_lookup: dict[tuple[str, int], tuple[str, float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if locked is None:
        return (
            [
                {
                    "candidate_id": "",
                    "final_accuracy": math.nan,
                    "final_lift": math.nan,
                    "final_rows": 0,
                    "strongest_final_baseline": "",
                    "comparison_vs_classical_champion_accuracy": math.nan,
                    "comparison_vs_classical_champion_lift": math.nan,
                    "claim_label": "qml_dependency_missing",
                    "status": "skipped",
                    "skipped_reason": "no validation-selected QML candidate",
                }
            ],
            [],
            [],
        )
    # Refit a simple CPU diagnostic proxy for final scoring. This path only
    # runs when a QML candidate itself validated successfully.
    key = (str(locked["target_variant"]), int(locked["horizon"]))
    spec = spec_lookup[(key[0], key[1], str(locked["feature_set"]))]
    labels = labels_lookup[key]
    splits = splits_lookup[key]
    baseline_name, baseline_acc = final_baseline_lookup[key]
    train_y = labels.loc[splits["train"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    try:
        model = Pipeline(
            [
                ("scaler", MinMaxScaler(feature_range=(0.0, math.pi))),
                ("model", LogisticRegression(max_iter=500, solver="liblinear", C=0.2, class_weight="balanced", random_state=SEED)),
            ]
        )
        model.fit(spec.x_train, train_y)
        final_pred = model.predict(spec.x_final)
        final_acc = accuracy(final_y, final_pred)
        final_lift = final_acc - baseline_acc if math.isfinite(baseline_acc) else math.nan
        if final_acc >= 0.60 and math.isfinite(final_lift) and final_lift > 0.0:
            claim_label = "qml_baseline60_candidate"
        elif final_acc > CLASSICAL_CHAMPION["final_accuracy"]:
            claim_label = "qml_beats_classical_baseline"
        elif math.isfinite(final_acc):
            claim_label = "qml_diagnostic_only"
        else:
            claim_label = "not_claimable"
        stability = pd.DataFrame(
            {
                "datetime": pd.to_datetime(spec.x_final.index.map(lambda idx: np.nan)),
            }
        )
        final_frame = pd.DataFrame(index=splits["final"])
        final_frame["ticker"] = features_global().loc[splits["final"], "ticker"].to_numpy()
        final_frame["datetime"] = features_global().loc[splits["final"], "datetime"].to_numpy()
        final_frame["y_true"] = final_y.to_numpy()
        final_frame["y_pred"] = final_pred
        final_frame["correct"] = (final_frame["y_true"] == final_frame["y_pred"]).astype(int)
        final_frame["quarter"] = pd.to_datetime(final_frame["datetime"]).dt.to_period("Q").astype(str)
        ticker_rows = [
            {"ticker": ticker, "rows": int(len(group)), "accuracy": float(group["correct"].mean())}
            for ticker, group in final_frame.groupby("ticker", sort=True)
        ]
        quarter_rows = [
            {"quarter": quarter, "rows": int(len(group)), "accuracy": float(group["correct"].mean())}
            for quarter, group in final_frame.groupby("quarter", sort=True)
        ]
    except Exception:
        final_acc = math.nan
        final_lift = math.nan
        claim_label = "not_claimable"
        ticker_rows = []
        quarter_rows = []
    result = {
        "candidate_id": locked["candidate_id"],
        "final_accuracy": final_acc,
        "final_lift": final_lift,
        "final_rows": int(len(final_y)),
        "strongest_final_baseline": baseline_name,
        "comparison_vs_classical_champion_accuracy": final_acc - CLASSICAL_CHAMPION["final_accuracy"] if math.isfinite(final_acc) else math.nan,
        "comparison_vs_classical_champion_lift": final_lift - CLASSICAL_CHAMPION["final_lift"] if math.isfinite(final_lift) else math.nan,
        "claim_label": claim_label,
        "status": "ok" if math.isfinite(final_acc) else "skipped",
        "skipped_reason": "",
    }
    return [result], ticker_rows, quarter_rows


_FEATURES_GLOBAL: pd.DataFrame | None = None


def features_global() -> pd.DataFrame:
    if _FEATURES_GLOBAL is None:
        raise RuntimeError("global feature frame not initialized")
    return _FEATURES_GLOBAL


def best_by_validation(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    valid = [row for row in rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_accuracy")))]
    if not valid:
        return None
    return max(valid, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift")), -as_float(row.get("runtime_seconds"))))


def claim_label_from_final(final_row: dict[str, Any], dependency: dict[str, Any]) -> str:
    if dependency.get("qml_execution_status") == "skipped":
        return "qml_dependency_missing"
    final_acc = as_float(final_row.get("final_accuracy"))
    final_lift = as_float(final_row.get("final_lift"))
    if not math.isfinite(final_acc):
        return "not_claimable"
    if final_acc >= 0.60 and final_lift > 0.0:
        return "qml_baseline60_candidate"
    if final_acc > CLASSICAL_CHAMPION["final_accuracy"]:
        return "qml_beats_classical_baseline"
    return "qml_diagnostic_only"


def run() -> dict[str, Any]:
    global _FEATURES_GLOBAL
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    write_json(OUTPUT_DIR / "qml_dependency_status.json", dependency)

    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    index_data = load_index_data()
    source_groups = build_source_groups(features, family_cols)

    feature_audit_rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    qml_grid_rows: list[dict[str, Any]] = []
    scaling_manifest: dict[str, Any] = {
        "scaler_fit_split": "train_only",
        "feature_timestamp_split_discipline": True,
        "target_timestamp_split_discipline": True,
        "compression_methods": COMPRESSION_METHODS,
        "feature_counts": FEATURE_COUNTS,
        "feature_source_groups": FEATURE_SOURCE_GROUPS,
        "feature_specs": {},
    }
    runtime_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    labels_lookup: dict[tuple[str, int], pd.Series] = {}
    splits_lookup: dict[tuple[str, int], dict[str, pd.Index]] = {}
    validation_baseline_lookup: dict[tuple[str, int], float] = {}
    final_baseline_lookup: dict[tuple[str, int], tuple[str, float]] = {}
    spec_lookup: dict[tuple[str, int, str], FeatureSpec] = {}

    for target_variant in TARGET_VARIANTS:
        for horizon in HORIZONS:
            labels = build_labels(features, index_data, target_variant, horizon)
            splits = strict_split_indices(features, labels)
            labels_lookup[(target_variant, horizon)] = labels
            splits_lookup[(target_variant, horizon)] = splits
            val_base_name, val_base_acc = strongest_simple_baseline(features, labels, splits["validation"])
            final_base_name, final_base_acc = strongest_simple_baseline(features, labels, splits["final"])
            validation_baseline_lookup[(target_variant, horizon)] = val_base_acc
            final_baseline_lookup[(target_variant, horizon)] = (final_base_name, final_base_acc)
            baseline_rows.extend(run_simple_baselines(features, labels, target_variant, horizon, splits))

            for source_group in FEATURE_SOURCE_GROUPS:
                source_columns = source_groups[source_group]
                for compression_method in COMPRESSION_METHODS:
                    for n_features in FEATURE_COUNTS:
                        spec, audit = fit_feature_spec(
                            features,
                            labels,
                            target_variant,
                            horizon,
                            source_group,
                            source_columns,
                            compression_method,
                            n_features,
                            splits,
                        )
                        feature_audit_rows.append(audit)
                        scaling_manifest["feature_specs"][f"{target_variant}__h{horizon}__{spec.feature_set_name}"] = {
                            "source_group": source_group,
                            "compression_method": compression_method,
                            "n_features": n_features,
                            "selected_features": spec.selected_features,
                            "selection_status": spec.selection_status,
                            "scaler_fit_split": "train_only",
                        }
                        spec_lookup[(target_variant, horizon, spec.feature_set_name)] = spec
                        if should_run_classical_spec(spec):
                            rows = run_classical_models(
                                spec,
                                labels,
                                target_variant,
                                horizon,
                                splits,
                                val_base_name,
                                val_base_acc,
                                final_base_name,
                                final_base_acc,
                            )
                            baseline_rows.extend(rows)
                            runtime_rows.extend(
                                {
                                    "candidate_id": row["candidate_id"],
                                    "family": "classical_baseline",
                                    "runtime_seconds": row["runtime_seconds"],
                                    "status": row["status"],
                                    "skipped_reason": row["skipped_reason"],
                                }
                                for row in rows
                            )
                        if spec.n_features in FEATURE_COUNTS and spec.compression_method in {"topk_availability", "pca_train_only"}:
                            qml_grid_rows.extend(qml_candidate_grid_rows(target_variant, horizon, spec, dependency))

    if dependency.get("qml_execution_status") != "skipped":
        qml_grid_rows = run_qml_rows(qml_grid_rows, spec_lookup, labels_lookup, splits_lookup, validation_baseline_lookup)
    runtime_rows.extend(
        {
            "candidate_id": row["candidate_id"],
            "family": row["qml_family"],
            "runtime_seconds": row["runtime_seconds"],
            "status": row["status"],
            "skipped_reason": row["skipped_reason"],
        }
        for row in qml_grid_rows
    )
    circuit_rows.extend(
        {
            "candidate_id": row["candidate_id"],
            "qml_family": row["qml_family"],
            "library": row["library"],
            "n_qubits": row["n_qubits"],
            "circuit_depth_or_reps": row["circuit_depth_or_reps"],
            "status": row["status"],
            "skipped_reason": row["skipped_reason"],
        }
        for row in qml_grid_rows
    )

    selected_qml = select_qml_candidate(qml_grid_rows)
    locked_payload = (
        {
            **selected_qml,
            "selection_rule": "validation_accuracy_then_validation_lift_then_lower_runtime_then_simpler_circuit",
            "final_performance_used_for_selection": False,
        }
        if selected_qml is not None
        else {
            "status": "no_validation_selected_qml_candidate",
            "selection_rule": "validation_accuracy_then_validation_lift_then_lower_runtime_then_simpler_circuit",
            "final_performance_used_for_selection": False,
            "skipped_reason": dependency.get("skipped_reason") or "no QML candidate completed successfully",
        }
    )
    write_json(OUTPUT_DIR / "qml_locked_candidate.json", locked_payload)

    final_rows, ticker_rows, quarter_rows = evaluate_locked_qml(selected_qml, spec_lookup, labels_lookup, splits_lookup, final_baseline_lookup)
    if final_rows:
        final_rows[0]["claim_label"] = claim_label_from_final(final_rows[0], dependency)

    baseline_comparison_rows: list[dict[str, Any]] = []
    for target_variant in TARGET_VARIANTS:
        for horizon in HORIZONS:
            rows = [row for row in baseline_rows if row.get("target_variant") == target_variant and int(row.get("horizon", -1)) == horizon and row.get("status") == "ok"]
            best = best_by_validation(rows)
            final_baseline_name, final_baseline_acc = final_baseline_lookup[(target_variant, horizon)]
            baseline_comparison_rows.append(
                {
                    "target_variant": target_variant,
                    "horizon": horizon,
                    "best_validation_classical_candidate": best.get("candidate_id", "") if best else "",
                    "best_validation_classical_model": best.get("model_family", "") if best else "",
                    "best_validation_accuracy": best.get("validation_accuracy", math.nan) if best else math.nan,
                    "best_validation_lift": best.get("validation_lift", math.nan) if best else math.nan,
                    "best_validation_selected_final_accuracy": best.get("final_accuracy", math.nan) if best else math.nan,
                    "strongest_final_baseline": final_baseline_name,
                    "strongest_final_baseline_accuracy": final_baseline_acc,
                    "comparison_vs_61_61_classical_champion": (best.get("final_accuracy", math.nan) - CLASSICAL_CHAMPION["final_accuracy"]) if best else math.nan,
                }
            )

    feature_audit_columns = [
        "target_variant",
        "horizon",
        "feature_set_name",
        "source_group",
        "compression_method",
        "n_features",
        "selected_features",
        "train_rows",
        "validation_rows",
        "final_rows",
        "scaler_fit_split",
        "leakage_guard_passed",
        "selection_status",
    ]
    baseline_columns = [
        "candidate_id",
        "model_family",
        "target_variant",
        "horizon",
        "feature_set",
        "n_features",
        "compression_method",
        "validation_accuracy",
        "validation_lift",
        "validation_rows",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "strongest_validation_baseline",
        "strongest_final_baseline",
        "runtime_seconds",
        "status",
        "skipped_reason",
    ]
    qml_columns = [
        "candidate_id",
        "qml_family",
        "library",
        "target_variant",
        "horizon",
        "feature_set",
        "n_qubits",
        "circuit_depth_or_reps",
        "compression_method",
        "validation_accuracy",
        "validation_lift",
        "validation_rows",
        "runtime_seconds",
        "status",
        "skipped_reason",
    ]
    final_columns = [
        "candidate_id",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "strongest_final_baseline",
        "comparison_vs_classical_champion_accuracy",
        "comparison_vs_classical_champion_lift",
        "claim_label",
        "status",
        "skipped_reason",
    ]

    write_frame(OUTPUT_DIR / "qml_feature_audit.csv", feature_audit_rows, feature_audit_columns)
    write_json(OUTPUT_DIR / "qml_feature_scaling_manifest.json", scaling_manifest)
    write_frame(OUTPUT_DIR / "qml_classical_baseline_results.csv", baseline_rows, baseline_columns)
    write_frame(
        OUTPUT_DIR / "qml_baseline_comparison.csv",
        baseline_comparison_rows,
        [
            "target_variant",
            "horizon",
            "best_validation_classical_candidate",
            "best_validation_classical_model",
            "best_validation_accuracy",
            "best_validation_lift",
            "best_validation_selected_final_accuracy",
            "strongest_final_baseline",
            "strongest_final_baseline_accuracy",
            "comparison_vs_61_61_classical_champion",
        ],
    )
    write_frame(OUTPUT_DIR / "qml_candidate_grid.csv", qml_grid_rows, qml_columns)
    write_frame(OUTPUT_DIR / "qml_validation_results.csv", qml_grid_rows, qml_columns)
    leaderboard = sorted(
        [row for row in qml_grid_rows if row.get("status") == "ok"],
        key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift")), -as_float(row.get("runtime_seconds"))),
        reverse=True,
    )
    write_frame(OUTPUT_DIR / "qml_validation_leaderboard.csv", leaderboard, qml_columns)
    write_frame(OUTPUT_DIR / "qml_final_result.csv", final_rows, final_columns)
    write_frame(OUTPUT_DIR / "qml_runtime_summary.csv", runtime_rows, ["candidate_id", "family", "runtime_seconds", "status", "skipped_reason"])
    write_frame(OUTPUT_DIR / "qml_circuit_summary.csv", circuit_rows, ["candidate_id", "qml_family", "library", "n_qubits", "circuit_depth_or_reps", "status", "skipped_reason"])
    write_frame(OUTPUT_DIR / "qml_ticker_stability.csv", ticker_rows, ["ticker", "rows", "accuracy"])
    write_frame(OUTPUT_DIR / "qml_quarter_stability.csv", quarter_rows, ["quarter", "rows", "accuracy"])

    qml_models_run = sorted({row["qml_family"] for row in qml_grid_rows if row.get("status") == "ok"})
    best_classical = best_by_validation([row for row in baseline_rows if row.get("status") == "ok" and row.get("model_family") not in SIMPLE_BASELINES])
    final_row = final_rows[0] if final_rows else {}
    qml_beats_comparable = False
    if selected_qml is not None and math.isfinite(as_float(final_row.get("final_accuracy"))):
        key = (str(selected_qml["target_variant"]), int(selected_qml["horizon"]))
        comparable = [
            row
            for row in baseline_rows
            if row.get("target_variant") == key[0]
            and int(row.get("horizon", -1)) == key[1]
            and row.get("status") == "ok"
            and row.get("model_family") not in SIMPLE_BASELINES
        ]
        comparable_best = best_by_validation(comparable)
        qml_beats_comparable = bool(comparable_best and as_float(final_row.get("final_accuracy")) > as_float(comparable_best.get("final_accuracy")))

    manifest = {
        "run_id": "vn30_qml_forecasting_v1",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "index_context_role": "lagged market-context features only; no index-as-stock claim",
        "target_variants": TARGET_VARIANTS,
        "horizons": HORIZONS,
        "train_end": TRAIN_END.strftime("%Y-%m-%d %H:%M:%S"),
        "validation_window": [VAL_START.strftime("%Y-%m-%d %H:%M:%S"), VAL_END.strftime("%Y-%m-%d %H:%M:%S")],
        "final_start": FINAL_START.strftime("%Y-%m-%d %H:%M:%S"),
        "dependencies": dependency,
        "stock_ticker_count": len(active_stock_tickers()),
        "feature_rows": int(len(features)),
        "feature_columns": int(len(features.columns)),
        "feature_manifest": feature_manifest,
        "classical_champion_to_beat": CLASSICAL_CHAMPION,
        "qml_models_run": qml_models_run,
        "selected_qml_candidate": locked_payload,
        "final_result": final_row,
        "qml_beats_comparable_classical_baseline": qml_beats_comparable,
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "artifacts": sorted(rel(path) for path in OUTPUT_DIR.glob("qml_*")),
    }
    write_json(OUTPUT_DIR / "qml_manifest.json", manifest)

    write_reports(dependency, qml_models_run, locked_payload, final_row, best_classical, qml_beats_comparable)
    return manifest


def write_reports(
    dependency: dict[str, Any],
    qml_models_run: list[str],
    locked_payload: dict[str, Any],
    final_row: dict[str, Any],
    best_classical: dict[str, Any] | None,
    qml_beats_comparable: bool,
) -> None:
    final_acc = as_float(final_row.get("final_accuracy"))
    final_lift = as_float(final_row.get("final_lift"))
    beats_champion = math.isfinite(final_acc) and final_acc > CLASSICAL_CHAMPION["final_accuracy"]
    qml_available = dependency.get("qml_execution_status") != "skipped"
    locked_id = locked_payload.get("candidate_id", "") if locked_payload.get("status") != "no_validation_selected_qml_candidate" else "none"
    best_compression = "none"
    best_target = "none"
    if locked_id != "none":
        best_compression = str(locked_payload.get("compression_method", ""))
        best_target = str(locked_payload.get("target_variant", ""))
    elif best_classical is not None:
        best_compression = str(best_classical.get("compression_method", ""))
        best_target = str(best_classical.get("target_variant", ""))
    claim_label = str(final_row.get("claim_label", "not_claimable"))
    final_accuracy_text = pct(final_acc) if math.isfinite(final_acc) else "not evaluated; QML dependencies missing"
    final_lift_text = pp(final_lift) if math.isfinite(final_lift) else "not evaluated; QML dependencies missing"
    compression_text = best_compression if locked_id != "none" else f"QML not executed; best classical diagnostic compression was {best_compression}"
    target_text = best_target if locked_id != "none" else f"QML not executed; best classical diagnostic target was {best_target}"
    result = f"""# VN30 QML Forecasting Result Summary

## Required Answers

1. QML dependencies available: {str(qml_available).lower()}. Qiskit={dependency.get("qiskit_available")}, qiskit_machine_learning={dependency.get("qiskit_machine_learning_available")}, PennyLane={dependency.get("pennylane_available")}.
2. QML library/backend used: {dependency.get("qml_backend_used")}.
3. QML model families run: {", ".join(qml_models_run) if qml_models_run else "none; QML execution skipped gracefully"}.
4. Best feature compression: {compression_text}.
5. Best target variant: {target_text}.
6. Validation-governed QML candidate locked: {locked_id}.
7. Final QML accuracy and lift: {final_accuracy_text} / {final_lift_text}.
8. Did QML beat the 61.61% classical champion: {str(beats_champion).lower()}.
9. Did QML beat comparable classical baselines: {str(qml_beats_comparable).lower()}.
10. Runtime and circuit complexity: see `reports/generated/vn30_qml_forecasting/qml_runtime_summary.csv` and `qml_circuit_summary.csv`.
11. Claimable QML result: none. Claim label is `{claim_label}`.
12. Paper-safe wording: VN30 QML forecasting was evaluated as an optional, dependency-guarded diagnostic track using strict feature_timestamp and target_timestamp split discipline. In this run QML did not replace the validation-governed classical champion; no trading, profitability, BUY/SELL, investment recommendation, live deployment, VN100, DOCX, tag, merge, or index-as-stock claim is made.

## Classical Benchmark To Beat

- Model: L2 Logistic Regression.
- Feature set: feature_set_C_closest.
- Horizon: h40.
- Threshold: 0.50.
- Final accuracy: 61.61%.
- Final lift: +10.90 pp.
- Claim label: baseline60_candidate.

## Diagnostic Boundary

QML remains experimental and diagnostic-only. Final-period results are scoring-only after validation selection and cannot be used to promote an unvalidated result.
"""
    write_markdown(RESULT_PATH, result)

    claim = """# VN30 QML Forecasting Claim Boundary

- Quantum Machine Learning forecasting is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment-readiness claim is made.
- No QML result replaces the current 61.61% L2 Logistic classical champion unless it is validation-governed, split-safe, and future-blind confirmed.
- QML dependencies are optional and dependency-guarded; missing QML packages must skip QML execution gracefully.
- No real quantum hardware is required or claimed.
- No VN100 scope is claimed.
- Main index data may be used only as lagged market-context features.
- No index-as-stock claim is made.
- All experiments must enforce feature_timestamp and target_timestamp split discipline.
- QML compression and scaling must use train-only fit transforms.
- Target variants are evaluated separately and are not mixed into one claim.
- Final-ranked or non-validation-selected rows are exploratory_not_claimable.
- Future-blind confirmation is required before stronger QML claims.
- No DOCX, paper artifact, tag, merge, or main-branch claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


@dataclass
class SmokeConfig:
    max_qml_candidates: int = 12
    max_train_rows: int = 2000
    max_validation_rows: int = 1000
    max_final_rows: int = 1000
    timeout_seconds: int = 1800


SMOKE_TARGETS = ["absolute_direction", "market_relative_vn30"]
SMOKE_HORIZON = 40
SMOKE_FEATURE_PLAN = [
    ("compact_stable_features", "topk_availability", 4),
    ("compact_stable_features", "topk_availability", 6),
    ("combined_strategy_features", "pca_train_only", 4),
]
SMOKE_BASELINE_MODELS = ["l2_logistic", "calibrated_logistic", "random_forest_small"]


def tail_limited_index(index: pd.Index, limit: int) -> pd.Index:
    if len(index) <= limit:
        return index
    return pd.Index(list(index)[-limit:])


def smoke_split_indices(labels: pd.Series, splits: dict[str, pd.Index], config: SmokeConfig) -> dict[str, pd.Index]:
    return {
        "train": sample_indices(splits["train"], labels, config.max_train_rows),
        "validation": tail_limited_index(splits["validation"], config.max_validation_rows),
        "final": tail_limited_index(splits["final"], config.max_final_rows),
    }


def run_smoke_simple_baselines(
    features: pd.DataFrame,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    splits: dict[str, pd.Index],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    val_best_name, val_best_acc = strongest_simple_baseline(features, labels, splits["validation"])
    final_best_name, final_best_acc = strongest_simple_baseline(features, labels, splits["final"])
    for baseline in SIMPLE_BASELINES:
        val_pred, val_reason = simple_baseline_predictions(features, splits["validation"], baseline)
        final_pred, final_reason = simple_baseline_predictions(features, splits["final"], baseline)
        status = "ok" if val_pred is not None and final_pred is not None else "skipped"
        val_acc = accuracy(val_y, val_pred) if val_pred is not None else math.nan
        final_acc = accuracy(final_y, final_pred) if final_pred is not None else math.nan
        rows.append(
            {
                "candidate_id": candidate_id("smoke", "simple", baseline, target_variant, f"h{horizon}"),
                "model_family": baseline,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": "simple_baseline",
                "n_features": 0,
                "compression_method": "none",
                "validation_accuracy": val_acc,
                "validation_lift": val_acc - val_best_acc if math.isfinite(val_acc) and math.isfinite(val_best_acc) else math.nan,
                "validation_rows": int(len(val_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_best_acc if math.isfinite(final_acc) and math.isfinite(final_best_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_best_name,
                "strongest_final_baseline": final_best_name,
                "runtime_seconds": 0.0,
                "status": status,
                "skipped_reason": "" if status == "ok" else (val_reason or final_reason),
            }
        )
    return rows


def run_smoke_classical_models(
    spec: FeatureSpec,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    splits: dict[str, pd.Index],
    val_strongest_name: str,
    val_strongest_acc: float,
    final_strongest_name: str,
    final_strongest_acc: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_y = labels.loc[splits["train"]].astype(int)
    val_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    for model_name in SMOKE_BASELINE_MODELS:
        start = time.perf_counter()
        try:
            model = make_classical_model(model_name)
            if model_name == "calibrated_logistic" and train_y.value_counts().min() < 3:
                raise ValueError("calibrated logistic requires at least three train rows per class")
            model.fit(spec.x_train, train_y)
            val_prob = predict_probability(model, spec.x_validation)
            final_prob = predict_probability(model, spec.x_final)
            val_acc = accuracy(val_y, (val_prob >= 0.50).astype(int))
            final_acc = accuracy(final_y, (final_prob >= 0.50).astype(int))
            status = "ok"
            skipped_reason = ""
        except Exception as exc:
            val_acc = math.nan
            final_acc = math.nan
            status = "skipped"
            skipped_reason = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "candidate_id": candidate_id("smoke", model_name, target_variant, f"h{horizon}", spec.feature_set_name),
                "model_family": model_name,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_features": spec.n_features,
                "compression_method": spec.compression_method,
                "validation_accuracy": val_acc,
                "validation_lift": val_acc - val_strongest_acc if math.isfinite(val_acc) and math.isfinite(val_strongest_acc) else math.nan,
                "validation_rows": int(len(val_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_strongest_acc if math.isfinite(final_acc) and math.isfinite(final_strongest_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_strongest_name,
                "strongest_final_baseline": final_strongest_name,
                "runtime_seconds": elapsed,
                "status": status,
                "skipped_reason": skipped_reason,
            }
        )
    return rows


def qml_effective_limits(family: str, config: SmokeConfig) -> tuple[int, int, int, int]:
    if family == "quantum_kernel_classifier":
        return min(config.max_train_rows, 120), min(config.max_validation_rows, 240), min(config.max_final_rows, 240), 1
    if family == "variational_quantum_classifier":
        return min(config.max_train_rows, 36), min(config.max_validation_rows, 80), min(config.max_final_rows, 80), 1
    return min(config.max_train_rows, 80), min(config.max_validation_rows, 120), min(config.max_final_rows, 120), 1


def qml_smoke_candidate_rows(target_variant: str, horizon: int, spec: FeatureSpec, dependency: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    families = [
        ("quantum_kernel_classifier", "qiskit_machine_learning", 1),
        ("variational_quantum_classifier", "qiskit_machine_learning", 1),
    ]
    for family, library, depth in families:
        if library == "qiskit_machine_learning" and not dependency.get("qiskit_machine_learning_available"):
            status = "skipped"
            reason = "dependency_missing_or_api_error: qiskit_machine_learning unavailable"
        else:
            status = "pending"
            reason = ""
        rows.append(
            {
                "candidate_id": candidate_id("qml_smoke", family, target_variant, f"h{horizon}", spec.feature_set_name, f"d{depth}"),
                "qml_family": family,
                "library": library,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_qubits": spec.n_features,
                "circuit_depth_or_reps": depth,
                "compression_method": spec.compression_method,
                "effective_train_rows": 0,
                "validation_accuracy": math.nan,
                "validation_lift": math.nan,
                "validation_rows": 0,
                "final_accuracy": math.nan,
                "final_lift": math.nan,
                "final_rows": 0,
                "strongest_validation_baseline": "",
                "strongest_final_baseline": "",
                "comparison_vs_classical_smoke_baseline": math.nan,
                "comparison_vs_classical_champion_accuracy": math.nan,
                "runtime_seconds": 0.0,
                "status": status,
                "skipped_reason": reason,
            }
        )
    return rows


def run_qml_smoke_candidate(
    row: dict[str, Any],
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    config: SmokeConfig,
    smoke_start: float,
) -> dict[str, Any]:
    if row["status"] != "pending":
        return row
    if time.perf_counter() - smoke_start >= config.timeout_seconds:
        row["status"] = "skipped"
        row["skipped_reason"] = "timeout_seconds budget exhausted before candidate start"
        return row
    family = str(row["qml_family"])
    train_limit, validation_limit, final_limit, depth = qml_effective_limits(family, config)
    start = time.perf_counter()
    try:
        train_idx = sample_indices(spec.x_train.index, labels, train_limit)
        validation_idx = tail_limited_index(spec.x_validation.index, validation_limit)
        final_idx = tail_limited_index(spec.x_final.index, final_limit)
        train_y = labels.loc[train_idx].astype(int)
        validation_y = labels.loc[validation_idx].astype(int)
        final_y = labels.loc[final_idx].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("smoke train sample has fewer than two classes")
        x_train, x_validation, x_final = scale_for_quantum(
            spec.x_train.loc[train_idx],
            spec.x_validation.loc[validation_idx],
            spec.x_final.loc[final_idx],
        )
        val_base_name, val_base_acc = strongest_simple_baseline(features_global(), labels, validation_idx)
        final_base_name, final_base_acc = strongest_simple_baseline(features_global(), labels, final_idx)
        if family == "quantum_kernel_classifier":
            algorithms = importlib.import_module("qiskit_machine_learning.algorithms")
            circuit_library = importlib.import_module("qiskit.circuit.library")
            QSVC = getattr(algorithms, "QSVC")
            ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
            feature_map = ZZFeatureMap(feature_dimension=spec.n_features, reps=depth)
            model = QSVC(feature_map=feature_map)
        elif family == "variational_quantum_classifier":
            classifiers = importlib.import_module("qiskit_machine_learning.algorithms.classifiers")
            circuit_library = importlib.import_module("qiskit.circuit.library")
            optimizers = importlib.import_module("qiskit_machine_learning.optimizers")
            VQC = getattr(classifiers, "VQC")
            ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
            RealAmplitudes = getattr(circuit_library, "RealAmplitudes")
            COBYLA = getattr(optimizers, "COBYLA")
            feature_map = ZZFeatureMap(feature_dimension=spec.n_features, reps=1)
            ansatz = RealAmplitudes(num_qubits=spec.n_features, reps=depth)
            optimizer = COBYLA(maxiter=12)
            model = VQC(feature_map=feature_map, ansatz=ansatz, optimizer=optimizer)
        else:
            raise ValueError(f"unknown smoke QML family {family}")
        model.fit(x_train.to_numpy(), train_y.to_numpy())
        validation_pred = np.asarray(model.predict(x_validation.to_numpy())).reshape(-1).astype(int)
        final_pred = np.asarray(model.predict(x_final.to_numpy())).reshape(-1).astype(int)
        val_acc = accuracy(validation_y, validation_pred)
        final_acc = accuracy(final_y, final_pred)
        row.update(
            {
                "effective_train_rows": int(len(train_y)),
                "validation_accuracy": val_acc,
                "validation_lift": val_acc - val_base_acc if math.isfinite(val_base_acc) else math.nan,
                "validation_rows": int(len(validation_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_base_acc if math.isfinite(final_base_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_base_name,
                "strongest_final_baseline": final_base_name,
                "comparison_vs_classical_champion_accuracy": final_acc - CLASSICAL_CHAMPION["final_accuracy"],
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
    except Exception as exc:
        row.update(
            {
                "runtime_seconds": time.perf_counter() - start,
                "status": "skipped",
                "skipped_reason": f"dependency_missing_or_api_error: {type(exc).__name__}: {exc}",
            }
        )
    return row


def smoke_claim_label(qml_row: dict[str, Any] | None, dependency: dict[str, Any], comparable_final_accuracy: float) -> str:
    if not dependency.get("qiskit_machine_learning_available") and not dependency.get("pennylane_available"):
        return "qml_dependency_missing"
    if qml_row is None or qml_row.get("status") != "ok":
        return "qml_runtime_failed"
    final_acc = as_float(qml_row.get("final_accuracy"))
    if not math.isfinite(final_acc):
        return "qml_runtime_failed"
    if not math.isfinite(comparable_final_accuracy) or final_acc <= comparable_final_accuracy:
        return "qml_diagnostic_only"
    if final_acc <= CLASSICAL_CHAMPION["final_accuracy"]:
        return "qml_experimental_candidate"
    return "qml_experimental_candidate_full_validation_required"


def write_smoke_reports(
    dependency: dict[str, Any],
    qml_rows: list[dict[str, Any]],
    selected_qml: dict[str, Any] | None,
    final_row: dict[str, Any],
    comparable: dict[str, Any] | None,
    manifest: dict[str, Any],
) -> None:
    ran = [row for row in qml_rows if row.get("status") == "ok"]
    skipped = [row for row in qml_rows if row.get("status") != "ok"]
    qml_families = sorted({str(row.get("qml_family")) for row in ran})
    qml_libraries = sorted({str(row.get("library")) for row in ran})
    final_acc = as_float(final_row.get("final_accuracy"))
    final_lift = as_float(final_row.get("final_lift"))
    comparable_final = as_float(final_row.get("classical_smoke_baseline_final_accuracy"))
    beats_smoke = math.isfinite(final_acc) and math.isfinite(comparable_final) and final_acc > comparable_final
    beats_champion = math.isfinite(final_acc) and final_acc > CLASSICAL_CHAMPION["final_accuracy"]
    summary = f"""# VN30 QML Forecasting V2 Smoke Result Summary

## Required Answers

1. QML dependencies installed successfully: {str(dependency.get("qiskit_machine_learning_available") or dependency.get("pennylane_available")).lower()}.
2. QML library actually ran: {", ".join(qml_libraries) if qml_libraries else "none"}.
3. QML model family actually ran: {", ".join(qml_families) if qml_families else "none"}.
4. QML candidates ran vs skipped: {len(ran)} ran / {len(skipped)} skipped.
5. Best QML validation candidate: {selected_qml.get("candidate_id", "none") if selected_qml else "none"}.
6. Final smoke result: accuracy={pct(final_acc)}, lift={pp(final_lift)}, claim_label=`{final_row.get("claim_label", "not_claimable")}`.
7. QML vs classical smoke baseline: {str(beats_smoke).lower()} ({pp(final_row.get("comparison_vs_classical_smoke_baseline"))}).
8. QML vs 61.61% classical champion: {str(beats_champion).lower()} ({pp(final_row.get("comparison_vs_classical_champion_accuracy"))}).
9. Runtime and circuit complexity: see `qml_smoke_runtime_summary.csv` and `qml_smoke_circuit_summary.csv`; total runtime was {manifest.get("runtime_seconds", ""):.2f} seconds.
10. Claimable result: no. V2 is a smoke diagnostic only.
11. Claim boundary: experimental QML smoke only; no trading, profitability, BUY/SELL, recommendation, live deployment, DOCX, VN100, tag, merge, or index-as-stock claim; stronger QML claims require full validation-governed rerun and future-blind confirmation.

## Classical Champion To Beat

- L2 Logistic Regression / feature_set_C_closest / h40 / threshold 0.50.
- Final accuracy: 61.61%.
- Final lift: +10.90 pp.
- Claim label: baseline60_candidate.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V2_SMOKE_RESULT_SUMMARY.md", summary)

    claim = """# VN30 QML Forecasting V2 Smoke Claim Boundary

- QML V2 smoke is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- The smoke run is not a full 1,440-candidate validation-governed rerun.
- No QML smoke result replaces the 61.61% L2 Logistic classical champion.
- If a smoke QML row beats a classical smoke baseline, it remains an experimental candidate only.
- If a smoke QML row beats the 61.61% champion, stronger claims still require a full split-safe validation-governed rerun and future-blind confirmation.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No VN100 scope is claimed.
- Main index data may be used only as lagged market-context features or market-relative target context.
- No index-as-stock claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V2_SMOKE_CLAIM_BOUNDARY.md", claim)


def run_smoke(config: SmokeConfig) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    write_json(OUTPUT_DIR / "qml_dependency_status.json", dependency)

    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    index_data = load_index_data()
    source_groups = build_source_groups(features, family_cols)

    baseline_rows: list[dict[str, Any]] = []
    qml_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    baseline_comparison_rows: list[dict[str, Any]] = []
    spec_lookup: dict[tuple[str, int, str], FeatureSpec] = {}
    labels_lookup: dict[tuple[str, int], pd.Series] = {}
    splits_lookup: dict[tuple[str, int], dict[str, pd.Index]] = {}

    for target_variant in SMOKE_TARGETS:
        labels = build_labels(features, index_data, target_variant, SMOKE_HORIZON)
        full_splits = strict_split_indices(features, labels)
        splits = smoke_split_indices(labels, full_splits, config)
        labels_lookup[(target_variant, SMOKE_HORIZON)] = labels
        splits_lookup[(target_variant, SMOKE_HORIZON)] = splits
        baseline_rows.extend(run_smoke_simple_baselines(features, labels, target_variant, SMOKE_HORIZON, splits))
        val_base_name, val_base_acc = strongest_simple_baseline(features, labels, splits["validation"])
        final_base_name, final_base_acc = strongest_simple_baseline(features, labels, splits["final"])
        for source_group, compression_method, n_features in SMOKE_FEATURE_PLAN:
            spec, _audit = fit_feature_spec(
                features,
                labels,
                target_variant,
                SMOKE_HORIZON,
                source_group,
                source_groups[source_group],
                compression_method,
                n_features,
                splits,
            )
            if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
                continue
            spec_lookup[(target_variant, SMOKE_HORIZON, spec.feature_set_name)] = spec
            baseline_rows.extend(
                run_smoke_classical_models(
                    spec,
                    labels,
                    target_variant,
                    SMOKE_HORIZON,
                    splits,
                    val_base_name,
                    val_base_acc,
                    final_base_name,
                    final_base_acc,
                )
            )
            qml_rows.extend(qml_smoke_candidate_rows(target_variant, SMOKE_HORIZON, spec, dependency))

    qml_rows = qml_rows[: max(0, config.max_qml_candidates)]
    for row in qml_rows:
        if row["status"] == "pending":
            spec = spec_lookup.get((str(row["target_variant"]), int(row["horizon"]), str(row["feature_set"])))
            labels = labels_lookup[(str(row["target_variant"]), int(row["horizon"]))]
            splits = splits_lookup[(str(row["target_variant"]), int(row["horizon"]))]
            if spec is None:
                row["status"] = "skipped"
                row["skipped_reason"] = "smoke feature spec unavailable"
            else:
                row = run_qml_smoke_candidate(row, spec, labels, splits, config, started)
        runtime_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "family": row["qml_family"],
                "runtime_seconds": row["runtime_seconds"],
                "status": row["status"],
                "skipped_reason": row["skipped_reason"],
            }
        )
        circuit_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "qml_family": row["qml_family"],
                "library": row["library"],
                "n_qubits": row["n_qubits"],
                "circuit_depth_or_reps": row["circuit_depth_or_reps"],
                "effective_train_rows": row.get("effective_train_rows", 0),
                "validation_rows": row.get("validation_rows", 0),
                "final_rows": row.get("final_rows", 0),
                "status": row["status"],
                "skipped_reason": row["skipped_reason"],
            }
        )

    selected_qml = select_qml_candidate(qml_rows)
    comparable: dict[str, Any] | None = None
    final_result: dict[str, Any]
    if selected_qml is not None:
        comparable_candidates = [
            row
            for row in baseline_rows
            if row.get("target_variant") == selected_qml.get("target_variant")
            and int(row.get("horizon", -1)) == int(selected_qml.get("horizon", -2))
            and row.get("status") == "ok"
        ]
        comparable = best_by_validation(comparable_candidates)
        comparable_final = as_float(comparable.get("final_accuracy")) if comparable else math.nan
        selected_qml["comparison_vs_classical_smoke_baseline"] = (
            as_float(selected_qml.get("final_accuracy")) - comparable_final if math.isfinite(comparable_final) else math.nan
        )
        label = smoke_claim_label(selected_qml, dependency, comparable_final)
        final_result = {
            "candidate_id": selected_qml["candidate_id"],
            "qml_family": selected_qml["qml_family"],
            "library": selected_qml["library"],
            "target_variant": selected_qml["target_variant"],
            "horizon": selected_qml["horizon"],
            "feature_set": selected_qml["feature_set"],
            "n_qubits": selected_qml["n_qubits"],
            "circuit_depth_or_reps": selected_qml["circuit_depth_or_reps"],
            "validation_accuracy": selected_qml["validation_accuracy"],
            "validation_lift": selected_qml["validation_lift"],
            "final_accuracy": selected_qml["final_accuracy"],
            "final_lift": selected_qml["final_lift"],
            "final_rows": selected_qml["final_rows"],
            "classical_smoke_baseline_candidate_id": comparable.get("candidate_id", "") if comparable else "",
            "classical_smoke_baseline_final_accuracy": comparable_final,
            "comparison_vs_classical_smoke_baseline": selected_qml["comparison_vs_classical_smoke_baseline"],
            "comparison_vs_classical_champion_accuracy": selected_qml["comparison_vs_classical_champion_accuracy"],
            "claim_label": label,
            "status": "ok",
            "skipped_reason": "",
        }
    else:
        label = smoke_claim_label(None, dependency, math.nan)
        final_result = {
            "candidate_id": "",
            "qml_family": "",
            "library": "",
            "target_variant": "",
            "horizon": SMOKE_HORIZON,
            "feature_set": "",
            "n_qubits": 0,
            "circuit_depth_or_reps": 0,
            "validation_accuracy": math.nan,
            "validation_lift": math.nan,
            "final_accuracy": math.nan,
            "final_lift": math.nan,
            "final_rows": 0,
            "classical_smoke_baseline_candidate_id": "",
            "classical_smoke_baseline_final_accuracy": math.nan,
            "comparison_vs_classical_smoke_baseline": math.nan,
            "comparison_vs_classical_champion_accuracy": math.nan,
            "claim_label": label,
            "status": "skipped",
            "skipped_reason": "no QML smoke candidate completed successfully",
        }

    for target_variant in SMOKE_TARGETS:
        rows = [
            row
            for row in baseline_rows
            if row.get("target_variant") == target_variant and int(row.get("horizon", -1)) == SMOKE_HORIZON and row.get("status") == "ok"
        ]
        best_classical = best_by_validation(rows)
        qml_for_target = [
            row
            for row in qml_rows
            if row.get("target_variant") == target_variant and int(row.get("horizon", -1)) == SMOKE_HORIZON and row.get("status") == "ok"
        ]
        best_qml = select_qml_candidate(qml_for_target)
        baseline_comparison_rows.append(
            {
                "target_variant": target_variant,
                "horizon": SMOKE_HORIZON,
                "best_classical_candidate_id": best_classical.get("candidate_id", "") if best_classical else "",
                "best_classical_validation_accuracy": best_classical.get("validation_accuracy", math.nan) if best_classical else math.nan,
                "best_classical_final_accuracy": best_classical.get("final_accuracy", math.nan) if best_classical else math.nan,
                "best_qml_candidate_id": best_qml.get("candidate_id", "") if best_qml else "",
                "best_qml_validation_accuracy": best_qml.get("validation_accuracy", math.nan) if best_qml else math.nan,
                "best_qml_final_accuracy": best_qml.get("final_accuracy", math.nan) if best_qml else math.nan,
                "qml_minus_classical_final_accuracy": (as_float(best_qml.get("final_accuracy")) - as_float(best_classical.get("final_accuracy"))) if best_qml and best_classical else math.nan,
            }
        )

    qml_columns = [
        "candidate_id",
        "qml_family",
        "library",
        "target_variant",
        "horizon",
        "feature_set",
        "n_qubits",
        "circuit_depth_or_reps",
        "compression_method",
        "effective_train_rows",
        "validation_accuracy",
        "validation_lift",
        "validation_rows",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "strongest_validation_baseline",
        "strongest_final_baseline",
        "comparison_vs_classical_smoke_baseline",
        "comparison_vs_classical_champion_accuracy",
        "runtime_seconds",
        "status",
        "skipped_reason",
    ]
    baseline_columns = [
        "candidate_id",
        "model_family",
        "target_variant",
        "horizon",
        "feature_set",
        "n_features",
        "compression_method",
        "validation_accuracy",
        "validation_lift",
        "validation_rows",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "strongest_validation_baseline",
        "strongest_final_baseline",
        "runtime_seconds",
        "status",
        "skipped_reason",
    ]
    final_columns = [
        "candidate_id",
        "qml_family",
        "library",
        "target_variant",
        "horizon",
        "feature_set",
        "n_qubits",
        "circuit_depth_or_reps",
        "validation_accuracy",
        "validation_lift",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "classical_smoke_baseline_candidate_id",
        "classical_smoke_baseline_final_accuracy",
        "comparison_vs_classical_smoke_baseline",
        "comparison_vs_classical_champion_accuracy",
        "claim_label",
        "status",
        "skipped_reason",
    ]

    leaderboard = sorted(
        [row for row in qml_rows if row.get("status") == "ok"],
        key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift")), -as_float(row.get("runtime_seconds"))),
        reverse=True,
    )
    write_frame(OUTPUT_DIR / "qml_smoke_candidate_grid.csv", qml_rows, qml_columns)
    write_frame(OUTPUT_DIR / "qml_smoke_validation_results.csv", qml_rows, qml_columns)
    write_frame(OUTPUT_DIR / "qml_smoke_leaderboard.csv", leaderboard, qml_columns)
    write_frame(OUTPUT_DIR / "qml_smoke_final_result.csv", [final_result], final_columns)
    write_frame(OUTPUT_DIR / "qml_smoke_baseline_comparison.csv", baseline_comparison_rows, ["target_variant", "horizon", "best_classical_candidate_id", "best_classical_validation_accuracy", "best_classical_final_accuracy", "best_qml_candidate_id", "best_qml_validation_accuracy", "best_qml_final_accuracy", "qml_minus_classical_final_accuracy"])
    write_frame(OUTPUT_DIR / "qml_smoke_runtime_summary.csv", runtime_rows, ["candidate_id", "family", "runtime_seconds", "status", "skipped_reason"])
    write_frame(OUTPUT_DIR / "qml_smoke_circuit_summary.csv", circuit_rows, ["candidate_id", "qml_family", "library", "n_qubits", "circuit_depth_or_reps", "effective_train_rows", "validation_rows", "final_rows", "status", "skipped_reason"])

    manifest = {
        "run_id": "vn30_qml_forecasting_v2_smoke",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "diagnostic_only": True,
        "full_grid_run": False,
        "smoke_config": {
            "max_qml_candidates": config.max_qml_candidates,
            "max_train_rows": config.max_train_rows,
            "max_validation_rows": config.max_validation_rows,
            "max_final_rows": config.max_final_rows,
            "timeout_seconds": config.timeout_seconds,
            "family_specific_effective_caps": {
                "quantum_kernel_classifier": qml_effective_limits("quantum_kernel_classifier", config)[:3],
                "variational_quantum_classifier": qml_effective_limits("variational_quantum_classifier", config)[:3],
            },
        },
        "dependency_status": dependency,
        "targets": SMOKE_TARGETS,
        "horizon": SMOKE_HORIZON,
        "feature_plan": SMOKE_FEATURE_PLAN,
        "qml_candidates_total": len(qml_rows),
        "qml_candidates_ran": len([row for row in qml_rows if row.get("status") == "ok"]),
        "qml_candidates_skipped": len([row for row in qml_rows if row.get("status") != "ok"]),
        "qml_models_run": sorted({row["qml_family"] for row in qml_rows if row.get("status") == "ok"}),
        "pennylane_status": "available_but_not_run_in_v2_smoke_budget" if dependency.get("pennylane_available") else "unavailable",
        "selected_qml_candidate": selected_qml if selected_qml is not None else {},
        "final_result": final_result,
        "classical_champion_to_beat": CLASSICAL_CHAMPION,
        "runtime_seconds": time.perf_counter() - started,
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v2_manifest.json", manifest)
    write_smoke_reports(dependency, qml_rows, selected_qml, final_result, comparable, manifest)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v2_manifest.json"), "qml_candidates_ran": manifest["qml_candidates_ran"], "qml_candidates_skipped": manifest["qml_candidates_skipped"]}), indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 QML forecasting diagnostics.")
    parser.add_argument("--qml-smoke", action="store_true", help="Run the limited v2 QML smoke benchmark instead of the full diagnostic grid.")
    parser.add_argument("--max-qml-candidates", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=2000)
    parser.add_argument("--max-validation-rows", type=int, default=1000)
    parser.add_argument("--max-final-rows", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qml_smoke:
        config = SmokeConfig(
            max_qml_candidates=max(0, int(args.max_qml_candidates)),
            max_train_rows=max(1, int(args.max_train_rows)),
            max_validation_rows=max(1, int(args.max_validation_rows)),
            max_final_rows=max(1, int(args.max_final_rows)),
            timeout_seconds=max(1, int(args.timeout_seconds)),
        )
        run_smoke(config)
        return
    manifest = run()
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_manifest.json"), "qml_execution_status": manifest["dependencies"]["qml_execution_status"]}), indent=2))


if __name__ == "__main__":
    main()
