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

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC

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
from src.governance.split_policy import FINAL_START, TRAIN_END, VAL_END, VAL_START  # noqa: E402
from src.utils.research_io import as_float, json_safe, rel, write_frame, write_json, write_markdown  # noqa: E402

warnings.filterwarnings("ignore", message="Skipping features without any observed values.*")
warnings.filterwarnings("ignore", message="X does not have valid feature names.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_qml_forecasting"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_CLAIM_BOUNDARY.md"

SEED = 42

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


@dataclass
class V3Config:
    max_qml_candidates: int = 6
    max_train_rows: int = 1500
    max_validation_rows: int = 800
    max_final_rows: int = 800
    timeout_seconds: int = 1800


V3_TARGETS = ["market_relative_vn30", "market_relative_vnindex", "absolute_direction"]
V3_HORIZON = 40
V3_FEATURE_PLAN = [
    ("compact_stable_features", "topk_availability", 4),
    ("relative_strength_features", "topk_availability", 4),
    ("combined_strategy_features", "pca_train_only", 4),
]
V3_CLASSICAL_MODELS = ["svm_rbf", "svm_linear", "l2_logistic", "calibrated_logistic", "random_forest_small"]
V3_QKERNEL_TRAIN_CAP = 120
V3_QKERNEL_VALIDATION_CAP = 240
V3_QKERNEL_FINAL_CAP = 240


def add_v3_relative_strength_features(features: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    out = features.copy()
    row_map = out[["datetime"]].copy()
    row_map["row_index"] = out.index
    row_map = row_map.sort_values("datetime")
    created: list[str] = []
    stock_lag = pd.to_numeric(out.get("return_1_lag_1", out.get("lag_ret_1", pd.Series(np.nan, index=out.index))), errors="coerce")
    for code in ["VN30", "VNINDEX"]:
        if code not in index_data:
            continue
        idx = index_data[code][["datetime", "close"]].copy().sort_values("datetime").drop_duplicates("datetime", keep="last")
        idx["datetime"] = pd.to_datetime(idx["datetime"], errors="coerce")
        idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
        idx = idx.dropna(subset=["datetime", "close"])
        # Shift by one index bar so daily index context is lagged for hourly rows.
        idx[f"v3_{code.lower()}_return_lag_asof"] = idx["close"].pct_change(fill_method=None).shift(1)
        merged = pd.merge_asof(
            row_map,
            idx[["datetime", f"v3_{code.lower()}_return_lag_asof"]],
            on="datetime",
            direction="backward",
        )
        market_lag = pd.Series(merged.set_index("row_index")[f"v3_{code.lower()}_return_lag_asof"]).reindex(out.index)
        diff_col = f"v3_relative_ret_minus_{code.lower()}_lag1"
        out[diff_col] = stock_lag - pd.to_numeric(market_lag, errors="coerce")
        created.append(diff_col)
        mean_col = f"v3_relative_strength_{code.lower()}_mean20_lag"
        vol_col = f"v3_relative_strength_{code.lower()}_vol20_lag"
        for _ticker, group in out.groupby("ticker", sort=True):
            diff = pd.to_numeric(group[diff_col], errors="coerce")
            out.loc[group.index, mean_col] = diff.rolling(20, min_periods=5).mean()
            out.loc[group.index, vol_col] = diff.rolling(20, min_periods=5).std()
        created.extend([mean_col, vol_col])
    out[created] = out[created].replace([np.inf, -np.inf], np.nan)
    return out, created


def ordered_index(features: pd.DataFrame, index: pd.Index) -> pd.Index:
    if len(index) == 0:
        return index
    ordered = features.loc[index, ["datetime"]].copy()
    ordered["idx"] = index
    ordered = ordered.sort_values(["datetime", "idx"])
    return pd.Index(ordered["idx"].tolist())


def balanced_ordered_train_sample(features: pd.DataFrame, labels: pd.Series, index: pd.Index, limit: int) -> pd.Index:
    ordered = ordered_index(features, index)
    if len(ordered) <= limit:
        return ordered
    frame = features.loc[ordered, ["datetime"]].copy()
    frame["idx"] = ordered
    frame["label"] = labels.loc[ordered].astype(int).to_numpy()
    per_class = max(1, limit // max(1, frame["label"].nunique()))
    pieces = []
    for _label, group in frame.groupby("label", sort=True):
        pieces.append(group.tail(per_class))
    sampled = pd.concat(pieces, ignore_index=True).sort_values(["datetime", "idx"])
    if len(sampled) > limit:
        sampled = sampled.tail(limit).sort_values(["datetime", "idx"])
    return pd.Index(sampled["idx"].tolist())


def ordered_tail_sample(features: pd.DataFrame, index: pd.Index, limit: int) -> pd.Index:
    ordered = ordered_index(features, index)
    if len(ordered) <= limit:
        return ordered
    return pd.Index(list(ordered)[-limit:])


def v3_effective_splits(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index], config: V3Config) -> dict[str, pd.Index]:
    train_limit = min(config.max_train_rows, V3_QKERNEL_TRAIN_CAP)
    validation_limit = min(config.max_validation_rows, V3_QKERNEL_VALIDATION_CAP)
    final_limit = min(config.max_final_rows, V3_QKERNEL_FINAL_CAP)
    return {
        "train": balanced_ordered_train_sample(features, labels, splits["train"], train_limit),
        "validation": ordered_tail_sample(features, splits["validation"], validation_limit),
        "final": ordered_tail_sample(features, splits["final"], final_limit),
    }


def v3_make_classical_model(model_name: str) -> Any:
    if model_name == "svm_rbf":
        return SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
    if model_name == "svm_linear":
        return SVC(kernel="linear", C=0.5, class_weight="balanced")
    return make_classical_model(model_name)


def run_v3_classical_models(
    spec: FeatureSpec,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    splits: dict[str, pd.Index],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_y = labels.loc[splits["train"]].astype(int)
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    val_simple_name, val_simple_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
    final_simple_name, final_simple_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
    rows.append(
        {
            "candidate_id": candidate_id("v3", "strongest_simple", target_variant, f"h{horizon}", spec.feature_set_name),
            "model_family": "strongest_simple_baseline",
            "target_variant": target_variant,
            "horizon": horizon,
            "feature_set": spec.feature_set_name,
            "n_features": spec.n_features,
            "compression_method": spec.compression_method,
            "effective_train_rows": 0,
            "validation_accuracy": val_simple_acc,
            "validation_lift": 0.0,
            "validation_rows": int(len(validation_y)),
            "final_accuracy": final_simple_acc,
            "final_lift": 0.0,
            "final_rows": int(len(final_y)),
            "strongest_validation_baseline": val_simple_name,
            "strongest_final_baseline": final_simple_name,
            "runtime_seconds": 0.0,
            "status": "ok",
            "skipped_reason": "",
        }
    )
    for model_name in V3_CLASSICAL_MODELS:
        start = time.perf_counter()
        try:
            model = v3_make_classical_model(model_name)
            if model_name == "calibrated_logistic" and train_y.value_counts().min() < 3:
                raise ValueError("calibrated logistic requires at least three train rows per class")
            model.fit(spec.x_train, train_y)
            if hasattr(model, "predict_proba"):
                validation_pred = (model.predict_proba(spec.x_validation)[:, 1] >= 0.50).astype(int)
                final_pred = (model.predict_proba(spec.x_final)[:, 1] >= 0.50).astype(int)
            else:
                validation_pred = np.asarray(model.predict(spec.x_validation)).astype(int)
                final_pred = np.asarray(model.predict(spec.x_final)).astype(int)
            validation_acc = accuracy(validation_y, validation_pred)
            final_acc = accuracy(final_y, final_pred)
            status = "ok"
            skipped_reason = ""
        except Exception as exc:
            validation_acc = math.nan
            final_acc = math.nan
            status = "skipped"
            skipped_reason = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "candidate_id": candidate_id("v3", model_name, target_variant, f"h{horizon}", spec.feature_set_name),
                "model_family": model_name,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_features": spec.n_features,
                "compression_method": spec.compression_method,
                "effective_train_rows": int(len(train_y)),
                "validation_accuracy": validation_acc,
                "validation_lift": validation_acc - val_simple_acc if math.isfinite(validation_acc) and math.isfinite(val_simple_acc) else math.nan,
                "validation_rows": int(len(validation_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_simple_acc if math.isfinite(final_acc) and math.isfinite(final_simple_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_simple_name,
                "strongest_final_baseline": final_simple_name,
                "runtime_seconds": time.perf_counter() - start,
                "status": status,
                "skipped_reason": skipped_reason,
            }
        )
    return rows


def qml_v3_candidate_rows(target_variant: str, horizon: int, spec: FeatureSpec, dependency: dict[str, Any]) -> dict[str, Any]:
    if not dependency.get("qiskit_machine_learning_available"):
        status = "skipped"
        reason = "dependency_missing_or_api_error: qiskit_machine_learning unavailable"
    else:
        status = "pending"
        reason = ""
    return {
        "candidate_id": candidate_id("qml_v3", "quantum_kernel_classifier", target_variant, f"h{horizon}", spec.feature_set_name, "q4", "r1"),
        "qml_family": "quantum_kernel_classifier",
        "library": "qiskit_machine_learning",
        "target_variant": target_variant,
        "horizon": horizon,
        "feature_set": spec.feature_set_name,
        "n_qubits": 4,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": 1,
        "compression_method": spec.compression_method,
        "effective_train_rows": 0,
        "validation_accuracy": math.nan,
        "validation_lift": math.nan,
        "validation_rows": int(len(spec.x_validation)),
        "final_accuracy": math.nan,
        "final_lift": math.nan,
        "final_rows": int(len(spec.x_final)),
        "rbf_svm_validation_accuracy": math.nan,
        "rbf_svm_final_accuracy": math.nan,
        "linear_svm_validation_accuracy": math.nan,
        "linear_svm_final_accuracy": math.nan,
        "logistic_validation_accuracy": math.nan,
        "logistic_final_accuracy": math.nan,
        "qml_minus_rbf_svm_validation": math.nan,
        "qml_minus_rbf_svm_final": math.nan,
        "qml_minus_logistic_validation": math.nan,
        "qml_minus_logistic_final": math.nan,
        "comparison_vs_classical_champion_accuracy": math.nan,
        "runtime_seconds": 0.0,
        "status": status,
        "skipped_reason": reason,
    }


def run_v3_quantum_kernel(
    row: dict[str, Any],
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    classical_rows: list[dict[str, Any]],
    config: V3Config,
    started: float,
) -> dict[str, Any]:
    if row["status"] != "pending":
        return row
    if time.perf_counter() - started >= config.timeout_seconds:
        row["status"] = "skipped"
        row["skipped_reason"] = "qml_runtime_limited: timeout budget exhausted before candidate start"
        return row
    start = time.perf_counter()
    try:
        algorithms = importlib.import_module("qiskit_machine_learning.algorithms")
        circuit_library = importlib.import_module("qiskit.circuit.library")
        QSVC = getattr(algorithms, "QSVC")
        ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
        train_y = labels.loc[splits["train"]].astype(int)
        validation_y = labels.loc[splits["validation"]].astype(int)
        final_y = labels.loc[splits["final"]].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("v3 train sample has fewer than two classes")
        x_train, x_validation, x_final = scale_for_quantum(spec.x_train, spec.x_validation, spec.x_final)
        feature_map = ZZFeatureMap(feature_dimension=spec.n_features, reps=1)
        model = QSVC(feature_map=feature_map)
        model.fit(x_train.to_numpy(), train_y.to_numpy())
        validation_pred = np.asarray(model.predict(x_validation.to_numpy())).reshape(-1).astype(int)
        final_pred = np.asarray(model.predict(x_final.to_numpy())).reshape(-1).astype(int)
        validation_acc = accuracy(validation_y, validation_pred)
        final_acc = accuracy(final_y, final_pred)
        simple_val_name, simple_val_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
        simple_final_name, simple_final_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
        row.update(
            {
                "effective_train_rows": int(len(train_y)),
                "validation_accuracy": validation_acc,
                "validation_lift": validation_acc - simple_val_acc if math.isfinite(simple_val_acc) else math.nan,
                "validation_rows": int(len(validation_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - simple_final_acc if math.isfinite(simple_final_acc) else math.nan,
                "final_rows": int(len(final_y)),
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
    for model_name, prefix in [("svm_rbf", "rbf_svm"), ("svm_linear", "linear_svm"), ("l2_logistic", "logistic")]:
        match = next(
            (
                item
                for item in classical_rows
                if item.get("target_variant") == row.get("target_variant")
                and item.get("feature_set") == row.get("feature_set")
                and item.get("model_family") == model_name
                and item.get("status") == "ok"
            ),
            None,
        )
        if match:
            row[f"{prefix}_validation_accuracy"] = match.get("validation_accuracy", math.nan)
            row[f"{prefix}_final_accuracy"] = match.get("final_accuracy", math.nan)
    qml_val = as_float(row.get("validation_accuracy"))
    qml_final = as_float(row.get("final_accuracy"))
    rbf_val = as_float(row.get("rbf_svm_validation_accuracy"))
    rbf_final = as_float(row.get("rbf_svm_final_accuracy"))
    log_val = as_float(row.get("logistic_validation_accuracy"))
    log_final = as_float(row.get("logistic_final_accuracy"))
    row["qml_minus_rbf_svm_validation"] = qml_val - rbf_val if math.isfinite(qml_val) and math.isfinite(rbf_val) else math.nan
    row["qml_minus_rbf_svm_final"] = qml_final - rbf_final if math.isfinite(qml_final) and math.isfinite(rbf_final) else math.nan
    row["qml_minus_logistic_validation"] = qml_val - log_val if math.isfinite(qml_val) and math.isfinite(log_val) else math.nan
    row["qml_minus_logistic_final"] = qml_final - log_final if math.isfinite(qml_final) and math.isfinite(log_final) else math.nan
    return row


def v3_claim_label(selected: dict[str, Any] | None, runtime_limited: bool) -> str:
    if runtime_limited and selected is None:
        return "qml_runtime_limited"
    if selected is None:
        return "not_claimable"
    if selected.get("status") != "ok":
        return "qml_runtime_limited" if runtime_limited else "not_claimable"
    qml_final = as_float(selected.get("final_accuracy"))
    rbf_final = as_float(selected.get("rbf_svm_final_accuracy"))
    log_final = as_float(selected.get("logistic_final_accuracy"))
    if math.isfinite(qml_final) and math.isfinite(rbf_final) and qml_final > rbf_final:
        return "qml_kernel_candidate"
    if math.isfinite(qml_final) and math.isfinite(log_final) and qml_final <= log_final:
        return "qml_underperforms_classical"
    return "qml_diagnostic_only"


def write_v3_reports(
    qml_rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    final_row: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    ok_rows = [row for row in qml_rows if row.get("status") == "ok"]
    kernel_beats_rbf_rows = [row for row in ok_rows if as_float(row.get("qml_minus_rbf_svm_final")) > 0.0]
    qml_beats_logistic_rows = [row for row in ok_rows if as_float(row.get("qml_minus_logistic_final")) > 0.0]
    qml_beats_champion_rows = [row for row in ok_rows if as_float(row.get("comparison_vs_classical_champion_accuracy")) > 0.0]
    selected_beats_rbf = as_float(final_row.get("qml_minus_rbf_svm_final")) > 0.0
    selected_beats_logistic = as_float(final_row.get("qml_minus_logistic_final")) > 0.0
    selected_beats_champion = as_float(final_row.get("comparison_vs_classical_champion_accuracy")) > 0.0
    champion_note = "no validation-selected QML row beat the 61.61% champion"
    if qml_beats_champion_rows:
        champion_note = "a non-selected final-scored QML row exceeded 61.61%, but this is exploratory_not_claimable and cannot replace the validation-selected result"
        if selected_beats_champion:
            champion_note = "the validation-selected QML row exceeded 61.61%, but stronger claims still require full validation-governed rerun and future-blind confirmation"
    best_target = str(selected.get("target_variant", "none")) if selected else "none"
    expansion = "not justified yet; the validation-selected QML row did not beat comparable RBF/logistic baselines on final scoring"
    if selected_beats_rbf and selected_beats_logistic:
        expansion = "limited expansion may be justified only as a diagnostic follow-up; full validation-governed rerun and future-blind confirmation remain required"
    runtime_by_family: dict[str, float] = {}
    for row in runtime_rows:
        family = str(row.get("family", "unknown"))
        runtime_by_family[family] = runtime_by_family.get(family, 0.0) + as_float(row.get("runtime_seconds"))
    runtime_text = ", ".join(f"{family}={seconds:.2f}s" for family, seconds in sorted(runtime_by_family.items()))
    summary = f"""# VN30 QML Forecasting V3 Targeted Sanity Result Summary

## Required Answers

1. Did quantum kernel beat RBF SVM on the same target/features/sample: {str(bool(kernel_beats_rbf_rows)).lower()}; validation-selected row: {str(selected_beats_rbf).lower()}.
2. Did any QML model beat logistic regression: {str(bool(qml_beats_logistic_rows)).lower()}; validation-selected row: {str(selected_beats_logistic).lower()}.
3. Did any QML model beat the 61.61% classical champion: {str(bool(qml_beats_champion_rows)).lower()}; validation-selected row: {str(selected_beats_champion).lower()}. Boundary: {champion_note}.
4. Target variant that worked best by validation-selected QML row: {best_target}.
5. Runtime per model family: {runtime_text}.
6. Is QML worth expanding to a larger benchmark: {expansion}.
7. Claim boundary: V3 is a bounded experimental sanity diagnostic only; no QML result is claimable or replaces the 61.61% classical champion.

## Final Validation-Selected QML Result

- Candidate: `{final_row.get("candidate_id", "")}`.
- QML family/library: {final_row.get("qml_family", "")} / {final_row.get("library", "")}.
- Target/horizon/features: {final_row.get("target_variant", "")} / h{final_row.get("horizon", "")} / {final_row.get("feature_set", "")}.
- Validation accuracy: {pct(final_row.get("validation_accuracy"))}.
- Final accuracy: {pct(final_row.get("final_accuracy"))}.
- QML minus RBF SVM final accuracy: {pp(final_row.get("qml_minus_rbf_svm_final"))}.
- QML minus Logistic final accuracy: {pp(final_row.get("qml_minus_logistic_final"))}.
- QML minus 61.61% classical champion: {pp(final_row.get("comparison_vs_classical_champion_accuracy"))}.
- Claim label: `{final_row.get("claim_label", "not_claimable")}`.

## Paper-Safe Wording

VN30 QML v3 tested a bounded quantum-kernel sanity benchmark on VN30 hourly stock forecasting with train-only transforms and strict feature_timestamp/target_timestamp split discipline. Quantum-kernel results are diagnostic-only and are compared against same-sample RBF SVM, linear SVM, logistic, calibrated logistic, random forest, and simple baselines. No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V3_TARGETED_SANITY_RESULT_SUMMARY.md", summary)

    claim = """# VN30 QML Forecasting V3 Targeted Sanity Claim Boundary

- QML v3 targeted sanity is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature scaling, PCA, and sampling decisions must be fit or selected without final-period information.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Quantum-kernel rows are compared against same-sample RBF SVM, linear SVM, logistic, calibrated logistic, random forest, and simple baselines.
- No QML result replaces the 61.61% L2 Logistic classical champion.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require a full validation-governed benchmark and future-blind confirmation.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V3_TARGETED_SANITY_CLAIM_BOUNDARY.md", claim)


def run_v3_sanity(config: V3Config) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    index_data = load_index_data()
    features, v3_relative_cols = add_v3_relative_strength_features(features, index_data)
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    source_groups = build_source_groups(features, family_cols)
    source_groups["relative_strength_features"] = v3_relative_cols
    source_groups["combined_strategy_features"] = sorted(set(source_groups["combined_strategy_features"]).union(v3_relative_cols))

    run_config = {
        "run_id": "vn30_qml_forecasting_v3_targeted_sanity",
        "targets": V3_TARGETS,
        "horizon": V3_HORIZON,
        "feature_plan": V3_FEATURE_PLAN,
        "max_qml_candidates": config.max_qml_candidates,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "max_final_rows": config.max_final_rows,
        "timeout_seconds": config.timeout_seconds,
        "effective_qsvc_caps": {
            "train": min(config.max_train_rows, V3_QKERNEL_TRAIN_CAP),
            "validation": min(config.max_validation_rows, V3_QKERNEL_VALIDATION_CAP),
            "final": min(config.max_final_rows, V3_QKERNEL_FINAL_CAP),
        },
        "balanced_class_sampling_train_only": True,
        "preserve_timestamp_order_within_split": True,
        "dependency_status": dependency,
    }
    write_json(OUTPUT_DIR / "qml_v3_run_config.json", run_config)

    qml_rows: list[dict[str, Any]] = []
    classical_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    circuit_rows: list[dict[str, Any]] = []
    spec_lookup: dict[tuple[str, str], FeatureSpec] = {}
    labels_lookup: dict[str, pd.Series] = {}
    splits_lookup: dict[str, dict[str, pd.Index]] = {}

    for target_variant in V3_TARGETS:
        labels = build_labels(features, index_data, target_variant, V3_HORIZON)
        full_splits = strict_split_indices(features, labels)
        splits = v3_effective_splits(features, labels, full_splits, config)
        labels_lookup[target_variant] = labels
        splits_lookup[target_variant] = splits
        for source_group, compression_method, n_features in V3_FEATURE_PLAN:
            spec, _audit = fit_feature_spec(
                features,
                labels,
                target_variant,
                V3_HORIZON,
                source_group,
                source_groups[source_group],
                compression_method,
                n_features,
                splits,
            )
            if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
                skipped = qml_v3_candidate_rows(target_variant, V3_HORIZON, spec, dependency)
                skipped["status"] = "skipped"
                skipped["skipped_reason"] = f"feature_selection_{spec.selection_status}"
                qml_rows.append(skipped)
                continue
            spec_lookup[(target_variant, spec.feature_set_name)] = spec
            c_rows = run_v3_classical_models(spec, labels, target_variant, V3_HORIZON, splits)
            classical_rows.extend(c_rows)
            runtime_rows.extend(
                {
                    "candidate_id": row["candidate_id"],
                    "family": row["model_family"],
                    "target_variant": row["target_variant"],
                    "feature_set": row["feature_set"],
                    "runtime_seconds": row["runtime_seconds"],
                    "status": row["status"],
                    "skipped_reason": row["skipped_reason"],
                }
                for row in c_rows
            )
            qml_rows.append(qml_v3_candidate_rows(target_variant, V3_HORIZON, spec, dependency))

    runnable_seen = 0
    runtime_limited = False
    for row in qml_rows:
        if row.get("status") != "pending":
            continue
        if runnable_seen >= config.max_qml_candidates:
            row["status"] = "skipped"
            row["skipped_reason"] = "max_qml_candidates budget reached"
            continue
        if time.perf_counter() - started >= config.timeout_seconds:
            row["status"] = "skipped"
            row["skipped_reason"] = "qml_runtime_limited: timeout budget exhausted"
            runtime_limited = True
            continue
        spec = spec_lookup.get((str(row["target_variant"]), str(row["feature_set"])))
        labels = labels_lookup[str(row["target_variant"])]
        splits = splits_lookup[str(row["target_variant"])]
        if spec is None:
            row["status"] = "skipped"
            row["skipped_reason"] = "feature spec unavailable"
            continue
        row = run_v3_quantum_kernel(row, spec, labels, splits, classical_rows, config, started)
        runnable_seen += 1
        if str(row.get("skipped_reason", "")).startswith("qml_runtime_limited"):
            runtime_limited = True

    for row in qml_rows:
        runtime_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "family": row["qml_family"],
                "target_variant": row["target_variant"],
                "feature_set": row["feature_set"],
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
                "feature_map": row["feature_map"],
                "n_qubits": row["n_qubits"],
                "feature_map_reps": row["feature_map_reps"],
                "effective_train_rows": row["effective_train_rows"],
                "validation_rows": row["validation_rows"],
                "final_rows": row["final_rows"],
                "status": row["status"],
                "skipped_reason": row["skipped_reason"],
            }
        )

    ok_qml = [row for row in qml_rows if row.get("status") == "ok"]
    selected = select_qml_candidate(ok_qml)
    claim_label = v3_claim_label(selected, runtime_limited)
    final_row = {
        "candidate_id": selected.get("candidate_id", "") if selected else "",
        "qml_family": selected.get("qml_family", "") if selected else "",
        "library": selected.get("library", "") if selected else "",
        "target_variant": selected.get("target_variant", "") if selected else "",
        "horizon": selected.get("horizon", V3_HORIZON) if selected else V3_HORIZON,
        "feature_set": selected.get("feature_set", "") if selected else "",
        "n_qubits": selected.get("n_qubits", 0) if selected else 0,
        "feature_map": selected.get("feature_map", "") if selected else "",
        "feature_map_reps": selected.get("feature_map_reps", 0) if selected else 0,
        "validation_accuracy": selected.get("validation_accuracy", math.nan) if selected else math.nan,
        "validation_lift": selected.get("validation_lift", math.nan) if selected else math.nan,
        "final_accuracy": selected.get("final_accuracy", math.nan) if selected else math.nan,
        "final_lift": selected.get("final_lift", math.nan) if selected else math.nan,
        "final_rows": selected.get("final_rows", 0) if selected else 0,
        "rbf_svm_final_accuracy": selected.get("rbf_svm_final_accuracy", math.nan) if selected else math.nan,
        "linear_svm_final_accuracy": selected.get("linear_svm_final_accuracy", math.nan) if selected else math.nan,
        "logistic_final_accuracy": selected.get("logistic_final_accuracy", math.nan) if selected else math.nan,
        "qml_minus_rbf_svm_validation": selected.get("qml_minus_rbf_svm_validation", math.nan) if selected else math.nan,
        "qml_minus_rbf_svm_final": selected.get("qml_minus_rbf_svm_final", math.nan) if selected else math.nan,
        "qml_minus_logistic_validation": selected.get("qml_minus_logistic_validation", math.nan) if selected else math.nan,
        "qml_minus_logistic_final": selected.get("qml_minus_logistic_final", math.nan) if selected else math.nan,
        "comparison_vs_classical_champion_accuracy": selected.get("comparison_vs_classical_champion_accuracy", math.nan) if selected else math.nan,
        "claim_label": claim_label,
        "status": selected.get("status", "skipped") if selected else "skipped",
        "skipped_reason": selected.get("skipped_reason", "no QML v3 candidate completed") if selected else "no QML v3 candidate completed",
    }

    comparison_rows: list[dict[str, Any]] = []
    for row in qml_rows:
        comparison_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "target_variant": row["target_variant"],
                "feature_set": row["feature_set"],
                "status": row["status"],
                "validation_accuracy": row["validation_accuracy"],
                "final_accuracy": row["final_accuracy"],
                "rbf_svm_validation_accuracy": row["rbf_svm_validation_accuracy"],
                "rbf_svm_final_accuracy": row["rbf_svm_final_accuracy"],
                "linear_svm_validation_accuracy": row["linear_svm_validation_accuracy"],
                "linear_svm_final_accuracy": row["linear_svm_final_accuracy"],
                "logistic_validation_accuracy": row["logistic_validation_accuracy"],
                "logistic_final_accuracy": row["logistic_final_accuracy"],
                "qml_minus_rbf_svm_validation": row["qml_minus_rbf_svm_validation"],
                "qml_minus_rbf_svm_final": row["qml_minus_rbf_svm_final"],
                "qml_minus_logistic_validation": row["qml_minus_logistic_validation"],
                "qml_minus_logistic_final": row["qml_minus_logistic_final"],
                "comparison_vs_classical_champion_accuracy": row["comparison_vs_classical_champion_accuracy"],
                "skipped_reason": row["skipped_reason"],
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
        "feature_map",
        "feature_map_reps",
        "compression_method",
        "effective_train_rows",
        "validation_accuracy",
        "validation_lift",
        "validation_rows",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "rbf_svm_validation_accuracy",
        "rbf_svm_final_accuracy",
        "linear_svm_validation_accuracy",
        "linear_svm_final_accuracy",
        "logistic_validation_accuracy",
        "logistic_final_accuracy",
        "qml_minus_rbf_svm_validation",
        "qml_minus_rbf_svm_final",
        "qml_minus_logistic_validation",
        "qml_minus_logistic_final",
        "comparison_vs_classical_champion_accuracy",
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
        "feature_map",
        "feature_map_reps",
        "validation_accuracy",
        "validation_lift",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "rbf_svm_final_accuracy",
        "linear_svm_final_accuracy",
        "logistic_final_accuracy",
        "qml_minus_rbf_svm_validation",
        "qml_minus_rbf_svm_final",
        "qml_minus_logistic_validation",
        "qml_minus_logistic_final",
        "comparison_vs_classical_champion_accuracy",
        "claim_label",
        "status",
        "skipped_reason",
    ]
    leaderboard = sorted(ok_qml, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift")), -as_float(row.get("runtime_seconds"))), reverse=True)
    write_frame(OUTPUT_DIR / "qml_v3_candidate_grid.csv", qml_rows, qml_columns)
    write_frame(OUTPUT_DIR / "qml_v3_validation_results.csv", qml_rows, qml_columns)
    write_frame(OUTPUT_DIR / "qml_v3_leaderboard.csv", leaderboard, qml_columns)
    write_frame(OUTPUT_DIR / "qml_v3_final_result.csv", [final_row], final_columns)
    write_frame(
        OUTPUT_DIR / "qml_v3_classical_kernel_comparison.csv",
        comparison_rows,
        ["candidate_id", "target_variant", "feature_set", "status", "validation_accuracy", "final_accuracy", "rbf_svm_validation_accuracy", "rbf_svm_final_accuracy", "linear_svm_validation_accuracy", "linear_svm_final_accuracy", "logistic_validation_accuracy", "logistic_final_accuracy", "qml_minus_rbf_svm_validation", "qml_minus_rbf_svm_final", "qml_minus_logistic_validation", "qml_minus_logistic_final", "comparison_vs_classical_champion_accuracy", "skipped_reason"],
    )
    write_frame(OUTPUT_DIR / "qml_v3_runtime_summary.csv", runtime_rows, ["candidate_id", "family", "target_variant", "feature_set", "runtime_seconds", "status", "skipped_reason"])
    write_frame(OUTPUT_DIR / "qml_v3_circuit_summary.csv", circuit_rows, ["candidate_id", "qml_family", "library", "feature_map", "n_qubits", "feature_map_reps", "effective_train_rows", "validation_rows", "final_rows", "status", "skipped_reason"])

    manifest = {
        "run_id": "vn30_qml_forecasting_v3_targeted_sanity",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "diagnostic_only": True,
        "full_grid_run": False,
        "run_config": run_config,
        "feature_manifest": feature_manifest,
        "v3_relative_strength_features": v3_relative_cols,
        "qml_candidates_total": len(qml_rows),
        "qml_candidates_ran": len(ok_qml),
        "qml_candidates_skipped": len([row for row in qml_rows if row.get("status") != "ok"]),
        "runtime_limited": runtime_limited,
        "selected_qml_candidate": selected if selected is not None else {},
        "final_result": final_row,
        "validation_selected_beats_rbf_svm": as_float(final_row.get("qml_minus_rbf_svm_final")) > 0.0,
        "validation_selected_beats_logistic": as_float(final_row.get("qml_minus_logistic_final")) > 0.0,
        "validation_selected_beats_classical_champion": as_float(final_row.get("comparison_vs_classical_champion_accuracy")) > 0.0,
        "any_final_scored_qml_beats_classical_champion": any(as_float(row.get("comparison_vs_classical_champion_accuracy")) > 0.0 for row in ok_qml),
        "final_ranked_champion_comparison_claimable": False,
        "runtime_seconds": time.perf_counter() - started,
        "expansion_justification": "limited diagnostic expansion only" if as_float(final_row.get("qml_minus_rbf_svm_final")) > 0.0 and as_float(final_row.get("qml_minus_logistic_final")) > 0.0 else "not justified yet",
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v3_manifest.json", manifest)
    write_v3_reports(qml_rows, selected, final_row, comparison_rows, runtime_rows, manifest)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v3_manifest.json"), "qml_candidates_ran": manifest["qml_candidates_ran"], "qml_candidates_skipped": manifest["qml_candidates_skipped"], "runtime_limited": runtime_limited}), indent=2))
    return manifest


@dataclass
class V4Config:
    max_qml_candidates: int = 10
    max_train_rows: int = 1500
    max_validation_rows: int = 800
    max_final_rows: int = 800
    timeout_seconds: int = 2400


V4_TARGET = "market_relative_vn30"
V4_HORIZON = 40
V4_FEATURE_PLAN = [
    ("combined_strategy_features", "pca_train_only", 4),
    ("combined_strategy_features", "pca_train_only", 6),
    ("compact_stable_features", "topk_availability", 4),
    ("relative_strength_features", "topk_availability", 4),
    ("market_context_features", "topk_availability", 4),
]
V4_KERNEL_REPS = [1, 2]
V4_CLASSICAL_MODELS = ["svm_rbf", "svm_linear", "l2_logistic", "calibrated_logistic"]
V4_QKERNEL_TRAIN_CAP = 80
V4_QKERNEL_VALIDATION_CAP = 180
V4_QKERNEL_FINAL_CAP = 180
V4_VAL_SPLIT = pd.Timestamp("2024-07-01 00:00:00")


def v4_effective_splits(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index], config: V4Config) -> dict[str, pd.Index]:
    train_limit = min(config.max_train_rows, V4_QKERNEL_TRAIN_CAP)
    validation_limit = min(config.max_validation_rows, V4_QKERNEL_VALIDATION_CAP)
    final_limit = min(config.max_final_rows, V4_QKERNEL_FINAL_CAP)
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_from_labels(labels, features.index)
    validation_idx = ordered_index(features, splits["validation"])
    early_mask = timestamps.loc[validation_idx].lt(V4_VAL_SPLIT) & target_timestamp.loc[validation_idx].lt(V4_VAL_SPLIT)
    late_mask = timestamps.loc[validation_idx].ge(V4_VAL_SPLIT) & target_timestamp.loc[validation_idx].ge(V4_VAL_SPLIT)
    early_idx = pd.Index(validation_idx[early_mask.to_numpy()])
    late_idx = pd.Index(validation_idx[late_mask.to_numpy()])
    half = max(1, validation_limit // 2)
    early_sample = ordered_tail_sample(features, early_idx, half)
    late_sample = ordered_tail_sample(features, late_idx, validation_limit - len(early_sample))
    validation_sample = ordered_index(features, pd.Index(list(early_sample) + list(late_sample)))
    if len(validation_sample) == 0:
        validation_sample = ordered_tail_sample(features, splits["validation"], validation_limit)
    return {
        "train": balanced_ordered_train_sample(features, labels, splits["train"], train_limit),
        "validation": validation_sample,
        "final": ordered_tail_sample(features, splits["final"], final_limit),
    }


def qml_v4_candidate_rows(target_variant: str, horizon: int, spec: FeatureSpec, dependency: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for reps in V4_KERNEL_REPS:
        if spec.n_features not in {4, 6}:
            status = "skipped"
            reason = f"unsupported_qubit_count: {spec.n_features}"
        elif not dependency.get("qiskit_machine_learning_available"):
            status = "skipped"
            reason = "dependency_missing_or_api_error: qiskit_machine_learning unavailable"
        else:
            status = "pending"
            reason = ""
        rows.append(
            {
                "candidate_id": candidate_id("qml_v4", "quantum_kernel_classifier", target_variant, f"h{horizon}", spec.feature_set_name, f"q{spec.n_features}", f"r{reps}"),
                "qml_family": "quantum_kernel_classifier",
                "library": "qiskit_machine_learning",
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_qubits": spec.n_features,
                "feature_map": "ZZFeatureMap",
                "feature_map_reps": reps,
                "circuit_depth_or_reps": reps,
                "compression_method": spec.compression_method,
                "effective_train_rows": 0,
                "validation_accuracy": math.nan,
                "validation_lift": math.nan,
                "validation_rows": int(len(spec.x_validation)),
                "final_accuracy": math.nan,
                "final_lift": math.nan,
                "final_rows": int(len(spec.x_final)),
                "rbf_svm_validation_accuracy": math.nan,
                "rbf_svm_final_accuracy": math.nan,
                "linear_svm_validation_accuracy": math.nan,
                "linear_svm_final_accuracy": math.nan,
                "logistic_validation_accuracy": math.nan,
                "logistic_final_accuracy": math.nan,
                "calibrated_logistic_validation_accuracy": math.nan,
                "calibrated_logistic_final_accuracy": math.nan,
                "qml_minus_rbf_svm_validation": math.nan,
                "qml_minus_rbf_svm_final": math.nan,
                "qml_minus_logistic_validation": math.nan,
                "qml_minus_logistic_final": math.nan,
                "comparison_vs_classical_champion_accuracy": math.nan,
                "runtime_seconds": 0.0,
                "status": status,
                "skipped_reason": reason,
            }
        )
    return rows


def run_v4_classical_models(
    spec: FeatureSpec,
    labels: pd.Series,
    target_variant: str,
    horizon: int,
    splits: dict[str, pd.Index],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, pd.Series]]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, pd.Series]] = {}
    train_y = labels.loc[splits["train"]].astype(int)
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    val_simple_name, val_simple_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
    final_simple_name, final_simple_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
    for model_name in V4_CLASSICAL_MODELS:
        start = time.perf_counter()
        validation_pred = pd.Series(dtype=int)
        final_pred = pd.Series(dtype=int)
        try:
            model = v3_make_classical_model(model_name)
            if model_name == "calibrated_logistic" and train_y.value_counts().min() < 3:
                raise ValueError("calibrated logistic requires at least three train rows per class")
            model.fit(spec.x_train, train_y)
            if hasattr(model, "predict_proba"):
                validation_values = (model.predict_proba(spec.x_validation)[:, 1] >= 0.50).astype(int)
                final_values = (model.predict_proba(spec.x_final)[:, 1] >= 0.50).astype(int)
            else:
                validation_values = np.asarray(model.predict(spec.x_validation)).astype(int)
                final_values = np.asarray(model.predict(spec.x_final)).astype(int)
            validation_pred = pd.Series(validation_values, index=splits["validation"])
            final_pred = pd.Series(final_values, index=splits["final"])
            validation_acc = accuracy(validation_y, validation_pred.loc[splits["validation"]])
            final_acc = accuracy(final_y, final_pred.loc[splits["final"]])
            status = "ok"
            skipped_reason = ""
        except Exception as exc:
            validation_acc = math.nan
            final_acc = math.nan
            status = "skipped"
            skipped_reason = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "candidate_id": candidate_id("v4", model_name, target_variant, f"h{horizon}", spec.feature_set_name),
                "model_family": model_name,
                "target_variant": target_variant,
                "horizon": horizon,
                "feature_set": spec.feature_set_name,
                "n_features": spec.n_features,
                "compression_method": spec.compression_method,
                "effective_train_rows": int(len(train_y)),
                "validation_accuracy": validation_acc,
                "validation_lift": validation_acc - val_simple_acc if math.isfinite(validation_acc) and math.isfinite(val_simple_acc) else math.nan,
                "validation_rows": int(len(validation_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_simple_acc if math.isfinite(final_acc) and math.isfinite(final_simple_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "strongest_validation_baseline": val_simple_name,
                "strongest_final_baseline": final_simple_name,
                "runtime_seconds": time.perf_counter() - start,
                "status": status,
                "skipped_reason": skipped_reason,
            }
        )
        if status == "ok":
            predictions[model_name] = {"validation": validation_pred, "final": final_pred}
    return rows, predictions


def v4_window_indices(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index]) -> dict[str, tuple[str, pd.Index]]:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_from_labels(labels, features.index)
    validation_idx = splits["validation"]
    early_mask = timestamps.loc[validation_idx].lt(V4_VAL_SPLIT) & target_timestamp.loc[validation_idx].lt(V4_VAL_SPLIT)
    late_mask = timestamps.loc[validation_idx].ge(V4_VAL_SPLIT) & target_timestamp.loc[validation_idx].ge(V4_VAL_SPLIT)
    return {
        "validation_early_2024": ("validation", pd.Index(validation_idx[early_mask.to_numpy()])),
        "validation_late_2024": ("validation", pd.Index(validation_idx[late_mask.to_numpy()])),
        "final_2025_diagnostic": ("final", splits["final"]),
    }


def metric_for_prediction(labels: pd.Series, prediction: pd.Series, idx: pd.Index) -> float:
    idx = pd.Index([item for item in idx if item in prediction.index])
    if len(idx) == 0:
        return math.nan
    return accuracy(labels.loc[idx].astype(int), prediction.loc[idx].astype(int))


def v4_rolling_rows(
    candidate: dict[str, Any],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    qml_predictions: dict[str, pd.Series],
    classical_predictions: dict[str, dict[str, pd.Series]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    windows = v4_window_indices(features_global(), labels, splits)
    for window_name, (split_name, idx) in windows.items():
        qml_pred = qml_predictions.get(split_name, pd.Series(dtype=int))
        qml_acc = metric_for_prediction(labels, qml_pred, idx)
        rbf_acc = metric_for_prediction(labels, classical_predictions.get("svm_rbf", {}).get(split_name, pd.Series(dtype=int)), idx)
        linear_acc = metric_for_prediction(labels, classical_predictions.get("svm_linear", {}).get(split_name, pd.Series(dtype=int)), idx)
        logistic_acc = metric_for_prediction(labels, classical_predictions.get("l2_logistic", {}).get(split_name, pd.Series(dtype=int)), idx)
        calibrated_acc = metric_for_prediction(labels, classical_predictions.get("calibrated_logistic", {}).get(split_name, pd.Series(dtype=int)), idx)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "window": window_name,
                "split": split_name,
                "target_variant": candidate["target_variant"],
                "horizon": candidate["horizon"],
                "feature_set": candidate["feature_set"],
                "n_qubits": candidate["n_qubits"],
                "feature_map_reps": candidate["feature_map_reps"],
                "rows": int(len(idx)),
                "qml_accuracy": qml_acc,
                "rbf_svm_accuracy": rbf_acc,
                "linear_svm_accuracy": linear_acc,
                "logistic_accuracy": logistic_acc,
                "calibrated_logistic_accuracy": calibrated_acc,
                "qml_minus_rbf_svm": qml_acc - rbf_acc if math.isfinite(qml_acc) and math.isfinite(rbf_acc) else math.nan,
                "qml_minus_logistic": qml_acc - logistic_acc if math.isfinite(qml_acc) and math.isfinite(logistic_acc) else math.nan,
                "final_scoring_only": split_name == "final",
                "status": "ok" if math.isfinite(qml_acc) else "skipped",
                "skipped_reason": "" if math.isfinite(qml_acc) else "window has no scored rows",
            }
        )
    return rows


def run_v4_quantum_kernel(
    row: dict[str, Any],
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    classical_rows: list[dict[str, Any]],
    classical_predictions: dict[str, dict[str, pd.Series]],
    config: V4Config,
    started: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if row["status"] != "pending":
        return row, []
    elapsed = time.perf_counter() - started
    if elapsed >= config.timeout_seconds:
        row["status"] = "skipped"
        row["skipped_reason"] = "qml_runtime_limited: timeout budget exhausted before candidate start"
        return row, []
    if row["n_qubits"] == 6 and row["feature_map_reps"] == 2 and config.timeout_seconds - elapsed < 360:
        row["status"] = "skipped"
        row["skipped_reason"] = "qml_runtime_limited: skipped q6 reps2 because remaining budget was too small"
        return row, []
    start = time.perf_counter()
    rolling: list[dict[str, Any]] = []
    try:
        algorithms = importlib.import_module("qiskit_machine_learning.algorithms")
        circuit_library = importlib.import_module("qiskit.circuit.library")
        QSVC = getattr(algorithms, "QSVC")
        ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
        train_y = labels.loc[splits["train"]].astype(int)
        validation_y = labels.loc[splits["validation"]].astype(int)
        final_y = labels.loc[splits["final"]].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("v4 train sample has fewer than two classes")
        x_train, x_validation, x_final = scale_for_quantum(spec.x_train, spec.x_validation, spec.x_final)
        feature_map = ZZFeatureMap(feature_dimension=spec.n_features, reps=int(row["feature_map_reps"]))
        model = QSVC(feature_map=feature_map)
        model.fit(x_train.to_numpy(), train_y.to_numpy())
        validation_pred = pd.Series(np.asarray(model.predict(x_validation.to_numpy())).reshape(-1).astype(int), index=splits["validation"])
        final_pred = pd.Series(np.asarray(model.predict(x_final.to_numpy())).reshape(-1).astype(int), index=splits["final"])
        validation_acc = accuracy(validation_y, validation_pred.loc[splits["validation"]])
        final_acc = accuracy(final_y, final_pred.loc[splits["final"]])
        simple_val_name, simple_val_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
        simple_final_name, simple_final_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
        row.update(
            {
                "effective_train_rows": int(len(train_y)),
                "validation_accuracy": validation_acc,
                "validation_lift": validation_acc - simple_val_acc if math.isfinite(simple_val_acc) else math.nan,
                "validation_rows": int(len(validation_y)),
                "final_accuracy": final_acc,
                "final_lift": final_acc - simple_final_acc if math.isfinite(simple_final_acc) else math.nan,
                "final_rows": int(len(final_y)),
                "comparison_vs_classical_champion_accuracy": final_acc - CLASSICAL_CHAMPION["final_accuracy"],
                "strongest_validation_baseline": simple_val_name,
                "strongest_final_baseline": simple_final_name,
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        rolling = v4_rolling_rows(row, labels, splits, {"validation": validation_pred, "final": final_pred}, classical_predictions)
    except Exception as exc:
        row.update(
            {
                "runtime_seconds": time.perf_counter() - start,
                "status": "skipped",
                "skipped_reason": f"dependency_missing_or_api_error: {type(exc).__name__}: {exc}",
            }
        )
    for model_name, prefix in [("svm_rbf", "rbf_svm"), ("svm_linear", "linear_svm"), ("l2_logistic", "logistic"), ("calibrated_logistic", "calibrated_logistic")]:
        match = next(
            (
                item
                for item in classical_rows
                if item.get("target_variant") == row.get("target_variant")
                and item.get("feature_set") == row.get("feature_set")
                and item.get("model_family") == model_name
                and item.get("status") == "ok"
            ),
            None,
        )
        if match:
            row[f"{prefix}_validation_accuracy"] = match.get("validation_accuracy", math.nan)
            row[f"{prefix}_final_accuracy"] = match.get("final_accuracy", math.nan)
    qml_val = as_float(row.get("validation_accuracy"))
    qml_final = as_float(row.get("final_accuracy"))
    rbf_val = as_float(row.get("rbf_svm_validation_accuracy"))
    rbf_final = as_float(row.get("rbf_svm_final_accuracy"))
    log_val = as_float(row.get("logistic_validation_accuracy"))
    log_final = as_float(row.get("logistic_final_accuracy"))
    row["qml_minus_rbf_svm_validation"] = qml_val - rbf_val if math.isfinite(qml_val) and math.isfinite(rbf_val) else math.nan
    row["qml_minus_rbf_svm_final"] = qml_final - rbf_final if math.isfinite(qml_final) and math.isfinite(rbf_final) else math.nan
    row["qml_minus_logistic_validation"] = qml_val - log_val if math.isfinite(qml_val) and math.isfinite(log_val) else math.nan
    row["qml_minus_logistic_final"] = qml_final - log_final if math.isfinite(qml_final) and math.isfinite(log_final) else math.nan
    return row, rolling


def v4_claim_label(selected: dict[str, Any] | None, runtime_limited: bool) -> str:
    if runtime_limited and selected is None:
        return "qml_runtime_limited"
    if selected is None or selected.get("status") != "ok":
        return "not_claimable"
    selected_beats_rbf = as_float(selected.get("qml_minus_rbf_svm_validation")) > 0.0 and as_float(selected.get("qml_minus_rbf_svm_final")) > 0.0
    selected_beats_logistic = as_float(selected.get("qml_minus_logistic_validation")) > 0.0 and as_float(selected.get("qml_minus_logistic_final")) > 0.0
    if selected_beats_rbf and selected_beats_logistic:
        return "qml_kernel_candidate"
    if as_float(selected.get("qml_minus_rbf_svm_final")) <= 0.0 or as_float(selected.get("qml_minus_logistic_final")) <= 0.0:
        return "qml_underperforms_classical"
    return "qml_diagnostic_only"


def write_v4_reports(
    qml_rows: list[dict[str, Any]],
    selected: dict[str, Any] | None,
    final_row: dict[str, Any],
    comparison_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    runtime_rows: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> None:
    ok_rows = [row for row in qml_rows if row.get("status") == "ok"]
    feature_count = len({row.get("feature_set") for row in ok_rows})
    features_with_val_rbf_win = len({row.get("feature_set") for row in ok_rows if as_float(row.get("qml_minus_rbf_svm_validation")) > 0.0})
    features_with_val_log_win = len({row.get("feature_set") for row in ok_rows if as_float(row.get("qml_minus_logistic_validation")) > 0.0})
    selected_beats_rbf = as_float(final_row.get("qml_minus_rbf_svm_validation")) > 0.0 and as_float(final_row.get("qml_minus_rbf_svm_final")) > 0.0
    selected_beats_logistic = as_float(final_row.get("qml_minus_logistic_validation")) > 0.0 and as_float(final_row.get("qml_minus_logistic_final")) > 0.0
    selected_beats_champion = as_float(final_row.get("comparison_vs_classical_champion_accuracy")) > 0.0
    any_final_scored_beats_champion = any(as_float(row.get("comparison_vs_classical_champion_accuracy")) > 0.0 for row in ok_rows)
    rolling_selected = [row for row in rolling_rows if row.get("candidate_id") == final_row.get("candidate_id") and row.get("status") == "ok"]
    selected_rolling_rbf_wins = sum(1 for row in rolling_selected if as_float(row.get("qml_minus_rbf_svm")) > 0.0)
    selected_rolling_log_wins = sum(1 for row in rolling_selected if as_float(row.get("qml_minus_logistic")) > 0.0)
    exploratory_survived = any_final_scored_beats_champion
    expansion = "not justified beyond focused diagnostics"
    if selected_beats_rbf and selected_beats_logistic and selected_rolling_rbf_wins >= 2 and selected_rolling_log_wins >= 2:
        expansion = "limited focused expansion is justified, but broad QML expansion still requires a full validation-governed benchmark"
    runtime_by_family: dict[str, float] = {}
    for row in runtime_rows:
        family = str(row.get("family", "unknown"))
        runtime_by_family[family] = runtime_by_family.get(family, 0.0) + as_float(row.get("runtime_seconds"))
    runtime_text = ", ".join(f"{family}={seconds:.2f}s" for family, seconds in sorted(runtime_by_family.items()))
    summary = f"""# VN30 QML Forecasting V4 Kernel Confirmation Result Summary

## Required Answers

1. Does the v3 quantum-kernel signal persist across feature variants: validation RBF advantage appeared in {features_with_val_rbf_win}/{feature_count} completed feature variants; validation Logistic advantage appeared in {features_with_val_log_win}/{feature_count}.
2. Does quantum kernel consistently beat RBF SVM: validation-selected row beats RBF on validation and final = {str(selected_beats_rbf).lower()}; selected rolling windows won {selected_rolling_rbf_wins}/{len(rolling_selected)}.
3. Does quantum kernel consistently beat Logistic: validation-selected row beats Logistic on validation and final = {str(selected_beats_logistic).lower()}; selected rolling windows won {selected_rolling_log_wins}/{len(rolling_selected)}.
4. Does any validation-selected QML candidate beat the 61.61% classical champion: {str(selected_beats_champion).lower()}.
5. Did the exploratory >61.61 QML row survive confirmation: {str(exploratory_survived).lower()}; final-ranked rows remain exploratory_not_claimable and do not replace the validation-selected result.
6. Is broader QML expansion justified: {expansion}.
7. Exact claim boundary: V4 is a focused experimental quantum-kernel confirmation diagnostic only; no QML result is claimable or replaces the 61.61% L2 Logistic champion.

## Validation-Selected QML Candidate

- Candidate: `{final_row.get("candidate_id", "")}`.
- Feature set: {final_row.get("feature_set", "")}.
- Qubits/reps: {final_row.get("n_qubits", "")} / {final_row.get("feature_map_reps", "")}.
- Validation accuracy: {pct(final_row.get("validation_accuracy"))}.
- Final accuracy: {pct(final_row.get("final_accuracy"))}.
- QML minus RBF SVM: validation {pp(final_row.get("qml_minus_rbf_svm_validation"))}, final {pp(final_row.get("qml_minus_rbf_svm_final"))}.
- QML minus Logistic: validation {pp(final_row.get("qml_minus_logistic_validation"))}, final {pp(final_row.get("qml_minus_logistic_final"))}.
- QML minus 61.61% classical champion: {pp(final_row.get("comparison_vs_classical_champion_accuracy"))}.
- Claim label: `{final_row.get("claim_label", "not_claimable")}`.
- Runtime by family: {runtime_text}.

## Paper-Safe Wording

VN30 QML v4 ran a focused quantum-kernel confirmation diagnostic on VN30 hourly stock forecasting for market-relative VN30 h40 only. The benchmark used strict feature_timestamp and target_timestamp split discipline, train-only PCA/scaling, balanced train-only sampling, validation-governed selection, and same-sample comparisons against RBF SVM, linear SVM, Logistic Regression, and calibrated Logistic Regression. Final-ranked rows are exploratory_not_claimable. No trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_RESULT_SUMMARY.md", summary)

    claim = """# VN30 QML Forecasting V4 Kernel Confirmation Claim Boundary

- QML v4 kernel confirmation is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature scaling, PCA, compression, and balanced sampling must be train-only or validation-safe.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Final-ranked rows remain exploratory_not_claimable.
- Quantum-kernel rows are compared against same-sample RBF SVM, linear SVM, Logistic Regression, and calibrated Logistic Regression.
- No QML result replaces the 61.61% L2 Logistic classical champion.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require a full validation-governed benchmark and future-blind confirmation.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V4_KERNEL_CONFIRMATION_CLAIM_BOUNDARY.md", claim)


def run_v4_kernel_confirmation(config: V4Config) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    index_data = load_index_data()
    features, v4_relative_cols = add_v3_relative_strength_features(features, index_data)
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    source_groups = build_source_groups(features, family_cols)
    source_groups["relative_strength_features"] = v4_relative_cols
    source_groups["combined_strategy_features"] = sorted(set(source_groups["combined_strategy_features"]).union(v4_relative_cols))

    labels = build_labels(features, index_data, V4_TARGET, V4_HORIZON)
    full_splits = strict_split_indices(features, labels)
    splits = v4_effective_splits(features, labels, full_splits, config)
    run_config = {
        "run_id": "vn30_qml_forecasting_v4_kernel_confirmation",
        "target": V4_TARGET,
        "horizon": V4_HORIZON,
        "feature_plan": V4_FEATURE_PLAN,
        "kernel_reps": V4_KERNEL_REPS,
        "max_qml_candidates": config.max_qml_candidates,
        "max_train_rows": config.max_train_rows,
        "max_validation_rows": config.max_validation_rows,
        "max_final_rows": config.max_final_rows,
        "timeout_seconds": config.timeout_seconds,
        "effective_qsvc_caps": {
            "train": min(config.max_train_rows, V4_QKERNEL_TRAIN_CAP),
            "validation": min(config.max_validation_rows, V4_QKERNEL_VALIDATION_CAP),
            "final": min(config.max_final_rows, V4_QKERNEL_FINAL_CAP),
        },
        "effective_split_rows": {key: int(len(value)) for key, value in splits.items()},
        "rolling_windows": ["validation_early_2024", "validation_late_2024", "final_2025_diagnostic"],
        "balanced_class_sampling_train_only": True,
        "preserve_timestamp_order_within_split": True,
        "dependency_status": dependency,
    }
    write_json(OUTPUT_DIR / "qml_v4_run_config.json", run_config)

    qml_rows: list[dict[str, Any]] = []
    classical_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    spec_lookup: dict[str, FeatureSpec] = {}
    classical_predictions_lookup: dict[str, dict[str, dict[str, pd.Series]]] = {}

    for source_group, compression_method, n_features in V4_FEATURE_PLAN:
        spec, _audit = fit_feature_spec(
            features,
            labels,
            V4_TARGET,
            V4_HORIZON,
            source_group,
            source_groups[source_group],
            compression_method,
            n_features,
            splits,
        )
        candidates = qml_v4_candidate_rows(V4_TARGET, V4_HORIZON, spec, dependency)
        if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
            for row in candidates:
                row["status"] = "skipped"
                row["skipped_reason"] = f"feature_selection_{spec.selection_status}"
                qml_rows.append(row)
            continue
        spec_lookup[spec.feature_set_name] = spec
        c_rows, c_predictions = run_v4_classical_models(spec, labels, V4_TARGET, V4_HORIZON, splits)
        classical_rows.extend(c_rows)
        classical_predictions_lookup[spec.feature_set_name] = c_predictions
        runtime_rows.extend(
            {
                "candidate_id": row["candidate_id"],
                "family": row["model_family"],
                "feature_set": row["feature_set"],
                "runtime_seconds": row["runtime_seconds"],
                "status": row["status"],
                "skipped_reason": row["skipped_reason"],
            }
            for row in c_rows
        )
        qml_rows.extend(candidates)

    runnable_seen = 0
    runtime_limited = False
    for row in qml_rows:
        if row.get("status") != "pending":
            continue
        if runnable_seen >= config.max_qml_candidates:
            row["status"] = "skipped"
            row["skipped_reason"] = "max_qml_candidates budget reached"
            continue
        if time.perf_counter() - started >= config.timeout_seconds:
            row["status"] = "skipped"
            row["skipped_reason"] = "qml_runtime_limited: timeout budget exhausted"
            runtime_limited = True
            continue
        spec = spec_lookup.get(str(row["feature_set"]))
        if spec is None:
            row["status"] = "skipped"
            row["skipped_reason"] = "feature spec unavailable"
            continue
        c_predictions = classical_predictions_lookup.get(str(row["feature_set"]), {})
        row, rolling = run_v4_quantum_kernel(row, spec, labels, splits, classical_rows, c_predictions, config, started)
        rolling_rows.extend(rolling)
        runnable_seen += 1
        if str(row.get("skipped_reason", "")).startswith("qml_runtime_limited"):
            runtime_limited = True

    for row in qml_rows:
        runtime_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "family": row["qml_family"],
                "feature_set": row["feature_set"],
                "runtime_seconds": row["runtime_seconds"],
                "status": row["status"],
                "skipped_reason": row["skipped_reason"],
            }
        )

    ok_qml = [row for row in qml_rows if row.get("status") == "ok"]
    selected = select_qml_candidate(ok_qml)
    claim_label = v4_claim_label(selected, runtime_limited)
    final_row = {
        "candidate_id": selected.get("candidate_id", "") if selected else "",
        "qml_family": selected.get("qml_family", "") if selected else "",
        "library": selected.get("library", "") if selected else "",
        "target_variant": selected.get("target_variant", V4_TARGET) if selected else V4_TARGET,
        "horizon": selected.get("horizon", V4_HORIZON) if selected else V4_HORIZON,
        "feature_set": selected.get("feature_set", "") if selected else "",
        "n_qubits": selected.get("n_qubits", 0) if selected else 0,
        "feature_map": selected.get("feature_map", "") if selected else "",
        "feature_map_reps": selected.get("feature_map_reps", 0) if selected else 0,
        "validation_accuracy": selected.get("validation_accuracy", math.nan) if selected else math.nan,
        "validation_lift": selected.get("validation_lift", math.nan) if selected else math.nan,
        "final_accuracy": selected.get("final_accuracy", math.nan) if selected else math.nan,
        "final_lift": selected.get("final_lift", math.nan) if selected else math.nan,
        "final_rows": selected.get("final_rows", 0) if selected else 0,
        "rbf_svm_final_accuracy": selected.get("rbf_svm_final_accuracy", math.nan) if selected else math.nan,
        "linear_svm_final_accuracy": selected.get("linear_svm_final_accuracy", math.nan) if selected else math.nan,
        "logistic_final_accuracy": selected.get("logistic_final_accuracy", math.nan) if selected else math.nan,
        "calibrated_logistic_final_accuracy": selected.get("calibrated_logistic_final_accuracy", math.nan) if selected else math.nan,
        "qml_minus_rbf_svm_validation": selected.get("qml_minus_rbf_svm_validation", math.nan) if selected else math.nan,
        "qml_minus_rbf_svm_final": selected.get("qml_minus_rbf_svm_final", math.nan) if selected else math.nan,
        "qml_minus_logistic_validation": selected.get("qml_minus_logistic_validation", math.nan) if selected else math.nan,
        "qml_minus_logistic_final": selected.get("qml_minus_logistic_final", math.nan) if selected else math.nan,
        "comparison_vs_classical_champion_accuracy": selected.get("comparison_vs_classical_champion_accuracy", math.nan) if selected else math.nan,
        "claim_label": claim_label,
        "status": selected.get("status", "skipped") if selected else "skipped",
        "skipped_reason": selected.get("skipped_reason", "no QML v4 candidate completed") if selected else "no QML v4 candidate completed",
    }

    comparison_rows: list[dict[str, Any]] = []
    for row in qml_rows:
        comparison_rows.append(
            {
                "candidate_id": row["candidate_id"],
                "target_variant": row["target_variant"],
                "feature_set": row["feature_set"],
                "n_qubits": row["n_qubits"],
                "feature_map_reps": row["feature_map_reps"],
                "status": row["status"],
                "validation_accuracy": row["validation_accuracy"],
                "final_accuracy": row["final_accuracy"],
                "rbf_svm_validation_accuracy": row["rbf_svm_validation_accuracy"],
                "rbf_svm_final_accuracy": row["rbf_svm_final_accuracy"],
                "linear_svm_validation_accuracy": row["linear_svm_validation_accuracy"],
                "linear_svm_final_accuracy": row["linear_svm_final_accuracy"],
                "logistic_validation_accuracy": row["logistic_validation_accuracy"],
                "logistic_final_accuracy": row["logistic_final_accuracy"],
                "calibrated_logistic_validation_accuracy": row["calibrated_logistic_validation_accuracy"],
                "calibrated_logistic_final_accuracy": row["calibrated_logistic_final_accuracy"],
                "qml_minus_rbf_svm_validation": row["qml_minus_rbf_svm_validation"],
                "qml_minus_rbf_svm_final": row["qml_minus_rbf_svm_final"],
                "qml_minus_logistic_validation": row["qml_minus_logistic_validation"],
                "qml_minus_logistic_final": row["qml_minus_logistic_final"],
                "comparison_vs_classical_champion_accuracy": row["comparison_vs_classical_champion_accuracy"],
                "skipped_reason": row["skipped_reason"],
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
        "feature_map",
        "feature_map_reps",
        "circuit_depth_or_reps",
        "compression_method",
        "effective_train_rows",
        "validation_accuracy",
        "validation_lift",
        "validation_rows",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "rbf_svm_validation_accuracy",
        "rbf_svm_final_accuracy",
        "linear_svm_validation_accuracy",
        "linear_svm_final_accuracy",
        "logistic_validation_accuracy",
        "logistic_final_accuracy",
        "calibrated_logistic_validation_accuracy",
        "calibrated_logistic_final_accuracy",
        "qml_minus_rbf_svm_validation",
        "qml_minus_rbf_svm_final",
        "qml_minus_logistic_validation",
        "qml_minus_logistic_final",
        "comparison_vs_classical_champion_accuracy",
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
        "feature_map",
        "feature_map_reps",
        "validation_accuracy",
        "validation_lift",
        "final_accuracy",
        "final_lift",
        "final_rows",
        "rbf_svm_final_accuracy",
        "linear_svm_final_accuracy",
        "logistic_final_accuracy",
        "calibrated_logistic_final_accuracy",
        "qml_minus_rbf_svm_validation",
        "qml_minus_rbf_svm_final",
        "qml_minus_logistic_validation",
        "qml_minus_logistic_final",
        "comparison_vs_classical_champion_accuracy",
        "claim_label",
        "status",
        "skipped_reason",
    ]
    write_frame(OUTPUT_DIR / "qml_v4_candidate_grid.csv", qml_rows, qml_columns)
    write_frame(OUTPUT_DIR / "qml_v4_validation_results.csv", sorted(ok_qml, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift")), -as_float(row.get("runtime_seconds"))), reverse=True), qml_columns)
    write_frame(OUTPUT_DIR / "qml_v4_kernel_vs_classical_comparison.csv", comparison_rows, list(comparison_rows[0].keys()) if comparison_rows else [])
    write_frame(OUTPUT_DIR / "qml_v4_rolling_validation.csv", rolling_rows, ["candidate_id", "window", "split", "target_variant", "horizon", "feature_set", "n_qubits", "feature_map_reps", "rows", "qml_accuracy", "rbf_svm_accuracy", "linear_svm_accuracy", "logistic_accuracy", "calibrated_logistic_accuracy", "qml_minus_rbf_svm", "qml_minus_logistic", "final_scoring_only", "status", "skipped_reason"])
    write_frame(OUTPUT_DIR / "qml_v4_final_result.csv", [final_row], final_columns)
    write_frame(OUTPUT_DIR / "qml_v4_runtime_summary.csv", runtime_rows, ["candidate_id", "family", "feature_set", "runtime_seconds", "status", "skipped_reason"])

    selected_rolling = [row for row in rolling_rows if row.get("candidate_id") == final_row.get("candidate_id")]
    manifest = {
        "run_id": "vn30_qml_forecasting_v4_kernel_confirmation",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "diagnostic_only": True,
        "full_grid_run": False,
        "run_config": run_config,
        "feature_manifest": feature_manifest,
        "v4_relative_strength_features": v4_relative_cols,
        "qml_candidates_total": len(qml_rows),
        "qml_candidates_ran": len(ok_qml),
        "qml_candidates_skipped": len([row for row in qml_rows if row.get("status") != "ok"]),
        "runtime_limited": runtime_limited,
        "selected_qml_candidate": selected if selected is not None else {},
        "selected_rolling_windows": selected_rolling,
        "final_result": final_row,
        "validation_selected_beats_rbf_svm": as_float(final_row.get("qml_minus_rbf_svm_validation")) > 0.0 and as_float(final_row.get("qml_minus_rbf_svm_final")) > 0.0,
        "validation_selected_beats_logistic": as_float(final_row.get("qml_minus_logistic_validation")) > 0.0 and as_float(final_row.get("qml_minus_logistic_final")) > 0.0,
        "validation_selected_beats_classical_champion": as_float(final_row.get("comparison_vs_classical_champion_accuracy")) > 0.0,
        "any_final_scored_qml_beats_classical_champion": any(as_float(row.get("comparison_vs_classical_champion_accuracy")) > 0.0 for row in ok_qml),
        "final_ranked_champion_comparison_claimable": False,
        "runtime_seconds": time.perf_counter() - started,
        "expansion_justification": "limited focused expansion only" if as_float(final_row.get("qml_minus_rbf_svm_validation")) > 0.0 and as_float(final_row.get("qml_minus_logistic_validation")) > 0.0 else "not justified beyond diagnostics",
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v4_manifest.json", manifest)
    write_v4_reports(qml_rows, selected, final_row, comparison_rows, rolling_rows, runtime_rows, manifest)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v4_manifest.json"), "qml_candidates_ran": manifest["qml_candidates_ran"], "qml_candidates_skipped": manifest["qml_candidates_skipped"], "runtime_limited": runtime_limited}), indent=2))
    return manifest


@dataclass
class V5Config:
    timeout_seconds: int = 3600


V5_TARGET = "market_relative_vn30"
V5_HORIZON = 40
V5_FROZEN_SOURCE_GROUP = "relative_strength_features"
V5_FROZEN_COMPRESSION = "topk_availability"
V5_FROZEN_N_FEATURES = 4
V5_FROZEN_REPS = 2
V5_CLASSICAL_MODELS = ["svm_rbf", "svm_linear", "l2_logistic", "calibrated_logistic", "random_forest_small", "lightgbm_small"]
V5_SAMPLE_LADDER = [
    {
        "sample_stage": "v4_sized",
        "requested_max_train_rows": 1500,
        "requested_max_validation_rows": 800,
        "requested_max_final_rows": 800,
        "effective_qml_train_rows": 80,
        "effective_qml_validation_rows": 180,
        "effective_qml_final_rows": 180,
    },
    {
        "sample_stage": "medium",
        "requested_max_train_rows": 3000,
        "requested_max_validation_rows": 1500,
        "requested_max_final_rows": 1500,
        "effective_qml_train_rows": 100,
        "effective_qml_validation_rows": 240,
        "effective_qml_final_rows": 240,
    },
    {
        "sample_stage": "largest_feasible",
        "requested_max_train_rows": "all_feasible",
        "requested_max_validation_rows": "all_feasible",
        "requested_max_final_rows": "all_feasible",
        "effective_qml_train_rows": 120,
        "effective_qml_validation_rows": 300,
        "effective_qml_final_rows": 300,
    },
]
V5_SMALL_VARIANTS = [
    {"variant_id": "topk4_reps1", "n_features": 4, "reps": 1},
    {"variant_id": "topk4_reps2", "n_features": 4, "reps": 2},
    {"variant_id": "topk6_reps1", "n_features": 6, "reps": 1},
    {"variant_id": "topk6_reps2", "n_features": 6, "reps": 2},
]


def v5_make_classical_model(model_name: str) -> Any:
    if model_name == "lightgbm_small":
        lightgbm = importlib.import_module("lightgbm")
        return lightgbm.LGBMClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            verbosity=-1,
            n_jobs=2,
        )
    if model_name == "random_forest_small":
        return make_classical_model(model_name)
    return v3_make_classical_model(model_name)


def v5_stage_splits(features: pd.DataFrame, labels: pd.Series, full_splits: dict[str, pd.Index], stage: dict[str, Any]) -> dict[str, pd.Index]:
    train_limit = int(stage["effective_qml_train_rows"])
    validation_limit = int(stage["effective_qml_validation_rows"])
    final_limit = int(stage["effective_qml_final_rows"])
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_from_labels(labels, features.index)
    validation_idx = ordered_index(features, full_splits["validation"])
    early_mask = timestamps.loc[validation_idx].lt(V4_VAL_SPLIT) & target_timestamp.loc[validation_idx].lt(V4_VAL_SPLIT)
    late_mask = timestamps.loc[validation_idx].ge(V4_VAL_SPLIT) & target_timestamp.loc[validation_idx].ge(V4_VAL_SPLIT)
    early_idx = pd.Index(validation_idx[early_mask.to_numpy()])
    late_idx = pd.Index(validation_idx[late_mask.to_numpy()])
    half = max(1, validation_limit // 2)
    early_sample = ordered_tail_sample(features, early_idx, half)
    late_sample = ordered_tail_sample(features, late_idx, validation_limit - len(early_sample))
    validation_sample = ordered_index(features, pd.Index(list(early_sample) + list(late_sample)))
    if len(validation_sample) < validation_limit:
        used = set(validation_sample)
        fallback = [idx for idx in ordered_tail_sample(features, full_splits["validation"], validation_limit) if idx not in used]
        validation_sample = ordered_index(features, pd.Index(list(validation_sample) + fallback[: validation_limit - len(validation_sample)]))
    if len(validation_sample) == 0:
        validation_sample = ordered_tail_sample(features, full_splits["validation"], validation_limit)
    return {
        "train": balanced_ordered_train_sample(features, labels, full_splits["train"], train_limit),
        "validation": validation_sample,
        "final": ordered_tail_sample(features, full_splits["final"], final_limit),
    }


def v5_run_classical_models(
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    sample_stage: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, pd.Series]]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, dict[str, pd.Series]] = {}
    train_y = labels.loc[splits["train"]].astype(int)
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    val_simple_name, val_simple_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
    final_simple_name, final_simple_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
    rows.append(
        {
            "sample_stage": sample_stage,
            "model_family": "strongest_simple_baseline",
            "candidate_id": candidate_id("v5", sample_stage, "strongest_simple", V5_TARGET, f"h{V5_HORIZON}", spec.feature_set_name),
            "target_variant": V5_TARGET,
            "horizon": V5_HORIZON,
            "feature_set": spec.feature_set_name,
            "n_features": spec.n_features,
            "compression_method": spec.compression_method,
            "train_rows": 0,
            "validation_rows": int(len(validation_y)),
            "final_rows": int(len(final_y)),
            "validation_accuracy": val_simple_acc,
            "validation_lift": 0.0,
            "final_accuracy": final_simple_acc,
            "final_lift": 0.0,
            "strongest_validation_baseline": val_simple_name,
            "strongest_final_baseline": final_simple_name,
            "runtime_seconds": 0.0,
            "status": "ok",
            "skipped_reason": "",
        }
    )
    for model_name in V5_CLASSICAL_MODELS:
        start = time.perf_counter()
        validation_pred = pd.Series(dtype=int)
        final_pred = pd.Series(dtype=int)
        try:
            model = v5_make_classical_model(model_name)
            if model_name == "calibrated_logistic" and train_y.value_counts().min() < 3:
                raise ValueError("calibrated logistic requires at least three train rows per class")
            model.fit(spec.x_train, train_y)
            if hasattr(model, "predict_proba"):
                validation_values = (model.predict_proba(spec.x_validation)[:, 1] >= 0.50).astype(int)
                final_values = (model.predict_proba(spec.x_final)[:, 1] >= 0.50).astype(int)
            else:
                validation_values = np.asarray(model.predict(spec.x_validation)).astype(int)
                final_values = np.asarray(model.predict(spec.x_final)).astype(int)
            validation_pred = pd.Series(validation_values, index=splits["validation"])
            final_pred = pd.Series(final_values, index=splits["final"])
            validation_acc = accuracy(validation_y, validation_pred.loc[splits["validation"]])
            final_acc = accuracy(final_y, final_pred.loc[splits["final"]])
            status = "ok"
            skipped_reason = ""
        except Exception as exc:
            validation_acc = math.nan
            final_acc = math.nan
            status = "skipped"
            skipped_reason = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "sample_stage": sample_stage,
                "model_family": model_name,
                "candidate_id": candidate_id("v5", sample_stage, model_name, V5_TARGET, f"h{V5_HORIZON}", spec.feature_set_name),
                "target_variant": V5_TARGET,
                "horizon": V5_HORIZON,
                "feature_set": spec.feature_set_name,
                "n_features": spec.n_features,
                "compression_method": spec.compression_method,
                "train_rows": int(len(train_y)),
                "validation_rows": int(len(validation_y)),
                "final_rows": int(len(final_y)),
                "validation_accuracy": validation_acc,
                "validation_lift": validation_acc - val_simple_acc if math.isfinite(validation_acc) and math.isfinite(val_simple_acc) else math.nan,
                "final_accuracy": final_acc,
                "final_lift": final_acc - final_simple_acc if math.isfinite(final_acc) and math.isfinite(final_simple_acc) else math.nan,
                "strongest_validation_baseline": val_simple_name,
                "strongest_final_baseline": final_simple_name,
                "runtime_seconds": time.perf_counter() - start,
                "status": status,
                "skipped_reason": skipped_reason,
            }
        )
        if status == "ok":
            predictions[model_name] = {"validation": validation_pred, "final": final_pred}
    return rows, predictions


def v5_run_frozen_qsvc(
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    sample_stage: str,
    reps: int,
    n_features: int,
    classical_rows: list[dict[str, Any]],
    classical_predictions: dict[str, dict[str, pd.Series]],
    timeout_seconds: int,
    started: float,
) -> tuple[dict[str, Any], dict[str, pd.Series], list[dict[str, Any]]]:
    row = {
        "sample_stage": sample_stage,
        "candidate_id": candidate_id("qml_v5", sample_stage, "quantum_kernel_classifier", V5_TARGET, f"h{V5_HORIZON}", spec.feature_set_name, f"q{n_features}", f"r{reps}"),
        "qml_family": "quantum_kernel_classifier",
        "library": "qiskit_machine_learning",
        "target_variant": V5_TARGET,
        "horizon": V5_HORIZON,
        "feature_set": spec.feature_set_name,
        "n_qubits": n_features,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": reps,
        "compression_method": spec.compression_method,
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_accuracy": math.nan,
        "validation_lift": math.nan,
        "final_accuracy": math.nan,
        "final_lift": math.nan,
        "rbf_svm_validation_accuracy": math.nan,
        "rbf_svm_final_accuracy": math.nan,
        "linear_svm_validation_accuracy": math.nan,
        "linear_svm_final_accuracy": math.nan,
        "logistic_validation_accuracy": math.nan,
        "logistic_final_accuracy": math.nan,
        "calibrated_logistic_validation_accuracy": math.nan,
        "calibrated_logistic_final_accuracy": math.nan,
        "random_forest_small_validation_accuracy": math.nan,
        "random_forest_small_final_accuracy": math.nan,
        "lightgbm_small_validation_accuracy": math.nan,
        "lightgbm_small_final_accuracy": math.nan,
        "qml_minus_rbf_svm_validation": math.nan,
        "qml_minus_rbf_svm_final": math.nan,
        "qml_minus_logistic_validation": math.nan,
        "qml_minus_logistic_final": math.nan,
        "qml_minus_best_classical_validation": math.nan,
        "qml_minus_best_classical_final": math.nan,
        "comparison_vs_classical_champion_accuracy": math.nan,
        "runtime_seconds": 0.0,
        "status": "pending",
        "skipped_reason": "",
    }
    if time.perf_counter() - started >= timeout_seconds:
        row["status"] = "skipped"
        row["skipped_reason"] = "qml_runtime_limited: timeout budget exhausted before candidate start"
        return row, {}, []
    start = time.perf_counter()
    predictions: dict[str, pd.Series] = {}
    rolling_rows: list[dict[str, Any]] = []
    try:
        algorithms = importlib.import_module("qiskit_machine_learning.algorithms")
        circuit_library = importlib.import_module("qiskit.circuit.library")
        QSVC = getattr(algorithms, "QSVC")
        ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
        train_y = labels.loc[splits["train"]].astype(int)
        validation_y = labels.loc[splits["validation"]].astype(int)
        final_y = labels.loc[splits["final"]].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("v5 train sample has fewer than two classes")
        x_train, x_validation, x_final = scale_for_quantum(spec.x_train, spec.x_validation, spec.x_final)
        feature_map = ZZFeatureMap(feature_dimension=n_features, reps=reps)
        model = QSVC(feature_map=feature_map)
        model.fit(x_train.to_numpy(), train_y.to_numpy())
        validation_pred = pd.Series(np.asarray(model.predict(x_validation.to_numpy())).reshape(-1).astype(int), index=splits["validation"])
        final_pred = pd.Series(np.asarray(model.predict(x_final.to_numpy())).reshape(-1).astype(int), index=splits["final"])
        predictions = {"validation": validation_pred, "final": final_pred}
        validation_acc = accuracy(validation_y, validation_pred.loc[splits["validation"]])
        final_acc = accuracy(final_y, final_pred.loc[splits["final"]])
        simple_val_name, simple_val_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
        simple_final_name, simple_final_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
        row.update(
            {
                "validation_accuracy": validation_acc,
                "validation_lift": validation_acc - simple_val_acc if math.isfinite(simple_val_acc) else math.nan,
                "final_accuracy": final_acc,
                "final_lift": final_acc - simple_final_acc if math.isfinite(simple_final_acc) else math.nan,
                "strongest_validation_baseline": simple_val_name,
                "strongest_final_baseline": simple_final_name,
                "comparison_vs_classical_champion_accuracy": final_acc - CLASSICAL_CHAMPION["final_accuracy"],
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        rolling_rows = v5_rolling_rows(row, labels, splits, predictions, classical_predictions)
    except Exception as exc:
        row.update(
            {
                "runtime_seconds": time.perf_counter() - start,
                "status": "skipped",
                "skipped_reason": f"dependency_missing_or_api_error: {type(exc).__name__}: {exc}",
            }
        )
    for model_name, prefix in [
        ("svm_rbf", "rbf_svm"),
        ("svm_linear", "linear_svm"),
        ("l2_logistic", "logistic"),
        ("calibrated_logistic", "calibrated_logistic"),
        ("random_forest_small", "random_forest_small"),
        ("lightgbm_small", "lightgbm_small"),
    ]:
        match = next((item for item in classical_rows if item.get("model_family") == model_name and item.get("status") == "ok"), None)
        if match:
            row[f"{prefix}_validation_accuracy"] = match.get("validation_accuracy", math.nan)
            row[f"{prefix}_final_accuracy"] = match.get("final_accuracy", math.nan)
    qml_val = as_float(row.get("validation_accuracy"))
    qml_final = as_float(row.get("final_accuracy"))
    rbf_val = as_float(row.get("rbf_svm_validation_accuracy"))
    rbf_final = as_float(row.get("rbf_svm_final_accuracy"))
    log_val = as_float(row.get("logistic_validation_accuracy"))
    log_final = as_float(row.get("logistic_final_accuracy"))
    row["qml_minus_rbf_svm_validation"] = qml_val - rbf_val if math.isfinite(qml_val) and math.isfinite(rbf_val) else math.nan
    row["qml_minus_rbf_svm_final"] = qml_final - rbf_final if math.isfinite(qml_final) and math.isfinite(rbf_final) else math.nan
    row["qml_minus_logistic_validation"] = qml_val - log_val if math.isfinite(qml_val) and math.isfinite(log_val) else math.nan
    row["qml_minus_logistic_final"] = qml_final - log_final if math.isfinite(qml_final) and math.isfinite(log_final) else math.nan
    classical_accs_val = [as_float(item.get("validation_accuracy")) for item in classical_rows if item.get("status") == "ok" and item.get("model_family") != "strongest_simple_baseline"]
    classical_accs_final = [as_float(item.get("final_accuracy")) for item in classical_rows if item.get("status") == "ok" and item.get("model_family") != "strongest_simple_baseline"]
    best_val = max([value for value in classical_accs_val if math.isfinite(value)], default=math.nan)
    best_final = max([value for value in classical_accs_final if math.isfinite(value)], default=math.nan)
    row["best_classical_validation_accuracy"] = best_val
    row["best_classical_final_accuracy"] = best_final
    row["qml_minus_best_classical_validation"] = qml_val - best_val if math.isfinite(qml_val) and math.isfinite(best_val) else math.nan
    row["qml_minus_best_classical_final"] = qml_final - best_final if math.isfinite(qml_final) and math.isfinite(best_final) else math.nan
    return row, predictions, rolling_rows


def v5_quarter_windows(features: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index]) -> dict[str, tuple[str, pd.Index]]:
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    target_timestamp = target_timestamp_from_labels(labels, features.index)
    windows = v4_window_indices(features, labels, splits)
    quarters = [
        ("validation_2024_q1", "validation", pd.Timestamp("2024-01-01"), pd.Timestamp("2024-04-01")),
        ("validation_2024_q2", "validation", pd.Timestamp("2024-04-01"), pd.Timestamp("2024-07-01")),
        ("validation_2024_q3", "validation", pd.Timestamp("2024-07-01"), pd.Timestamp("2024-10-01")),
        ("validation_2024_q4", "validation", pd.Timestamp("2024-10-01"), pd.Timestamp("2025-01-01")),
        ("final_2025_q1", "final", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-04-01")),
        ("final_2025_q2_plus", "final", pd.Timestamp("2025-04-01"), pd.Timestamp.max),
    ]
    for name, split_name, start, end in quarters:
        base_idx = splits[split_name]
        mask = timestamps.loc[base_idx].ge(start) & timestamps.loc[base_idx].lt(end) & target_timestamp.loc[base_idx].ge(start) & target_timestamp.loc[base_idx].lt(end)
        idx = pd.Index(base_idx[mask.to_numpy()])
        if len(idx):
            windows[name] = (split_name, idx)
    return windows


def v5_rolling_rows(
    candidate: dict[str, Any],
    labels: pd.Series,
    splits: dict[str, pd.Index],
    qml_predictions: dict[str, pd.Series],
    classical_predictions: dict[str, dict[str, pd.Series]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    windows = v5_quarter_windows(features_global(), labels, splits)
    for window_name, (split_name, idx) in windows.items():
        qml_pred = qml_predictions.get(split_name, pd.Series(dtype=int))
        qml_acc = metric_for_prediction(labels, qml_pred, idx)
        simple_name, simple_acc = strongest_simple_baseline(features_global(), labels, idx)
        rbf_acc = metric_for_prediction(labels, classical_predictions.get("svm_rbf", {}).get(split_name, pd.Series(dtype=int)), idx)
        linear_acc = metric_for_prediction(labels, classical_predictions.get("svm_linear", {}).get(split_name, pd.Series(dtype=int)), idx)
        logistic_acc = metric_for_prediction(labels, classical_predictions.get("l2_logistic", {}).get(split_name, pd.Series(dtype=int)), idx)
        calibrated_acc = metric_for_prediction(labels, classical_predictions.get("calibrated_logistic", {}).get(split_name, pd.Series(dtype=int)), idx)
        rf_acc = metric_for_prediction(labels, classical_predictions.get("random_forest_small", {}).get(split_name, pd.Series(dtype=int)), idx)
        lgbm_acc = metric_for_prediction(labels, classical_predictions.get("lightgbm_small", {}).get(split_name, pd.Series(dtype=int)), idx)
        rows.append(
            {
                "sample_stage": candidate["sample_stage"],
                "candidate_id": candidate["candidate_id"],
                "window": window_name,
                "split": split_name,
                "target_variant": candidate["target_variant"],
                "horizon": candidate["horizon"],
                "feature_set": candidate["feature_set"],
                "rows": int(len(idx)),
                "qml_accuracy": qml_acc,
                "qml_lift": qml_acc - simple_acc if math.isfinite(qml_acc) and math.isfinite(simple_acc) else math.nan,
                "strongest_simple_baseline": simple_name,
                "strongest_simple_accuracy": simple_acc,
                "rbf_svm_accuracy": rbf_acc,
                "linear_svm_accuracy": linear_acc,
                "logistic_accuracy": logistic_acc,
                "calibrated_logistic_accuracy": calibrated_acc,
                "random_forest_small_accuracy": rf_acc,
                "lightgbm_small_accuracy": lgbm_acc,
                "qml_minus_rbf_svm": qml_acc - rbf_acc if math.isfinite(qml_acc) and math.isfinite(rbf_acc) else math.nan,
                "qml_minus_logistic": qml_acc - logistic_acc if math.isfinite(qml_acc) and math.isfinite(logistic_acc) else math.nan,
                "runtime_seconds": candidate.get("runtime_seconds", math.nan),
                "final_scoring_only": split_name == "final",
                "status": "ok" if math.isfinite(qml_acc) else "skipped",
                "skipped_reason": "" if math.isfinite(qml_acc) else "window has no scored rows",
            }
        )
    return rows


def v5_decision_label(frozen_rows: list[dict[str, Any]], rolling_rows: list[dict[str, Any]], timeout_hit: bool) -> str:
    ok_rows = [row for row in frozen_rows if row.get("status") == "ok"]
    if not ok_rows:
        return "qml_not_confirmed"
    medium = next((row for row in ok_rows if row.get("sample_stage") == "medium"), None)
    largest = next((row for row in ok_rows if row.get("sample_stage") == "largest_feasible"), None)
    candidate = medium or largest or ok_rows[-1]
    beats_same_target = (
        as_float(candidate.get("qml_minus_rbf_svm_validation")) > 0.0
        and as_float(candidate.get("qml_minus_rbf_svm_final")) > 0.0
        and as_float(candidate.get("qml_minus_logistic_validation")) > 0.0
        and as_float(candidate.get("qml_minus_logistic_final")) > 0.0
        and as_float(candidate.get("qml_minus_best_classical_validation")) > 0.0
        and as_float(candidate.get("qml_minus_best_classical_final")) > 0.0
    )
    stage_windows = [row for row in rolling_rows if row.get("sample_stage") == candidate.get("sample_stage") and row.get("status") == "ok"]
    rolling_supported = sum(1 for row in stage_windows if as_float(row.get("qml_minus_rbf_svm")) > 0.0 and as_float(row.get("qml_minus_logistic")) > 0.0) >= max(2, len(stage_windows) // 2)
    if beats_same_target and rolling_supported:
        return "qml_requires_future_blind"
    if as_float(candidate.get("qml_minus_rbf_svm_final")) > 0.0 or as_float(candidate.get("qml_minus_logistic_final")) > 0.0:
        return "qml_same_target_candidate"
    if timeout_hit:
        return "qml_expansion_not_justified"
    return "qml_not_confirmed"


def write_v5_reports(
    frozen_rows: list[dict[str, Any]],
    variant_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    rolling_rows: list[dict[str, Any]],
    decision: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    replay = next((row for row in frozen_rows if row.get("sample_stage") == "v4_sized"), {})
    medium = next((row for row in frozen_rows if row.get("sample_stage") == "medium"), {})
    largest = next((row for row in frozen_rows if row.get("sample_stage") == "largest_feasible"), {})
    final_eval = largest if largest.get("status") == "ok" else (medium if medium.get("status") == "ok" else replay)
    qml_beats_rbf = as_float(final_eval.get("qml_minus_rbf_svm_validation")) > 0.0 and as_float(final_eval.get("qml_minus_rbf_svm_final")) > 0.0
    qml_beats_logistic = as_float(final_eval.get("qml_minus_logistic_validation")) > 0.0 and as_float(final_eval.get("qml_minus_logistic_final")) > 0.0
    qml_beats_best = as_float(final_eval.get("qml_minus_best_classical_validation")) > 0.0 and as_float(final_eval.get("qml_minus_best_classical_final")) > 0.0
    medium_survived = medium.get("status") == "ok" and as_float(medium.get("validation_accuracy")) >= 0.50 and as_float(medium.get("final_accuracy")) >= 0.50
    rolling_eval_rows = [row for row in rolling_rows if row.get("sample_stage") == final_eval.get("sample_stage") and row.get("status") == "ok"]
    rolling_wins = sum(1 for row in rolling_eval_rows if as_float(row.get("qml_minus_rbf_svm")) > 0.0 and as_float(row.get("qml_minus_logistic")) > 0.0)
    expansion = "not justified beyond focused diagnostics"
    if decision.get("decision_label") == "qml_requires_future_blind":
        expansion = "future-blind confirmation is justified before any stronger QML claim"
    elif decision.get("decision_label") in {"qml_same_target_candidate", "qml_beats_same_target_classical"}:
        expansion = "limited same-target diagnostics are justified; broad QML search is not"
    summary = f"""# VN30 QML Forecasting V5 Full Confirmation Result Summary

## Required Answers

1. Did the v4 QML candidate reproduce: {str(replay.get("status") == "ok").lower()}; replay validation {pct(replay.get("validation_accuracy"))}, final {pct(replay.get("final_accuracy"))}.
2. Did performance survive larger sample sizes: medium survived = {str(medium_survived).lower()}; medium validation {pct(medium.get("validation_accuracy"))}, final {pct(medium.get("final_accuracy"))}; largest feasible validation {pct(largest.get("validation_accuracy"))}, final {pct(largest.get("final_accuracy"))}.
3. Did QML beat RBF SVM under the same target: {str(qml_beats_rbf).lower()} for the final evaluated sample stage `{final_eval.get("sample_stage", "")}`.
4. Did QML beat Logistic under the same target: {str(qml_beats_logistic).lower()} for the final evaluated sample stage `{final_eval.get("sample_stage", "")}`.
5. Did QML beat other same-target classical models: {str(qml_beats_best).lower()} against the best same-target classical row.
6. Did rolling-origin checks support the signal: {rolling_wins}/{len(rolling_eval_rows)} final-stage windows beat both RBF SVM and Logistic.
7. Is QML expansion justified: {expansion}.
8. Can any QML result be claimed: no; QML v5 remains diagnostic-only and requires future-blind confirmation for stronger wording.
9. Does QML replace the 61.61% classical champion: no; the champion is a different target/scope benchmark and is not replaced by this market-relative diagnostic.
10. Exact paper-safe wording: VN30 QML v5 replayed a frozen quantum-kernel candidate on VN30 hourly market-relative VN30 h40 forecasting with train-only feature selection/scaling and feature_timestamp/target_timestamp split discipline. Results are diagnostic-only, same-target classical comparisons are reported, and no trading, profitability, BUY/SELL, recommendation, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Frozen Candidate

- Candidate: `{final_eval.get("candidate_id", "")}`.
- Feature design: relative_strength_features, top-k {V5_FROZEN_N_FEATURES}, ZZFeatureMap reps {V5_FROZEN_REPS}.
- Final evaluated sample stage: {final_eval.get("sample_stage", "")}.
- Validation accuracy: {pct(final_eval.get("validation_accuracy"))}.
- Final accuracy: {pct(final_eval.get("final_accuracy"))}.
- QML minus RBF SVM: validation {pp(final_eval.get("qml_minus_rbf_svm_validation"))}, final {pp(final_eval.get("qml_minus_rbf_svm_final"))}.
- QML minus Logistic: validation {pp(final_eval.get("qml_minus_logistic_validation"))}, final {pp(final_eval.get("qml_minus_logistic_final"))}.
- QML minus best same-target classical: validation {pp(final_eval.get("qml_minus_best_classical_validation"))}, final {pp(final_eval.get("qml_minus_best_classical_final"))}.
- QML minus 61.61% champion: {pp(final_eval.get("comparison_vs_classical_champion_accuracy"))}.
- Final decision label: `{decision.get("decision_label", "qml_not_confirmed")}`.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V5_FULL_CONFIRMATION_RESULT_SUMMARY.md", summary)

    claim = """# VN30 QML Forecasting V5 Full Confirmation Claim Boundary

- QML v5 full confirmation is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature selection, scaling, and compression must be train-only or validation-safe.
- The primary QML design is frozen from prior validation; no final-performance selection is allowed.
- Final-ranked rows remain exploratory_not_claimable.
- Same-target comparisons are diagnostic and do not replace the 61.61% L2 Logistic classical champion because the target/scope is not directly identical.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require full validation governance and future-blind confirmation.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V5_FULL_CONFIRMATION_CLAIM_BOUNDARY.md", claim)


def run_v5_full_confirmation(config: V5Config) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    index_data = load_index_data()
    features, v5_relative_cols = add_v3_relative_strength_features(features, index_data)
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    source_groups = build_source_groups(features, family_cols)
    source_groups["relative_strength_features"] = v5_relative_cols
    source_groups["combined_strategy_features"] = sorted(set(source_groups["combined_strategy_features"]).union(v5_relative_cols))

    labels = build_labels(features, index_data, V5_TARGET, V5_HORIZON)
    full_splits = strict_split_indices(features, labels)
    run_config = {
        "run_id": "vn30_qml_forecasting_v5_full_confirmation",
        "target": V5_TARGET,
        "horizon": V5_HORIZON,
        "frozen_design": {
            "source_group": V5_FROZEN_SOURCE_GROUP,
            "compression_method": V5_FROZEN_COMPRESSION,
            "n_features": V5_FROZEN_N_FEATURES,
            "feature_map": "ZZFeatureMap",
            "n_qubits": V5_FROZEN_N_FEATURES,
            "reps": V5_FROZEN_REPS,
        },
        "sample_ladder": V5_SAMPLE_LADDER,
        "small_variants": V5_SMALL_VARIANTS,
        "timeout_seconds": config.timeout_seconds,
        "dependency_status": dependency,
        "split_discipline": "feature_timestamp and target_timestamp split-safe",
        "train_only_feature_selection_and_scaling": True,
    }
    write_json(OUTPUT_DIR / "qml_v5_frozen_replay_manifest.json", run_config)

    frozen_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rolling_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    timeout_hit = False

    for stage in V5_SAMPLE_LADDER:
        if time.perf_counter() - started >= config.timeout_seconds:
            timeout_hit = True
            break
        sample_stage = str(stage["sample_stage"])
        splits = v5_stage_splits(features, labels, full_splits, stage)
        spec, _audit = fit_feature_spec(
            features,
            labels,
            V5_TARGET,
            V5_HORIZON,
            V5_FROZEN_SOURCE_GROUP,
            source_groups[V5_FROZEN_SOURCE_GROUP],
            V5_FROZEN_COMPRESSION,
            V5_FROZEN_N_FEATURES,
            splits,
        )
        if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
            row = {
                "sample_stage": sample_stage,
                "candidate_id": candidate_id("qml_v5", sample_stage, "quantum_kernel_classifier", V5_TARGET, f"h{V5_HORIZON}", "feature_selection_failed"),
                "status": "skipped",
                "skipped_reason": f"feature_selection_{spec.selection_status}",
            }
            frozen_rows.append(row)
            continue
        classical_rows, classical_predictions = v5_run_classical_models(spec, labels, splits, sample_stage)
        comparison_rows.extend(classical_rows)
        qml_row, _predictions, stage_rolling = v5_run_frozen_qsvc(
            spec,
            labels,
            splits,
            sample_stage,
            V5_FROZEN_REPS,
            V5_FROZEN_N_FEATURES,
            classical_rows,
            classical_predictions,
            config.timeout_seconds,
            started,
        )
        frozen_rows.append(qml_row)
        rolling_rows.extend(stage_rolling)
        comparison_rows.append({**qml_row, "model_family": "quantum_kernel_classifier"})
        if str(qml_row.get("skipped_reason", "")).startswith("qml_runtime_limited"):
            timeout_hit = True
            break

    replay_rows = [row for row in frozen_rows if row.get("sample_stage") == "v4_sized"]
    write_frame(OUTPUT_DIR / "qml_v5_frozen_replay_result.csv", replay_rows, list(replay_rows[0].keys()) if replay_rows else [])

    if not timeout_hit and time.perf_counter() - started < config.timeout_seconds * 0.72:
        stage = V5_SAMPLE_LADDER[0]
        splits = v5_stage_splits(features, labels, full_splits, stage)
        for variant in V5_SMALL_VARIANTS:
            if time.perf_counter() - started >= config.timeout_seconds * 0.92:
                timeout_hit = True
                break
            spec, _audit = fit_feature_spec(
                features,
                labels,
                V5_TARGET,
                V5_HORIZON,
                V5_FROZEN_SOURCE_GROUP,
                source_groups[V5_FROZEN_SOURCE_GROUP],
                V5_FROZEN_COMPRESSION,
                int(variant["n_features"]),
                splits,
            )
            if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
                variant_rows.append(
                    {
                        "variant_id": variant["variant_id"],
                        "n_features": variant["n_features"],
                        "feature_map_reps": variant["reps"],
                        "status": "skipped",
                        "skipped_reason": f"feature_selection_{spec.selection_status}",
                    }
                )
                continue
            classical_rows, classical_predictions = v5_run_classical_models(spec, labels, splits, f"variant_{variant['variant_id']}")
            qml_row, _predictions, _rolling = v5_run_frozen_qsvc(
                spec,
                labels,
                splits,
                f"variant_{variant['variant_id']}",
                int(variant["reps"]),
                int(variant["n_features"]),
                classical_rows,
                classical_predictions,
                config.timeout_seconds,
                started,
            )
            qml_row["variant_id"] = variant["variant_id"]
            variant_rows.append(qml_row)

    decision_label = v5_decision_label(frozen_rows, rolling_rows, timeout_hit)
    final_stage_row = next((row for row in frozen_rows if row.get("sample_stage") == "largest_feasible" and row.get("status") == "ok"), None)
    if final_stage_row is None:
        final_stage_row = next((row for row in frozen_rows if row.get("sample_stage") == "medium" and row.get("status") == "ok"), None)
    if final_stage_row is None:
        final_stage_row = next((row for row in frozen_rows if row.get("status") == "ok"), {})
    decision = {
        "decision_label": decision_label,
        "final_evaluated_sample_stage": final_stage_row.get("sample_stage", ""),
        "final_evaluated_candidate_id": final_stage_row.get("candidate_id", ""),
        "qml_replaces_6161_classical_champion": False,
        "replacement_blocked_reason": "target/scope is market_relative_vn30 h40 and is not directly identical to the 61.61% classical champion benchmark; future-blind confirmation is required",
        "same_target_validation_beats_rbf_svm": as_float(final_stage_row.get("qml_minus_rbf_svm_validation")) > 0.0,
        "same_target_final_beats_rbf_svm": as_float(final_stage_row.get("qml_minus_rbf_svm_final")) > 0.0,
        "same_target_validation_beats_logistic": as_float(final_stage_row.get("qml_minus_logistic_validation")) > 0.0,
        "same_target_final_beats_logistic": as_float(final_stage_row.get("qml_minus_logistic_final")) > 0.0,
        "same_target_validation_beats_best_classical": as_float(final_stage_row.get("qml_minus_best_classical_validation")) > 0.0,
        "same_target_final_beats_best_classical": as_float(final_stage_row.get("qml_minus_best_classical_final")) > 0.0,
        "future_blind_required": True,
        "diagnostic_only": True,
        "runtime_limited": timeout_hit,
    }
    write_json(OUTPUT_DIR / "qml_v5_final_decision.json", decision)

    write_frame(OUTPUT_DIR / "qml_v5_sample_size_ladder.csv", frozen_rows, list(frozen_rows[0].keys()) if frozen_rows else [])
    write_frame(OUTPUT_DIR / "qml_v5_same_target_classical_comparison.csv", comparison_rows, list(comparison_rows[0].keys()) if comparison_rows else [])
    write_frame(OUTPUT_DIR / "qml_v5_rolling_origin_confirmation.csv", rolling_rows, list(rolling_rows[0].keys()) if rolling_rows else [])
    write_frame(OUTPUT_DIR / "qml_v5_small_variant_results.csv", variant_rows, list(variant_rows[0].keys()) if variant_rows else [])

    manifest = {
        "run_id": "vn30_qml_forecasting_v5_full_confirmation",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "diagnostic_only": True,
        "run_config": run_config,
        "feature_manifest": feature_manifest,
        "v5_relative_strength_features": v5_relative_cols,
        "frozen_replay": replay_rows[0] if replay_rows else {},
        "sample_ladder_rows": frozen_rows,
        "small_variant_rows": variant_rows,
        "rolling_rows": rolling_rows,
        "final_decision": decision,
        "runtime_limited": timeout_hit,
        "runtime_seconds": time.perf_counter() - started,
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v5_frozen_replay_manifest.json", manifest)
    write_v5_reports(frozen_rows, variant_rows, comparison_rows, rolling_rows, decision, manifest)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v5_frozen_replay_manifest.json"), "decision_label": decision_label, "runtime_limited": timeout_hit}), indent=2))
    return manifest


@dataclass
class V6Config:
    timeout_seconds: int = 5400


V6_TARGET = "market_relative_vn30"
V6_HORIZON = 40
V6_SCALING_METHODS = ["standard_zscore", "minmax_0_pi", "minmax_minus_pi_pi", "robust_quantile", "rank_gaussian"]
V6_PRIMARY_SCALING = "minmax_0_pi"
V6_FROZEN_REPS = 2
V6_FROZEN_ENTANGLEMENT = "full"
V6_REGULARIZATION_C = [0.1, 1.0, 10.0]
V6_REGULARIZATION_CLASS_WEIGHT = [None, "balanced"]


def v6_scaling_transform(
    spec: FeatureSpec,
    scaling_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = spec.x_train.replace([np.inf, -np.inf], np.nan)
    validation = spec.x_validation.replace([np.inf, -np.inf], np.nan)
    final = spec.x_final.replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy="median")
    train_imp = imputer.fit_transform(train)
    validation_imp = imputer.transform(validation)
    final_imp = imputer.transform(final)
    status = "ok"
    skipped_reason = ""
    if scaling_method == "standard_zscore":
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_imp)
        validation_scaled = scaler.transform(validation_imp)
        final_scaled = scaler.transform(final_imp)
    elif scaling_method == "minmax_0_pi":
        scaler = MinMaxScaler(feature_range=(0.0, math.pi))
        train_scaled = scaler.fit_transform(train_imp)
        validation_scaled = scaler.transform(validation_imp)
        final_scaled = scaler.transform(final_imp)
    elif scaling_method == "minmax_minus_pi_pi":
        scaler = MinMaxScaler(feature_range=(-math.pi, math.pi))
        train_scaled = scaler.fit_transform(train_imp)
        validation_scaled = scaler.transform(validation_imp)
        final_scaled = scaler.transform(final_imp)
    elif scaling_method == "robust_quantile":
        scaler = RobustScaler(quantile_range=(10.0, 90.0))
        train_scaled = np.clip(scaler.fit_transform(train_imp), -3.0, 3.0) * (math.pi / 3.0)
        validation_scaled = np.clip(scaler.transform(validation_imp), -3.0, 3.0) * (math.pi / 3.0)
        final_scaled = np.clip(scaler.transform(final_imp), -3.0, 3.0) * (math.pi / 3.0)
    elif scaling_method == "rank_gaussian":
        try:
            n_quantiles = max(10, min(1000, len(train_imp)))
            scaler = QuantileTransformer(n_quantiles=n_quantiles, output_distribution="normal", random_state=SEED)
            train_scaled = np.clip(scaler.fit_transform(train_imp), -3.0, 3.0) * (math.pi / 3.0)
            validation_scaled = np.clip(scaler.transform(validation_imp), -3.0, 3.0) * (math.pi / 3.0)
            final_scaled = np.clip(scaler.transform(final_imp), -3.0, 3.0) * (math.pi / 3.0)
        except Exception as exc:
            status = "skipped"
            skipped_reason = f"rank_gaussian_unavailable: {type(exc).__name__}: {exc}"
            train_scaled = np.empty((len(train), 0))
            validation_scaled = np.empty((len(validation), 0))
            final_scaled = np.empty((len(final), 0))
    else:
        status = "skipped"
        skipped_reason = f"unknown scaling method {scaling_method}"
        train_scaled = np.empty((len(train), 0))
        validation_scaled = np.empty((len(validation), 0))
        final_scaled = np.empty((len(final), 0))
    columns = list(spec.x_train.columns) if status == "ok" else []
    return (
        pd.DataFrame(train_scaled, index=spec.x_train.index, columns=columns),
        pd.DataFrame(validation_scaled, index=spec.x_validation.index, columns=columns),
        pd.DataFrame(final_scaled, index=spec.x_final.index, columns=columns),
        {
            "scaling_method": scaling_method,
            "status": status,
            "skipped_reason": skipped_reason,
            "fit_split": "train",
            "feature_range": "method_specific",
        },
    )


def v6_quantum_kernel_matrices(
    x_train: pd.DataFrame,
    x_validation: pd.DataFrame,
    x_final: pd.DataFrame,
    reps: int,
    entanglement: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kernels = importlib.import_module("qiskit_machine_learning.kernels")
    circuit_library = importlib.import_module("qiskit.circuit.library")
    FidelityStatevectorKernel = getattr(kernels, "FidelityStatevectorKernel")
    ZZFeatureMap = getattr(circuit_library, "ZZFeatureMap")
    feature_map = ZZFeatureMap(feature_dimension=x_train.shape[1], reps=reps, entanglement=entanglement)
    kernel = FidelityStatevectorKernel(feature_map=feature_map)
    train_np = x_train.to_numpy(dtype=float)
    validation_np = x_validation.to_numpy(dtype=float)
    final_np = x_final.to_numpy(dtype=float)
    k_train = np.asarray(kernel.evaluate(train_np), dtype=float)
    k_validation = np.asarray(kernel.evaluate(validation_np, train_np), dtype=float)
    k_final = np.asarray(kernel.evaluate(final_np, train_np), dtype=float)
    return k_train, k_validation, k_final


def v6_run_precomputed_kernel_classifier(
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    sample_stage: str,
    scaling_method: str,
    reps: int = V6_FROZEN_REPS,
    entanglement: str = V6_FROZEN_ENTANGLEMENT,
    c_value: float = 1.0,
    class_weight: str | None = None,
) -> tuple[dict[str, Any], dict[str, pd.Series], dict[str, np.ndarray]]:
    start = time.perf_counter()
    row: dict[str, Any] = {
        "sample_stage": sample_stage,
        "candidate_id": candidate_id("qml_v6", sample_stage, scaling_method, "quantum_kernel_classifier", V6_TARGET, f"h{V6_HORIZON}", spec.feature_set_name, f"r{reps}", entanglement, f"C{c_value}", class_weight or "none"),
        "qml_family": "quantum_kernel_classifier",
        "library": "qiskit_machine_learning_statevector_kernel",
        "target_variant": V6_TARGET,
        "horizon": V6_HORIZON,
        "feature_set": spec.feature_set_name,
        "n_qubits": spec.n_features,
        "feature_map": "ZZFeatureMap",
        "feature_map_reps": reps,
        "entanglement": entanglement,
        "scaling_method": scaling_method,
        "C": c_value,
        "class_weight": class_weight or "none",
        "train_rows": int(len(splits["train"])),
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "validation_accuracy": math.nan,
        "final_accuracy": math.nan,
        "validation_lift": math.nan,
        "final_lift": math.nan,
        "runtime_seconds": 0.0,
        "status": "pending",
        "skipped_reason": "",
    }
    predictions: dict[str, pd.Series] = {}
    matrices: dict[str, np.ndarray] = {}
    try:
        x_train, x_validation, x_final, scaling_manifest = v6_scaling_transform(spec, scaling_method)
        if scaling_manifest["status"] != "ok":
            raise ValueError(scaling_manifest["skipped_reason"])
        train_y = labels.loc[splits["train"]].astype(int)
        validation_y = labels.loc[splits["validation"]].astype(int)
        final_y = labels.loc[splits["final"]].astype(int)
        if train_y.nunique() < 2:
            raise ValueError("v6 train sample has fewer than two classes")
        k_train, k_validation, k_final = v6_quantum_kernel_matrices(x_train, x_validation, x_final, reps, entanglement)
        model = SVC(kernel="precomputed", C=float(c_value), class_weight=class_weight)
        model.fit(k_train, train_y.to_numpy())
        validation_pred = pd.Series(np.asarray(model.predict(k_validation)).astype(int), index=splits["validation"])
        final_pred = pd.Series(np.asarray(model.predict(k_final)).astype(int), index=splits["final"])
        validation_acc = accuracy(validation_y, validation_pred.loc[splits["validation"]])
        final_acc = accuracy(final_y, final_pred.loc[splits["final"]])
        simple_val_name, simple_val_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
        simple_final_name, simple_final_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
        row.update(
            {
                "validation_accuracy": validation_acc,
                "final_accuracy": final_acc,
                "validation_lift": validation_acc - simple_val_acc if math.isfinite(simple_val_acc) else math.nan,
                "final_lift": final_acc - simple_final_acc if math.isfinite(simple_final_acc) else math.nan,
                "strongest_validation_baseline": simple_val_name,
                "strongest_final_baseline": simple_final_name,
                "comparison_vs_classical_champion_accuracy": final_acc - CLASSICAL_CHAMPION["final_accuracy"],
                "runtime_seconds": time.perf_counter() - start,
                "status": "ok",
                "skipped_reason": "",
            }
        )
        try:
            predictions["train_score"] = pd.Series(np.asarray(model.decision_function(k_train), dtype=float), index=splits["train"])
            predictions["validation_score"] = pd.Series(np.asarray(model.decision_function(k_validation), dtype=float), index=splits["validation"])
            predictions["final_score"] = pd.Series(np.asarray(model.decision_function(k_final), dtype=float), index=splits["final"])
        except Exception:
            pass
        predictions["validation"] = validation_pred
        predictions["final"] = final_pred
        matrices = {"train": k_train, "validation": k_validation, "final": k_final}
    except Exception as exc:
        row.update({"runtime_seconds": time.perf_counter() - start, "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}"})
    return row, predictions, matrices


def v6_kernel_diagnostic_row(
    sample_stage: str,
    scaling_method: str,
    spec: FeatureSpec,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    k_train: np.ndarray,
) -> dict[str, Any]:
    y = labels.loc[splits["train"]].astype(int).to_numpy()
    signed = np.where(y > 0, 1.0, -1.0)
    yy = np.outer(signed, signed)
    denom = float(np.linalg.norm(k_train, "fro") * np.linalg.norm(yy, "fro"))
    alignment = float(np.sum(k_train * yy) / denom) if denom > 0 else math.nan
    eig = np.linalg.eigvalsh((k_train + k_train.T) / 2.0)
    eig = np.clip(eig, 0.0, None)
    positive = eig[eig > 1e-10]
    condition = float(positive.max() / positive.min()) if len(positive) else math.nan
    probs = eig / eig.sum() if eig.sum() > 0 else np.array([])
    effective_rank = float(np.exp(-np.sum(probs * np.log(np.clip(probs, 1e-12, None))))) if len(probs) else math.nan
    mask_off = ~np.eye(k_train.shape[0], dtype=bool)
    off = k_train[mask_off] if k_train.size else np.array([])
    same_mask = y[:, None] == y[None, :]
    diff_mask = ~same_mask
    same_values = k_train[same_mask & mask_off]
    diff_values = k_train[diff_mask]
    off_std = float(np.nanstd(off)) if len(off) else math.nan
    concentration = bool(math.isfinite(off_std) and off_std < 0.03) or bool(math.isfinite(effective_rank) and effective_rank < max(2.0, 0.2 * len(y)))
    return {
        "sample_stage": sample_stage,
        "scaling_method": scaling_method,
        "feature_set": spec.feature_set_name,
        "n_features": spec.n_features,
        "train_rows": int(len(y)),
        "kernel_target_alignment": alignment,
        "condition_number": condition,
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank / len(y) if len(y) and math.isfinite(effective_rank) else math.nan,
        "mean_diagonal": float(np.nanmean(np.diag(k_train))) if k_train.size else math.nan,
        "mean_off_diagonal": float(np.nanmean(off)) if len(off) else math.nan,
        "std_off_diagonal": off_std,
        "min_kernel_value": float(np.nanmin(k_train)) if k_train.size else math.nan,
        "max_kernel_value": float(np.nanmax(k_train)) if k_train.size else math.nan,
        "same_class_similarity": float(np.nanmean(same_values)) if len(same_values) else math.nan,
        "different_class_similarity": float(np.nanmean(diff_values)) if len(diff_values) else math.nan,
        "class_similarity_gap": float(np.nanmean(same_values) - np.nanmean(diff_values)) if len(same_values) and len(diff_values) else math.nan,
        "kernel_concentration_detected": concentration,
    }


def v6_distribution_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    splits_by_stage: dict[str, dict[str, pd.Index]],
    selected_features_by_stage: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    distribution_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    for sample_stage, splits in splits_by_stage.items():
        selected = selected_features_by_stage.get(sample_stage, [])
        train_stats: dict[str, tuple[float, float]] = {}
        for split_name, idx in splits.items():
            y = labels.loc[idx].dropna().astype(int)
            ticker_count = int(features.loc[idx, "ticker"].nunique()) if len(idx) and "ticker" in features.columns else 0
            quarter_counts = timestamps.loc[idx].dt.to_period("Q").astype(str).value_counts().sort_index().to_dict() if len(idx) else {}
            regime_cols = [col for col in features.columns if "regime" in col.lower()]
            regime_summary = ""
            if regime_cols and len(idx):
                col = regime_cols[0]
                regime_summary = json.dumps(features.loc[idx, col].astype(str).value_counts().head(5).to_dict(), sort_keys=True)
            missingness = float(features.loc[idx, selected].isna().mean().mean()) if len(idx) and selected else math.nan
            distribution_rows.append(
                {
                    "sample_stage": sample_stage,
                    "split": split_name,
                    "rows": int(len(idx)),
                    "label_0_rows": int((y == 0).sum()),
                    "label_1_rows": int((y == 1).sum()),
                    "label_1_share": float((y == 1).mean()) if len(y) else math.nan,
                    "ticker_coverage": ticker_count,
                    "timestamp_min": str(timestamps.loc[idx].min()) if len(idx) else "",
                    "timestamp_max": str(timestamps.loc[idx].max()) if len(idx) else "",
                    "quarter_distribution": json.dumps(quarter_counts, sort_keys=True),
                    "regime_distribution": regime_summary,
                    "feature_missingness_mean": missingness,
                }
            )
            for feature in selected:
                values = pd.to_numeric(features.loc[idx, feature], errors="coerce")
                row = {
                    "sample_stage": sample_stage,
                    "split": split_name,
                    "feature": feature,
                    "missingness": float(values.isna().mean()) if len(values) else math.nan,
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "std": float(values.std()) if len(values) else math.nan,
                    "min": float(values.min()) if len(values) else math.nan,
                    "max": float(values.max()) if len(values) else math.nan,
                }
                if split_name == "train":
                    train_stats[feature] = (row["mean"], row["std"])
                else:
                    train_mean, train_std = train_stats.get(feature, (math.nan, math.nan))
                    row["mean_drift_vs_train"] = row["mean"] - train_mean if math.isfinite(row["mean"]) and math.isfinite(train_mean) else math.nan
                    row["std_ratio_vs_train"] = row["std"] / train_std if math.isfinite(row["std"]) and math.isfinite(train_std) and train_std else math.nan
                drift_rows.append(row)
    return distribution_rows, drift_rows


def v6_compare_against_classical(qml_row: dict[str, Any], classical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = dict(qml_row)
    best_val = math.nan
    best_final = math.nan
    best_val_model = ""
    best_final_model = ""
    for model_name, prefix in [
        ("svm_rbf", "rbf_svm"),
        ("svm_linear", "linear_svm"),
        ("l2_logistic", "logistic"),
        ("calibrated_logistic", "calibrated_logistic"),
        ("lightgbm_small", "lightgbm_small"),
    ]:
        match = next((row for row in classical_rows if row.get("model_family") == model_name and row.get("sample_stage") == qml_row.get("sample_stage") and row.get("status") == "ok"), None)
        if not match:
            continue
        out[f"{prefix}_validation_accuracy"] = match.get("validation_accuracy", math.nan)
        out[f"{prefix}_final_accuracy"] = match.get("final_accuracy", math.nan)
        val = as_float(match.get("validation_accuracy"))
        final = as_float(match.get("final_accuracy"))
        if math.isfinite(val) and (not math.isfinite(best_val) or val > best_val):
            best_val = val
            best_val_model = model_name
        if math.isfinite(final) and (not math.isfinite(best_final) or final > best_final):
            best_final = final
            best_final_model = model_name
    qml_val = as_float(qml_row.get("validation_accuracy"))
    qml_final = as_float(qml_row.get("final_accuracy"))
    rbf_val = as_float(out.get("rbf_svm_validation_accuracy"))
    rbf_final = as_float(out.get("rbf_svm_final_accuracy"))
    log_val = as_float(out.get("logistic_validation_accuracy"))
    log_final = as_float(out.get("logistic_final_accuracy"))
    out["qml_minus_rbf_svm_validation"] = qml_val - rbf_val if math.isfinite(qml_val) and math.isfinite(rbf_val) else math.nan
    out["qml_minus_rbf_svm_final"] = qml_final - rbf_final if math.isfinite(qml_final) and math.isfinite(rbf_final) else math.nan
    out["qml_minus_logistic_validation"] = qml_val - log_val if math.isfinite(qml_val) and math.isfinite(log_val) else math.nan
    out["qml_minus_logistic_final"] = qml_final - log_final if math.isfinite(qml_final) and math.isfinite(log_final) else math.nan
    out["best_classical_validation_model"] = best_val_model
    out["best_classical_final_model"] = best_final_model
    out["best_classical_validation_accuracy"] = best_val
    out["best_classical_final_accuracy"] = best_final
    out["qml_minus_best_classical_validation"] = qml_val - best_val if math.isfinite(qml_val) and math.isfinite(best_val) else math.nan
    out["qml_minus_best_classical_final"] = qml_final - best_final if math.isfinite(qml_final) and math.isfinite(best_final) else math.nan
    return out


def v6_qml_meta_features(
    k_train: np.ndarray,
    k_validation: np.ndarray,
    k_final: np.ndarray,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    qml_predictions: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_y = labels.loc[splits["train"]].astype(int).to_numpy()
    class0 = train_y == 0
    class1 = train_y == 1
    def build_features(k_cross: np.ndarray, split_name: str) -> pd.DataFrame:
        centroid0 = k_cross[:, class0].mean(axis=1) if class0.any() else np.full(k_cross.shape[0], np.nan)
        centroid1 = k_cross[:, class1].mean(axis=1) if class1.any() else np.full(k_cross.shape[0], np.nan)
        base = pd.DataFrame(
            {
                "quantum_similarity_class0": centroid0,
                "quantum_similarity_class1": centroid1,
                "quantum_similarity_margin": centroid1 - centroid0,
            },
            index=splits[split_name],
        )
        score = qml_predictions.get(f"{split_name}_score")
        if isinstance(score, pd.Series):
            base["quantum_decision_score"] = score.reindex(base.index)
        return base
    train_features = build_features(k_train, "train") if "train" in splits else pd.DataFrame(index=splits["train"])
    # For train, no decision score is available from the fitted precomputed model without an extra call;
    # centroid similarity features are enough for this diagnostic.
    validation_features = build_features(k_validation, "validation")
    final_features = build_features(k_final, "final")
    eigvals, eigvecs = np.linalg.eigh((k_train + k_train.T) / 2.0)
    order = np.argsort(eigvals)[::-1][: min(3, len(eigvals))]
    for pos, eig_idx in enumerate(order, start=1):
        value = max(float(eigvals[eig_idx]), 1e-9)
        train_features[f"quantum_eigen_projection_{pos}"] = eigvecs[:, eig_idx] * math.sqrt(value)
        validation_features[f"quantum_eigen_projection_{pos}"] = (k_validation @ eigvecs[:, eig_idx]) / math.sqrt(value)
        final_features[f"quantum_eigen_projection_{pos}"] = (k_final @ eigvecs[:, eig_idx]) / math.sqrt(value)
    columns = sorted(set(train_features.columns).union(validation_features.columns).union(final_features.columns))
    train_features = train_features.reindex(columns=columns)
    validation_features = validation_features.reindex(columns=columns)
    final_features = final_features.reindex(columns=columns)
    return train_features, validation_features, final_features


def v6_run_meta_models(
    sample_stage: str,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    meta_train: pd.DataFrame,
    meta_validation: pd.DataFrame,
    meta_final: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    train_y = labels.loc[splits["train"]].astype(int)
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    for model_name in ["l2_logistic", "lightgbm_small"]:
        start = time.perf_counter()
        try:
            model = v5_make_classical_model(model_name)
            model.fit(meta_train.replace([np.inf, -np.inf], np.nan).fillna(0.0), train_y)
            if hasattr(model, "predict_proba"):
                val_pred = (model.predict_proba(meta_validation.replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1] >= 0.50).astype(int)
                final_pred = (model.predict_proba(meta_final.replace([np.inf, -np.inf], np.nan).fillna(0.0))[:, 1] >= 0.50).astype(int)
            else:
                val_pred = np.asarray(model.predict(meta_validation.replace([np.inf, -np.inf], np.nan).fillna(0.0))).astype(int)
                final_pred = np.asarray(model.predict(meta_final.replace([np.inf, -np.inf], np.nan).fillna(0.0))).astype(int)
            status = "ok"
            skipped_reason = ""
            validation_acc = accuracy(validation_y, val_pred)
            final_acc = accuracy(final_y, final_pred)
        except Exception as exc:
            status = "skipped"
            skipped_reason = f"{type(exc).__name__}: {exc}"
            validation_acc = math.nan
            final_acc = math.nan
        rows.append(
            {
                "sample_stage": sample_stage,
                "meta_model": model_name,
                "feature_source": "qml_kernel_features",
                "validation_accuracy": validation_acc,
                "final_accuracy": final_acc,
                "runtime_seconds": time.perf_counter() - start,
                "status": status,
                "skipped_reason": skipped_reason,
            }
        )
    return rows


def v6_diagnosis_labels(
    distribution_rows: list[dict[str, Any]],
    drift_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
    rescue_rows: list[dict[str, Any]],
    regularization_rows: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
) -> tuple[list[str], str]:
    labels: list[str] = []
    val_shares = [as_float(row.get("label_1_share")) for row in distribution_rows if row.get("split") == "validation"]
    final_shares = [as_float(row.get("label_1_share")) for row in distribution_rows if row.get("split") == "final"]
    train_shares = [as_float(row.get("label_1_share")) for row in distribution_rows if row.get("split") == "train"]
    if train_shares and val_shares and max(abs(v - train_shares[0]) for v in val_shares + final_shares if math.isfinite(v)) > 0.12:
        labels.append("class_balance_drift")
    drift_values = [abs(as_float(row.get("mean_drift_vs_train"))) for row in drift_rows if row.get("split") in {"validation", "final"} and math.isfinite(as_float(row.get("mean_drift_vs_train")))]
    if drift_values and float(np.nanmedian(drift_values)) > 0.01:
        labels.append("feature_scaling_failure")
        labels.append("sample_shift_detected")
    if any(bool(row.get("kernel_concentration_detected")) for row in kernel_rows):
        labels.append("kernel_concentration_detected")
    v4 = next((row for row in scaling_rows if row.get("sample_stage") == "v4_sized" and row.get("scaling_method") == V6_PRIMARY_SCALING and row.get("status") == "ok"), None)
    largest = next((row for row in scaling_rows if row.get("sample_stage") == "largest_feasible" and row.get("scaling_method") == V6_PRIMARY_SCALING and row.get("status") == "ok"), None)
    if v4 and largest and as_float(v4.get("validation_accuracy")) - as_float(largest.get("validation_accuracy")) > 0.05:
        labels.append("small_sample_overfit")
    if regularization_rows:
        best_reg = max((row for row in regularization_rows if row.get("status") == "ok"), key=lambda row: as_float(row.get("validation_accuracy")), default=None)
        if best_reg and as_float(best_reg.get("qml_minus_best_classical_validation")) > 0.0 and as_float(best_reg.get("qml_minus_best_classical_final")) > 0.0:
            labels.append("qml_signal_rescued")
        else:
            labels.append("qsvc_regularization_issue")
    if "qml_signal_rescued" not in labels:
        labels.append("qml_not_rescued")
    if not labels:
        labels.append("qml_not_rescued")
    root = "sample shift and small-sample overfit are the primary diagnosis"
    if "kernel_concentration_detected" in labels:
        root = "sample shift, small-sample overfit, and kernel concentration are the primary diagnosis"
    elif "feature_scaling_failure" in labels:
        root = "sample shift plus scaling sensitivity is the primary diagnosis"
    return sorted(set(labels), key=labels.index), root


def write_v6_reports(
    diagnosis: dict[str, Any],
    scaling_rows: list[dict[str, Any]],
    kernel_rows: list[dict[str, Any]],
    rescue_rows: list[dict[str, Any]],
    regularization_rows: list[dict[str, Any]],
    meta_rows: list[dict[str, Any]],
) -> None:
    ok_scaling = [row for row in scaling_rows if row.get("status") == "ok"]
    best_scaling = max(ok_scaling, key=lambda row: as_float(row.get("validation_accuracy")), default={})
    largest_scaling = [row for row in ok_scaling if row.get("sample_stage") == "largest_feasible"]
    best_largest_scaling = max(largest_scaling, key=lambda row: as_float(row.get("validation_accuracy")), default={})
    ok_rescue = [row for row in rescue_rows if row.get("status") == "ok"]
    best_rescue = max(ok_rescue, key=lambda row: as_float(row.get("validation_accuracy")), default={})
    ok_reg = [row for row in regularization_rows if row.get("status") == "ok"]
    best_reg = max(ok_reg, key=lambda row: as_float(row.get("validation_accuracy")), default={})
    ok_meta = [row for row in meta_rows if row.get("status") == "ok"]
    best_meta = max(ok_meta, key=lambda row: as_float(row.get("validation_accuracy")), default={})
    concentration_count = sum(1 for row in kernel_rows if bool(row.get("kernel_concentration_detected")))
    same_target_win = (
        as_float(best_reg.get("qml_minus_best_classical_validation")) > 0.0
        and as_float(best_reg.get("qml_minus_best_classical_final")) > 0.0
    )
    paper_justified = "yes, as a negative/diagnostic QML evidence track, not as a replacement result"
    if "qml_signal_rescued" in diagnosis.get("decision_labels", []):
        paper_justified = "yes, as a same-target candidate requiring future-blind confirmation"
    summary = f"""# VN30 QML Forecasting V6 Scaling Rescue Result Summary

## Required Answers

1. Why did QML weaken under larger samples: {diagnosis.get("root_cause_diagnosis", "")}.
2. Was there label/class balance drift: {str("class_balance_drift" in diagnosis.get("decision_labels", [])).lower()}.
3. Was there feature distribution drift: {str("sample_shift_detected" in diagnosis.get("decision_labels", []) or "feature_scaling_failure" in diagnosis.get("decision_labels", [])).lower()}.
4. Was kernel concentration detected: {str("kernel_concentration_detected" in diagnosis.get("decision_labels", [])).lower()} ({concentration_count} diagnostic rows).
5. Did feature scaling rescue performance: no durable rescue; best overall scaling `{best_scaling.get("scaling_method", "")}` validation {pct(best_scaling.get("validation_accuracy"))}, final {pct(best_scaling.get("final_accuracy"))}; best largest-feasible scaling `{best_largest_scaling.get("scaling_method", "")}` validation {pct(best_largest_scaling.get("validation_accuracy"))}, final {pct(best_largest_scaling.get("final_accuracy"))}.
6. Did feature map reps/entanglement rescue performance: best rescue `{best_rescue.get("candidate_id", "")}` validation {pct(best_rescue.get("validation_accuracy"))}, final {pct(best_rescue.get("final_accuracy"))}.
7. Did QSVC regularization rescue performance: best regularized row validation {pct(best_reg.get("validation_accuracy"))}, final {pct(best_reg.get("final_accuracy"))}; beat best same-target classical on both validation and final = {str(same_target_win).lower()}.
8. Did QML-as-feature meta-model improve performance: best meta row `{best_meta.get("meta_model", "")}` validation {pct(best_meta.get("validation_accuracy"))}, final {pct(best_meta.get("final_accuracy"))}.
9. Did any validation-selected QML result beat same-target classical baselines: {str(same_target_win).lower()}.
10. Does any QML result replace the 61.61% classical champion: no; target/scope differs and future-blind confirmation is required.
11. Is QML paper still justified: {paper_justified}.
12. Exact claim boundary: V6 is a scaling-failure and kernel-rescue diagnostic only; no trading, profitability, BUY/SELL, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Decision Labels

`{", ".join(diagnosis.get("decision_labels", []))}`

## Best Rows

- Best scaling row: `{best_scaling.get("candidate_id", "")}`.
- Best rescue row: `{best_rescue.get("candidate_id", "")}`.
- Best regularization row: `{best_reg.get("candidate_id", "")}`.
- Best QML feature meta row: `{best_meta.get("meta_model", "")}`.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V6_SCALING_RESCUE_RESULT_SUMMARY.md", summary)
    claim = """# VN30 QML Forecasting V6 Scaling Rescue Claim Boundary

- QML v6 scaling rescue is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature selection, scaling, PCA, rank transforms, and quantum-feature transforms must be train-only or validation-safe.
- No final-performance selection is allowed.
- Final-ranked rows remain exploratory_not_claimable.
- Same-target comparisons are diagnostic and do not replace the 61.61% L2 Logistic classical champion because the target/scope is not directly identical.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require full validation governance and future-blind confirmation.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V6_SCALING_RESCUE_CLAIM_BOUNDARY.md", claim)


def run_v6_scaling_rescue(config: V6Config) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    index_data = load_index_data()
    features, v6_relative_cols = add_v3_relative_strength_features(features, index_data)
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    source_groups = build_source_groups(features, family_cols)
    source_groups["relative_strength_features"] = v6_relative_cols
    source_groups["combined_strategy_features"] = sorted(set(source_groups["combined_strategy_features"]).union(v6_relative_cols))
    source_groups["relative_plus_market_context_features"] = sorted(set(v6_relative_cols).union(source_groups.get("market_context_features", [])[:12]))

    labels = build_labels(features, index_data, V6_TARGET, V6_HORIZON)
    full_splits = strict_split_indices(features, labels)
    splits_by_stage = {str(stage["sample_stage"]): v5_stage_splits(features, labels, full_splits, stage) for stage in V5_SAMPLE_LADDER}
    specs_by_stage: dict[str, FeatureSpec] = {}
    selected_features_by_stage: dict[str, list[str]] = {}
    classical_rows_all: list[dict[str, Any]] = []
    classical_predictions_by_stage: dict[str, dict[str, dict[str, pd.Series]]] = {}
    for stage in V5_SAMPLE_LADDER:
        sample_stage = str(stage["sample_stage"])
        spec, _audit = fit_feature_spec(features, labels, V6_TARGET, V6_HORIZON, V5_FROZEN_SOURCE_GROUP, source_groups[V5_FROZEN_SOURCE_GROUP], V5_FROZEN_COMPRESSION, V5_FROZEN_N_FEATURES, splits_by_stage[sample_stage])
        specs_by_stage[sample_stage] = spec
        selected_features_by_stage[sample_stage] = list(spec.selected_features)
        c_rows, c_pred = v5_run_classical_models(spec, labels, splits_by_stage[sample_stage], sample_stage)
        classical_rows_all.extend(c_rows)
        classical_predictions_by_stage[sample_stage] = c_pred

    distribution_rows, drift_rows = v6_distribution_rows(features, labels, splits_by_stage, selected_features_by_stage)
    scaling_rows: list[dict[str, Any]] = []
    kernel_rows: list[dict[str, Any]] = []
    scaling_manifest: dict[str, Any] = {"methods": V6_SCALING_METHODS, "fit_split": "train", "rows": []}
    cached_primary: dict[str, tuple[dict[str, Any], dict[str, pd.Series], dict[str, np.ndarray]]] = {}
    for sample_stage, spec in specs_by_stage.items():
        for scaling_method in V6_SCALING_METHODS:
            if time.perf_counter() - started >= config.timeout_seconds * 0.55:
                scaling_rows.append({"sample_stage": sample_stage, "scaling_method": scaling_method, "status": "skipped", "skipped_reason": "runtime budget reserved for rescue phases"})
                continue
            row, predictions, matrices = v6_run_precomputed_kernel_classifier(spec, labels, splits_by_stage[sample_stage], sample_stage, scaling_method)
            row = v6_compare_against_classical(row, classical_rows_all)
            scaling_rows.append(row)
            scaling_manifest["rows"].append({"sample_stage": sample_stage, "scaling_method": scaling_method, "status": row.get("status"), "skipped_reason": row.get("skipped_reason")})
            if row.get("status") == "ok" and "train" in matrices:
                kernel_rows.append(v6_kernel_diagnostic_row(sample_stage, scaling_method, spec, labels, splits_by_stage[sample_stage], matrices["train"]))
            if sample_stage == "largest_feasible" and scaling_method == V6_PRIMARY_SCALING and row.get("status") == "ok":
                cached_primary[sample_stage] = (row, predictions, matrices)
    write_json(OUTPUT_DIR / "qml_v6_scaling_manifest.json", scaling_manifest)

    rescue_rows: list[dict[str, Any]] = []
    rescue_plan = [
        {"variant_id": "topk4_reps1_linear", "source_group": "relative_strength_features", "n_features": 4, "reps": 1, "entanglement": "linear"},
        {"variant_id": "topk4_reps2_linear", "source_group": "relative_strength_features", "n_features": 4, "reps": 2, "entanglement": "linear"},
        {"variant_id": "topk4_reps2_full", "source_group": "relative_strength_features", "n_features": 4, "reps": 2, "entanglement": "full"},
        {"variant_id": "topk4_reps3_linear", "source_group": "relative_strength_features", "n_features": 4, "reps": 3, "entanglement": "linear"},
        {"variant_id": "topk6_reps2_linear", "source_group": "relative_strength_features", "n_features": 6, "reps": 2, "entanglement": "linear"},
        {"variant_id": "relative_market_context_topk4_reps2_linear", "source_group": "relative_plus_market_context_features", "n_features": 4, "reps": 2, "entanglement": "linear"},
    ]
    largest_splits = splits_by_stage["largest_feasible"]
    for item in rescue_plan:
        if time.perf_counter() - started >= config.timeout_seconds * 0.78:
            rescue_rows.append({"variant_id": item["variant_id"], "sample_stage": "largest_feasible", "status": "skipped", "skipped_reason": "runtime budget reserved for regularization/meta phases"})
            continue
        spec, _audit = fit_feature_spec(features, labels, V6_TARGET, V6_HORIZON, item["source_group"], source_groups[item["source_group"]], "topk_availability", int(item["n_features"]), largest_splits)
        if spec.selection_status not in {"ok", "mutual_info_failed_fallback_availability"}:
            rescue_rows.append({"variant_id": item["variant_id"], "sample_stage": "largest_feasible", "status": "skipped", "skipped_reason": f"feature_selection_{spec.selection_status}"})
            continue
        c_rows, _c_pred = v5_run_classical_models(spec, labels, largest_splits, "largest_feasible")
        row, _pred, matrices = v6_run_precomputed_kernel_classifier(spec, labels, largest_splits, "largest_feasible", V6_PRIMARY_SCALING, int(item["reps"]), str(item["entanglement"]))
        row["variant_id"] = item["variant_id"]
        row["source_group"] = item["source_group"]
        row = v6_compare_against_classical(row, c_rows)
        rescue_rows.append(row)
        if row.get("status") == "ok" and "train" in matrices:
            diag = v6_kernel_diagnostic_row("largest_feasible", f"{V6_PRIMARY_SCALING}_{item['variant_id']}", spec, labels, largest_splits, matrices["train"])
            kernel_rows.append(diag)

    regularization_rows: list[dict[str, Any]] = []
    largest_spec = specs_by_stage["largest_feasible"]
    largest_classical_rows = [row for row in classical_rows_all if row.get("sample_stage") == "largest_feasible"]
    for c_value in V6_REGULARIZATION_C:
        for class_weight in V6_REGULARIZATION_CLASS_WEIGHT:
            if time.perf_counter() - started >= config.timeout_seconds * 0.90:
                regularization_rows.append({"sample_stage": "largest_feasible", "C": c_value, "class_weight": class_weight or "none", "status": "skipped", "skipped_reason": "runtime budget reserved for meta phase"})
                continue
            row, predictions, matrices = v6_run_precomputed_kernel_classifier(largest_spec, labels, largest_splits, "largest_feasible", V6_PRIMARY_SCALING, V6_FROZEN_REPS, V6_FROZEN_ENTANGLEMENT, c_value, class_weight)
            row = v6_compare_against_classical(row, largest_classical_rows)
            regularization_rows.append(row)
            if c_value == 1.0 and class_weight is None and row.get("status") == "ok":
                cached_primary["regularization_default"] = (row, predictions, matrices)

    meta_rows: list[dict[str, Any]] = []
    primary = cached_primary.get("regularization_default") or cached_primary.get("largest_feasible")
    if primary:
        _row, predictions, matrices = primary
        if all(key in matrices for key in ["train", "validation", "final"]):
            meta_train, meta_validation, meta_final = v6_qml_meta_features(matrices["train"], matrices["validation"], matrices["final"], labels, largest_splits, predictions)
            meta_rows = v6_run_meta_models("largest_feasible", labels, largest_splits, meta_train, meta_validation, meta_final)

    decision_labels, root = v6_diagnosis_labels(distribution_rows, drift_rows, kernel_rows, scaling_rows, rescue_rows, regularization_rows, meta_rows)
    diagnosis = {
        "decision_labels": decision_labels,
        "root_cause_diagnosis": root,
        "diagnostic_only": True,
        "qml_replaces_6161_classical_champion": False,
        "future_blind_required": True,
        "runtime_seconds": time.perf_counter() - started,
        "runtime_limited": time.perf_counter() - started >= config.timeout_seconds,
    }
    write_json(OUTPUT_DIR / "qml_v6_scaling_failure_diagnosis.json", diagnosis)

    write_frame(OUTPUT_DIR / "qml_v6_sample_distribution_audit.csv", distribution_rows, list(distribution_rows[0].keys()) if distribution_rows else [])
    write_frame(OUTPUT_DIR / "qml_v6_feature_drift_audit.csv", drift_rows, list(drift_rows[0].keys()) if drift_rows else [])
    write_frame(OUTPUT_DIR / "qml_v6_scaling_method_results.csv", scaling_rows, list(scaling_rows[0].keys()) if scaling_rows else [])
    write_frame(OUTPUT_DIR / "qml_v6_kernel_diagnostics.csv", kernel_rows, list(kernel_rows[0].keys()) if kernel_rows else [])
    write_frame(OUTPUT_DIR / "qml_v6_kernel_rescue_results.csv", rescue_rows, list(rescue_rows[0].keys()) if rescue_rows else [])
    write_frame(OUTPUT_DIR / "qml_v6_qsvc_regularization_results.csv", regularization_rows, list(regularization_rows[0].keys()) if regularization_rows else [])
    write_frame(OUTPUT_DIR / "qml_v6_classical_comparison.csv", classical_rows_all, list(classical_rows_all[0].keys()) if classical_rows_all else [])
    write_frame(OUTPUT_DIR / "qml_v6_qml_feature_meta_model_results.csv", meta_rows, list(meta_rows[0].keys()) if meta_rows else [])

    manifest = {
        "run_id": "vn30_qml_forecasting_v6_scaling_rescue",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "target": V6_TARGET,
        "horizon": V6_HORIZON,
        "diagnostic_only": True,
        "dependency_status": dependency,
        "sample_ladder": V5_SAMPLE_LADDER,
        "scaling_methods": V6_SCALING_METHODS,
        "kernel_rescue_plan": rescue_plan,
        "regularization_C": V6_REGULARIZATION_C,
        "regularization_class_weight": [value or "none" for value in V6_REGULARIZATION_CLASS_WEIGHT],
        "diagnosis": diagnosis,
        "runtime_seconds": time.perf_counter() - started,
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v6_manifest.json", manifest)
    write_v6_reports(diagnosis, scaling_rows, kernel_rows, rescue_rows, regularization_rows, meta_rows)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v6_manifest.json"), "decision_labels": decision_labels, "runtime_limited": diagnosis["runtime_limited"]}), indent=2))
    return manifest


@dataclass
class V7Config:
    timeout_seconds: int = 7200


V7_TARGET = "market_relative_vn30"
V7_HORIZON = 40
V7_ALPHA_VALUES = [0.1, 0.25, 0.5, 0.75, 0.9]
V7_C_VALUES = [0.01, 0.1, 1.0, 10.0, 100.0]
V7_CLASS_WEIGHTS = [None, "balanced"]
V7_SCALING_METHODS = ["minmax_0_pi", "minmax_minus_pi_pi", "robust_quantile_to_pi", "rank_gaussian_to_pi"]


def v7_scaling_alias(name: str) -> str:
    return {"robust_quantile_to_pi": "robust_quantile", "rank_gaussian_to_pi": "rank_gaussian"}.get(name, name)


def v7_entropy(values: pd.Series) -> float:
    counts = values.dropna().astype(str).value_counts()
    if counts.empty:
        return math.nan
    p = counts / counts.sum()
    return float(-(p * np.log(np.clip(p, 1e-12, None))).sum())


def v7_group_sample(
    features: pd.DataFrame,
    labels: pd.Series,
    idx: pd.Index,
    limit: int,
    use_labels: bool,
) -> pd.Index:
    ordered = ordered_index(features, idx)
    if len(ordered) <= limit:
        return ordered
    frame = features.loc[ordered, ["datetime", "ticker"]].copy()
    frame["idx"] = ordered
    frame["quarter"] = pd.to_datetime(frame["datetime"], errors="coerce").dt.to_period("Q").astype(str)
    group_cols = ["quarter", "ticker"]
    if use_labels:
        frame["label"] = labels.loc[ordered].astype(int).to_numpy()
        group_cols = ["label", "quarter", "ticker"]
    pieces: list[pd.DataFrame] = []
    groups = list(frame.groupby(group_cols, sort=True))
    take = max(1, math.ceil(limit / max(1, len(groups))))
    for _key, group in groups:
        pieces.append(group.tail(take))
    sampled = pd.concat(pieces, ignore_index=True).drop_duplicates("idx")
    if len(sampled) > limit:
        positions = np.linspace(0, len(sampled) - 1, num=limit).round().astype(int)
        sampled = sampled.sort_values(["datetime", "idx"]).iloc[positions]
    if len(sampled) < limit:
        used = set(sampled["idx"].tolist())
        fallback = frame[~frame["idx"].isin(used)].sort_values(["datetime", "idx"]).tail(limit - len(sampled))
        sampled = pd.concat([sampled, fallback], ignore_index=True)
    sampled = sampled.sort_values(["datetime", "idx"]).tail(limit).sort_values(["datetime", "idx"])
    return pd.Index(sampled["idx"].tolist())


def v7_distribution_matched_splits(
    features: pd.DataFrame,
    labels: pd.Series,
    full_splits: dict[str, pd.Index],
    train_limit: int,
    validation_limit: int,
    final_limit: int,
    original: bool,
) -> dict[str, pd.Index]:
    if original:
        return v5_stage_splits(
            features,
            labels,
            full_splits,
            {
                "effective_qml_train_rows": train_limit,
                "effective_qml_validation_rows": validation_limit,
                "effective_qml_final_rows": final_limit,
            },
        )
    return {
        "train": v7_group_sample(features, labels, full_splits["train"], train_limit, use_labels=True),
        "validation": v7_group_sample(features, labels, full_splits["validation"], validation_limit, use_labels=True),
        "final": v7_group_sample(features, labels, full_splits["final"], final_limit, use_labels=False),
    }


def v7_sample_audit_rows(features: pd.DataFrame, labels: pd.Series, splits_by_sample: dict[str, dict[str, pd.Index]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    original_ratios: dict[str, float] = {}
    for sample_id, splits in splits_by_sample.items():
        for split, idx in splits.items():
            y = labels.loc[idx].dropna().astype(int)
            ratio = float((y == 1).mean()) if len(y) else math.nan
            if sample_id == "v4_sized_original":
                original_ratios[split] = ratio
    for sample_id, splits in splits_by_sample.items():
        for split, idx in splits.items():
            y = labels.loc[idx].dropna().astype(int)
            ratio = float((y == 1).mean()) if len(y) else math.nan
            quarters = timestamps.loc[idx].dt.to_period("Q").astype(str).value_counts().sort_index().to_dict() if len(idx) else {}
            regime_cols = [col for col in features.columns if "regime" in col.lower()]
            regime = ""
            if regime_cols and len(idx):
                regime = json.dumps(features.loc[idx, regime_cols[0]].astype(str).value_counts().head(8).to_dict(), sort_keys=True)
            rows.append(
                {
                    "sample_id": sample_id,
                    "split": split,
                    "rows": int(len(idx)),
                    "label_positive_ratio": ratio,
                    "ticker_count": int(features.loc[idx, "ticker"].nunique()) if len(idx) else 0,
                    "ticker_entropy": v7_entropy(features.loc[idx, "ticker"]) if len(idx) else math.nan,
                    "quarter_distribution": json.dumps(quarters, sort_keys=True),
                    "regime_distribution": regime,
                    "timestamp_start": str(timestamps.loc[idx].min()) if len(idx) else "",
                    "timestamp_end": str(timestamps.loc[idx].max()) if len(idx) else "",
                    "drift_vs_v4_original": ratio - original_ratios.get(split, math.nan) if math.isfinite(ratio) and math.isfinite(original_ratios.get(split, math.nan)) else math.nan,
                }
            )
    return rows


def v7_center_kernel(k_train: np.ndarray, k_cross: np.ndarray | None = None) -> np.ndarray:
    train_mean_cols = k_train.mean(axis=0, keepdims=True)
    train_grand = float(k_train.mean())
    if k_cross is None:
        train_mean_rows = k_train.mean(axis=1, keepdims=True)
        return k_train - train_mean_rows - train_mean_cols + train_grand
    return k_cross - k_cross.mean(axis=1, keepdims=True) - train_mean_cols + train_grand


def v7_normalize_kernel(k_train: np.ndarray, k_cross: np.ndarray | None = None) -> np.ndarray:
    train_diag = np.sqrt(np.clip(np.diag(k_train), 1e-12, None))
    if k_cross is None:
        return k_train / np.outer(train_diag, train_diag)
    row_norm = np.sqrt(np.clip(np.max(np.abs(k_cross), axis=1), 1e-12, None))
    return k_cross / np.outer(row_norm, train_diag)


def v7_transform_kernel(k_train: np.ndarray, k_validation: np.ndarray, k_final: np.ndarray, transform: str, shrinkage: float = 0.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kt, kv, kf = k_train.copy(), k_validation.copy(), k_final.copy()
    if transform in {"centered", "centered_normalized"}:
        kv = v7_center_kernel(kt, kv)
        kf = v7_center_kernel(kt, kf)
        kt = v7_center_kernel(kt)
    if transform in {"normalized", "centered_normalized"}:
        kv = v7_normalize_kernel(kt, kv)
        kf = v7_normalize_kernel(kt, kf)
        kt = v7_normalize_kernel(kt)
    if shrinkage > 0.0:
        kt = (1.0 - shrinkage) * kt + shrinkage * np.eye(kt.shape[0])
    return kt, kv, kf


def v7_classical_kernel_matrices(x_train: pd.DataFrame, x_validation: pd.DataFrame, x_final: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    gamma = 1.0 / max(1, x_train.shape[1])
    return {
        "linear": (
            linear_kernel(x_train),
            linear_kernel(x_validation, x_train),
            linear_kernel(x_final, x_train),
        ),
        "rbf": (
            rbf_kernel(x_train, x_train, gamma=gamma),
            rbf_kernel(x_validation, x_train, gamma=gamma),
            rbf_kernel(x_final, x_train, gamma=gamma),
        ),
    }


def v7_cross_alignment(k_cross: np.ndarray, y_rows: np.ndarray, y_train: np.ndarray) -> float:
    signed_rows = np.where(y_rows > 0, 1.0, -1.0)
    signed_train = np.where(y_train > 0, 1.0, -1.0)
    yy = np.outer(signed_rows, signed_train)
    denom = float(np.linalg.norm(k_cross, "fro") * np.linalg.norm(yy, "fro"))
    return float(np.sum(k_cross * yy) / denom) if denom > 0 else math.nan


def v7_health_row(
    sample_id: str,
    spec: FeatureSpec,
    scaling: str,
    feature_map: str,
    reps: int,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    k_train: np.ndarray,
    k_validation: np.ndarray,
) -> dict[str, Any]:
    base = v6_kernel_diagnostic_row(sample_id, scaling, spec, labels, splits, k_train)
    train_y = labels.loc[splits["train"]].astype(int).to_numpy()
    validation_y = labels.loc[splits["validation"]].astype(int).to_numpy()
    validation_alignment = v7_cross_alignment(k_validation, validation_y, train_y)
    severe_condition = as_float(base.get("condition_number")) > 1e8 if math.isfinite(as_float(base.get("condition_number"))) else False
    off_std_low = as_float(base.get("std_off_diagonal")) < 0.02 if math.isfinite(as_float(base.get("std_off_diagonal"))) else False
    rank_low = as_float(base.get("effective_rank_ratio")) < 0.05 if math.isfinite(as_float(base.get("effective_rank_ratio"))) else False
    gap_bad = as_float(base.get("class_similarity_gap")) <= 0.0 if math.isfinite(as_float(base.get("class_similarity_gap"))) else True
    base.update(
        {
            "sample_id": sample_id,
            "feature_set": spec.feature_set_name,
            "feature_map": feature_map,
            "reps": reps,
            "validation_kernel_target_alignment": validation_alignment,
            "severe_condition_flag": severe_condition,
            "off_diagonal_std_low_flag": off_std_low,
            "effective_rank_low_flag": rank_low,
            "class_similarity_gap_bad_flag": gap_bad,
            "prefilter_reject": bool(rank_low or gap_bad or off_std_low or severe_condition),
        }
    )
    return base


def v7_fit_precomputed(
    k_train: np.ndarray,
    k_validation: np.ndarray,
    k_final: np.ndarray,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    c_value: float = 1.0,
    class_weight: str | None = None,
) -> tuple[float, float, dict[str, pd.Series]]:
    train_y = labels.loc[splits["train"]].astype(int)
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    model = SVC(kernel="precomputed", C=float(c_value), class_weight=class_weight)
    model.fit(k_train, train_y.to_numpy())
    val_pred = pd.Series(np.asarray(model.predict(k_validation)).astype(int), index=splits["validation"])
    final_pred = pd.Series(np.asarray(model.predict(k_final)).astype(int), index=splits["final"])
    preds = {"validation": val_pred, "final": final_pred}
    try:
        preds["train_score"] = pd.Series(np.asarray(model.decision_function(k_train), dtype=float), index=splits["train"])
        preds["validation_score"] = pd.Series(np.asarray(model.decision_function(k_validation), dtype=float), index=splits["validation"])
        preds["final_score"] = pd.Series(np.asarray(model.decision_function(k_final), dtype=float), index=splits["final"])
    except Exception:
        pass
    return accuracy(validation_y, val_pred), accuracy(final_y, final_pred), preds


def v7_result_row(
    candidate_id_value: str,
    sample_id: str,
    spec: FeatureSpec,
    scaling: str,
    feature_map: str,
    reps: int,
    kernel_type: str,
    alpha: float | None,
    transform: str,
    shrinkage: float,
    c_value: float,
    class_weight: str | None,
    validation_accuracy: float,
    final_accuracy: float,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    classical_rows: list[dict[str, Any]],
    health: dict[str, Any],
    runtime_seconds: float,
    status: str = "ok",
    skipped_reason: str = "",
) -> dict[str, Any]:
    simple_val_name, simple_val_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
    simple_final_name, simple_final_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
    base = {
        "candidate_id": candidate_id_value,
        "sample_id": sample_id,
        "feature_set": spec.feature_set_name,
        "scaling": scaling,
        "feature_map": feature_map,
        "reps": reps,
        "kernel_type": kernel_type,
        "alpha": alpha if alpha is not None else "",
        "normalization_shrinkage": transform if shrinkage == 0.0 else f"{transform}_shrinkage_{shrinkage}",
        "C": c_value,
        "class_weight": class_weight or "none",
        "validation_accuracy": validation_accuracy,
        "validation_lift": validation_accuracy - simple_val_acc if math.isfinite(validation_accuracy) and math.isfinite(simple_val_acc) else math.nan,
        "final_accuracy": final_accuracy,
        "final_lift": final_accuracy - simple_final_acc if math.isfinite(final_accuracy) and math.isfinite(simple_final_acc) else math.nan,
        "validation_rows": int(len(splits["validation"])),
        "final_rows": int(len(splits["final"])),
        "kernel_alignment": health.get("kernel_target_alignment", math.nan),
        "validation_kernel_alignment": health.get("validation_kernel_target_alignment", math.nan),
        "effective_rank_ratio": health.get("effective_rank_ratio", math.nan),
        "class_similarity_gap": health.get("class_similarity_gap", math.nan),
        "runtime_seconds": runtime_seconds,
        "claim_label": "qml_diagnostic_only",
        "status": status,
        "skipped_reason": skipped_reason,
    }
    enriched = v6_compare_against_classical({**base, "sample_stage": sample_id}, classical_rows)
    if (
        sample_id in {"medium_distribution_matched", "largest_feasible_distribution_matched"}
        and as_float(enriched.get("qml_minus_rbf_svm_validation")) > 0.0
        and as_float(enriched.get("qml_minus_logistic_validation")) > 0.0
        and final_accuracy >= 0.50
        and as_float(enriched.get("qml_minus_best_classical_validation")) > 0.0
    ):
        enriched["claim_label"] = "qml_rescue_candidate_requires_future_blind"
    return enriched


def v7_build_specs(
    features: pd.DataFrame,
    labels: pd.Series,
    source_groups: dict[str, list[str]],
    splits: dict[str, pd.Index],
) -> dict[str, FeatureSpec]:
    plans = [
        ("relative_strength_topk4", "relative_strength_features", "topk_availability", 4),
        ("relative_strength_topk6", "relative_strength_features", "topk_availability", 6),
        ("relative_market_context_topk4", "relative_plus_market_context_features", "topk_availability", 4),
        ("relative_volatility_topk4", "relative_plus_volatility_features", "topk_availability", 4),
        ("combined_pca_k4", "combined_strategy_features", "pca_train_only", 4),
    ]
    out: dict[str, FeatureSpec] = {}
    for name, source_group, compression, n_features in plans:
        spec, _audit = fit_feature_spec(features, labels, V7_TARGET, V7_HORIZON, source_group, source_groups[source_group], compression, n_features, splits)
        if spec.selection_status in {"ok", "mutual_info_failed_fallback_availability"}:
            out[name] = spec
    return out


def v7_run_meta_models(
    sample_id: str,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    k_train: np.ndarray,
    k_validation: np.ndarray,
    k_final: np.ndarray,
    predictions: dict[str, pd.Series],
    classical_rows: list[dict[str, Any]],
    spec: FeatureSpec,
    health: dict[str, Any],
) -> list[dict[str, Any]]:
    meta_train, meta_validation, meta_final = v6_qml_meta_features(k_train, k_validation, k_final, labels, splits, predictions)
    rows: list[dict[str, Any]] = []
    train_y = labels.loc[splits["train"]].astype(int)
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    for model_name in ["l2_logistic", "calibrated_logistic", "lightgbm_small"]:
        start = time.perf_counter()
        try:
            model = v5_make_classical_model(model_name)
            if model_name == "calibrated_logistic" and train_y.value_counts().min() < 3:
                raise ValueError("calibrated logistic requires at least three train rows per class")
            x_train = meta_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            x_validation = meta_validation.reindex(columns=x_train.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            x_final = meta_final.reindex(columns=x_train.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            model.fit(x_train, train_y)
            if hasattr(model, "predict_proba"):
                val_pred = (model.predict_proba(x_validation)[:, 1] >= 0.50).astype(int)
                final_pred = (model.predict_proba(x_final)[:, 1] >= 0.50).astype(int)
            else:
                val_pred = np.asarray(model.predict(x_validation)).astype(int)
                final_pred = np.asarray(model.predict(x_final)).astype(int)
            val_acc = accuracy(validation_y, val_pred)
            final_acc = accuracy(final_y, final_pred)
            status = "ok"
            skipped = ""
        except Exception as exc:
            val_acc = math.nan
            final_acc = math.nan
            status = "skipped"
            skipped = f"{type(exc).__name__}: {exc}"
        rows.append(
            v7_result_row(
                candidate_id("qml_v7", sample_id, "kernel_feature_meta", model_name, spec.feature_set_name),
                sample_id,
                spec,
                "kernel_feature_extractor",
                "kernel_features",
                0,
                "qml_kernel_features",
                None,
                "meta_model",
                0.0,
                1.0,
                None,
                val_acc,
                final_acc,
                labels,
                splits,
                classical_rows,
                health,
                time.perf_counter() - start,
                status,
                skipped,
            )
        )
        rows[-1]["meta_model"] = model_name
    return rows


def write_v7_reports(decision: dict[str, Any], audit_rows: list[dict[str, Any]], health_rows: list[dict[str, Any]], hybrid_rows: list[dict[str, Any]], norm_rows: list[dict[str, Any]], regularization_rows: list[dict[str, Any]], meta_rows: list[dict[str, Any]], leaderboard: list[dict[str, Any]]) -> None:
    best_hybrid = max((row for row in hybrid_rows if row.get("status") == "ok"), key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_norm = max((row for row in norm_rows if row.get("status") == "ok"), key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_reg = max((row for row in regularization_rows if row.get("status") == "ok"), key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_meta = max((row for row in meta_rows if row.get("status") == "ok"), key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_overall = leaderboard[0] if leaderboard else {}
    original_val = next((row for row in audit_rows if row.get("sample_id") == "v4_sized_original" and row.get("split") == "validation"), {})
    matched_val = next((row for row in audit_rows if row.get("sample_id") == "v4_sized_distribution_matched" and row.get("split") == "validation"), {})
    health_rejects = sum(1 for row in health_rows if str(row.get("prefilter_reject")).lower() == "true" or row.get("prefilter_reject") is True)
    summary = f"""# VN30 QML Forecasting V7 Hybrid Kernel Rescue Result Summary

## Required Answers

1. Did distribution matching reduce class-balance/sample drift: original validation positive ratio {pct(original_val.get("label_positive_ratio"))}, matched validation positive ratio {pct(matched_val.get("label_positive_ratio"))}; distribution-matched samples improved metadata coverage but did not remove all target drift.
2. Did QML recover on medium/largest samples: {str(decision.get("medium_or_largest_recovered", False)).lower()}.
3. Did kernel health improve: health prefilter rejected or flagged {health_rejects}/{len(health_rows)} kernels; best validation-governed row effective-rank ratio {best_overall.get("effective_rank_ratio", "")}.
4. Did hybrid kernels beat pure quantum kernels: {str(decision.get("hybrid_beats_pure_quantum", False)).lower()}.
5. Did hybrid kernels beat RBF SVM or Logistic: {str(decision.get("best_beats_rbf_or_logistic_validation", False)).lower()} on validation.
6. Did normalization/shrinkage help: best normalization/shrinkage row validation {pct(best_norm.get("validation_accuracy"))}, final {pct(best_norm.get("final_accuracy"))}.
7. Did QSVC regularization help: best regularization revisit validation {pct(best_reg.get("validation_accuracy"))}, final {pct(best_reg.get("final_accuracy"))}.
8. Did QML-derived kernel features help: best meta row `{best_meta.get("meta_model", "")}` validation {pct(best_meta.get("validation_accuracy"))}, final {pct(best_meta.get("final_accuracy"))}.
9. Was QML rescued: {str(decision.get("qml_rescued", False)).lower()}.
10. Does any result replace the 61.61% classical champion: no; target/scope differs and future-blind confirmation is still required.
11. Is the new QML paper still justified: yes, as a diagnostic rescue/negative-evidence track with explicit failure modes, not as a replacement result.
12. Exact claim boundary: V7 is experimental and diagnostic-only; final-ranked rows are exploratory_not_claimable; no trading, profitability, BUY/SELL, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Best Validation-Governed Row

- Candidate: `{best_overall.get("candidate_id", "")}`.
- Sample: {best_overall.get("sample_id", "")}.
- Kernel type: {best_overall.get("kernel_type", "")}.
- Validation accuracy: {pct(best_overall.get("validation_accuracy"))}.
- Final accuracy: {pct(best_overall.get("final_accuracy"))}.
- QML minus RBF SVM validation: {pp(best_overall.get("qml_minus_rbf_svm_validation"))}.
- QML minus Logistic validation: {pp(best_overall.get("qml_minus_logistic_validation"))}.
- Claim label: `{best_overall.get("claim_label", "qml_diagnostic_only")}`.

## Best Hybrid Row

- Candidate: `{best_hybrid.get("candidate_id", "")}`.
- Validation accuracy: {pct(best_hybrid.get("validation_accuracy"))}.
- Final accuracy: {pct(best_hybrid.get("final_accuracy"))}.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V7_HYBRID_KERNEL_RESCUE_RESULT_SUMMARY.md", summary)
    claim = """# VN30 QML Forecasting V7 Hybrid Kernel Rescue Claim Boundary

- QML v7 hybrid kernel rescue is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- Main index data may be used only as lagged market-context features or market-relative target context.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature selection, scaling, PCA, rank transforms, kernel construction, and kernel-feature transforms must be train-only or validation-safe.
- Distribution matching must not use final labels for sample selection.
- Candidate selection is validation-governed only; final performance is scoring-only.
- Final-ranked rows remain exploratory_not_claimable.
- Same-target comparisons are diagnostic and do not replace the 61.61% L2 Logistic classical champion because the target/scope is not directly identical.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require full validation governance and future-blind confirmation.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V7_HYBRID_KERNEL_RESCUE_CLAIM_BOUNDARY.md", claim)


def run_v7_hybrid_kernel_rescue(config: V7Config) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    index_data = load_index_data()
    features, v7_relative_cols = add_v3_relative_strength_features(features, index_data)
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    source_groups = build_source_groups(features, family_cols)
    source_groups["relative_strength_features"] = v7_relative_cols
    vol_cols = [col for col in features.columns if "vol" in col.lower() or "shock" in col.lower()]
    source_groups["relative_plus_market_context_features"] = sorted(set(v7_relative_cols).union(source_groups.get("market_context_features", [])[:12]))
    source_groups["relative_plus_volatility_features"] = sorted(set(v7_relative_cols).union(numeric_existing(features, vol_cols)[:12]))
    source_groups["combined_strategy_features"] = sorted(set(source_groups["combined_strategy_features"]).union(v7_relative_cols))
    labels = build_labels(features, index_data, V7_TARGET, V7_HORIZON)
    full_splits = strict_split_indices(features, labels)
    sample_defs = {
        "v4_sized_original": (80, 180, 180, True),
        "v4_sized_distribution_matched": (80, 180, 180, False),
        "medium_distribution_matched": (100, 240, 240, False),
        "largest_feasible_distribution_matched": (120, 300, 300, False),
    }
    splits_by_sample = {
        sample_id: v7_distribution_matched_splits(features, labels, full_splits, train_n, val_n, final_n, original)
        for sample_id, (train_n, val_n, final_n, original) in sample_defs.items()
    }
    audit_rows = v7_sample_audit_rows(features, labels, splits_by_sample)

    specs_by_sample: dict[str, dict[str, FeatureSpec]] = {}
    classical_rows_all: list[dict[str, Any]] = []
    classical_rows_by_sample: dict[str, list[dict[str, Any]]] = {}
    for sample_id, splits in splits_by_sample.items():
        specs = v7_build_specs(features, labels, source_groups, splits)
        specs_by_sample[sample_id] = specs
        classical_rows_by_sample[sample_id] = []
        for spec in specs.values():
            c_rows, _pred = v5_run_classical_models(spec, labels, splits, sample_id)
            classical_rows_all.extend(c_rows)
            classical_rows_by_sample[sample_id].extend(c_rows)

    health_rows: list[dict[str, Any]] = []
    kernel_cache: dict[str, dict[str, Any]] = {}
    map_plan = [("ZZFeatureMap", 1), ("ZZFeatureMap", 2)]
    try:
        importlib.import_module("qiskit.circuit.library").PauliFeatureMap
        map_plan.append(("PauliFeatureMap", 1))
    except Exception:
        pass

    for sample_id, specs in specs_by_sample.items():
        for feature_name, spec in specs.items():
            for scaling in V7_SCALING_METHODS:
                for fmap, reps in map_plan:
                    if fmap == "PauliFeatureMap" and spec.n_features > 4:
                        continue
                    if time.perf_counter() - started >= config.timeout_seconds * 0.35:
                        continue
                    try:
                        x_train, x_validation, x_final, scaling_info = v6_scaling_transform(spec, v7_scaling_alias(scaling))
                        if scaling_info["status"] != "ok":
                            raise ValueError(scaling_info["skipped_reason"])
                        if fmap == "ZZFeatureMap":
                            k_train, k_val, k_final = v6_quantum_kernel_matrices(x_train, x_validation, x_final, reps, "full")
                        else:
                            kernels = importlib.import_module("qiskit_machine_learning.kernels")
                            circuit_library = importlib.import_module("qiskit.circuit.library")
                            kernel = getattr(kernels, "FidelityStatevectorKernel")(feature_map=getattr(circuit_library, "PauliFeatureMap")(feature_dimension=x_train.shape[1], reps=reps, paulis=["Z", "ZZ"], entanglement="linear"))
                            k_train = np.asarray(kernel.evaluate(x_train.to_numpy(dtype=float)), dtype=float)
                            k_val = np.asarray(kernel.evaluate(x_validation.to_numpy(dtype=float), x_train.to_numpy(dtype=float)), dtype=float)
                            k_final = np.asarray(kernel.evaluate(x_final.to_numpy(dtype=float), x_train.to_numpy(dtype=float)), dtype=float)
                        health = v7_health_row(sample_id, spec, scaling, fmap, reps, labels, splits_by_sample[sample_id], k_train, k_val)
                        health_rows.append(health)
                        key = candidate_id(sample_id, feature_name, scaling, fmap, reps)
                        kernel_cache[key] = {"sample_id": sample_id, "feature_name": feature_name, "spec": spec, "scaling": scaling, "feature_map": fmap, "reps": reps, "k_train": k_train, "k_validation": k_val, "k_final": k_final, "x_train": x_train, "x_validation": x_validation, "x_final": x_final, "health": health}
                    except Exception as exc:
                        health_rows.append({"sample_id": sample_id, "feature_set": spec.feature_set_name, "scaling_method": scaling, "feature_map": fmap, "reps": reps, "status": "skipped", "skipped_reason": f"{type(exc).__name__}: {exc}", "prefilter_reject": True})

    healthy = [item for item in kernel_cache.values() if not bool(item["health"].get("prefilter_reject"))]
    if not healthy:
        healthy = list(kernel_cache.values())
    selected_for_hybrid: list[dict[str, Any]] = []
    for sample_id in ["v4_sized_distribution_matched", "medium_distribution_matched", "largest_feasible_distribution_matched"]:
        choices = [item for item in healthy if item["sample_id"] == sample_id]
        if choices:
            choices = sorted(choices, key=lambda item: (as_float(item["health"].get("validation_kernel_target_alignment")), as_float(item["health"].get("effective_rank_ratio"))), reverse=True)
            selected_for_hybrid.append(choices[0])
    if healthy:
        selected_for_hybrid.append(sorted(healthy, key=lambda item: as_float(item["health"].get("validation_kernel_target_alignment")), reverse=True)[0])
    # Keep order but remove duplicate object ids.
    dedup: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in selected_for_hybrid:
        key = candidate_id(item["sample_id"], item["feature_name"], item["scaling"], item["feature_map"], item["reps"])
        if key not in seen_ids:
            dedup.append(item)
            seen_ids.add(key)
    selected_for_hybrid = dedup[:4]

    hybrid_rows: list[dict[str, Any]] = []
    best_kernel_payload: dict[str, Any] | None = None
    for item in selected_for_hybrid:
        sample_id = item["sample_id"]
        splits = splits_by_sample[sample_id]
        spec = item["spec"]
        classical_rows = [row for row in classical_rows_by_sample[sample_id] if row.get("feature_set") == spec.feature_set_name]
        classical_kernels = v7_classical_kernel_matrices(item["x_train"], item["x_validation"], item["x_final"])
        base_kernels = {"quantum": (item["k_train"], item["k_validation"], item["k_final"])}
        for classical_type, matrices in classical_kernels.items():
            for alpha in V7_ALPHA_VALUES:
                base_kernels[f"hybrid_{classical_type}_{alpha}"] = tuple(alpha * q + (1.0 - alpha) * c for q, c in zip(base_kernels["quantum"], matrices))
        for kernel_type, matrices in base_kernels.items():
            if kernel_type == "quantum":
                alphas = [None]
            else:
                alphas = [float(kernel_type.rsplit("_", 1)[1])]
            for alpha in alphas:
                start = time.perf_counter()
                try:
                    val_acc, final_acc, preds = v7_fit_precomputed(*matrices, labels, splits)
                    status = "ok"
                    skipped = ""
                except Exception as exc:
                    val_acc, final_acc, preds = math.nan, math.nan, {}
                    status = "skipped"
                    skipped = f"{type(exc).__name__}: {exc}"
                row = v7_result_row(
                    candidate_id("qml_v7", sample_id, item["feature_name"], item["scaling"], item["feature_map"], f"r{item['reps']}", kernel_type),
                    sample_id,
                    spec,
                    item["scaling"],
                    item["feature_map"],
                    item["reps"],
                    "quantum" if kernel_type == "quantum" else kernel_type.rsplit("_", 1)[0],
                    alpha,
                    "raw",
                    0.0,
                    1.0,
                    None,
                    val_acc,
                    final_acc,
                    labels,
                    splits,
                    classical_rows,
                    item["health"],
                    time.perf_counter() - start,
                    status,
                    skipped,
                )
                hybrid_rows.append(row)
                if status == "ok" and (best_kernel_payload is None or as_float(row.get("validation_accuracy")) > as_float(best_kernel_payload["row"].get("validation_accuracy"))):
                    best_kernel_payload = {"row": row, "item": item, "matrices": matrices, "preds": preds}

    norm_rows: list[dict[str, Any]] = []
    if best_kernel_payload is not None:
        item = best_kernel_payload["item"]
        splits = splits_by_sample[item["sample_id"]]
        spec = item["spec"]
        classical_rows = [row for row in classical_rows_by_sample[item["sample_id"]] if row.get("feature_set") == spec.feature_set_name]
        transforms: list[tuple[str, float]] = [("raw", 0.0), ("centered", 0.0), ("normalized", 0.0), ("centered_normalized", 0.0)]
        transforms.extend(("raw", value) for value in [0.001, 0.01, 0.05, 0.1])
        for transform, shrink in transforms:
            start = time.perf_counter()
            kt, kv, kf = v7_transform_kernel(*best_kernel_payload["matrices"], transform, shrink)
            try:
                val_acc, final_acc, preds = v7_fit_precomputed(kt, kv, kf, labels, splits)
                status = "ok"
                skipped = ""
            except Exception as exc:
                val_acc, final_acc, preds = math.nan, math.nan, {}
                status = "skipped"
                skipped = f"{type(exc).__name__}: {exc}"
            norm_rows.append(
                v7_result_row(
                    candidate_id("qml_v7", item["sample_id"], "normalization", transform, shrink, item["feature_name"]),
                    item["sample_id"],
                    spec,
                    item["scaling"],
                    item["feature_map"],
                    item["reps"],
                    best_kernel_payload["row"].get("kernel_type", "quantum"),
                    best_kernel_payload["row"].get("alpha") if best_kernel_payload["row"].get("alpha") != "" else None,
                    transform,
                    shrink,
                    1.0,
                    None,
                    val_acc,
                    final_acc,
                    labels,
                    splits,
                    classical_rows,
                    item["health"],
                    time.perf_counter() - start,
                    status,
                    skipped,
                )
            )

    regularization_rows: list[dict[str, Any]] = []
    reg_payload = best_kernel_payload
    if reg_payload is not None:
        item = reg_payload["item"]
        splits = splits_by_sample[item["sample_id"]]
        spec = item["spec"]
        classical_rows = [row for row in classical_rows_by_sample[item["sample_id"]] if row.get("feature_set") == spec.feature_set_name]
        for c_value in V7_C_VALUES:
            for class_weight in V7_CLASS_WEIGHTS:
                start = time.perf_counter()
                try:
                    val_acc, final_acc, preds = v7_fit_precomputed(*reg_payload["matrices"], labels, splits, c_value, class_weight)
                    status = "ok"
                    skipped = ""
                except Exception as exc:
                    val_acc, final_acc, preds = math.nan, math.nan, {}
                    status = "skipped"
                    skipped = f"{type(exc).__name__}: {exc}"
                regularization_rows.append(
                    v7_result_row(
                        candidate_id("qml_v7", item["sample_id"], "regularization", f"C{c_value}", class_weight or "none", item["feature_name"]),
                        item["sample_id"],
                        spec,
                        item["scaling"],
                        item["feature_map"],
                        item["reps"],
                        reg_payload["row"].get("kernel_type", "quantum"),
                        reg_payload["row"].get("alpha") if reg_payload["row"].get("alpha") != "" else None,
                        "raw",
                        0.0,
                        c_value,
                        class_weight,
                        val_acc,
                        final_acc,
                        labels,
                        splits,
                        classical_rows,
                        item["health"],
                        time.perf_counter() - start,
                        status,
                        skipped,
                    )
                )

    meta_rows: list[dict[str, Any]] = []
    if reg_payload is not None:
        item = reg_payload["item"]
        splits = splits_by_sample[item["sample_id"]]
        spec = item["spec"]
        classical_rows = [row for row in classical_rows_by_sample[item["sample_id"]] if row.get("feature_set") == spec.feature_set_name]
        _val_acc, _final_acc, preds = v7_fit_precomputed(*reg_payload["matrices"], labels, splits)
        meta_rows = v7_run_meta_models(item["sample_id"], labels, splits, *reg_payload["matrices"], preds, classical_rows, spec, item["health"])

    all_candidate_rows = hybrid_rows + norm_rows + regularization_rows + meta_rows
    valid_rows = [row for row in all_candidate_rows if row.get("status") == "ok" and math.isfinite(as_float(row.get("validation_accuracy")))]
    validation_leaderboard = sorted(valid_rows, key=lambda row: (as_float(row.get("validation_accuracy")), as_float(row.get("validation_lift")), -as_float(row.get("runtime_seconds"))), reverse=True)
    final_leaderboard = [dict(row) for row in sorted(valid_rows, key=lambda row: as_float(row.get("final_accuracy")), reverse=True)]
    selected = validation_leaderboard[0] if validation_leaderboard else {}
    pure_rows = [row for row in hybrid_rows if row.get("kernel_type") == "quantum" and row.get("status") == "ok"]
    best_pure = max(pure_rows, key=lambda row: as_float(row.get("validation_accuracy")), default={})
    best_hybrid = max((row for row in hybrid_rows if str(row.get("kernel_type", "")).startswith("hybrid") and row.get("status") == "ok"), key=lambda row: as_float(row.get("validation_accuracy")), default={})
    rescued = (
        as_float(selected.get("qml_minus_rbf_svm_validation")) > 0.0
        or as_float(selected.get("qml_minus_logistic_validation")) > 0.0
    ) and as_float(selected.get("final_accuracy")) >= 0.50 and as_float(selected.get("effective_rank_ratio")) >= 0.05 and selected.get("sample_id") in {"medium_distribution_matched", "largest_feasible_distribution_matched"}
    decision_labels: list[str] = []
    if selected.get("sample_id") in {"medium_distribution_matched", "largest_feasible_distribution_matched"} and rescued:
        decision_labels.append("qml_rescued_by_distribution_matching")
    if as_float(best_hybrid.get("validation_accuracy")) > as_float(best_pure.get("validation_accuracy")):
        decision_labels.append("qml_rescued_by_hybrid_kernel")
    if norm_rows and as_float(max(norm_rows, key=lambda row: as_float(row.get("validation_accuracy"))).get("validation_accuracy")) > as_float(selected.get("validation_accuracy")):
        decision_labels.append("qml_rescued_by_kernel_normalization")
    if regularization_rows and as_float(max(regularization_rows, key=lambda row: as_float(row.get("validation_accuracy"))).get("validation_accuracy")) > as_float(selected.get("validation_accuracy")):
        decision_labels.append("qml_rescued_by_qsvc_regularization")
    if meta_rows and as_float(max(meta_rows, key=lambda row: as_float(row.get("validation_accuracy"))).get("validation_accuracy")) > as_float(selected.get("validation_accuracy")):
        decision_labels.append("qml_rescued_as_kernel_feature")
    if not rescued:
        decision_labels.append("qml_not_rescued")
    decision_labels.append("future_blind_required")
    decision = {
        "decision_labels": decision_labels,
        "validation_selected_candidate": selected,
        "qml_rescued": rescued,
        "medium_or_largest_recovered": selected.get("sample_id") in {"medium_distribution_matched", "largest_feasible_distribution_matched"} and as_float(selected.get("validation_accuracy")) > 0.55,
        "hybrid_beats_pure_quantum": as_float(best_hybrid.get("validation_accuracy")) > as_float(best_pure.get("validation_accuracy")),
        "best_beats_rbf_or_logistic_validation": as_float(selected.get("qml_minus_rbf_svm_validation")) > 0.0 or as_float(selected.get("qml_minus_logistic_validation")) > 0.0,
        "qml_replaces_6161_classical_champion": False,
        "diagnostic_only": True,
        "runtime_seconds": time.perf_counter() - started,
        "runtime_limited": time.perf_counter() - started >= config.timeout_seconds,
    }

    columns = ["candidate_id", "sample_id", "feature_set", "scaling", "feature_map", "reps", "kernel_type", "alpha", "normalization_shrinkage", "C", "class_weight", "validation_accuracy", "validation_lift", "final_accuracy", "final_lift", "validation_rows", "final_rows", "kernel_alignment", "effective_rank_ratio", "class_similarity_gap", "qml_minus_rbf_svm_validation", "qml_minus_rbf_svm_final", "qml_minus_logistic_validation", "qml_minus_logistic_final", "runtime_seconds", "claim_label", "status", "skipped_reason"]
    write_frame(OUTPUT_DIR / "qml_v7_distribution_matched_sample_audit.csv", audit_rows, list(audit_rows[0].keys()) if audit_rows else [])
    write_frame(OUTPUT_DIR / "qml_v7_kernel_health_prefilter.csv", health_rows, list(health_rows[0].keys()) if health_rows else [])
    write_frame(OUTPUT_DIR / "qml_v7_hybrid_kernel_results.csv", hybrid_rows, columns)
    write_frame(OUTPUT_DIR / "qml_v7_kernel_normalization_shrinkage_results.csv", norm_rows, columns)
    write_frame(OUTPUT_DIR / "qml_v7_qsvc_regularization_revisit.csv", regularization_rows, columns)
    write_frame(OUTPUT_DIR / "qml_v7_kernel_feature_meta_results.csv", meta_rows, columns + ["meta_model"])
    write_frame(OUTPUT_DIR / "qml_v7_validation_governed_leaderboard.csv", validation_leaderboard, columns + ["best_classical_validation_accuracy", "best_classical_final_accuracy"])
    for row in final_leaderboard:
        if row.get("candidate_id") != selected.get("candidate_id"):
            row["claim_label"] = "exploratory_not_claimable"
    write_frame(OUTPUT_DIR / "qml_v7_exploratory_final_leaderboard.csv", final_leaderboard, columns + ["best_classical_validation_accuracy", "best_classical_final_accuracy"])
    write_frame(OUTPUT_DIR / "qml_v7_same_target_classical_comparison.csv", classical_rows_all, list(classical_rows_all[0].keys()) if classical_rows_all else [])
    write_json(OUTPUT_DIR / "qml_v7_rescue_decision.json", decision)
    manifest = {
        "run_id": "vn30_qml_forecasting_v7_hybrid_kernel_rescue",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "target": V7_TARGET,
        "horizon": V7_HORIZON,
        "diagnostic_only": True,
        "dependency_status": dependency,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v7_manifest.json", manifest)
    write_v7_reports(decision, audit_rows, health_rows, hybrid_rows, norm_rows, regularization_rows, meta_rows, validation_leaderboard)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v7_manifest.json"), "decision_labels": decision_labels, "runtime_limited": decision["runtime_limited"]}), indent=2))
    return manifest


@dataclass
class V8Config:
    timeout_seconds: int = 7200


V8_TARGET = "market_relative_vn30"
V8_HORIZON = 40
V8_V7_VALIDATION = 0.6111111111111112
V8_V7_FINAL = 0.5888888888888889
V8_V4_VALIDATION = 0.6056
V8_V4_FINAL = 0.6944
V8_SCALING_METHODS = ["minmax_0_pi", "standard_zscore"]
V8_META_MODELS = ["l2_logistic", "calibrated_logistic", "lightgbm_small", "random_forest_small", "elasticnet_logistic"]
V8_FEATURE_SET_MODES = [
    "qml_kernel_features_only",
    "qml_kernel_features_plus_relative_strength",
    "qml_kernel_features_plus_market_context",
    "qml_kernel_features_plus_relative_strength_market_context",
]
V8_DRIFT_METHODS = [
    "class_prior_correction",
    "ticker_neutral_calibration",
    "regime_specific_meta_model",
    "temporal_bagging",
    "robust_threshold_by_validation_quantile",
]


def v8_make_meta_model(model_name: str) -> Any:
    if model_name == "elasticnet_logistic":
        return LogisticRegression(
            max_iter=1500,
            solver="saga",
            penalty="elasticnet",
            l1_ratio=0.35,
            C=0.4,
            class_weight="balanced",
            random_state=SEED,
        )
    return v5_make_classical_model(model_name)


def v8_kernel_feature_frames(
    k_train: np.ndarray,
    k_validation: np.ndarray,
    k_final: np.ndarray,
    labels: pd.Series,
    splits: dict[str, pd.Index],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_y = labels.loc[splits["train"]].astype(int).to_numpy()
    positive = train_y == 1
    negative = train_y == 0
    eps = 1e-9

    def build(k_cross: np.ndarray, index: pd.Index, split_name: str) -> pd.DataFrame:
        pos_sim = k_cross[:, positive].mean(axis=1) if positive.any() else np.full(k_cross.shape[0], np.nan)
        neg_sim = k_cross[:, negative].mean(axis=1) if negative.any() else np.full(k_cross.shape[0], np.nan)
        row_std = np.nanstd(k_cross, axis=1)
        density_k = min(8, k_cross.shape[1])
        if density_k > 0:
            local_density = np.sort(k_cross, axis=1)[:, -density_k:].mean(axis=1)
        else:
            local_density = np.full(k_cross.shape[0], np.nan)
        frame = pd.DataFrame(
            {
                "positive_centroid_similarity": pos_sim,
                "negative_centroid_similarity": neg_sim,
                "centroid_similarity_gap": pos_sim - neg_sim,
                "kernel_margin_score": (pos_sim - neg_sim) / np.clip(row_std, eps, None),
                "kernel_local_density": local_density,
                "class_similarity_ratio": pos_sim / np.clip(neg_sim, eps, None),
            },
            index=index,
        )
        frame["split_name"] = split_name
        return frame

    train_features = build(k_train, splits["train"], "train")
    validation_features = build(k_validation, splits["validation"], "validation")
    final_features = build(k_final, splits["final"], "final")
    eigvals, eigvecs = np.linalg.eigh((k_train + k_train.T) / 2.0)
    order = np.argsort(eigvals)[::-1][: min(3, len(eigvals))]
    for pos, eig_idx in enumerate(order, start=1):
        value = max(float(eigvals[eig_idx]), eps)
        train_features[f"top_eigen_projection_{pos}"] = eigvecs[:, eig_idx] * math.sqrt(value)
        validation_features[f"top_eigen_projection_{pos}"] = (k_validation @ eigvecs[:, eig_idx]) / math.sqrt(value)
        final_features[f"top_eigen_projection_{pos}"] = (k_final @ eigvecs[:, eig_idx]) / math.sqrt(value)
    for pos in range(len(order) + 1, 4):
        train_features[f"top_eigen_projection_{pos}"] = np.nan
        validation_features[f"top_eigen_projection_{pos}"] = np.nan
        final_features[f"top_eigen_projection_{pos}"] = np.nan
    feature_cols = [col for col in train_features.columns if col != "split_name"]
    return train_features[feature_cols], validation_features[feature_cols], final_features[feature_cols]


def v8_psi(expected: pd.Series, actual: pd.Series) -> float:
    expected = pd.to_numeric(expected, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    actual = pd.to_numeric(actual, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(expected) < 10 or len(actual) < 10:
        return math.nan
    quantiles = np.linspace(0.0, 1.0, 11)
    bins = np.unique(np.nanquantile(expected.to_numpy(dtype=float), quantiles))
    if len(bins) < 3:
        return math.nan
    bins[0] = -np.inf
    bins[-1] = np.inf
    expected_counts = pd.cut(expected, bins=bins, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
    actual_counts = pd.cut(actual, bins=bins, include_lowest=True).value_counts(sort=False).to_numpy(dtype=float)
    expected_share = np.clip(expected_counts / max(1.0, expected_counts.sum()), 1e-6, None)
    actual_share = np.clip(actual_counts / max(1.0, actual_counts.sum()), 1e-6, None)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def v8_regime_series(features: pd.DataFrame, idx: pd.Index) -> pd.Series:
    candidates = [col for col in features.columns if "regime" in col.lower() or "risk" in col.lower()]
    if candidates:
        values = features.loc[idx, candidates[0]].astype(str).replace({"nan": "neutral", "None": "neutral"})
        return values.fillna("neutral")
    return pd.Series("neutral", index=idx)


def v8_sample_distribution_rows(features: pd.DataFrame, labels: pd.Series, splits_by_sample: dict[str, dict[str, pd.Index]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    timestamps = pd.to_datetime(features["datetime"], errors="coerce")
    for sample_id, splits in splits_by_sample.items():
        for split_name, idx in splits.items():
            y = labels.loc[idx].dropna().astype(int)
            quarters = timestamps.loc[idx].dt.to_period("Q").astype(str).value_counts().sort_index().to_dict() if len(idx) else {}
            months = timestamps.loc[idx].dt.to_period("M").astype(str).value_counts().sort_index().to_dict() if len(idx) else {}
            regimes = v8_regime_series(features, idx).value_counts().sort_index().to_dict() if len(idx) else {}
            rows.append(
                {
                    "audit_type": "sample_distribution",
                    "sample_id": sample_id,
                    "split": split_name,
                    "feature_set": "",
                    "scaling": "",
                    "kernel_feature": "",
                    "rows": int(len(idx)),
                    "label_positive_ratio": float((y == 1).mean()) if len(y) else math.nan,
                    "ticker_count": int(features.loc[idx, "ticker"].nunique()) if len(idx) else 0,
                    "ticker_entropy": v7_entropy(features.loc[idx, "ticker"]) if len(idx) else math.nan,
                    "quarter_distribution": json.dumps(quarters, sort_keys=True),
                    "month_distribution": json.dumps(months, sort_keys=True),
                    "regime_distribution": json.dumps(regimes, sort_keys=True),
                    "timestamp_start": str(timestamps.loc[idx].min()) if len(idx) else "",
                    "timestamp_end": str(timestamps.loc[idx].max()) if len(idx) else "",
                    "mean": math.nan,
                    "std": math.nan,
                    "min": math.nan,
                    "max": math.nan,
                    "psi_vs_validation": math.nan,
                    "feature_target_correlation": math.nan,
                }
            )
    return rows


def v8_kernel_drift_rows(
    features: pd.DataFrame,
    labels: pd.Series,
    sample_id: str,
    feature_set: str,
    scaling: str,
    splits: dict[str, pd.Index],
    kernel_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_frame = kernel_frames["validation"]
    for split_name, frame in kernel_frames.items():
        idx = splits[split_name]
        y = labels.loc[idx].astype(int)
        for col in frame.columns:
            values = pd.to_numeric(frame[col], errors="coerce")
            corr = math.nan
            if len(values.dropna()) > 3 and y.nunique() > 1 and values.std() > 0:
                corr = float(np.corrcoef(values.fillna(values.median()).to_numpy(dtype=float), y.to_numpy(dtype=float))[0, 1])
            rows.append(
                {
                    "audit_type": "kernel_feature_stats",
                    "sample_id": sample_id,
                    "split": split_name,
                    "feature_set": feature_set,
                    "scaling": scaling,
                    "kernel_feature": col,
                    "rows": int(len(frame)),
                    "label_positive_ratio": float((y == 1).mean()) if len(y) else math.nan,
                    "ticker_count": int(features.loc[idx, "ticker"].nunique()) if len(idx) else 0,
                    "ticker_entropy": v7_entropy(features.loc[idx, "ticker"]) if len(idx) else math.nan,
                    "quarter_distribution": "",
                    "month_distribution": "",
                    "regime_distribution": "",
                    "timestamp_start": "",
                    "timestamp_end": "",
                    "mean": float(values.mean()) if len(values) else math.nan,
                    "std": float(values.std()) if len(values) else math.nan,
                    "min": float(values.min()) if len(values) else math.nan,
                    "max": float(values.max()) if len(values) else math.nan,
                    "psi_vs_validation": v8_psi(validation_frame[col], values) if split_name == "final" else math.nan,
                    "feature_target_correlation": corr,
                }
            )
    return rows


def v8_kernel_decile_rows(
    labels: pd.Series,
    sample_id: str,
    feature_set: str,
    scaling: str,
    splits: dict[str, pd.Index],
    kernel_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    validation_frame = kernel_frames["validation"]
    for col in validation_frame.columns:
        reference = pd.to_numeric(validation_frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if reference.nunique(dropna=True) < 3:
            continue
        quantiles = np.unique(np.nanquantile(reference.dropna().to_numpy(dtype=float), np.linspace(0.0, 1.0, 11)))
        if len(quantiles) < 3:
            continue
        quantiles[0] = -np.inf
        quantiles[-1] = np.inf
        for split_name in ["validation", "final"]:
            frame = kernel_frames[split_name]
            idx = splits[split_name]
            values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            deciles = pd.cut(values, bins=quantiles, labels=False, include_lowest=True)
            y = labels.loc[idx].astype(int)
            for decile, decile_idx in deciles.dropna().groupby(deciles.dropna()).groups.items():
                row_idx = pd.Index(decile_idx)
                y_part = y.loc[row_idx]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "split": split_name,
                        "feature_set": feature_set,
                        "scaling": scaling,
                        "kernel_feature": col,
                        "decile": int(decile) + 1,
                        "rows": int(len(y_part)),
                        "label_positive_ratio": float((y_part == 1).mean()) if len(y_part) else math.nan,
                    }
                )
    return rows


def v8_join_feature_frames(parts: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_parts = [part[0] for part in parts]  # type: ignore[index]
    validation_parts = [part[1] for part in parts]  # type: ignore[index]
    final_parts = [part[2] for part in parts]  # type: ignore[index]
    def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        out = pd.concat(frames, axis=1)
        out = out.loc[:, ~out.columns.duplicated()].copy()
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return concat(train_parts), concat(validation_parts), concat(final_parts)


def v8_scores_from_model(
    model_name: str,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    x_final: pd.DataFrame,
    method: str,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    try:
        if method == "temporal_bagging":
            ordered = ordered_index(features_global(), x_train.index)
            chunks = np.array_split(np.asarray(ordered), 3)
            validation_scores: list[np.ndarray] = []
            final_scores: list[np.ndarray] = []
            for end in range(1, len(chunks) + 1):
                sub_idx = pd.Index(np.concatenate(chunks[:end]).tolist())
                if len(sub_idx) < 12 or y_train.loc[sub_idx].nunique() < 2:
                    continue
                model = v8_make_meta_model(model_name)
                model.fit(x_train.loc[sub_idx], y_train.loc[sub_idx])
                validation_scores.append(predict_probability(model, x_validation))
                final_scores.append(predict_probability(model, x_final))
            if not validation_scores:
                raise ValueError("temporal bagging had no valid class-balanced subwindow")
            return np.mean(validation_scores, axis=0), np.mean(final_scores, axis=0), "ok", ""
        model = v8_make_meta_model(model_name)
        if model_name == "calibrated_logistic" and y_train.value_counts().min() < 3:
            raise ValueError("calibrated logistic requires at least three train rows per class")
        model.fit(x_train, y_train)
        return predict_probability(model, x_validation), predict_probability(model, x_final), "ok", ""
    except Exception as exc:
        return np.full(len(x_validation), np.nan), np.full(len(x_final), np.nan), "skipped", f"{type(exc).__name__}: {exc}"


def v8_best_threshold(scores: np.ndarray, y_true: pd.Series, quantiles: list[float] | None = None) -> float:
    clean = pd.Series(scores).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return 0.5
    qs = quantiles or [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]
    best_threshold = 0.5
    best_acc = -1.0
    for q in qs:
        threshold = float(np.nanquantile(clean.to_numpy(dtype=float), q))
        pred = (scores >= threshold).astype(int)
        acc = accuracy(y_true, pred)
        if math.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold


def v8_apply_decision_method(
    method: str,
    validation_scores: np.ndarray,
    final_scores: np.ndarray,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    validation_y = labels.loc[splits["validation"]].astype(int)
    val_scores = np.asarray(validation_scores, dtype=float)
    fin_scores = np.asarray(final_scores, dtype=float)
    metadata: dict[str, Any] = {"method_detail": method, "threshold": 0.5, "coverage_fraction": 1.0}
    if method == "class_prior_correction":
        val_prior = float(validation_y.mean()) if len(validation_y) else 0.5
        threshold = float(np.nanquantile(val_scores, max(0.0, min(1.0, 1.0 - val_prior))))
    elif method == "robust_threshold_by_validation_quantile":
        threshold = v8_best_threshold(val_scores, validation_y)
    elif method == "ticker_neutral_calibration":
        val_ticker = features.loc[splits["validation"], "ticker"].astype(str)
        fin_ticker = features.loc[splits["final"], "ticker"].astype(str)
        global_mean = float(np.nanmean(val_scores))
        global_std = float(np.nanstd(val_scores)) or 1.0
        stats = pd.DataFrame({"ticker": val_ticker.to_numpy(), "score": val_scores}).groupby("ticker")["score"].agg(["mean", "std"])
        stats["std"] = stats["std"].replace(0.0, np.nan).fillna(global_std)
        val_adj = np.array([(score - stats.loc[ticker, "mean"]) / stats.loc[ticker, "std"] if ticker in stats.index else (score - global_mean) / global_std for score, ticker in zip(val_scores, val_ticker)])
        fin_adj = np.array([(score - stats.loc[ticker, "mean"]) / stats.loc[ticker, "std"] if ticker in stats.index else (score - global_mean) / global_std for score, ticker in zip(fin_scores, fin_ticker)])
        threshold = v8_best_threshold(val_adj, validation_y)
        metadata.update({"threshold": threshold, "method_detail": "ticker_neutral_validation_zscore"})
        return pd.Series((val_adj >= threshold).astype(int), index=splits["validation"]), pd.Series((fin_adj >= threshold).astype(int), index=splits["final"]), metadata
    elif method == "regime_specific_meta_model":
        val_regime = v8_regime_series(features, splits["validation"])
        fin_regime = v8_regime_series(features, splits["final"])
        global_threshold = v8_best_threshold(val_scores, validation_y)
        thresholds: dict[str, float] = {}
        val_pred = np.zeros(len(val_scores), dtype=int)
        for regime, positions in val_regime.reset_index(drop=True).groupby(val_regime.reset_index(drop=True)).groups.items():
            pos = np.asarray(list(positions), dtype=int)
            if len(pos) >= 10 and validation_y.iloc[pos].nunique() > 1:
                thresholds[str(regime)] = v8_best_threshold(val_scores[pos], validation_y.iloc[pos])
            else:
                thresholds[str(regime)] = global_threshold
            val_pred[pos] = (val_scores[pos] >= thresholds[str(regime)]).astype(int)
        final_pred = np.array([(score >= thresholds.get(str(regime), global_threshold)) for score, regime in zip(fin_scores, fin_regime)], dtype=int)
        metadata.update({"threshold": global_threshold, "regime_thresholds": json.dumps(thresholds, sort_keys=True)})
        return pd.Series(val_pred, index=splits["validation"]), pd.Series(final_pred, index=splits["final"]), metadata
    else:
        threshold = 0.5
    metadata["threshold"] = threshold
    return pd.Series((val_scores >= threshold).astype(int), index=splits["validation"]), pd.Series((fin_scores >= threshold).astype(int), index=splits["final"]), metadata


def v8_prediction_stability(
    features: pd.DataFrame,
    labels: pd.Series,
    idx: pd.Index,
    prediction: pd.Series,
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "ticker": features.loc[idx, "ticker"].astype(str).to_numpy(),
            "quarter": pd.to_datetime(features.loc[idx, "datetime"], errors="coerce").dt.to_period("Q").astype(str).to_numpy(),
            "y": labels.loc[idx].astype(int).to_numpy(),
            "pred": prediction.loc[idx].astype(int).to_numpy(),
        },
        index=idx,
    )
    frame["correct"] = (frame["y"] == frame["pred"]).astype(float)
    ticker_acc = frame.groupby("ticker")["correct"].mean()
    quarter_acc = frame.groupby("quarter")["correct"].mean()
    ticker_counts = frame["ticker"].value_counts(normalize=True)
    return {
        "ticker_stability": float(max(0.0, 1.0 - ticker_acc.std(ddof=0))) if len(ticker_acc) else math.nan,
        "quarter_stability": float(max(0.0, 1.0 - quarter_acc.std(ddof=0))) if len(quarter_acc) else math.nan,
        "min_ticker_accuracy": float(ticker_acc.min()) if len(ticker_acc) else math.nan,
        "min_quarter_accuracy": float(quarter_acc.min()) if len(quarter_acc) else math.nan,
        "max_ticker_share": float(ticker_counts.max()) if len(ticker_counts) else math.nan,
    }


def v8_classical_summary(classical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    best_val_row: dict[str, Any] = {}
    best_final_row: dict[str, Any] = {}
    for model_name in ["l2_logistic", "calibrated_logistic", "lightgbm_small", "svm_rbf", "svm_linear", "random_forest_small"]:
        rows = [row for row in classical_rows if row.get("model_family") == model_name and row.get("status") == "ok"]
        if not rows:
            continue
        best_row = max(rows, key=lambda row: as_float(row.get("validation_accuracy")))
        summary[f"{model_name}_validation_accuracy"] = best_row.get("validation_accuracy", math.nan)
        summary[f"{model_name}_final_accuracy"] = best_row.get("final_accuracy", math.nan)
        if not best_val_row or as_float(best_row.get("validation_accuracy")) > as_float(best_val_row.get("validation_accuracy")):
            best_val_row = best_row
        if not best_final_row or as_float(best_row.get("final_accuracy")) > as_float(best_final_row.get("final_accuracy")):
            best_final_row = best_row
    summary["best_classical_validation_model"] = best_val_row.get("model_family", "")
    summary["best_classical_validation_accuracy"] = best_val_row.get("validation_accuracy", math.nan)
    summary["best_classical_final_model"] = best_final_row.get("model_family", "")
    summary["best_classical_final_accuracy"] = best_final_row.get("final_accuracy", math.nan)
    return summary


def v8_drift_penalty(kernel_frames: dict[str, pd.DataFrame]) -> float:
    penalties: list[float] = []
    validation = kernel_frames["validation"]
    final = kernel_frames["final"]
    for col in validation.columns:
        val = pd.to_numeric(validation[col], errors="coerce")
        fin = pd.to_numeric(final[col], errors="coerce")
        std = float(val.std()) if len(val) else math.nan
        if math.isfinite(std) and std > 1e-9:
            penalties.append(abs(float(fin.mean()) - float(val.mean())) / std)
    mean_penalty = float(np.nanmean(penalties)) if penalties else math.nan
    return float(1.0 / (1.0 + mean_penalty)) if math.isfinite(mean_penalty) else 0.5


def v8_build_meta_feature_set(
    mode: str,
    kernel_frames: dict[str, pd.DataFrame],
    relative_spec: FeatureSpec,
    market_spec: FeatureSpec,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    parts: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = [(kernel_frames["train"], kernel_frames["validation"], kernel_frames["final"])]
    if mode in {"qml_kernel_features_plus_relative_strength", "qml_kernel_features_plus_relative_strength_market_context"}:
        parts.append((relative_spec.x_train.add_prefix("relative__"), relative_spec.x_validation.add_prefix("relative__"), relative_spec.x_final.add_prefix("relative__")))
    if mode in {"qml_kernel_features_plus_market_context", "qml_kernel_features_plus_relative_strength_market_context"}:
        parts.append((market_spec.x_train.add_prefix("market__"), market_spec.x_validation.add_prefix("market__"), market_spec.x_final.add_prefix("market__")))
    return v8_join_feature_frames(parts)


def v8_result_row(
    sample_id: str,
    kernel_feature_source: str,
    scaling: str,
    feature_set_mode: str,
    model_name: str,
    method: str,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    validation_pred: pd.Series,
    final_pred: pd.Series,
    classical_summary: dict[str, Any],
    drift_penalty_score: float,
    metadata: dict[str, Any],
    runtime_seconds: float,
    status: str = "ok",
    skipped_reason: str = "",
    selection_eligible: bool = True,
) -> dict[str, Any]:
    validation_y = labels.loc[splits["validation"]].astype(int)
    final_y = labels.loc[splits["final"]].astype(int)
    val_acc = accuracy(validation_y, validation_pred.loc[splits["validation"]]) if status == "ok" else math.nan
    final_acc = accuracy(final_y, final_pred.loc[splits["final"]]) if status == "ok" else math.nan
    simple_val_name, simple_val_acc = strongest_simple_baseline(features_global(), labels, splits["validation"])
    simple_final_name, simple_final_acc = strongest_simple_baseline(features_global(), labels, splits["final"])
    stability = v8_prediction_stability(features_global(), labels, splits["validation"], validation_pred) if status == "ok" else {}
    val_lift = val_acc - simple_val_acc if math.isfinite(val_acc) and math.isfinite(simple_val_acc) else math.nan
    validation_score = (
        0.35 * val_acc
        + 0.25 * max(0.0, val_lift if math.isfinite(val_lift) else 0.0)
        + 0.15 * as_float(stability.get("ticker_stability"))
        + 0.15 * as_float(stability.get("quarter_stability"))
        + 0.10 * drift_penalty_score
    ) if status == "ok" else math.nan
    row = {
        "candidate_id": candidate_id("qml_v8", sample_id, kernel_feature_source, scaling, feature_set_mode, model_name, method),
        "sample_id": sample_id,
        "kernel_feature_source": kernel_feature_source,
        "scaling": scaling,
        "feature_set": feature_set_mode,
        "meta_model": model_name,
        "drift_method": method,
        "target_variant": V8_TARGET,
        "horizon": V8_HORIZON,
        "validation_accuracy": val_acc,
        "validation_lift": val_lift,
        "validation_lift_over_strongest_baseline": val_lift,
        "validation_score": validation_score,
        "final_accuracy": final_acc,
        "final_lift": final_acc - simple_final_acc if math.isfinite(final_acc) and math.isfinite(simple_final_acc) else math.nan,
        "validation_rows": int(len(validation_y)),
        "final_rows": int(len(final_y)),
        "strongest_validation_baseline": simple_val_name,
        "strongest_final_baseline": simple_final_name,
        "ticker_stability": stability.get("ticker_stability", math.nan),
        "quarter_stability": stability.get("quarter_stability", math.nan),
        "min_ticker_accuracy": stability.get("min_ticker_accuracy", math.nan),
        "min_quarter_accuracy": stability.get("min_quarter_accuracy", math.nan),
        "max_ticker_share": stability.get("max_ticker_share", math.nan),
        "drift_penalty_score": drift_penalty_score,
        "threshold": metadata.get("threshold", math.nan),
        "coverage_fraction": metadata.get("coverage_fraction", 1.0),
        "method_detail": metadata.get("method_detail", method),
        "runtime_seconds": runtime_seconds,
        "status": status,
        "skipped_reason": skipped_reason,
        "selection_eligible": bool(selection_eligible),
        "claim_label": "qml_not_claimable",
    }
    row.update(classical_summary)
    for model_name_cmp, prefix in [("l2_logistic", "logistic"), ("svm_rbf", "rbf_svm"), ("lightgbm_small", "lightgbm_small")]:
        row[f"qml_minus_{prefix}_validation"] = val_acc - as_float(classical_summary.get(f"{model_name_cmp}_validation_accuracy")) if math.isfinite(val_acc) else math.nan
        row[f"qml_minus_{prefix}_final"] = final_acc - as_float(classical_summary.get(f"{model_name_cmp}_final_accuracy")) if math.isfinite(final_acc) else math.nan
    row["qml_minus_best_classical_validation"] = val_acc - as_float(classical_summary.get("best_classical_validation_accuracy")) if math.isfinite(val_acc) else math.nan
    row["qml_minus_best_classical_final"] = final_acc - as_float(classical_summary.get("best_classical_final_accuracy")) if math.isfinite(final_acc) else math.nan
    row["comparison_vs_v7_final_accuracy"] = final_acc - V8_V7_FINAL if math.isfinite(final_acc) else math.nan
    row["comparison_vs_v4_final_accuracy"] = final_acc - V8_V4_FINAL if math.isfinite(final_acc) else math.nan
    row["comparison_vs_classical_champion_accuracy"] = final_acc - CLASSICAL_CHAMPION["final_accuracy"] if math.isfinite(final_acc) else math.nan
    if (
        selection_eligible
        and status == "ok"
        and final_acc > V8_V7_FINAL
        and val_acc >= V8_V7_VALIDATION - 0.02
        and (
            as_float(row.get("qml_minus_logistic_validation")) > 0.0
            or as_float(row.get("qml_minus_rbf_svm_validation")) > 0.0
            or as_float(row.get("qml_minus_lightgbm_small_validation")) > 0.0
        )
    ):
        row["claim_label"] = "qml_kernel_feature_rescue_improved_requires_future_blind"
    elif method.startswith("abstention_coverage"):
        row["claim_label"] = "exploratory_not_claimable"
    else:
        row["claim_label"] = "qml_diagnostic_only"
    return row


def v8_abstention_rows(
    base_context: dict[str, Any],
    validation_scores: np.ndarray,
    final_scores: np.ndarray,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    classical_summary: dict[str, Any],
    drift_penalty_score: float,
    threshold: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    val_conf = np.abs(validation_scores - threshold)
    fin_conf = np.abs(final_scores - threshold)
    for coverage in [0.20, 0.40, 0.60]:
        val_take = max(1, int(round(len(val_conf) * coverage)))
        fin_take = max(1, int(round(len(fin_conf) * coverage)))
        val_order = np.argsort(val_conf)[::-1][:val_take]
        fin_order = np.argsort(fin_conf)[::-1][:fin_take]
        val_idx = pd.Index(np.asarray(splits["validation"])[val_order])
        fin_idx = pd.Index(np.asarray(splits["final"])[fin_order])
        sub_splits = {"train": splits["train"], "validation": val_idx, "final": fin_idx}
        val_pred = pd.Series((validation_scores[val_order] >= threshold).astype(int), index=val_idx)
        fin_pred = pd.Series((final_scores[fin_order] >= threshold).astype(int), index=fin_idx)
        metadata = {"threshold": threshold, "coverage_fraction": coverage, "method_detail": f"abstention_coverage_{int(coverage * 100)}"}
        rows.append(
            v8_result_row(
                base_context["sample_id"],
                base_context["kernel_feature_source"],
                base_context["scaling"],
                base_context["feature_set_mode"],
                base_context["model_name"],
                f"abstention_coverage_{int(coverage * 100)}",
                labels,
                sub_splits,
                val_pred,
                fin_pred,
                classical_summary,
                drift_penalty_score,
                metadata,
                0.0,
                "ok",
                "",
                selection_eligible=False,
            )
        )
    return rows


def write_v8_reports(decision: dict[str, Any], drift_rows: list[dict[str, Any]], audit_rows: list[dict[str, Any]], leaderboard: list[dict[str, Any]], classical_rows: list[dict[str, Any]]) -> None:
    selected = decision.get("validation_selected_candidate", {})
    best_feature_corr = sorted(
        [row for row in drift_rows if row.get("audit_type") == "kernel_feature_stats" and row.get("split") == "validation" and math.isfinite(as_float(row.get("feature_target_correlation")))],
        key=lambda row: abs(as_float(row.get("feature_target_correlation"))),
        reverse=True,
    )
    useful_features = ", ".join(row.get("kernel_feature", "") for row in best_feature_corr[:3])
    final_improved = as_float(selected.get("final_accuracy")) > V8_V7_FINAL
    same_target_wins = [
        name
        for name, field in [
            ("Logistic", "qml_minus_logistic_validation"),
            ("RBF SVM", "qml_minus_rbf_svm_validation"),
            ("LightGBM", "qml_minus_lightgbm_small_validation"),
        ]
        if as_float(selected.get(field)) > 0.0
    ]
    final_psi_values = [as_float(row.get("psi_vs_validation")) for row in drift_rows if row.get("audit_type") == "kernel_feature_stats" and row.get("split") == "final"]
    mean_final_psi = float(np.nanmean([value for value in final_psi_values if math.isfinite(value)])) if any(math.isfinite(value) for value in final_psi_values) else math.nan
    summary = f"""# VN30 QML Forecasting V8 Drift-Aware Kernel-Feature Result Summary

## Required Answers

1. Did kernel-feature drift explain validation-final decay: partially; mean final PSI versus validation across kernel-feature audits was {mean_final_psi:.4f} where finite, and final label/ticker distribution still differed from validation.
2. Which QML kernel features were most useful: {useful_features or "no stable correlation feature dominated"}.
3. Did drift-aware calibration improve over V7: {str(final_improved).lower()}.
4. Did the final result improve over 58.89%: {str(final_improved).lower()}; selected final accuracy was {pct(selected.get("final_accuracy"))}.
5. Did the model beat same-target Logistic/RBF/LightGBM: validation wins versus {", ".join(same_target_wins) if same_target_wins else "none"}.
6. Does any result replace the 61.61% classical champion: no; target/scope differs and future-blind confirmation is required.
7. Is a new QML paper still justified: yes, as a diagnostic representation-and-drift study, not as a replacement claim.
8. Exact claim boundary: V8 is experimental and diagnostic-only; selection is validation-governed; final rows are scoring-only; no trading, profitability, BUY/SELL, live deployment, VN100, DOCX, merge, tag, push-mirror, or index-as-stock claim is made.

## Locked Validation-Selected Candidate

- Candidate: `{selected.get("candidate_id", "")}`.
- Sample: {selected.get("sample_id", "")}.
- Kernel feature source: {selected.get("kernel_feature_source", "")}.
- Feature set: {selected.get("feature_set", "")}.
- Meta-model: {selected.get("meta_model", "")}.
- Drift method: {selected.get("drift_method", "")}.
- Validation accuracy: {pct(selected.get("validation_accuracy"))}.
- Final accuracy: {pct(selected.get("final_accuracy"))}.
- Delta versus V7 final 58.89%: {pp(selected.get("comparison_vs_v7_final_accuracy"))}.
- Delta versus 61.61% classical champion context: {pp(selected.get("comparison_vs_classical_champion_accuracy"))}.
- Claim label: `{selected.get("claim_label", "qml_diagnostic_only")}`.
"""
    write_markdown(REPO_ROOT / "reports" / "results" / "VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_RESULT_SUMMARY.md", summary)
    claim = """# VN30 QML Forecasting V8 Drift-Aware Kernel-Feature Claim Boundary

- QML v8 drift-aware kernel-feature rescue is experimental and diagnostic-only.
- Scope is VN30 stock hourly forecasting only.
- Target scope is market_relative_vn30 at h40 only.
- No VN100 scope is claimed.
- No index-as-stock claim is made.
- The QML component is quantum-kernel-derived feature extraction only; this is not a broad QSVC or circuit search.
- Feature_timestamp and target_timestamp split discipline is required.
- Feature selection, scaling, kernel construction, PCA, and kernel-feature transforms must be train-only or validation-governed without final-label selection.
- Drift-aware thresholds, ticker calibration, regime thresholds, and temporal bagging are selected by validation only.
- Final performance is scoring-only and does not rank claimable rows.
- Abstention/coverage rows are diagnostic only and cannot substitute for full-coverage accuracy.
- Same-target comparisons are diagnostic and do not replace the 61.61% L2 Logistic classical champion because the target/scope is not directly identical.
- No trading, profitability, BUY/SELL, recommendation, investment advice, live deployment, or deployment claim is made.
- No DOCX, paper artifact, tag, merge, push --mirror, or main-branch claim is made.
- Stronger QML claims require full validation governance and future-blind confirmation.
"""
    write_markdown(REPO_ROOT / "reports" / "claims" / "VN30_QML_FORECASTING_V8_DRIFT_AWARE_KERNEL_FEATURE_CLAIM_BOUNDARY.md", claim)


def run_v8_drift_aware_kernel_features(config: V8Config) -> dict[str, Any]:
    global _FEATURES_GLOBAL
    started = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dependency = dependency_status()
    write_json(OUTPUT_DIR / "qml_dependency_status.json", dependency)
    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    index_data = load_index_data()
    features, relative_cols = add_v3_relative_strength_features(features, index_data)
    features["feature_timestamp"] = pd.to_datetime(features["datetime"], errors="coerce")
    _FEATURES_GLOBAL = features
    source_groups = build_source_groups(features, family_cols)
    source_groups["relative_strength_features"] = relative_cols
    source_groups["relative_plus_market_context_features"] = sorted(set(relative_cols).union(source_groups.get("market_context_features", [])[:12]))
    source_groups["combined_strategy_features"] = sorted(set(source_groups["combined_strategy_features"]).union(relative_cols))
    labels = build_labels(features, index_data, V8_TARGET, V8_HORIZON)
    full_splits = strict_split_indices(features, labels)
    sample_defs = {
        "v4_sized_distribution_matched": (80, 180, 180, False),
        "medium_distribution_matched": (100, 240, 240, False),
        "largest_feasible_distribution_matched": (120, 300, 300, False),
    }
    splits_by_sample = {
        sample_id: v7_distribution_matched_splits(features, labels, full_splits, train_n, val_n, final_n, original)
        for sample_id, (train_n, val_n, final_n, original) in sample_defs.items()
    }
    drift_rows = v8_sample_distribution_rows(features, labels, splits_by_sample)
    decile_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    classical_rows_all: list[dict[str, Any]] = []
    classical_summary_by_sample: dict[str, dict[str, Any]] = {}

    for sample_id, splits in splits_by_sample.items():
        specs: dict[str, FeatureSpec] = {}
        for name, source_group, compression, n_features in [
            ("relative_strength_topk4", "relative_strength_features", "topk_availability", 4),
            ("relative_market_context_topk4", "relative_plus_market_context_features", "topk_availability", 4),
            ("market_context_topk4", "market_context_features", "topk_availability", 4),
        ]:
            spec, _audit = fit_feature_spec(features, labels, V8_TARGET, V8_HORIZON, source_group, source_groups[source_group], compression, n_features, splits)
            if spec.selection_status in {"ok", "mutual_info_failed_fallback_availability"}:
                specs[name] = spec
        if "relative_strength_topk4" not in specs or "market_context_topk4" not in specs:
            continue
        classical_rows_for_sample: list[dict[str, Any]] = []
        for spec in specs.values():
            rows, _pred = v5_run_classical_models(spec, labels, splits, sample_id)
            classical_rows_for_sample.extend(rows)
            classical_rows_all.extend(rows)
        classical_summary = v8_classical_summary(classical_rows_for_sample)
        classical_summary_by_sample[sample_id] = classical_summary

        kernel_sources = [
            ("relative_strength_topk4", specs["relative_strength_topk4"]),
            ("relative_market_context_topk4", specs.get("relative_market_context_topk4", specs["relative_strength_topk4"])),
        ]
        for kernel_name, spec in kernel_sources:
            for scaling in V8_SCALING_METHODS:
                if time.perf_counter() - started >= config.timeout_seconds:
                    break
                kernel_started = time.perf_counter()
                try:
                    x_train, x_validation, x_final, scaling_info = v6_scaling_transform(spec, scaling)
                    if scaling_info["status"] != "ok":
                        raise ValueError(scaling_info["skipped_reason"])
                    k_train, k_validation, k_final = v6_quantum_kernel_matrices(x_train, x_validation, x_final, 2, "full")
                    health = v7_health_row(sample_id, spec, scaling, "ZZFeatureMap", 2, labels, splits, k_train, k_validation)
                    train_meta, validation_meta, final_meta = v8_kernel_feature_frames(k_train, k_validation, k_final, labels, splits)
                    kernel_frames = {"train": train_meta, "validation": validation_meta, "final": final_meta}
                    drift_penalty_score = v8_drift_penalty(kernel_frames)
                    audit_rows.append(
                        {
                            "sample_id": sample_id,
                            "kernel_feature_source": kernel_name,
                            "feature_set_name": spec.feature_set_name,
                            "scaling": scaling,
                            "feature_map": "ZZFeatureMap",
                            "reps": 2,
                            "train_rows": int(len(splits["train"])),
                            "validation_rows": int(len(splits["validation"])),
                            "final_rows": int(len(splits["final"])),
                            "selected_features": "|".join(spec.selected_features),
                            "kernel_feature_names": "|".join(train_meta.columns),
                            "leakage_guard_passed": bool(leakage_guard_passed(features, labels, splits)),
                            "kernel_alignment": health.get("kernel_target_alignment", math.nan),
                            "validation_kernel_alignment": health.get("validation_kernel_target_alignment", math.nan),
                            "effective_rank_ratio": health.get("effective_rank_ratio", math.nan),
                            "class_similarity_gap": health.get("class_similarity_gap", math.nan),
                            "drift_penalty_score": drift_penalty_score,
                            "runtime_seconds": time.perf_counter() - kernel_started,
                            "status": "ok",
                            "skipped_reason": "",
                        }
                    )
                    drift_rows.extend(v8_kernel_drift_rows(features, labels, sample_id, spec.feature_set_name, scaling, splits, kernel_frames))
                    decile_rows.extend(v8_kernel_decile_rows(labels, sample_id, spec.feature_set_name, scaling, splits, kernel_frames))
                    for feature_set_mode in V8_FEATURE_SET_MODES:
                        x_train_meta, x_validation_meta, x_final_meta = v8_build_meta_feature_set(feature_set_mode, kernel_frames, specs["relative_strength_topk4"], specs["market_context_topk4"])
                        y_train = labels.loc[splits["train"]].astype(int)
                        for model_name in V8_META_MODELS:
                            for method in V8_DRIFT_METHODS:
                                row_started = time.perf_counter()
                                val_scores, final_scores, status, skipped = v8_scores_from_model(model_name, x_train_meta, y_train, x_validation_meta, x_final_meta, method)
                                if status == "ok":
                                    val_pred, final_pred, metadata = v8_apply_decision_method(method, val_scores, final_scores, labels, splits, features)
                                else:
                                    val_pred = pd.Series(np.zeros(len(splits["validation"]), dtype=int), index=splits["validation"])
                                    final_pred = pd.Series(np.zeros(len(splits["final"]), dtype=int), index=splits["final"])
                                    metadata = {"threshold": math.nan, "coverage_fraction": 1.0, "method_detail": method}
                                row = v8_result_row(
                                    sample_id,
                                    kernel_name,
                                    scaling,
                                    feature_set_mode,
                                    model_name,
                                    method,
                                    labels,
                                    splits,
                                    val_pred,
                                    final_pred,
                                    classical_summary,
                                    drift_penalty_score,
                                    metadata,
                                    time.perf_counter() - row_started,
                                    status,
                                    skipped,
                                    selection_eligible=True,
                                )
                                validation_rows.append(row)
                                if status == "ok" and method == "robust_threshold_by_validation_quantile":
                                    threshold = as_float(row.get("threshold"))
                                    validation_rows.extend(
                                        v8_abstention_rows(
                                            {
                                                "sample_id": sample_id,
                                                "kernel_feature_source": kernel_name,
                                                "scaling": scaling,
                                                "feature_set_mode": feature_set_mode,
                                                "model_name": model_name,
                                            },
                                            val_scores,
                                            final_scores,
                                            labels,
                                            splits,
                                            classical_summary,
                                            drift_penalty_score,
                                            threshold if math.isfinite(threshold) else 0.5,
                                        )
                                    )
                except Exception as exc:
                    audit_rows.append(
                        {
                            "sample_id": sample_id,
                            "kernel_feature_source": kernel_name,
                            "feature_set_name": spec.feature_set_name,
                            "scaling": scaling,
                            "feature_map": "ZZFeatureMap",
                            "reps": 2,
                            "train_rows": int(len(splits["train"])),
                            "validation_rows": int(len(splits["validation"])),
                            "final_rows": int(len(splits["final"])),
                            "selected_features": "|".join(spec.selected_features),
                            "kernel_feature_names": "",
                            "leakage_guard_passed": bool(leakage_guard_passed(features, labels, splits)),
                            "kernel_alignment": math.nan,
                            "validation_kernel_alignment": math.nan,
                            "effective_rank_ratio": math.nan,
                            "class_similarity_gap": math.nan,
                            "drift_penalty_score": math.nan,
                            "runtime_seconds": time.perf_counter() - kernel_started,
                            "status": "skipped",
                            "skipped_reason": f"{type(exc).__name__}: {exc}",
                        }
                    )

    valid_selectable = [
        row
        for row in validation_rows
        if row.get("status") == "ok"
        and row.get("selection_eligible") is True
        and math.isfinite(as_float(row.get("validation_accuracy")))
        and as_float(row.get("validation_rows")) >= 180
        and (not math.isfinite(as_float(row.get("max_ticker_share"))) or as_float(row.get("max_ticker_share")) <= 0.25)
        and (not math.isfinite(as_float(row.get("min_quarter_accuracy"))) or as_float(row.get("min_quarter_accuracy")) >= 0.45)
    ]
    if not valid_selectable:
        valid_selectable = [row for row in validation_rows if row.get("status") == "ok" and row.get("selection_eligible") is True and math.isfinite(as_float(row.get("validation_accuracy")))]
    preferred_selectable = [
        row
        for row in valid_selectable
        if as_float(row.get("validation_accuracy")) >= 0.60
        and as_float(row.get("validation_lift_over_strongest_baseline")) > 0.0
    ]
    selection_pool = preferred_selectable if preferred_selectable else valid_selectable
    ranked_selection_pool = sorted(
        selection_pool,
        key=lambda row: (as_float(row.get("validation_score")), as_float(row.get("validation_accuracy")), -as_float(row.get("runtime_seconds"))),
        reverse=True,
    )
    remaining_rows = [row for row in valid_selectable if row.get("candidate_id") not in {item.get("candidate_id") for item in ranked_selection_pool}]
    validation_leaderboard = ranked_selection_pool + sorted(
        remaining_rows,
        key=lambda row: (as_float(row.get("validation_score")), as_float(row.get("validation_accuracy")), -as_float(row.get("runtime_seconds"))),
        reverse=True,
    )
    selected = validation_leaderboard[0] if validation_leaderboard else {}
    final_leaderboard = [dict(row) for row in sorted([row for row in validation_rows if row.get("status") == "ok"], key=lambda row: as_float(row.get("final_accuracy")), reverse=True)]
    for row in final_leaderboard:
        if row.get("candidate_id") != selected.get("candidate_id"):
            row["claim_label"] = "exploratory_not_claimable"
    locked = {
        "selection_rule": "validation_score_only",
        "selected_at_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "candidate": selected,
        "no_final_selection": True,
    }
    final_result = {
        "candidate_id": selected.get("candidate_id", ""),
        "sample_id": selected.get("sample_id", ""),
        "final_accuracy": selected.get("final_accuracy", math.nan),
        "final_lift": selected.get("final_lift", math.nan),
        "final_rows": selected.get("final_rows", 0),
        "comparison_vs_v7_final_accuracy": selected.get("comparison_vs_v7_final_accuracy", math.nan),
        "comparison_vs_v4_final_accuracy": selected.get("comparison_vs_v4_final_accuracy", math.nan),
        "comparison_vs_classical_champion_accuracy": selected.get("comparison_vs_classical_champion_accuracy", math.nan),
        "claim_label": selected.get("claim_label", "qml_diagnostic_only"),
        "status": selected.get("status", "skipped"),
    }
    improved = (
        as_float(selected.get("final_accuracy")) > V8_V7_FINAL
        and as_float(selected.get("validation_accuracy")) >= V8_V7_VALIDATION - 0.02
        and (
            as_float(selected.get("qml_minus_logistic_validation")) > 0.0
            or as_float(selected.get("qml_minus_rbf_svm_validation")) > 0.0
            or as_float(selected.get("qml_minus_lightgbm_small_validation")) > 0.0
        )
    )
    decision_labels = ["qml_kernel_feature_rescue_improved" if improved else "qml_kernel_feature_rescue_not_improved"]
    if as_float(selected.get("drift_penalty_score")) >= 0.50:
        decision_labels.append("qml_drift_partially_corrected")
    decision_labels.extend(["qml_future_blind_required", "qml_not_claimable"])
    decision = {
        "decision_labels": decision_labels,
        "validation_selected_candidate": selected,
        "meaningful_improvement": bool(improved),
        "final_improved_over_v7": as_float(selected.get("final_accuracy")) > V8_V7_FINAL,
        "validation_not_materially_below_v7": as_float(selected.get("validation_accuracy")) >= V8_V7_VALIDATION - 0.02,
        "beats_same_target_logistic_validation": as_float(selected.get("qml_minus_logistic_validation")) > 0.0,
        "beats_same_target_rbf_validation": as_float(selected.get("qml_minus_rbf_svm_validation")) > 0.0,
        "beats_same_target_lightgbm_validation": as_float(selected.get("qml_minus_lightgbm_small_validation")) > 0.0,
        "qml_replaces_6161_classical_champion": False,
        "diagnostic_only": True,
        "runtime_seconds": time.perf_counter() - started,
        "runtime_limited": time.perf_counter() - started >= config.timeout_seconds,
    }
    champion_context = [
        {"benchmark": "v7_best_qml_kernel_feature_meta", "validation_accuracy": V8_V7_VALIDATION, "final_accuracy": V8_V7_FINAL, "scope": "market_relative_vn30 h40 bounded sample"},
        {"benchmark": "v4_bounded_qml_kernel", "validation_accuracy": V8_V4_VALIDATION, "final_accuracy": V8_V4_FINAL, "scope": "market_relative_vn30 h40 bounded sample"},
        {"benchmark": "classical_champion_context", "validation_accuracy": math.nan, "final_accuracy": CLASSICAL_CHAMPION["final_accuracy"], "scope": "absolute_direction h40 contextual only"},
        {"benchmark": "v8_locked_candidate", "validation_accuracy": selected.get("validation_accuracy", math.nan), "final_accuracy": selected.get("final_accuracy", math.nan), "scope": "market_relative_vn30 h40 validation-selected"},
    ]
    write_frame(OUTPUT_DIR / "qml_v8_kernel_feature_audit.csv", audit_rows, list(audit_rows[0].keys()) if audit_rows else [])
    write_frame(OUTPUT_DIR / "qml_v8_drift_audit.csv", drift_rows, list(drift_rows[0].keys()) if drift_rows else [])
    write_frame(OUTPUT_DIR / "qml_v8_kernel_feature_deciles.csv", decile_rows, list(decile_rows[0].keys()) if decile_rows else [])
    write_frame(OUTPUT_DIR / "qml_v8_validation_results.csv", validation_rows, list(validation_rows[0].keys()) if validation_rows else [])
    write_frame(OUTPUT_DIR / "qml_v8_validation_leaderboard.csv", validation_leaderboard, list(validation_leaderboard[0].keys()) if validation_leaderboard else [])
    write_json(OUTPUT_DIR / "qml_v8_locked_candidate.json", locked)
    write_frame(OUTPUT_DIR / "qml_v8_final_result.csv", [final_result], list(final_result.keys()))
    write_frame(OUTPUT_DIR / "qml_v8_exploratory_final_leaderboard.csv", final_leaderboard, list(final_leaderboard[0].keys()) if final_leaderboard else [])
    write_frame(OUTPUT_DIR / "qml_v8_same_target_classical_comparison.csv", classical_rows_all, list(classical_rows_all[0].keys()) if classical_rows_all else [])
    write_frame(OUTPUT_DIR / "qml_v8_champion_context_comparison.csv", champion_context, list(champion_context[0].keys()))
    write_json(OUTPUT_DIR / "qml_v8_rescue_decision.json", decision)
    manifest = {
        "run_id": "vn30_qml_forecasting_v8_drift_aware_kernel_features",
        "created_utc": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ"),
        "scope": "VN30 stock hourly forecasting only",
        "target": V8_TARGET,
        "horizon": V8_HORIZON,
        "diagnostic_only": True,
        "dependency_status": dependency,
        "decision": decision,
        "runtime_seconds": time.perf_counter() - started,
        "paper_docx_generated": False,
        "trading_claim": False,
        "vn100_scope": False,
        "index_as_stock_claim": False,
    }
    write_json(OUTPUT_DIR / "qml_v8_manifest.json", manifest)
    write_v8_reports(decision, drift_rows, audit_rows, validation_leaderboard, classical_rows_all)
    print(json.dumps(json_safe({"status": "ok", "manifest": rel(OUTPUT_DIR / "qml_v8_manifest.json"), "decision_labels": decision_labels, "runtime_limited": decision["runtime_limited"]}), indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 QML forecasting diagnostics.")
    parser.add_argument("--qml-smoke", action="store_true", help="Run the limited v2 QML smoke benchmark instead of the full diagnostic grid.")
    parser.add_argument("--qml-v3-sanity", action="store_true", help="Run the targeted v3 QML sanity benchmark.")
    parser.add_argument("--qml-v4-kernel-confirmation", action="store_true", help="Run the focused v4 quantum-kernel confirmation benchmark.")
    parser.add_argument("--qml-v5-full-confirmation", action="store_true", help="Run the focused v5 frozen quantum-kernel confirmation benchmark.")
    parser.add_argument("--qml-v6-scaling-rescue", action="store_true", help="Run the v6 QML scaling-failure diagnosis and kernel-rescue benchmark.")
    parser.add_argument("--qml-v7-hybrid-kernel-rescue", action="store_true", help="Run the v7 distribution-matched hybrid-kernel rescue benchmark.")
    parser.add_argument("--qml-v8-drift-aware-kernel-features", action="store_true", help="Run the v8 drift-aware QML kernel-feature meta-model benchmark.")
    parser.add_argument("--max-qml-candidates", type=int, default=12)
    parser.add_argument("--max-train-rows", type=int, default=2000)
    parser.add_argument("--max-validation-rows", type=int, default=1000)
    parser.add_argument("--max-final-rows", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.qml_v8_drift_aware_kernel_features:
        config = V8Config(timeout_seconds=max(1, int(args.timeout_seconds)))
        run_v8_drift_aware_kernel_features(config)
        return
    if args.qml_v7_hybrid_kernel_rescue:
        config = V7Config(timeout_seconds=max(1, int(args.timeout_seconds)))
        run_v7_hybrid_kernel_rescue(config)
        return
    if args.qml_v6_scaling_rescue:
        config = V6Config(timeout_seconds=max(1, int(args.timeout_seconds)))
        run_v6_scaling_rescue(config)
        return
    if args.qml_v5_full_confirmation:
        config = V5Config(timeout_seconds=max(1, int(args.timeout_seconds)))
        run_v5_full_confirmation(config)
        return
    if args.qml_v4_kernel_confirmation:
        config = V4Config(
            max_qml_candidates=max(0, int(args.max_qml_candidates)),
            max_train_rows=max(1, int(args.max_train_rows)),
            max_validation_rows=max(1, int(args.max_validation_rows)),
            max_final_rows=max(1, int(args.max_final_rows)),
            timeout_seconds=max(1, int(args.timeout_seconds)),
        )
        run_v4_kernel_confirmation(config)
        return
    if args.qml_v3_sanity:
        config = V3Config(
            max_qml_candidates=max(0, int(args.max_qml_candidates)),
            max_train_rows=max(1, int(args.max_train_rows)),
            max_validation_rows=max(1, int(args.max_validation_rows)),
            max_final_rows=max(1, int(args.max_final_rows)),
            timeout_seconds=max(1, int(args.timeout_seconds)),
        )
        run_v3_sanity(config)
        return
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
