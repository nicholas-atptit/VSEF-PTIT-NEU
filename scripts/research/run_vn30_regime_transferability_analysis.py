"""VN30 latent-regime transferability analysis.

This script uses existing local VN30 artifacts only. It reconstructs the h40
feature/label matrix from the repository's local hourly feature builders, fits
latent regimes on train-window ex-ante market state vectors, and computes
validation-governed transferability diagnostics. Final-window outputs are
descriptive scoring-only.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from sklearn.mixture import GaussianMixture
except Exception:  # pragma: no cover - optional dependency
    GaussianMixture = None

try:
    from scipy.stats import chi2_contingency, pearsonr, spearmanr
except Exception:  # pragma: no cover - optional dependency
    chi2_contingency = None
    pearsonr = None
    spearmanr = None

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_hourly_validation_safe_improvement_tracks import (  # noqa: E402
    BASE_FEATURE_FAMILY,
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    build_feature_families,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    add_absolute_labels,
    rel,
    strict_target_split_indices,
    target_timestamp_from_labels,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_transferability"
SOURCE_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_model_universe_benchmark"
ROW_PREDICTIONS_PATH = SOURCE_OUTPUT_DIR / "row_predictions.csv"

HORIZON = 40
RANDOM_STATE = 42
N_REGIMES = 3
MIN_TRAIN_ROWS_PER_REGIME = 80
EPS = 1e-6


@dataclass(frozen=True)
class FittedRegimeModel:
    regime: str
    model: Pipeline
    train_rows: int
    positive_ratio: float


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
        return value.isoformat()
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    shown = frame.head(max_rows).copy()
    headers = [str(col) for col in shown.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for _, row in shown.iterrows():
        values = []
        for col in shown.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def pct(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(number):
        return ""
    return f"{number * 100.0:.2f}%"


def logistic_l2_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            (
                "model",
                LogisticRegression(
                    C=0.3,
                    class_weight="balanced",
                    max_iter=1000,
                    penalty="l2",
                    random_state=RANDOM_STATE,
                    solver="liblinear",
                ),
            ),
        ]
    )


def predict_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(x)[:, 1]
    return np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)


def classification_metrics(y_true: pd.Series, probability: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y = y_true.astype(int).to_numpy()
    prob = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    pred = (prob >= threshold).astype(int)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)) if len(y) else math.nan,
        "log_loss": float(log_loss(y, prob, labels=[0, 1])) if len(y) else math.nan,
        "positive_ratio": float(np.mean(y)) if len(y) else math.nan,
        "predicted_positive_ratio": float(np.mean(pred)) if len(y) else math.nan,
        "mean_probability": float(np.mean(prob)) if len(y) else math.nan,
    }


def choose_existing_reference_predictions() -> pd.DataFrame:
    usecols = [
        "datetime",
        "ticker",
        "market_direction_regime",
        "volatility_regime",
        "regime_router_key",
        "model_group",
        "model_id",
        "feature_family",
        "horizon",
        "threshold_policy",
        "threshold",
        "candidate_id",
        "split",
        "y_true",
        "y_score_or_probability",
        "y_pred",
        "correct",
    ]
    if not ROW_PREDICTIONS_PATH.exists():
        return pd.DataFrame(columns=usecols)

    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(ROW_PREDICTIONS_PATH, usecols=usecols, chunksize=150_000, low_memory=False):
        mask = (
            chunk["model_id"].astype(str).eq("logistic_l2")
            & chunk["feature_family"].astype(str).eq(BASE_FEATURE_FAMILY)
            & pd.to_numeric(chunk["horizon"], errors="coerce").eq(HORIZON)
        )
        if mask.any():
            pieces.append(chunk.loc[mask].copy())
    if not pieces:
        return pd.DataFrame(columns=usecols)

    frame = pd.concat(pieces, ignore_index=True)
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["threshold"] = pd.to_numeric(frame["threshold"], errors="coerce")
    preferred = frame[frame["threshold_policy"].astype(str).eq("validation_selected_threshold")]
    if preferred.empty:
        preferred = frame[frame["threshold_policy"].astype(str).eq("fixed_0.50")]
    if preferred.empty:
        preferred = frame
    candidate_counts = preferred.groupby("candidate_id", dropna=False).size().sort_values(ascending=False)
    selected_candidate = str(candidate_counts.index[0])
    return preferred[preferred["candidate_id"].astype(str).eq(selected_candidate)].copy()


def write_skipped_report(reason: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped = pd.DataFrame(
        [
            {
                "status": "skipped_with_reason",
                "reason": reason,
                "data_fetch": False,
                "provider_behavior_changed": False,
                "horizon": HORIZON,
            }
        ]
    )
    write_frame(OUTPUT_DIR / "skipped_with_reason.csv", skipped)
    write_markdown(
        OUTPUT_DIR / "regime_transferability_summary.md",
        "\n".join(
            [
                "# VN30 Regime Transferability Summary",
                "",
                f"Status: skipped_with_reason.",
                "",
                f"Reason: {reason}",
                "",
                "No data fetch, provider change, full benchmark rerun, DOCX/PDF edit, or trading claim was made.",
            ]
        ),
    )


def add_universe_market_state_proxies(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    work = out[["datetime", "ticker", "return_1", "volume"]].copy()
    work["datetime"] = pd.to_datetime(work["datetime"], errors="coerce")
    work["return_1"] = pd.to_numeric(work["return_1"], errors="coerce")
    work["volume"] = pd.to_numeric(work["volume"], errors="coerce")
    state = (
        work.groupby("datetime")
        .agg(
            universe_avg_return_raw=("return_1", "mean"),
            universe_dispersion_raw=("return_1", "std"),
            universe_positive_share_raw=("return_1", lambda values: float((values > 0.0).mean())),
            universe_total_volume_raw=("volume", "sum"),
        )
        .sort_index()
        .reset_index()
    )
    avg_return = pd.to_numeric(state["universe_avg_return_raw"], errors="coerce")
    total_volume = pd.to_numeric(state["universe_total_volume_raw"], errors="coerce")
    state["universe_avg_return_lag_1_state"] = avg_return.shift(1)
    state["universe_avg_return_lag_5_state"] = avg_return.shift(5)
    state["universe_avg_return_lag_20_state"] = avg_return.shift(20)
    state["universe_mean_return_20_lag_state"] = avg_return.rolling(20, min_periods=5).mean().shift(1)
    state["universe_mean_return_60_lag_state"] = avg_return.rolling(60, min_periods=15).mean().shift(1)
    state["universe_vol_20_lag_state"] = avg_return.rolling(20, min_periods=5).std().shift(1)
    state["universe_vol_60_lag_state"] = avg_return.rolling(60, min_periods=15).std().shift(1)
    state["universe_dispersion_lag_1_state"] = pd.to_numeric(state["universe_dispersion_raw"], errors="coerce").shift(1)
    state["universe_positive_share_lag_1_state"] = pd.to_numeric(state["universe_positive_share_raw"], errors="coerce").shift(1)
    state["universe_volume_shock_20_lag_state"] = (total_volume / total_volume.rolling(20, min_periods=5).mean() - 1.0).shift(1)

    add_cols = [
        "universe_avg_return_lag_1_state",
        "universe_avg_return_lag_5_state",
        "universe_avg_return_lag_20_state",
        "universe_mean_return_20_lag_state",
        "universe_mean_return_60_lag_state",
        "universe_vol_20_lag_state",
        "universe_vol_60_lag_state",
        "universe_dispersion_lag_1_state",
        "universe_positive_share_lag_1_state",
        "universe_volume_shock_20_lag_state",
    ]
    out = out.merge(state[["datetime", *add_cols]].drop_duplicates("datetime", keep="last"), on="datetime", how="left")
    return out


def train_available_columns(features: pd.DataFrame, columns: list[str], train_idx: pd.Index, min_nonnull: int = 30) -> list[str]:
    available: list[str] = []
    for col in columns:
        if col not in features.columns:
            continue
        values = pd.to_numeric(features.loc[train_idx, col], errors="coerce")
        if int(values.notna().sum()) >= min_nonnull:
            available.append(col)
    return available


def select_state_columns(features: pd.DataFrame, train_idx: pd.Index) -> tuple[list[str], list[str], list[str], list[str]]:
    core_return = [
        "vn30_ret_lag_1_ctx",
        "vn30_ret_lag_5_ctx",
        "vn30_ret_lag_20_ctx",
        "vn30_mean_20_lag_ctx",
        "vn30_trend_60_lag_ctx",
    ]
    core_volatility = [
        "vn30_vol_20_lag_ctx",
        "vn30_vol_60_lag_ctx",
    ]
    fallback_return = [
        "vnindex_ret_lag_1_ctx",
        "vnindex_ret_lag_5_ctx",
        "vnindex_ret_lag_20_ctx",
        "vnindex_mean_20_lag_ctx",
        "vnindex_trend_60_lag_ctx",
    ]
    fallback_volatility = [
        "vnindex_vol_20_lag_ctx",
        "vnindex_vol_60_lag_ctx",
    ]
    universe_return = [
        "universe_avg_return_lag_1_state",
        "universe_avg_return_lag_5_state",
        "universe_avg_return_lag_20_state",
        "universe_mean_return_20_lag_state",
        "universe_mean_return_60_lag_state",
    ]
    universe_volatility = [
        "universe_vol_20_lag_state",
        "universe_vol_60_lag_state",
    ]
    breadth = [
        "breadth_positive_lag_1",
        "breadth_avg_return_lag_1",
        "breadth_dispersion_lag_1",
        "breadth_trend_lag",
        "universe_positive_share_lag_1_state",
        "universe_dispersion_lag_1_state",
        "universe_volume_shock_20_lag_state",
    ]

    return_cols = train_available_columns(features, core_return, train_idx)
    vol_cols = train_available_columns(features, core_volatility, train_idx)
    if len(return_cols) < 2:
        return_cols = train_available_columns(features, fallback_return, train_idx)
    if len(vol_cols) < 1:
        vol_cols = train_available_columns(features, fallback_volatility, train_idx)
    if len(return_cols) < 2:
        return_cols = train_available_columns(features, universe_return, train_idx)
    if len(vol_cols) < 1:
        vol_cols = train_available_columns(features, universe_volatility, train_idx)
    breadth_cols = train_available_columns(features, breadth, train_idx)
    state_cols = [*return_cols, *vol_cols, *breadth_cols]
    return state_cols, return_cols, vol_cols, breadth_cols


def build_market_state(features: pd.DataFrame, state_cols: list[str]) -> pd.DataFrame:
    keep = ["datetime", *state_cols]
    state = features[keep].drop_duplicates("datetime", keep="last").sort_values("datetime").reset_index(drop=True)
    state["datetime"] = pd.to_datetime(state["datetime"], errors="coerce")
    state = state.dropna(subset=["datetime"])
    for col in state_cols:
        state[col] = pd.to_numeric(state[col], errors="coerce")
    state[state_cols] = state[state_cols].replace([np.inf, -np.inf], np.nan)
    return state


def fit_latent_regimes(
    market_state: pd.DataFrame,
    state_cols: list[str],
    distance_cols: list[str],
    train_datetimes: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if GaussianMixture is None:
        raise RuntimeError("sklearn.mixture.GaussianMixture is unavailable")
    train_set = set(pd.to_datetime(train_datetimes, errors="coerce").dropna().unique())
    fit_mask = market_state["datetime"].isin(train_set)
    train_state = market_state.loc[fit_mask, ["datetime", *state_cols]].dropna(subset=state_cols).copy()
    if len(train_state) < max(30, N_REGIMES * 10):
        raise RuntimeError(f"insufficient train state rows for {N_REGIMES}-regime GMM: {len(train_state)}")

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_state[state_cols])
    gmm = GaussianMixture(n_components=N_REGIMES, covariance_type="full", random_state=RANDOM_STATE, n_init=10)
    gmm.fit(train_scaled)

    assignable = market_state[["datetime", *state_cols]].dropna(subset=state_cols).copy()
    scaled_all = scaler.transform(assignable[state_cols])
    raw_labels = gmm.predict(scaled_all)
    assignable["_raw_regime"] = raw_labels

    train_assignable = assignable[assignable["datetime"].isin(train_set)].copy()
    ordering = (
        train_assignable.groupby("_raw_regime")[state_cols]
        .mean()
        .assign(_sort_return=lambda x: x[[col for col in state_cols if "ret" in col or "mean" in col or "trend" in col]].mean(axis=1))
        .assign(_sort_volatility=lambda x: x[[col for col in state_cols if "vol" in col or "dispersion" in col]].mean(axis=1))
        .sort_values(["_sort_return", "_sort_volatility"], ascending=[True, True])
    )
    raw_to_name = {int(raw): f"latent_regime_{rank}" for rank, raw in enumerate(ordering.index)}
    assignable["latent_regime"] = assignable["_raw_regime"].map(raw_to_name)

    assignment = market_state[["datetime"]].merge(assignable[["datetime", "latent_regime"]], on="datetime", how="left")
    assignment["latent_regime"] = assignment["latent_regime"].fillna("unassigned")

    scaled_frame = pd.DataFrame(scaled_all, columns=state_cols)
    scaled_frame["datetime"] = assignable["datetime"].to_numpy()
    scaled_frame["latent_regime"] = assignable["latent_regime"].to_numpy()
    centroids = scaled_frame.groupby("latent_regime", sort=True)[state_cols].mean().reset_index()

    distance_rows: list[dict[str, Any]] = []
    centroid_index = centroids.set_index("latent_regime")
    rd_cols = [col for col in distance_cols if col in centroid_index.columns]
    if not rd_cols:
        rd_cols = state_cols
    for regime_i in centroid_index.index:
        vi = centroid_index.loc[regime_i, rd_cols].to_numpy(dtype=float)
        for regime_j in centroid_index.index:
            vj = centroid_index.loc[regime_j, rd_cols].to_numpy(dtype=float)
            distance_rows.append(
                {
                    "train_regime": regime_i,
                    "test_regime": regime_j,
                    "regime_distance": float(np.linalg.norm(vi - vj)),
                    "rd_columns": ",".join(rd_cols),
                }
            )
    distances = pd.DataFrame(distance_rows)

    fit_report = {
        "status": "ok",
        "method": "GaussianMixture",
        "n_regimes": N_REGIMES,
        "fit_rows_unique_datetimes": int(len(train_state)),
        "state_columns": state_cols,
        "rd_columns": rd_cols,
        "train_only_fit": True,
        "random_state": RANDOM_STATE,
        "bic_train": float(gmm.bic(train_scaled)),
        "aic_train": float(gmm.aic(train_scaled)),
        "raw_to_ordered_regime": raw_to_name,
    }
    return assignment, centroids, distances, fit_report


def attach_regimes(
    features: pd.DataFrame,
    labels: pd.Series,
    target_timestamp: pd.Series,
    splits: dict[str, pd.Index],
    regime_assignment: pd.DataFrame,
    state_cols: list[str],
) -> pd.DataFrame:
    frame = features[["datetime", "ticker", *state_cols]].copy()
    frame["target_timestamp"] = target_timestamp.reindex(features.index)
    frame["y_true"] = labels.reindex(features.index)
    split_name = pd.Series("", index=features.index, dtype=object)
    for name, idx in splits.items():
        split_name.loc[idx] = name
    frame["split"] = split_name
    frame = frame.merge(regime_assignment, on="datetime", how="left")
    frame["latent_regime"] = frame["latent_regime"].fillna("unassigned")
    return frame


def regime_summary(analysis_frame: pd.DataFrame, return_cols: list[str], vol_cols: list[str]) -> pd.DataFrame:
    return_col = return_cols[0] if return_cols else ""
    vol_col = vol_cols[0] if vol_cols else ""
    rows: list[dict[str, Any]] = []
    for (split, regime), group in analysis_frame[analysis_frame["split"].ne("")].groupby(["split", "latent_regime"], sort=True):
        valid_labels = pd.to_numeric(group["y_true"], errors="coerce").dropna()
        rows.append(
            {
                "split": split,
                "latent_regime": regime,
                "rows": int(len(group)),
                "unique_datetimes": int(group["datetime"].nunique()),
                "mean_return": float(pd.to_numeric(group[return_col], errors="coerce").mean()) if return_col else math.nan,
                "volatility": float(pd.to_numeric(group[vol_col], errors="coerce").mean()) if vol_col else math.nan,
                "label_ratio": float(valid_labels.mean()) if len(valid_labels) else math.nan,
                "return_column": return_col,
                "volatility_column": vol_col,
            }
        )
    return pd.DataFrame(rows)


def fit_global_and_regime_models(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    analysis_frame: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Pipeline, dict[str, FittedRegimeModel], pd.DataFrame]:
    train_idx = splits["train"]
    y_train = labels.loc[train_idx].astype(int)
    if y_train.nunique() < 2:
        raise RuntimeError("global train split has fewer than two classes")
    global_model = logistic_l2_pipeline()
    global_model.fit(features.loc[train_idx, feature_cols], y_train)

    model_rows: list[dict[str, Any]] = [
        {
            "model_scope": "global",
            "latent_regime": "all",
            "status": "ok",
            "train_rows": int(len(train_idx)),
            "positive_ratio": float(y_train.mean()),
            "reason": "",
        }
    ]
    regime_models: dict[str, FittedRegimeModel] = {}
    train_regimes = analysis_frame.loc[train_idx, "latent_regime"].astype(str)
    for regime in sorted(reg for reg in train_regimes.unique() if reg != "unassigned"):
        regime_idx = train_regimes[train_regimes.eq(regime)].index
        regime_y = labels.loc[regime_idx].astype(int)
        if len(regime_idx) < MIN_TRAIN_ROWS_PER_REGIME:
            reason = f"train rows below minimum {MIN_TRAIN_ROWS_PER_REGIME}"
            model_rows.append({"model_scope": "regime_specific", "latent_regime": regime, "status": "skipped_with_reason", "train_rows": int(len(regime_idx)), "positive_ratio": float(regime_y.mean()) if len(regime_y) else math.nan, "reason": reason})
            continue
        if regime_y.nunique() < 2:
            reason = "train regime has fewer than two target classes"
            model_rows.append({"model_scope": "regime_specific", "latent_regime": regime, "status": "skipped_with_reason", "train_rows": int(len(regime_idx)), "positive_ratio": float(regime_y.mean()) if len(regime_y) else math.nan, "reason": reason})
            continue
        model = logistic_l2_pipeline()
        model.fit(features.loc[regime_idx, feature_cols], regime_y)
        regime_models[regime] = FittedRegimeModel(regime=regime, model=model, train_rows=int(len(regime_idx)), positive_ratio=float(regime_y.mean()))
        model_rows.append({"model_scope": "regime_specific", "latent_regime": regime, "status": "ok", "train_rows": int(len(regime_idx)), "positive_ratio": float(regime_y.mean()), "reason": ""})
    return global_model, regime_models, pd.DataFrame(model_rows)


def score_global_and_regime(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    analysis_frame: pd.DataFrame,
    feature_cols: list[str],
    global_model: Pipeline,
    regime_models: dict[str, FittedRegimeModel],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    row_pieces: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    for split_name in ["validation", "final"]:
        idx = splits[split_name]
        if len(idx) == 0:
            continue
        y_true = labels.loc[idx].astype(int)
        global_prob = predict_probability(global_model, features.loc[idx, feature_cols])
        global_metrics = classification_metrics(y_true, global_prob)
        metric_rows.append({"split": split_name, "model_scope": "global", "latent_regime": "all", **global_metrics})

        work = analysis_frame.loc[idx, ["datetime", "ticker", "latent_regime"]].copy()
        work["split"] = split_name
        work["y_true"] = y_true.to_numpy(dtype=int)
        work["global_probability"] = global_prob
        work["global_prediction"] = (global_prob >= 0.5).astype(int)
        work["same_regime_probability"] = np.nan
        work["same_regime_prediction"] = np.nan

        for regime, group in work.groupby("latent_regime", sort=True):
            group_idx = group.index
            group_y = labels.loc[group_idx].astype(int)
            group_prob = work.loc[group_idx, "global_probability"].to_numpy(dtype=float)
            conditional_rows.append({"split": split_name, "model_scope": "global", "latent_regime": regime, **classification_metrics(group_y, group_prob)})
            if regime in regime_models:
                same_prob = predict_probability(regime_models[regime].model, features.loc[group_idx, feature_cols])
                work.loc[group_idx, "same_regime_probability"] = same_prob
                work.loc[group_idx, "same_regime_prediction"] = (same_prob >= 0.5).astype(int)
                conditional_rows.append({"split": split_name, "model_scope": "same_regime_model", "latent_regime": regime, **classification_metrics(group_y, same_prob)})

        available = work["same_regime_probability"].notna()
        if available.any():
            regime_metrics = classification_metrics(
                work.loc[available, "y_true"].astype(int),
                work.loc[available, "same_regime_probability"].to_numpy(dtype=float),
            )
            metric_rows.append({"split": split_name, "model_scope": "same_regime_model", "latent_regime": "all_available", **regime_metrics})
            rig = global_metrics["log_loss"] - regime_metrics["log_loss"]
            metric_rows.append(
                {
                    "split": split_name,
                    "model_scope": "regime_information_gain",
                    "latent_regime": "global_minus_same_regime",
                    "rows": int(available.sum()),
                    "accuracy": math.nan,
                    "log_loss": float(rig),
                    "positive_ratio": math.nan,
                    "predicted_positive_ratio": math.nan,
                    "mean_probability": math.nan,
                }
            )
        row_pieces.append(work.reset_index(drop=True))
    predictions = pd.concat(row_pieces, ignore_index=True) if row_pieces else pd.DataFrame()
    return predictions, pd.DataFrame(metric_rows), pd.DataFrame(conditional_rows)


def transfer_matrices(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    analysis_frame: pd.DataFrame,
    feature_cols: list[str],
    regime_models: dict[str, FittedRegimeModel],
    distances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matrix_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for split_name in ["validation", "final"]:
        test_idx = splits[split_name]
        test_regimes = sorted(reg for reg in analysis_frame.loc[test_idx, "latent_regime"].astype(str).unique() if reg != "unassigned")
        for train_regime, fitted in sorted(regime_models.items()):
            for test_regime in test_regimes:
                cell_idx = analysis_frame.loc[test_idx].index[analysis_frame.loc[test_idx, "latent_regime"].astype(str).eq(test_regime)]
                if len(cell_idx) == 0:
                    continue
                y_true = labels.loc[cell_idx].astype(int)
                probability = predict_probability(fitted.model, features.loc[cell_idx, feature_cols])
                metrics = classification_metrics(y_true, probability)
                matrix_rows.append(
                    {
                        "split": split_name,
                        "train_regime": train_regime,
                        "test_regime": test_regime,
                        **metrics,
                    }
                )
                for row_idx, prob in zip(cell_idx, probability):
                    prediction_rows.append(
                        {
                            "split": split_name,
                            "datetime": analysis_frame.loc[row_idx, "datetime"],
                            "ticker": analysis_frame.loc[row_idx, "ticker"],
                            "train_regime": train_regime,
                            "test_regime": test_regime,
                            "y_true": int(labels.loc[row_idx]),
                            "probability": float(prob),
                            "prediction": int(prob >= 0.5),
                        }
                    )
    matrix = pd.DataFrame(matrix_rows)
    if matrix.empty:
        return matrix, pd.DataFrame(), pd.DataFrame(prediction_rows)

    diag = (
        matrix[matrix["train_regime"].eq(matrix["test_regime"])]
        .set_index(["split", "train_regime"])["accuracy"]
        .to_dict()
    )
    matrix["same_regime_accuracy"] = [
        diag.get((row["split"], row["train_regime"]), math.nan) for _, row in matrix.iterrows()
    ]
    matrix["transfer_retention_ratio"] = matrix["accuracy"] / matrix["same_regime_accuracy"].replace(0.0, np.nan)
    matrix["transfer_gap"] = matrix["same_regime_accuracy"] - matrix["accuracy"]
    matrix = matrix.merge(distances, on=["train_regime", "test_regime"], how="left")
    matrix["cell_role"] = np.where(matrix["train_regime"].eq(matrix["test_regime"]), "same_regime", "cross_regime")

    long_for_matrix = matrix.copy()
    pivot_rows: list[dict[str, Any]] = []
    for split_name, split_frame in long_for_matrix.groupby("split", sort=True):
        for value_col in ["accuracy", "log_loss", "transfer_retention_ratio", "transfer_gap", "regime_distance"]:
            pivot = split_frame.pivot(index="train_regime", columns="test_regime", values=value_col).reset_index()
            pivot.insert(0, "metric", value_col)
            pivot.insert(0, "split", split_name)
            pivot_rows.extend(pivot.to_dict("records"))
    matrix_wide = pd.DataFrame(pivot_rows)
    return matrix, matrix_wide, pd.DataFrame(prediction_rows)


def regime_conditional_test(scored_predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if scored_predictions.empty:
        return pd.DataFrame(rows)
    for split_name, split_frame in scored_predictions.groupby("split", sort=True):
        work = split_frame.dropna(subset=["global_prediction"]).copy()
        work["correct"] = (work["y_true"].astype(int) == work["global_prediction"].astype(int)).astype(int)
        grouped = work.groupby("latent_regime")["correct"].agg(["count", "mean"]).reset_index()
        accuracy_gap = float(grouped["mean"].max() - grouped["mean"].min()) if not grouped.empty else math.nan
        p_value = math.nan
        chi2_stat = math.nan
        cramers_v = math.nan
        if chi2_contingency is not None and work["latent_regime"].nunique() > 1:
            table = pd.crosstab(work["latent_regime"], work["correct"])
            if table.shape[0] > 1 and table.shape[1] > 1:
                chi2_stat, p_value, _dof, _expected = chi2_contingency(table)
                n = float(table.to_numpy().sum())
                min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
                cramers_v = math.sqrt(float(chi2_stat) / (n * min_dim)) if n > 0 and min_dim > 0 else math.nan
        rows.append(
            {
                "split": split_name,
                "model_scope": "global_logistic_l2",
                "test": "regime_conditional_accuracy_difference",
                "rows": int(len(work)),
                "regime_count": int(work["latent_regime"].nunique()),
                "accuracy_gap_max_minus_min": accuracy_gap,
                "chi2_stat": float(chi2_stat) if math.isfinite(float(chi2_stat)) else math.nan,
                "p_value": float(p_value) if math.isfinite(float(p_value)) else math.nan,
                "cramers_v": float(cramers_v) if math.isfinite(float(cramers_v)) else math.nan,
            }
        )
    return pd.DataFrame(rows)


def ols_summary(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2:
        return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan}
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))
    denom = float(np.sum((x - x_mean) ** 2))
    if denom <= 0:
        return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan}
    slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
    intercept = y_mean - slope * x_mean
    pred = intercept + slope * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else math.nan
    return {"ols_intercept": intercept, "ols_slope": slope, "ols_r_squared": r2}


def transfer_distance_tests(matrix: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if matrix.empty:
        return pd.DataFrame(rows)
    cross = matrix[matrix["cell_role"].eq("cross_regime")].copy()
    for split_name, split_frame in cross.groupby("split", sort=True):
        for metric in ["transfer_retention_ratio", "transfer_gap"]:
            work = split_frame[["regime_distance", metric]].replace([np.inf, -np.inf], np.nan).dropna()
            x = work["regime_distance"].to_numpy(dtype=float)
            y = work[metric].to_numpy(dtype=float)
            pearson_corr = math.nan
            pearson_p = math.nan
            spearman_corr = math.nan
            spearman_p = math.nan
            if len(work) >= 2:
                if pearsonr is not None:
                    pearson_corr, pearson_p = pearsonr(x, y)
                else:
                    pearson_corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else math.nan
                if spearmanr is not None:
                    spearman_corr, spearman_p = spearmanr(x, y)
                else:
                    spearman_corr = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
            expected_direction = "negative" if metric == "transfer_retention_ratio" else "positive"
            rows.append(
                {
                    "split": split_name,
                    "metric": metric,
                    "n_cross_cells": int(len(work)),
                    "expected_slope_direction": expected_direction,
                    "pearson_corr": float(pearson_corr) if math.isfinite(float(pearson_corr)) else math.nan,
                    "pearson_p_value": float(pearson_p) if math.isfinite(float(pearson_p)) else math.nan,
                    "spearman_corr": float(spearman_corr) if math.isfinite(float(spearman_corr)) else math.nan,
                    "spearman_p_value": float(spearman_p) if math.isfinite(float(spearman_p)) else math.nan,
                    **ols_summary(x, y),
                }
            )
    return pd.DataFrame(rows)


def existing_prediction_regime_metrics(regime_assignment: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    existing = choose_existing_reference_predictions()
    if existing.empty:
        return existing, pd.DataFrame(
            [
                {
                    "status": "skipped_with_reason",
                    "reason": "no h40 Logistic L2 baseline_C_closest row predictions found",
                }
            ]
        )
    existing = existing.merge(regime_assignment, on="datetime", how="left")
    existing["latent_regime"] = existing["latent_regime"].fillna("unassigned")
    for col in ["y_true", "y_score_or_probability", "y_pred", "correct"]:
        existing[col] = pd.to_numeric(existing[col], errors="coerce")
    rows: list[dict[str, Any]] = []
    selected_candidate = str(existing["candidate_id"].dropna().iloc[0]) if existing["candidate_id"].notna().any() else ""
    for (split_name, regime), group in existing.groupby(["split", "latent_regime"], sort=True):
        clean = group.dropna(subset=["y_true", "y_score_or_probability", "y_pred"])
        if clean.empty:
            continue
        probability = np.clip(clean["y_score_or_probability"].to_numpy(dtype=float), EPS, 1.0 - EPS)
        metrics = classification_metrics(clean["y_true"].astype(int), probability, threshold=0.5)
        observed_accuracy = float(clean["correct"].mean()) if clean["correct"].notna().any() else float((clean["y_true"].astype(int) == clean["y_pred"].astype(int)).mean())
        metrics["stored_prediction_accuracy"] = observed_accuracy
        rows.append(
            {
                "status": "ok",
                "source": rel(ROW_PREDICTIONS_PATH),
                "candidate_id": selected_candidate,
                "split": split_name,
                "latent_regime": regime,
                **metrics,
                "scope_note": "Existing row predictions use their original benchmark split discipline; final rows are scoring-only.",
            }
        )
    return existing, pd.DataFrame(rows)


def artifact_inventory(
    features: pd.DataFrame,
    family_cols: dict[str, list[str]],
    state_cols: list[str],
    return_cols: list[str],
    vol_cols: list[str],
    breadth_cols: list[str],
    splits: dict[str, pd.Index],
    labels: pd.Series,
    target_timestamp: pd.Series,
) -> pd.DataFrame:
    rows = [
        {"artifact": "reconstructed_feature_matrix", "path": "local feature builders", "status": "available", "detail": f"rows={len(features)}, columns={len(features.columns)}"},
        {"artifact": "baseline_feature_family", "path": "build_feature_families()", "status": "available" if BASE_FEATURE_FAMILY in family_cols else "missing", "detail": f"features={len(family_cols.get(BASE_FEATURE_FAMILY, []))}"},
        {"artifact": "target_labels_h40", "path": "add_absolute_labels(features, 40)", "status": "available", "detail": f"valid_labels={int(labels.notna().sum())}"},
        {"artifact": "target_timestamp_h40", "path": "label attrs", "status": "available" if target_timestamp.notna().any() else "missing", "detail": f"min={target_timestamp.dropna().min() if target_timestamp.notna().any() else ''}; max={target_timestamp.dropna().max() if target_timestamp.notna().any() else ''}"},
        {"artifact": "strict_train_validation_final_splits", "path": "strict_target_split_indices()", "status": "available", "detail": "; ".join(f"{name}={len(idx)}" for name, idx in splits.items())},
        {"artifact": "row_level_predictions", "path": rel(ROW_PREDICTIONS_PATH), "status": "available" if ROW_PREDICTIONS_PATH.exists() else "missing", "detail": "h40 Logistic L2 filtered when present"},
        {"artifact": "market_index_return_state", "path": "lagged VN30/VNINDEX context features", "status": "available" if return_cols else "missing", "detail": ", ".join(return_cols)},
        {"artifact": "market_index_volatility_state", "path": "lagged VN30/VNINDEX context features", "status": "available" if vol_cols else "missing", "detail": ", ".join(vol_cols)},
        {"artifact": "breadth_proxy_state", "path": "lagged breadth features", "status": "available" if breadth_cols else "missing", "detail": ", ".join(breadth_cols)},
        {"artifact": "latent_regime_state_columns", "path": "train-only GaussianMixture input", "status": "available" if state_cols else "missing", "detail": ", ".join(state_cols)},
    ]
    return pd.DataFrame(rows)


def summary_markdown(
    fit_report: dict[str, Any],
    inventory: pd.DataFrame,
    regime_summary_frame: pd.DataFrame,
    global_regime_metrics: pd.DataFrame,
    transfer_tests: pd.DataFrame,
    conditional_tests: pd.DataFrame,
    model_audit: pd.DataFrame,
    existing_metrics: pd.DataFrame,
) -> str:
    validation_rig = global_regime_metrics[
        global_regime_metrics["split"].eq("validation")
        & global_regime_metrics["model_scope"].eq("regime_information_gain")
    ]
    rig_text = ""
    if not validation_rig.empty:
        rig_text = f"{float(validation_rig.iloc[0]['log_loss']):.6f}"

    validation_transfer = transfer_tests[transfer_tests["split"].eq("validation")] if not transfer_tests.empty else pd.DataFrame()
    lines = [
        "# VN30 Regime Transferability Summary",
        "",
        "## Feasibility Verdict",
        "",
        "Feasible with existing local VN30 artifacts and lightweight h40 Logistic L2 refitting. No market data fetch, provider behavior change, full benchmark rerun, DOCX/PDF edit, commit, push, or tag was performed.",
        "",
        "## Primary Scope",
        "",
        f"- Paper title: Regime-Dependent Transferability of Machine Learning Forecasts in an Emerging Equity Market: Evidence from VN30 Walk-Forward Benchmarks",
        f"- Horizon: h{HORIZON}.",
        "- Primary empirical window: validation. Final-window outputs are descriptive scoring-only.",
        "- Latent regimes: train-only GaussianMixture on lagged market return/volatility state vectors, with lagged breadth included when available.",
        "- Primary reference model: Logistic L2 on baseline_C_closest features with fixed 0.50 threshold for transfer-matrix comparability.",
        "",
        "## Artifact Inventory",
        "",
        markdown_table(inventory, max_rows=20),
        "",
        "## Latent Regime Fit",
        "",
        f"- Status: {fit_report.get('status', '')}.",
        f"- Method: {fit_report.get('method', '')}.",
        f"- Train-only fit rows: {fit_report.get('fit_rows_unique_datetimes', '')} unique timestamps.",
        f"- State columns: {', '.join(fit_report.get('state_columns', []))}.",
        f"- RD columns: {', '.join(fit_report.get('rd_columns', []))}.",
        "",
        "## Regime Summaries",
        "",
        markdown_table(regime_summary_frame[regime_summary_frame["split"].eq("validation")], max_rows=20),
        "",
        "## Metrics Computed",
        "",
        f"- Validation RIG, defined as global log-loss minus same-regime model log-loss: {rig_text}.",
        "- Regime-conditional accuracy/log-loss summaries were computed for global and same-regime Logistic L2 models.",
        "- Transfer accuracy, log-loss, TRR, TG, and RD matrices were computed for validation and descriptive final splits.",
        "- RD-vs-TRR and RD-vs-TG tests were computed with correlation and simple OLS diagnostics.",
        "",
        "## Transfer Distance Tests",
        "",
        markdown_table(validation_transfer, max_rows=20),
        "",
        "## Regime-Conditional Forecasting Test",
        "",
        markdown_table(conditional_tests[conditional_tests["split"].eq("validation")] if not conditional_tests.empty else conditional_tests, max_rows=20),
        "",
        "## Model Fit Audit",
        "",
        markdown_table(model_audit, max_rows=20),
        "",
        "## Existing Row Prediction Check",
        "",
        markdown_table(existing_metrics.head(12) if not existing_metrics.empty else existing_metrics, max_rows=12),
        "",
        "## Claim Boundary",
        "",
        "- These outputs support empirical diagnostics of regime-dependent transferability only.",
        "- Do not interpret final-window scores as selection evidence or claim promotion.",
        "- Do not promote any result to trading, profitability, investment, live-deployment, or out-of-sample market readiness claims.",
        "- Latent regimes are reconstructed from local lagged features; they are not external economic regime labels.",
    ]
    return "\n".join(lines)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if GaussianMixture is None:
        write_skipped_report("sklearn.mixture.GaussianMixture is unavailable")
        return 0

    features, family_cols, feature_manifest = build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    labels = add_absolute_labels(features, HORIZON)
    target_timestamp = target_timestamp_from_labels(labels).reindex(features.index)
    splits = strict_target_split_indices(features, labels, TRAIN_END, VAL_START, VAL_END, FINAL_START)

    features = add_universe_market_state_proxies(features)
    state_cols, return_cols, vol_cols, breadth_cols = select_state_columns(features, splits["train"])
    if not return_cols or not vol_cols:
        write_skipped_report("missing train-available return or volatility state columns for latent regime construction")
        return 0

    market_state = build_market_state(features, state_cols)
    train_datetimes = features.loc[splits["train"], "datetime"]
    try:
        regime_assignment, centroids, distances, fit_report = fit_latent_regimes(market_state, state_cols, [*return_cols, *vol_cols], train_datetimes)
    except RuntimeError as exc:
        write_skipped_report(str(exc))
        return 0

    analysis_frame = attach_regimes(features, labels, target_timestamp, splits, regime_assignment, state_cols)
    valid_splits = analysis_frame["split"].isin(["train", "validation", "final"])
    write_frame(OUTPUT_DIR / "row_regime_assignments_h40.csv", analysis_frame.loc[valid_splits].sort_values(["split", "datetime", "ticker"]))
    write_frame(OUTPUT_DIR / "timestamp_regime_assignments.csv", regime_assignment.sort_values("datetime"))
    write_frame(OUTPUT_DIR / "regime_state_centroids_standardized.csv", centroids)
    write_frame(OUTPUT_DIR / "regime_distance_matrix.csv", distances)
    write_json(OUTPUT_DIR / "latent_regime_fit.json", {**fit_report, "feature_manifest": feature_manifest})

    feature_cols = [
        col
        for col in family_cols.get(BASE_FEATURE_FAMILY, [])
        if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not feature_cols:
        write_skipped_report("missing numeric baseline_C_closest feature columns")
        return 0

    inventory = artifact_inventory(features, family_cols, state_cols, return_cols, vol_cols, breadth_cols, splits, labels, target_timestamp)
    write_frame(OUTPUT_DIR / "artifact_inventory.csv", inventory)

    summary = regime_summary(analysis_frame, return_cols, vol_cols)
    write_frame(OUTPUT_DIR / "regime_summary.csv", summary)

    global_model, regime_models, model_audit = fit_global_and_regime_models(features, labels, splits, analysis_frame, feature_cols)
    write_frame(OUTPUT_DIR / "model_fit_audit.csv", model_audit)

    scored_predictions, global_regime_metrics, conditional_metrics = score_global_and_regime(
        features,
        labels,
        splits,
        analysis_frame,
        feature_cols,
        global_model,
        regime_models,
    )
    write_frame(OUTPUT_DIR / "global_vs_regime_predictions.csv", scored_predictions)
    write_frame(OUTPUT_DIR / "global_vs_regime_metrics.csv", global_regime_metrics)
    write_frame(OUTPUT_DIR / "regime_conditional_metrics.csv", conditional_metrics)

    conditional_tests = regime_conditional_test(scored_predictions)
    write_frame(OUTPUT_DIR / "regime_conditional_tests.csv", conditional_tests)

    transfer_long, transfer_wide, transfer_predictions = transfer_matrices(
        features,
        labels,
        splits,
        analysis_frame,
        feature_cols,
        regime_models,
        distances,
    )
    write_frame(OUTPUT_DIR / "transfer_matrix_long.csv", transfer_long)
    write_frame(OUTPUT_DIR / "transfer_matrices_wide.csv", transfer_wide)
    write_frame(OUTPUT_DIR / "transfer_predictions.csv", transfer_predictions)

    transfer_tests = transfer_distance_tests(transfer_long)
    write_frame(OUTPUT_DIR / "transfer_distance_tests.csv", transfer_tests)

    existing_predictions, existing_metrics = existing_prediction_regime_metrics(regime_assignment)
    if not existing_predictions.empty:
        write_frame(OUTPUT_DIR / "existing_h40_logistic_row_predictions_with_latent_regime.csv", existing_predictions)
    write_frame(OUTPUT_DIR / "existing_h40_logistic_regime_metrics.csv", existing_metrics)

    write_markdown(
        OUTPUT_DIR / "regime_transferability_summary.md",
        summary_markdown(
            fit_report,
            inventory,
            summary,
            global_regime_metrics,
            transfer_tests,
            conditional_tests,
            model_audit,
            existing_metrics,
        ),
    )

    print(f"Wrote VN30 regime transferability outputs to {rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
