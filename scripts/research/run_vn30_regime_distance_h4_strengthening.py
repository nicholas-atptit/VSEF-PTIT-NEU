"""Additional VN30 H4 strengthening audit.

This runner tests whether the H4 distance-transfer mechanism receives stronger
legitimate support under additional pre-specified robustness designs. It uses
existing local artifacts and local feature builders only. It does not fetch
data, edit paper/DOCX/PDF files, touch QML files, commit, push, or tag.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.run_vn30_regime_distance_robustness import (  # noqa: E402
    EPS,
    RANDOM_STATE,
    balanced_accuracy_binary,
    finite_float,
    infer_state_groups,
    logistic_l2_pipeline,
    markdown_table,
    pairwise_cosine,
    pairwise_euclidean,
    safe_log_loss,
    write_frame,
    write_json,
    write_markdown,
)

TRANSFER_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_transferability"
ROBUSTNESS_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_distance_robustness"
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_distance_h4_strengthening"

HORIZONS = [20, 40, 60, 80]
PRIMARY_HORIZON = 40
REGIME_COUNTS = [2, 4]
MIN_TRAIN_ROWS_PER_REGIME = 80
MIN_COMPLETE_CELLS = 6

CURRENT_STATUS = "limited metric-specific weak support"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required local artifact is missing: {rel(path)}")


def import_transferability_module() -> Any:
    return importlib.import_module("scripts.research.run_vn30_regime_transferability_analysis")


def ece_binary(y_true: np.ndarray, probability: np.ndarray, n_bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    if len(y) == 0:
        return math.nan
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = float(len(y))
    error = 0.0
    for idx in range(n_bins):
        left = edges[idx]
        right = edges[idx + 1]
        if idx == n_bins - 1:
            mask = (prob >= left) & (prob <= right)
        else:
            mask = (prob >= left) & (prob < right)
        if not mask.any():
            continue
        bin_weight = float(mask.sum()) / total
        error += bin_weight * abs(float(prob[mask].mean()) - float(y[mask].mean()))
    return float(error)


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    prob = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    if len(y) == 0:
        return math.nan
    return float(np.mean((prob - y) ** 2))


def safe_ols_value(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan, "ols_p_value": math.nan}
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan, "ols_p_value": math.nan}
    if float(np.ptp(x)) <= 1e-12 or float(np.ptp(y)) <= 1e-12:
        return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan, "ols_p_value": math.nan}
    try:
        from scipy.stats import linregress

        result = linregress(x, y)
        return {
            "ols_intercept": float(result.intercept),
            "ols_slope": float(result.slope),
            "ols_r_squared": float(result.rvalue**2),
            "ols_p_value": float(result.pvalue),
        }
    except Exception:
        x_mean = float(np.mean(x))
        y_mean = float(np.mean(y))
        denom = float(np.sum((x - x_mean) ** 2))
        if denom <= 0.0:
            return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan, "ols_p_value": math.nan}
        slope = float(np.sum((x - x_mean) * (y - y_mean)) / denom)
        intercept = y_mean - slope * x_mean
        pred = intercept + slope * x
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y_mean) ** 2))
        return {
            "ols_intercept": intercept,
            "ols_slope": slope,
            "ols_r_squared": 1.0 - ss_res / ss_tot if ss_tot > 0.0 else math.nan,
            "ols_p_value": math.nan,
        }


def safe_correlation_value(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return math.nan, math.nan, math.nan, math.nan
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return math.nan, math.nan, math.nan, math.nan
    if float(np.ptp(x)) <= 1e-12 or float(np.ptp(y)) <= 1e-12:
        return math.nan, math.nan, math.nan, math.nan
    try:
        from scipy.stats import pearsonr, spearmanr

        pearson_corr, pearson_p = pearsonr(x, y)
        spearman_corr, spearman_p = spearmanr(x, y)
        return float(pearson_corr), float(pearson_p), float(spearman_corr), float(spearman_p)
    except Exception:
        pearson_corr = float(np.corrcoef(x, y)[0, 1])
        spearman_corr = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
        return pearson_corr, math.nan, spearman_corr, math.nan


def feature_bundle() -> tuple[Any, pd.DataFrame, dict[str, list[str]], list[str]]:
    source = import_transferability_module()
    features, family_cols, _manifest = source.build_feature_families()
    features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
    features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
    features = source.add_universe_market_state_proxies(features)
    feature_cols = [
        col
        for col in family_cols.get(source.BASE_FEATURE_FAMILY, [])
        if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
    ]
    if not feature_cols:
        raise RuntimeError("missing numeric baseline_C_closest feature columns")
    return source, features, family_cols, feature_cols


def strict_labels_and_splits(source: Any, features: pd.DataFrame, horizon: int) -> tuple[pd.Series, dict[str, pd.Index]]:
    labels = source.add_absolute_labels(features, horizon)
    splits = source.strict_target_split_indices(
        features,
        labels,
        source.TRAIN_END,
        source.VAL_START,
        source.VAL_END,
        source.FINAL_START,
    )
    return labels, splits


def saved_k3_assignment() -> pd.DataFrame:
    path = TRANSFER_DIR / "timestamp_regime_assignments.csv"
    require_file(path)
    assignment = pd.read_csv(path)
    assignment["datetime"] = pd.to_datetime(assignment["datetime"], errors="coerce")
    assignment["latent_regime"] = assignment["latent_regime"].astype(str)
    return assignment[["datetime", "latent_regime"]].dropna(subset=["datetime"])


def saved_k3_distances() -> pd.DataFrame:
    path = ROBUSTNESS_DIR / "h4_distance_definitions.csv"
    if not path.exists():
        return pd.DataFrame()
    distances_path = ROBUSTNESS_DIR / "h4_pair_level_transfer_cells.csv"
    if not distances_path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(distances_path, nrows=6)
    cols = [
        col
        for col in frame.columns
        if col.startswith("RD_") or col.startswith("FRD_")
    ]
    return frame[["train_regime", "test_regime", *cols]].drop_duplicates(["train_regime", "test_regime"])


def build_analysis_frame(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    analysis = features[["datetime", "ticker"]].copy()
    analysis["y_true"] = labels.reindex(features.index)
    split_name = pd.Series("", index=features.index, dtype=object)
    for name, idx in splits.items():
        split_name.loc[idx] = name
    analysis["split"] = split_name
    analysis = analysis.merge(assignment, on="datetime", how="left")
    analysis["latent_regime"] = analysis["latent_regime"].fillna("unassigned").astype(str)
    return analysis


def predict_probability(model: Any, x: pd.DataFrame) -> np.ndarray:
    probability = model.predict_proba(x)[:, 1]
    return np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)


def fit_regime_models(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    analysis: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    models: dict[str, Any] = {}
    train_idx = splits["train"]
    train_regimes = analysis.loc[train_idx, "latent_regime"].astype(str)
    for regime in sorted(reg for reg in train_regimes.unique() if reg != "unassigned"):
        regime_idx = train_regimes[train_regimes.eq(regime)].index
        y_train = labels.loc[regime_idx].astype(int)
        if len(regime_idx) < MIN_TRAIN_ROWS_PER_REGIME:
            rows.append(
                {
                    "latent_regime": regime,
                    "status": "skipped_with_reason",
                    "train_rows": int(len(regime_idx)),
                    "positive_ratio": float(y_train.mean()) if len(y_train) else math.nan,
                    "reason": f"train rows below minimum {MIN_TRAIN_ROWS_PER_REGIME}",
                }
            )
            continue
        if y_train.nunique() < 2:
            rows.append(
                {
                    "latent_regime": regime,
                    "status": "skipped_with_reason",
                    "train_rows": int(len(regime_idx)),
                    "positive_ratio": float(y_train.mean()) if len(y_train) else math.nan,
                    "reason": "train regime has fewer than two classes",
                }
            )
            continue
        model = logistic_l2_pipeline()
        model.fit(features.loc[regime_idx, feature_cols], y_train)
        models[regime] = model
        rows.append(
            {
                "latent_regime": regime,
                "status": "computed",
                "train_rows": int(len(regime_idx)),
                "positive_ratio": float(y_train.mean()),
                "reason": "",
            }
        )
    return models, pd.DataFrame(rows)


def transfer_predictions_for_models(
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    analysis: pd.DataFrame,
    feature_cols: list[str],
    models: dict[str, Any],
    horizon: int,
    regime_count: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    idx = splits["validation"]
    if len(idx) == 0:
        return pd.DataFrame()
    validation = analysis.loc[idx, ["datetime", "ticker", "latent_regime"]].copy()
    validation = validation[validation["latent_regime"].ne("unassigned")]
    test_regimes = sorted(validation["latent_regime"].unique())
    for train_regime, model in sorted(models.items()):
        for test_regime in test_regimes:
            cell_idx = validation.index[validation["latent_regime"].eq(test_regime)]
            if len(cell_idx) == 0:
                continue
            probability = predict_probability(model, features.loc[cell_idx, feature_cols])
            piece = analysis.loc[cell_idx, ["datetime", "ticker"]].copy()
            piece["split"] = "validation"
            piece["horizon"] = int(horizon)
            piece["regime_count"] = int(regime_count)
            piece["train_regime"] = train_regime
            piece["test_regime"] = test_regime
            piece["y_true"] = labels.loc[cell_idx].astype(int).to_numpy()
            piece["probability"] = probability
            piece["prediction"] = (probability >= 0.5).astype(int)
            rows.append(piece)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def coefficient_distances(models: dict[str, Any], prefix: str = "FRD") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    coefs = {regime: np.asarray(model.named_steps["model"].coef_[0], dtype=float) for regime, model in models.items()}
    for train_regime in sorted(coefs):
        vi = coefs[train_regime]
        ni = float(np.linalg.norm(vi))
        for test_regime in sorted(coefs):
            vj = coefs[test_regime]
            nj = float(np.linalg.norm(vj))
            cosine_distance = math.nan
            if ni > 0.0 and nj > 0.0:
                cosine_distance = float(1.0 - np.clip(float(np.dot(vi, vj) / (ni * nj)), -1.0, 1.0))
            rows.append(
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    f"{prefix}_coefficient_l2_distance": float(np.linalg.norm(vi - vj)),
                    f"{prefix}_coefficient_cosine_distance": cosine_distance,
                }
            )
    return pd.DataFrame(rows)


def probability_distribution_distances(predictions: pd.DataFrame, prefix: str = "FRD") -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    base_cols = ["horizon", "regime_count"] if {"horizon", "regime_count"}.issubset(predictions.columns) else []
    rows: list[dict[str, Any]] = []
    grouped = predictions.groupby(base_cols, sort=True) if base_cols else [((), predictions)]
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = {col: val for col, val in zip(base_cols, keys)}
        pivot = group.pivot_table(
            index=["datetime", "ticker"],
            columns="train_regime",
            values="probability",
            aggfunc="mean",
        )
        regimes = sorted(str(col) for col in pivot.columns)
        for train_regime in regimes:
            for test_regime in regimes:
                pair = pivot[[train_regime, test_regime]].dropna()
                if pair.empty:
                    distance = math.nan
                    common_rows = 0
                else:
                    diff = pair[train_regime].to_numpy(dtype=float) - pair[test_regime].to_numpy(dtype=float)
                    distance = float(math.sqrt(float(np.mean(diff**2))))
                    common_rows = int(len(pair))
                rows.append(
                    {
                        **key_payload,
                        "train_regime": train_regime,
                        "test_regime": test_regime,
                        f"{prefix}_probability_distribution_distance": distance,
                        f"{prefix}_probability_common_rows": common_rows,
                    }
                )
    return pd.DataFrame(rows)


def cell_metrics(predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        clean = group.dropna(subset=["y_true", "probability", "prediction"]).copy()
        row = {col: key for col, key in zip(group_cols, keys)}
        if clean.empty:
            row.update(
                {
                    "rows": 0,
                    "accuracy": math.nan,
                    "balanced_accuracy": math.nan,
                    "log_loss": math.nan,
                    "brier_score": math.nan,
                    "calibration_error": math.nan,
                }
            )
        else:
            y = clean["y_true"].astype(int).to_numpy()
            prob = clean["probability"].to_numpy(dtype=float)
            pred = clean["prediction"].astype(int).to_numpy()
            row.update(
                {
                    "rows": int(len(clean)),
                    "accuracy": float((y == pred).mean()),
                    "balanced_accuracy": balanced_accuracy_binary(y, pred),
                    "log_loss": safe_log_loss(y, prob),
                    "brier_score": brier_score(y, prob),
                    "calibration_error": ece_binary(y, prob),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def attach_transfer_loss(cells: pd.DataFrame, same_keys: list[str]) -> pd.DataFrame:
    if cells.empty:
        return cells
    cells = cells.copy()
    cells["cell_role"] = np.where(cells["train_regime"].astype(str).eq(cells["test_regime"].astype(str)), "same_regime", "cross_regime")
    same = cells[cells["cell_role"].eq("same_regime")].copy()
    same = same.rename(
        columns={
            "rows": "same_rows",
            "accuracy": "same_accuracy",
            "balanced_accuracy": "same_balanced_accuracy",
            "log_loss": "same_log_loss",
            "brier_score": "same_brier_score",
            "calibration_error": "same_calibration_error",
        }
    )
    keep = [
        *same_keys,
        "same_rows",
        "same_accuracy",
        "same_balanced_accuracy",
        "same_log_loss",
        "same_brier_score",
        "same_calibration_error",
    ]
    out = cells.merge(same[keep], on=same_keys, how="left")
    out = out.rename(
        columns={
            "rows": "cross_rows",
            "accuracy": "cross_accuracy",
            "balanced_accuracy": "cross_balanced_accuracy",
            "log_loss": "cross_log_loss",
            "brier_score": "cross_brier_score",
            "calibration_error": "cross_calibration_error",
        }
    )
    out["TRR_accuracy"] = out["cross_accuracy"] / out["same_accuracy"].replace(0.0, np.nan)
    out["TG_accuracy"] = out["same_accuracy"] - out["cross_accuracy"]
    out["TRR_balanced_accuracy"] = out["cross_balanced_accuracy"] / out["same_balanced_accuracy"].replace(0.0, np.nan)
    out["TG_balanced_accuracy"] = out["same_balanced_accuracy"] - out["cross_balanced_accuracy"]
    out["logloss_gap"] = out["cross_log_loss"] - out["same_log_loss"]
    out["brier_score_gap"] = out["cross_brier_score"] - out["same_brier_score"]
    out["calibration_error_gap"] = out["cross_calibration_error"] - out["same_calibration_error"]
    return out[out["cell_role"].eq("cross_regime")].copy()


def build_transfer_cells(
    predictions: pd.DataFrame,
    distance_frame: pd.DataFrame,
    observation_level: str,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    base_keys = ["split", "horizon", "regime_count", "train_regime"]
    group_cols = [*base_keys, "test_regime"]
    if observation_level == "ticker_level":
        group_cols.append("ticker")
        same_keys = [*base_keys, "ticker"]
    else:
        same_keys = base_keys
    cells = cell_metrics(predictions, group_cols)
    cells = attach_transfer_loss(cells, same_keys)
    merge_keys = ["train_regime", "test_regime"]
    if {"horizon", "regime_count"}.issubset(distance_frame.columns):
        merge_keys = ["horizon", "regime_count", *merge_keys]
    elif "regime_count" in distance_frame.columns:
        merge_keys = ["regime_count", *merge_keys]
    elif "horizon" in distance_frame.columns:
        merge_keys = ["horizon", *merge_keys]
    return cells.merge(distance_frame, on=merge_keys, how="left")


def fit_train_only_gmm(
    source: Any,
    features: pd.DataFrame,
    labels: pd.Series,
    splits: dict[str, pd.Index],
    n_regimes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    state_cols, return_cols, vol_cols, breadth_cols = source.select_state_columns(features, splits["train"])
    if not return_cols or not vol_cols:
        raise RuntimeError("missing train-available return or volatility state columns")
    market_state = source.build_market_state(features, state_cols)
    train_set = set(pd.to_datetime(features.loc[splits["train"], "datetime"], errors="coerce").dropna().unique())
    train_state = market_state[market_state["datetime"].isin(train_set)].dropna(subset=state_cols).copy()
    if len(train_state) < max(30, n_regimes * 10):
        raise RuntimeError(f"insufficient train state rows for K={n_regimes}: {len(train_state)}")
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_state[state_cols])
    gmm = GaussianMixture(n_components=n_regimes, covariance_type="full", random_state=RANDOM_STATE, n_init=10)
    gmm.fit(train_scaled)

    assignable = market_state[["datetime", *state_cols]].dropna(subset=state_cols).copy()
    scaled_all = scaler.transform(assignable[state_cols])
    assignable["_raw_regime"] = gmm.predict(scaled_all)
    train_assignable = assignable[assignable["datetime"].isin(train_set)].copy()
    ordering = (
        train_assignable.groupby("_raw_regime")[state_cols]
        .mean()
        .assign(_sort_return=lambda x: x[[col for col in state_cols if "ret" in col or "mean" in col or "trend" in col]].mean(axis=1))
        .assign(_sort_volatility=lambda x: x[[col for col in state_cols if "vol" in col or "dispersion" in col]].mean(axis=1))
        .sort_values(["_sort_return", "_sort_volatility"], ascending=[True, True])
    )
    raw_to_name = {int(raw): f"k{n_regimes}_regime_{rank}" for rank, raw in enumerate(ordering.index)}
    assignable["latent_regime"] = assignable["_raw_regime"].map(raw_to_name)
    assignment = market_state[["datetime"]].merge(assignable[["datetime", "latent_regime"]], on="datetime", how="left")
    assignment["latent_regime"] = assignment["latent_regime"].fillna("unassigned")

    scaled = pd.DataFrame(scaled_all, columns=state_cols)
    scaled["latent_regime"] = assignable["latent_regime"].to_numpy()
    centroids = scaled.groupby("latent_regime", sort=True)[state_cols].mean().reset_index()
    meta = {
        "n_regimes": int(n_regimes),
        "fit_rows_unique_datetimes": int(len(train_state)),
        "state_columns": state_cols,
        "rd_columns": [*return_cols, *vol_cols],
        "return_cols": return_cols,
        "vol_cols": vol_cols,
        "breadth_cols": breadth_cols,
        "bic_train": float(gmm.bic(train_scaled)),
        "aic_train": float(gmm.aic(train_scaled)),
        "train_only_fit": True,
    }
    return assignment, centroids, meta


def rd_distance_variants(centroids: pd.DataFrame, meta: dict[str, Any], regime_count: int, horizon: int | None = None) -> pd.DataFrame:
    centroid_cols = [col for col in centroids.columns if col != "latent_regime"]
    state_cols = [col for col in meta.get("state_columns", []) if col in centroid_cols] or centroid_cols
    rd_cols = [col for col in meta.get("rd_columns", []) if col in centroid_cols] or state_cols
    groups = infer_state_groups(state_cols, centroid_cols)
    variants = [
        pairwise_euclidean(centroids, rd_cols, "RD_standardized_euclidean"),
        pairwise_euclidean(centroids, [*groups["return_cols"], *groups["vol_cols"]], "RD_return_volatility"),
        pairwise_euclidean(
            centroids,
            list(dict.fromkeys([*groups["return_cols"], *groups["vol_cols"], *groups["breadth_cols"]])),
            "RD_return_volatility_breadth",
        ),
    ]
    cosine_frame, reason = pairwise_cosine(centroids, state_cols, "RD_cosine")
    if not reason:
        variants.append(cosine_frame)
    out = variants[0]
    for frame in variants[1:]:
        extra = [col for col in frame.columns if col not in {"train_regime", "test_regime"}]
        out = out.merge(frame[["train_regime", "test_regime", *extra]], on=["train_regime", "test_regime"], how="outer")
    out["regime_count"] = int(regime_count)
    if horizon is not None:
        out["horizon"] = int(horizon)
    return out


def expected_direction(metric: str) -> str:
    if metric.startswith("TRR"):
        return "negative"
    return "positive"


def metric_family(metric: str) -> str:
    if "balanced" in metric:
        return "balanced_accuracy"
    if "accuracy" in metric:
        return "accuracy"
    if "logloss" in metric:
        return "logloss"
    if "brier" in metric:
        return "brier"
    if "calibration" in metric:
        return "calibration"
    return metric


def run_tests(
    frame: pd.DataFrame,
    analysis_scope: str,
    experiment: str,
    observation_level: str,
    horizon: int | str,
    regime_count: int,
) -> pd.DataFrame:
    transfer_metrics = [
        "TRR_accuracy",
        "TG_accuracy",
        "TRR_balanced_accuracy",
        "TG_balanced_accuracy",
        "logloss_gap",
        "brier_score_gap",
        "calibration_error_gap",
    ]
    distance_cols = [
        col
        for col in frame.columns
        if col.startswith("RD_") or col.startswith("FRD_")
    ]
    rows: list[dict[str, Any]] = []
    for distance_variant in distance_cols:
        for transfer_metric in transfer_metrics:
            if transfer_metric not in frame.columns:
                continue
            work = frame[[distance_variant, transfer_metric]].replace([np.inf, -np.inf], np.nan).dropna()
            x = work[distance_variant].to_numpy(dtype=float)
            y = work[transfer_metric].to_numpy(dtype=float)
            pearson_corr, pearson_p, spearman_corr, spearman_p = safe_correlation_value(x, y)
            row = {
                "analysis_scope": analysis_scope,
                "experiment": experiment,
                "split": "validation",
                "horizon": horizon,
                "regime_count": int(regime_count),
                "observation_level": observation_level,
                "distance_variant": distance_variant,
                "transfer_metric": transfer_metric,
                "transfer_metric_family": metric_family(transfer_metric),
                "n_observations": int(len(work)),
                "expected_direction": expected_direction(transfer_metric),
                "pearson_corr": pearson_corr,
                "pearson_p_value": pearson_p,
                "spearman_corr": spearman_corr,
                "spearman_p_value": spearman_p,
                **safe_ols_value(x, y),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def sign_ok(value: float, direction: str) -> bool:
    if not math.isfinite(value):
        return False
    return value < 0.0 if direction == "negative" else value > 0.0


def classify_tests(tests: pd.DataFrame) -> pd.DataFrame:
    if tests.empty:
        return tests
    out = tests.copy()
    prelim: list[str] = []
    significant: list[bool] = []
    expected: list[bool] = []
    near_zero: list[bool] = []
    for _, row in out.iterrows():
        n = int(row.get("n_observations", 0))
        slope = finite_float(row.get("ols_slope"))
        r2 = finite_float(row.get("ols_r_squared"))
        pearson_corr = finite_float(row.get("pearson_corr"))
        spearman_corr = finite_float(row.get("spearman_corr"))
        p_values = [
            finite_float(row.get("pearson_p_value")),
            finite_float(row.get("spearman_p_value")),
            finite_float(row.get("ols_p_value")),
        ]
        p_values = [value for value in p_values if math.isfinite(value)]
        min_p = min(p_values) if p_values else math.nan
        direction = str(row.get("expected_direction", ""))
        slope_expected = sign_ok(slope, direction)
        zero = (
            (not math.isfinite(pearson_corr) or abs(pearson_corr) < 0.05)
            and (not math.isfinite(spearman_corr) or abs(spearman_corr) < 0.05)
            and (not math.isfinite(r2) or r2 < 0.002)
        )
        sig = slope_expected and math.isfinite(min_p) and min_p < 0.05
        if n < MIN_COMPLETE_CELLS or not math.isfinite(slope):
            status = "inconclusive"
        elif not slope_expected or zero:
            status = "not supported"
        elif sig:
            status = "candidate_significant"
        else:
            status = "weak"
        prelim.append(status)
        significant.append(sig)
        expected.append(slope_expected)
        near_zero.append(zero)
    out["preliminary_status"] = prelim
    out["significant_expected_direction"] = significant
    out["expected_direction_observed"] = expected
    out["near_zero_relationship"] = near_zero
    out["support_status"] = out["preliminary_status"]

    significant_mask = out["preliminary_status"].eq("candidate_significant")
    for idx, row in out[significant_mask].iterrows():
        same_metric_group = out[
            out["significant_expected_direction"]
            & out["analysis_scope"].eq(row["analysis_scope"])
            & out["experiment"].eq(row["experiment"])
            & out["observation_level"].eq(row["observation_level"])
            & out["distance_variant"].eq(row["distance_variant"])
            & out["horizon"].eq(row["horizon"])
        ]
        same_horizon_group = out[
            out["significant_expected_direction"]
            & out["analysis_scope"].eq(row["analysis_scope"])
            & out["observation_level"].eq(row["observation_level"])
            & out["distance_variant"].eq(row["distance_variant"])
            & out["transfer_metric"].eq(row["transfer_metric"])
            & out["horizon"].ne("pooled")
        ]
        metric_families = same_metric_group["transfer_metric_family"].nunique()
        horizon_count = same_horizon_group["horizon"].nunique()
        if metric_families >= 2 or horizon_count >= 2:
            out.loc[idx, "support_status"] = "supported"
            out.loc[idx, "support_basis"] = (
                f"expected direction with p < 0.05 and consistency across "
                f"{metric_families} metric families / {horizon_count} horizons"
            )
        else:
            out.loc[idx, "support_status"] = "weak"
            out.loc[idx, "support_basis"] = "isolated p < 0.05 in one narrow metric/horizon"
    out.loc[out["support_basis"].isna() if "support_basis" in out.columns else [], "support_basis"] = ""
    out["support_status"] = out["support_status"].replace({"candidate_significant": "weak"})
    if "support_basis" not in out.columns:
        out["support_basis"] = ""
    return out


def tests_for_cells(
    cells: pd.DataFrame,
    analysis_scope: str,
    experiment: str,
    observation_level: str,
    regime_count: int,
    include_pooled: bool = False,
) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    if cells.empty:
        return pd.DataFrame()
    for horizon, group in cells.groupby("horizon", sort=True):
        pieces.append(run_tests(group, analysis_scope, experiment, observation_level, int(horizon), regime_count))
    if include_pooled and cells["horizon"].nunique() > 1:
        pieces.append(run_tests(cells, analysis_scope, experiment, observation_level, "pooled", regime_count))
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def frd_baseline_tests(source: Any, features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment = saved_k3_assignment()
    labels, splits = strict_labels_and_splits(source, features, PRIMARY_HORIZON)
    analysis = build_analysis_frame(features, labels, splits, assignment)
    models, model_audit = fit_regime_models(features, labels, splits, analysis, feature_cols)
    predictions = transfer_predictions_for_models(features, labels, splits, analysis, feature_cols, models, PRIMARY_HORIZON, 3)
    coef_dist = coefficient_distances(models, "FRD")
    prob_dist = probability_distribution_distances(predictions, "FRD")
    distance_frame = coef_dist.merge(prob_dist.drop(columns=["horizon", "regime_count"], errors="ignore"), on=["train_regime", "test_regime"], how="outer")
    pair_cells = build_transfer_cells(predictions, distance_frame, "pair_level")
    ticker_cells = build_transfer_cells(predictions, distance_frame, "ticker_level")
    tests = pd.concat(
        [
            tests_for_cells(pair_cells, "frd", "frd_baseline_h40", "pair_level", 3),
            tests_for_cells(ticker_cells, "frd", "frd_baseline_h40", "ticker_level", 3),
        ],
        ignore_index=True,
    )
    tests = classify_tests(tests)
    cells = pd.concat(
        [
            pair_cells.assign(observation_level="pair_level"),
            ticker_cells.assign(observation_level="ticker_level"),
        ],
        ignore_index=True,
    )
    return tests, cells, model_audit


def multihorizon_tests(source: Any, features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assignment = saved_k3_assignment()
    distance_frame_base = saved_k3_distances()
    all_predictions: list[pd.DataFrame] = []
    model_audits: list[pd.DataFrame] = []
    distance_frames: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        labels, splits = strict_labels_and_splits(source, features, horizon)
        if len(splits["train"]) == 0 or len(splits["validation"]) == 0:
            continue
        analysis = build_analysis_frame(features, labels, splits, assignment)
        models, model_audit = fit_regime_models(features, labels, splits, analysis, feature_cols)
        model_audit["horizon"] = int(horizon)
        model_audits.append(model_audit)
        predictions = transfer_predictions_for_models(features, labels, splits, analysis, feature_cols, models, horizon, 3)
        all_predictions.append(predictions)
        coef_dist = coefficient_distances(models, "FRD")
        prob_dist = probability_distribution_distances(predictions, "FRD")
        distance_frame = coef_dist.merge(prob_dist.drop(columns=["horizon", "regime_count"], errors="ignore"), on=["train_regime", "test_regime"], how="outer")
        if not distance_frame_base.empty:
            distance_frame = distance_frame.merge(distance_frame_base, on=["train_regime", "test_regime"], how="left")
        distance_frame["horizon"] = int(horizon)
        distance_frame["regime_count"] = 3
        distance_frames.append(distance_frame)
    if not all_predictions:
        skipped = pd.DataFrame(
            [
                {
                    "analysis_scope": "multihorizon",
                    "experiment": "multihorizon_k3",
                    "support_status": "inconclusive",
                    "reason": "h20/h40/h60/h80 validation labels/features were not available",
                }
            ]
        )
        return skipped, pd.DataFrame(), pd.DataFrame()
    predictions_all = pd.concat(all_predictions, ignore_index=True)
    distance_all = pd.concat(distance_frames, ignore_index=True)
    pair_cells = build_transfer_cells(predictions_all, distance_all, "pair_level")
    ticker_cells = build_transfer_cells(predictions_all, distance_all, "ticker_level")
    tests = pd.concat(
        [
            tests_for_cells(pair_cells, "multihorizon", "multihorizon_k3", "pair_level", 3, include_pooled=True),
            tests_for_cells(ticker_cells, "multihorizon", "multihorizon_k3", "ticker_level", 3, include_pooled=True),
        ],
        ignore_index=True,
    )
    tests = classify_tests(tests)
    cells = pd.concat(
        [
            pair_cells.assign(observation_level="pair_level"),
            ticker_cells.assign(observation_level="ticker_level"),
        ],
        ignore_index=True,
    )
    return tests, cells, pd.concat(model_audits, ignore_index=True) if model_audits else pd.DataFrame()


def regime_count_tests(source: Any, features: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels, splits = strict_labels_and_splits(source, features, PRIMARY_HORIZON)
    test_pieces: list[pd.DataFrame] = []
    cell_pieces: list[pd.DataFrame] = []
    audit_rows: list[dict[str, Any]] = []
    for n_regimes in REGIME_COUNTS:
        try:
            assignment, centroids, meta = fit_train_only_gmm(source, features, labels, splits, n_regimes)
            analysis = build_analysis_frame(features, labels, splits, assignment)
            models, model_audit = fit_regime_models(features, labels, splits, analysis, feature_cols)
            predictions = transfer_predictions_for_models(features, labels, splits, analysis, feature_cols, models, PRIMARY_HORIZON, n_regimes)
            distance_frame = rd_distance_variants(centroids, meta, n_regimes, PRIMARY_HORIZON)
            coef_dist = coefficient_distances(models, "FRD")
            prob_dist = probability_distribution_distances(predictions, "FRD")
            frd = coef_dist.merge(prob_dist.drop(columns=["horizon", "regime_count"], errors="ignore"), on=["train_regime", "test_regime"], how="outer")
            frd["horizon"] = PRIMARY_HORIZON
            frd["regime_count"] = n_regimes
            distance_frame = distance_frame.merge(frd, on=["horizon", "regime_count", "train_regime", "test_regime"], how="left")
            pair_cells = build_transfer_cells(predictions, distance_frame, "pair_level")
            ticker_cells = build_transfer_cells(predictions, distance_frame, "ticker_level")
            tests = pd.concat(
                [
                    tests_for_cells(pair_cells, "regime_count", f"k{n_regimes}_h40", "pair_level", n_regimes),
                    tests_for_cells(ticker_cells, "regime_count", f"k{n_regimes}_h40", "ticker_level", n_regimes),
                ],
                ignore_index=True,
            )
            test_pieces.append(classify_tests(tests))
            cell_pieces.append(pair_cells.assign(observation_level="pair_level"))
            cell_pieces.append(ticker_cells.assign(observation_level="ticker_level"))
            audit_rows.append(
                {
                    "regime_count": n_regimes,
                    "status": "computed",
                    "fit_rows_unique_datetimes": meta["fit_rows_unique_datetimes"],
                    "bic_train": meta["bic_train"],
                    "aic_train": meta["aic_train"],
                    "model_status_counts": json.dumps(model_audit["status"].value_counts().to_dict(), sort_keys=True),
                    "reason": "",
                }
            )
        except Exception as exc:
            audit_rows.append(
                {
                    "regime_count": n_regimes,
                    "status": "skipped_with_reason",
                    "fit_rows_unique_datetimes": 0,
                    "bic_train": math.nan,
                    "aic_train": math.nan,
                    "model_status_counts": "{}",
                    "reason": str(exc)[:240],
                }
            )
    return (
        pd.concat(test_pieces, ignore_index=True) if test_pieces else pd.DataFrame(),
        pd.concat(cell_pieces, ignore_index=True) if cell_pieces else pd.DataFrame(),
        pd.DataFrame(audit_rows),
    )


def final_verdict(tests: pd.DataFrame) -> str:
    if tests.empty or tests["support_status"].eq("inconclusive").all():
        return "inconclusive"
    supported = tests[tests["support_status"].eq("supported")]
    weak = tests[tests["support_status"].eq("weak")]
    if not supported.empty:
        scope_count = supported["analysis_scope"].nunique()
        distance_count = supported["distance_variant"].nunique()
        metric_family_count = supported["transfer_metric_family"].nunique()
        horizon_count = supported[supported["horizon"].ne("pooled")]["horizon"].nunique()
        if scope_count >= 2 and distance_count >= 2 and metric_family_count >= 2 and horizon_count >= 2:
            return "generally supported"
        return "partially supported"
    if not weak.empty:
        return "limited metric-specific weak support"
    return "not supported"


def status_improved(verdict: str) -> bool:
    rank = {
        "not supported": 0,
        "inconclusive": 0,
        "limited metric-specific weak support": 1,
        "partially supported": 2,
        "generally supported": 3,
    }
    return rank.get(verdict, 0) > rank[CURRENT_STATUS]


def summary_frame(tests: pd.DataFrame, verdict: str, audits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    counts = tests["support_status"].value_counts().to_dict() if not tests.empty else {}
    rows = [
        {
            "scope": "overall",
            "previous_h4_status": CURRENT_STATUS,
            "final_verdict": verdict,
            "h4_status_improved": status_improved(verdict),
            "supported_tests": int(counts.get("supported", 0)),
            "weak_tests": int(counts.get("weak", 0)),
            "not_supported_tests": int(counts.get("not supported", 0)),
            "inconclusive_tests": int(counts.get("inconclusive", 0)),
            "claim_safe_wording": claim_safe_wording(verdict),
        }
    ]
    for scope, frame in tests.groupby("analysis_scope", sort=True) if not tests.empty else []:
        scope_counts = frame["support_status"].value_counts().to_dict()
        rows.append(
            {
                "scope": scope,
                "previous_h4_status": CURRENT_STATUS,
                "final_verdict": verdict,
                "h4_status_improved": status_improved(verdict),
                "supported_tests": int(scope_counts.get("supported", 0)),
                "weak_tests": int(scope_counts.get("weak", 0)),
                "not_supported_tests": int(scope_counts.get("not supported", 0)),
                "inconclusive_tests": int(scope_counts.get("inconclusive", 0)),
                "claim_safe_wording": claim_safe_wording(verdict),
            }
        )
    for name, audit in audits.items():
        if audit.empty:
            continue
        rows.append(
            {
                "scope": f"{name}_feasibility",
                "previous_h4_status": CURRENT_STATUS,
                "final_verdict": verdict,
                "h4_status_improved": status_improved(verdict),
                "supported_tests": 0,
                "weak_tests": 0,
                "not_supported_tests": 0,
                "inconclusive_tests": int(audit["status"].astype(str).ne("computed").sum()) if "status" in audit.columns else 0,
                "claim_safe_wording": claim_safe_wording(verdict),
            }
        )
    return pd.DataFrame(rows)


def claim_safe_wording(verdict: str) -> str:
    if verdict == "generally supported":
        return "H4 is generally supported by validation-governed robustness diagnostics, but RD remains diagnostic and non-causal."
    if verdict == "partially supported":
        return "H4 is partially supported under specific robustness designs, not established as a general law."
    if verdict == "limited metric-specific weak support":
        return "H4 remains limited to metric-specific weak support and should not be stated as generally supported."
    if verdict == "not supported":
        return "H4 is not supported under the tested strengthening variants."
    return "H4 remains inconclusive under the tested strengthening variants."


def audit_markdown(
    tests: pd.DataFrame,
    summary: pd.DataFrame,
    frd_tests: pd.DataFrame,
    mh_tests: pd.DataFrame,
    rc_tests: pd.DataFrame,
    regime_audit: pd.DataFrame,
    verdict: str,
) -> str:
    support_rows = tests[tests["support_status"].isin(["supported", "weak"])].copy() if not tests.empty else pd.DataFrame()
    fail_rows = tests[tests["support_status"].eq("not supported")].copy() if not tests.empty else pd.DataFrame()
    columns = [
        "analysis_scope",
        "experiment",
        "horizon",
        "regime_count",
        "observation_level",
        "distance_variant",
        "transfer_metric",
        "n_observations",
        "pearson_corr",
        "pearson_p_value",
        "spearman_corr",
        "spearman_p_value",
        "ols_slope",
        "ols_r_squared",
        "support_status",
        "support_basis",
    ]
    support_show = support_rows[[col for col in columns if col in support_rows.columns]].sort_values(
        ["support_status", "analysis_scope", "experiment", "distance_variant", "transfer_metric"],
        ascending=[True, True, True, True, True],
    ) if not support_rows.empty else support_rows
    fail_summary = (
        fail_rows.groupby(["analysis_scope", "observation_level"], sort=True)
        .size()
        .reset_index(name="not_supported_tests")
    ) if not fail_rows.empty else pd.DataFrame()
    return "\n".join(
        [
            "# H4 Strengthening Audit",
            "",
            "## Scope",
            "",
            "This audit tests additional pre-specified H4 strengthening designs using existing local artifacts and local feature builders only. It does not fetch new data, edit the paper draft, edit DOCX/PDF files, touch QML files, commit, push, or tag.",
            "",
            "## Previous Status",
            "",
            f"Previous H4 status: {CURRENT_STATUS}. The prior robustness pack found 82 not-supported tests, 2 weak tests, and 0 supported tests, with weak evidence only in ticker-level RD_cosine balanced-accuracy metrics.",
            "",
            "## Designs Tested",
            "",
            "- Forecast Relationship Distance: coefficient L2, coefficient cosine, and predicted-probability distribution distance.",
            "- Probability-loss transfer metrics: Brier score gap and calibration error gap, in addition to TRR/TG/log-loss gap.",
            "- Multi-horizon validation diagnostics for h20/h40/h60/h80 when local labels/features are available.",
            "- Regime-count robustness for K=2 and K=4 train-only GMM assignments when lightweight fitting is feasible.",
            "",
            "## Statistical Criteria",
            "",
            "A test is supported only when it has the expected direction, p < 0.05, and consistency across at least two related metrics or two horizons. Isolated significant rows and nonsignificant expected-direction rows are weak. Wrong-direction or near-zero relationships are not supported. Insufficient or unstable estimates are inconclusive.",
            "",
            "## Summary",
            "",
            markdown_table(summary, max_rows=20),
            "",
            "## Supporting Or Weak Rows",
            "",
            markdown_table(support_show, max_rows=40),
            "",
            "## Not-Supported Test Counts",
            "",
            markdown_table(fail_summary, max_rows=20),
            "",
            "## FRD Tests",
            "",
            markdown_table(frd_tests["support_status"].value_counts().rename_axis("support_status").reset_index(name="tests") if not frd_tests.empty else frd_tests, max_rows=10),
            "",
            "## Multi-Horizon Tests",
            "",
            markdown_table(mh_tests["support_status"].value_counts().rename_axis("support_status").reset_index(name="tests") if not mh_tests.empty else mh_tests, max_rows=10),
            "",
            "## Regime-Count Robustness",
            "",
            markdown_table(regime_audit, max_rows=10),
            "",
            markdown_table(rc_tests["support_status"].value_counts().rename_axis("support_status").reset_index(name="tests") if not rc_tests.empty else rc_tests, max_rows=10),
            "",
            "## Final Verdict",
            "",
            f"Final verdict: {verdict}.",
            "",
            claim_safe_wording(verdict),
            "",
            "RD and FRD are diagnostic distances, not causal mechanisms. No trading, profitability, investment, live-deployment, final65, or generalization claim is made.",
        ]
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_file(TRANSFER_DIR / "timestamp_regime_assignments.csv")
    require_file(ROBUSTNESS_DIR / "h4_distance_metric_tests.csv")

    source, features, _family_cols, feature_cols = feature_bundle()

    frd_tests, frd_cells, frd_model_audit = frd_baseline_tests(source, features, feature_cols)
    mh_tests, mh_cells, mh_model_audit = multihorizon_tests(source, features, feature_cols)
    rc_tests, rc_cells, regime_audit = regime_count_tests(source, features, feature_cols)

    all_tests = pd.concat(
        [frame for frame in [frd_tests, mh_tests, rc_tests] if not frame.empty],
        ignore_index=True,
    )
    verdict = final_verdict(all_tests)
    audits = {"frd_model": frd_model_audit, "multihorizon_model": mh_model_audit, "regime_count": regime_audit}
    summary = summary_frame(all_tests, verdict, audits)

    write_frame(OUTPUT_DIR / "h4_strengthening_metric_tests.csv", all_tests)
    write_frame(OUTPUT_DIR / "h4_strengthening_summary.csv", summary)
    write_frame(OUTPUT_DIR / "h4_frd_tests.csv", frd_tests)
    write_frame(OUTPUT_DIR / "h4_frd_transfer_cells.csv", frd_cells)
    write_frame(OUTPUT_DIR / "h4_multihorizon_tests.csv", mh_tests)
    write_frame(OUTPUT_DIR / "h4_multihorizon_transfer_cells.csv", mh_cells)
    write_frame(OUTPUT_DIR / "h4_regime_count_robustness.csv", rc_tests)
    write_frame(OUTPUT_DIR / "h4_regime_count_transfer_cells.csv", rc_cells)
    write_frame(OUTPUT_DIR / "h4_regime_count_fit_audit.csv", regime_audit)
    write_json(
        OUTPUT_DIR / "h4_strengthening_manifest.json",
        {
            "status": "ok",
            "output_dir": rel(OUTPUT_DIR),
            "previous_h4_status": CURRENT_STATUS,
            "final_verdict": verdict,
            "h4_status_improved": status_improved(verdict),
            "source_artifacts": [
                rel(TRANSFER_DIR),
                rel(ROBUSTNESS_DIR),
                "local feature builders",
            ],
            "data_fetch": False,
            "provider_behavior_changed": False,
            "paper_draft_edited": False,
            "docx_pdf_edited": False,
            "qml_touched": False,
            "commit_push_tag": False,
        },
    )
    write_markdown(
        OUTPUT_DIR / "H4_STRENGTHENING_AUDIT.md",
        audit_markdown(all_tests, summary, frd_tests, mh_tests, rc_tests, regime_audit, verdict),
    )

    print(f"Wrote H4 strengthening audit outputs to {rel(OUTPUT_DIR)}")
    print(f"Final verdict: {verdict}")
    print(f"H4 status improved: {status_improved(verdict)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
