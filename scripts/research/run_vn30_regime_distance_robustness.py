"""VN30 H4 regime-distance robustness analysis.

This runner uses existing local VN30 regime-transfer artifacts only for the
H4 distance-transfer robustness pack. It reuses saved h40 latent-regime
assignments and transfer predictions. The optional coefficient-distance
diagnostic refits only local train-window Logistic L2 regime models; it does
not refit latent regimes, fetch market data, or change provider behavior.
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
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.stats import linregress, pearsonr, spearmanr
except Exception:  # pragma: no cover - scipy is optional in this repo
    linregress = None
    pearsonr = None
    spearmanr = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TRANSFER_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_transferability"
PAPER_PACK_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_transferability_paper_pack"
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_regime_distance_robustness"

HORIZON = 40
EPS = 1e-6
RANDOM_STATE = 42
MIN_TRAIN_ROWS_PER_REGIME = 80

CURRENT_H4_STATUS = "not supported"
CURRENT_VALIDATION_TRR_PEARSON = 0.0651101
CURRENT_VALIDATION_TRR_P = 0.902473
CURRENT_VALIDATION_TG_PEARSON = -0.0255052
CURRENT_VALIDATION_TG_P = 0.961751


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)


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
        values: list[str] = []
        for col in shown.columns:
            value = row[col]
            if isinstance(value, float):
                values.append("" if not math.isfinite(value) else f"{value:.6g}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def finite_float(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def safe_log_loss(y_true: np.ndarray, probability: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    prob = np.clip(np.asarray(probability, dtype=float), EPS, 1.0 - EPS)
    if len(y) == 0:
        return math.nan
    return float(log_loss(y, prob, labels=[0, 1]))


def balanced_accuracy_binary(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = np.asarray(prediction, dtype=int)
    if len(y) == 0:
        return math.nan
    recalls: list[float] = []
    for klass in [0, 1]:
        mask = y == klass
        if not mask.any():
            return math.nan
        recalls.append(float((pred[mask] == klass).mean()))
    return float(np.mean(recalls))


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


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required local artifact is missing: {rel(path)}")


def load_fit_metadata() -> dict[str, Any]:
    path = TRANSFER_DIR / "latent_regime_fit.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clean_regime_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["train_regime", "test_regime", "latent_regime", "split", "ticker"]:
        if col in out.columns:
            out[col] = out[col].astype(str)
    if "datetime" in out.columns:
        out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    return out


def select_columns(existing: list[str], candidates: list[str]) -> list[str]:
    existing_set = set(existing)
    return [col for col in candidates if col in existing_set]


def infer_state_groups(state_cols: list[str], centroid_cols: list[str]) -> dict[str, list[str]]:
    centroid_set = set(centroid_cols)
    usable = [col for col in state_cols if col in centroid_set]

    def is_return_col(col: str) -> bool:
        lower = col.lower()
        if lower.startswith("breadth_"):
            return False
        if "positive" in lower or "dispersion" in lower or "volume" in lower:
            return False
        return "return" in lower or "_ret" in lower or "mean" in lower or "trend" in lower

    def is_vol_col(col: str) -> bool:
        lower = col.lower()
        return "vol" in lower and "volume" not in lower

    def is_breadth_col(col: str) -> bool:
        lower = col.lower()
        return (
            lower.startswith("breadth_")
            or "positive_share" in lower
            or "positive_lag" in lower
            or "dispersion" in lower
            or "volume_shock" in lower
        )

    return_cols = [col for col in usable if is_return_col(col)]
    vol_cols = [col for col in usable if is_vol_col(col)]
    breadth_cols = [col for col in usable if is_breadth_col(col)]
    return {
        "state_cols": usable,
        "return_cols": return_cols,
        "vol_cols": vol_cols,
        "breadth_cols": breadth_cols,
    }


def pairwise_euclidean(centroids: pd.DataFrame, columns: list[str], variant: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    indexed = centroids.set_index("latent_regime")
    for train_regime in indexed.index:
        vi = indexed.loc[train_regime, columns].to_numpy(dtype=float)
        for test_regime in indexed.index:
            vj = indexed.loc[test_regime, columns].to_numpy(dtype=float)
            rows.append(
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    variant: float(np.linalg.norm(vi - vj)),
                }
            )
    return pd.DataFrame(rows)


def pairwise_cosine(centroids: pd.DataFrame, columns: list[str], variant: str) -> tuple[pd.DataFrame, str]:
    rows: list[dict[str, Any]] = []
    indexed = centroids.set_index("latent_regime")
    skipped_reason = ""
    for train_regime in indexed.index:
        vi = indexed.loc[train_regime, columns].to_numpy(dtype=float)
        ni = float(np.linalg.norm(vi))
        for test_regime in indexed.index:
            vj = indexed.loc[test_regime, columns].to_numpy(dtype=float)
            nj = float(np.linalg.norm(vj))
            if ni <= 0.0 or nj <= 0.0:
                distance = math.nan
                skipped_reason = "zero-norm centroid encountered"
            else:
                cosine = float(np.dot(vi, vj) / (ni * nj))
                distance = float(1.0 - np.clip(cosine, -1.0, 1.0))
            rows.append({"train_regime": train_regime, "test_regime": test_regime, variant: distance})
    return pd.DataFrame(rows), skipped_reason


def drop_duplicate_numeric_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for col in columns:
        series = pd.to_numeric(frame[col], errors="coerce")
        duplicate_of_existing = False
        for kept_col in kept:
            kept_series = pd.to_numeric(frame[kept_col], errors="coerce")
            both = pd.concat([series, kept_series], axis=1).dropna()
            if both.empty:
                continue
            if float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max()) <= 1e-12:
                duplicate_of_existing = True
                break
        if duplicate_of_existing:
            dropped.append(col)
        else:
            kept.append(col)
    return kept, dropped


def mahalanobis_distances(
    row_assignments: pd.DataFrame,
    centroids: pd.DataFrame,
    state_cols: list[str],
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    available_cols = select_columns(list(row_assignments.columns), select_columns(list(centroids.columns), state_cols))
    if len(available_cols) < 2:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": "fewer than two shared state columns",
            "columns_used": "",
            "n_columns": 0,
        }

    train_state = row_assignments[row_assignments["split"].astype(str).eq("train")].copy()
    train_state["datetime"] = pd.to_datetime(train_state["datetime"], errors="coerce")
    train_state = train_state.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last")
    for col in available_cols:
        train_state[col] = pd.to_numeric(train_state[col], errors="coerce")
    train_state = train_state[["datetime", *available_cols]].replace([np.inf, -np.inf], np.nan).dropna(subset=available_cols)
    if len(train_state) < max(30, len(available_cols) * 3):
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": f"insufficient complete train timestamps after NA removal: {len(train_state)}",
            "columns_used": ",".join(available_cols),
            "n_columns": len(available_cols),
        }

    nonconstant_cols = []
    dropped_constant: list[str] = []
    for col in available_cols:
        std = float(pd.to_numeric(train_state[col], errors="coerce").std(ddof=0))
        if math.isfinite(std) and std > 0.0:
            nonconstant_cols.append(col)
        else:
            dropped_constant.append(col)
    deduped_cols, dropped_duplicate = drop_duplicate_numeric_columns(train_state, nonconstant_cols)
    if len(deduped_cols) < 2:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": "fewer than two nonconstant nonduplicate columns for covariance inversion",
            "columns_used": ",".join(deduped_cols),
            "n_columns": len(deduped_cols),
        }

    scaler = StandardScaler()
    x = scaler.fit_transform(train_state[deduped_cols])
    cov = np.cov(x, rowvar=False)
    condition = float(np.linalg.cond(cov))
    if not math.isfinite(condition) or condition > 1e8:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": f"unstable covariance inversion; condition_number={condition:.6g}",
            "columns_used": ",".join(deduped_cols),
            "n_columns": len(deduped_cols),
            "condition_number": condition,
            "dropped_constant_columns": ",".join(dropped_constant),
            "dropped_duplicate_columns": ",".join(dropped_duplicate),
        }
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError as exc:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": f"covariance inversion failed: {exc}",
            "columns_used": ",".join(deduped_cols),
            "n_columns": len(deduped_cols),
            "condition_number": condition,
            "dropped_constant_columns": ",".join(dropped_constant),
            "dropped_duplicate_columns": ",".join(dropped_duplicate),
        }

    indexed = centroids.set_index("latent_regime")
    rows: list[dict[str, Any]] = []
    for train_regime in indexed.index:
        vi = indexed.loc[train_regime, deduped_cols].to_numpy(dtype=float)
        for test_regime in indexed.index:
            vj = indexed.loc[test_regime, deduped_cols].to_numpy(dtype=float)
            diff = vi - vj
            distance_sq = float(diff.T @ inv_cov @ diff)
            rows.append(
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    variant: float(math.sqrt(max(distance_sq, 0.0))),
                }
            )
    return pd.DataFrame(rows), {
        "status": "computed",
        "skipped_reason": "",
        "columns_used": ",".join(deduped_cols),
        "n_columns": len(deduped_cols),
        "condition_number": condition,
        "dropped_constant_columns": ",".join(dropped_constant),
        "dropped_duplicate_columns": ",".join(dropped_duplicate),
    }


def probability_relationship_distances(transfer_predictions: pd.DataFrame, variant: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    validation = transfer_predictions[transfer_predictions["split"].astype(str).eq("validation")].copy()
    if validation.empty:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": "validation transfer predictions are unavailable",
            "columns_used": "probability",
            "n_columns": 1,
        }
    validation["probability"] = pd.to_numeric(validation["probability"], errors="coerce")
    validation = validation.dropna(subset=["datetime", "ticker", "train_regime", "probability"])
    pivot = validation.pivot_table(
        index=["datetime", "ticker"],
        columns="train_regime",
        values="probability",
        aggfunc="mean",
    )
    regimes = sorted(str(col) for col in pivot.columns)
    if len(regimes) < 2:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": "fewer than two regime-specific probability vectors",
            "columns_used": "probability",
            "n_columns": 1,
        }
    rows: list[dict[str, Any]] = []
    for train_regime in regimes:
        for test_regime in regimes:
            pair = pivot[[train_regime, test_regime]].dropna()
            if pair.empty:
                distance = math.nan
                n_common = 0
            else:
                diff = pair[train_regime].to_numpy(dtype=float) - pair[test_regime].to_numpy(dtype=float)
                distance = float(math.sqrt(float(np.mean(diff**2))))
                n_common = int(len(pair))
            rows.append(
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    variant: distance,
                    f"{variant}_common_rows": n_common,
                }
            )
    return pd.DataFrame(rows), {
        "status": "computed",
        "skipped_reason": "",
        "columns_used": "validation probability vectors from transfer_predictions.csv",
        "n_columns": 1,
        "notes": "Root-mean-square probability-vector distance over common validation datetime/ticker rows.",
    }


def coefficient_relationship_distances(
    timestamp_assignments: pd.DataFrame,
    variant: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        source = importlib.import_module("scripts.research.run_vn30_regime_transferability_analysis")
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": f"could not import transferability runner: {exc}",
            "columns_used": "",
            "n_columns": 0,
        }

    required_names = [
        "BASE_FEATURE_FAMILY",
        "FINAL_START",
        "TRAIN_END",
        "VAL_END",
        "VAL_START",
        "add_absolute_labels",
        "build_feature_families",
        "strict_target_split_indices",
    ]
    missing_names = [name for name in required_names if not hasattr(source, name)]
    if missing_names:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": "missing helper(s): " + ", ".join(missing_names),
            "columns_used": "",
            "n_columns": 0,
        }

    try:
        features, family_cols, _feature_manifest = source.build_feature_families()
        features = features.sort_values(["ticker", "datetime"]).reset_index(drop=True).copy()
        features["datetime"] = pd.to_datetime(features["datetime"], errors="coerce")
        labels = source.add_absolute_labels(features, HORIZON)
        splits = source.strict_target_split_indices(
            features,
            labels,
            source.TRAIN_END,
            source.VAL_START,
            source.VAL_END,
            source.FINAL_START,
        )
        family_name = source.BASE_FEATURE_FAMILY
        feature_cols = [
            col
            for col in family_cols.get(family_name, [])
            if col in features.columns and pd.api.types.is_numeric_dtype(features[col])
        ]
        if not feature_cols:
            return pd.DataFrame(), {
                "status": "skipped_with_reason",
                "skipped_reason": f"no numeric {family_name} feature columns",
                "columns_used": "",
                "n_columns": 0,
            }

        assignment = timestamp_assignments[["datetime", "latent_regime"]].copy()
        assignment["datetime"] = pd.to_datetime(assignment["datetime"], errors="coerce")
        analysis = features[["datetime", "ticker"]].merge(assignment, on="datetime", how="left")
        analysis["latent_regime"] = analysis["latent_regime"].fillna("unassigned").astype(str)

        train_idx = splits["train"]
        train_regimes = analysis.loc[train_idx, "latent_regime"].astype(str)
        coefficients: dict[str, np.ndarray] = {}
        model_rows: list[dict[str, Any]] = []
        for regime in sorted(reg for reg in train_regimes.unique() if reg != "unassigned"):
            regime_idx = train_regimes[train_regimes.eq(regime)].index
            y_train = labels.loc[regime_idx].astype(int)
            if len(regime_idx) < MIN_TRAIN_ROWS_PER_REGIME:
                model_rows.append(
                    {
                        "latent_regime": regime,
                        "status": "skipped_with_reason",
                        "train_rows": int(len(regime_idx)),
                        "reason": f"train rows below minimum {MIN_TRAIN_ROWS_PER_REGIME}",
                    }
                )
                continue
            if y_train.nunique() < 2:
                model_rows.append(
                    {
                        "latent_regime": regime,
                        "status": "skipped_with_reason",
                        "train_rows": int(len(regime_idx)),
                        "reason": "train regime has fewer than two target classes",
                    }
                )
                continue
            model = logistic_l2_pipeline()
            model.fit(features.loc[regime_idx, feature_cols], y_train)
            coefficients[regime] = np.asarray(model.named_steps["model"].coef_[0], dtype=float)
            model_rows.append({"latent_regime": regime, "status": "computed", "train_rows": int(len(regime_idx)), "reason": ""})
    except Exception as exc:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": f"local Logistic L2 coefficient reconstruction failed: {exc}",
            "columns_used": "",
            "n_columns": 0,
        }

    if len(coefficients) < 2:
        return pd.DataFrame(), {
            "status": "skipped_with_reason",
            "skipped_reason": "fewer than two reconstructed regime-specific coefficient vectors",
            "columns_used": ",".join(feature_cols),
            "n_columns": len(feature_cols),
            "model_fit_rows": json.dumps(model_rows, ensure_ascii=False),
        }

    rows: list[dict[str, Any]] = []
    for train_regime in sorted(coefficients):
        vi = coefficients[train_regime]
        for test_regime in sorted(coefficients):
            vj = coefficients[test_regime]
            rows.append(
                {
                    "train_regime": train_regime,
                    "test_regime": test_regime,
                    variant: float(np.linalg.norm(vi - vj)),
                }
            )
    return pd.DataFrame(rows), {
        "status": "computed",
        "skipped_reason": "",
        "columns_used": ",".join(feature_cols),
        "n_columns": len(feature_cols),
        "notes": "L2 distance between local train-window regime-specific Logistic L2 coefficient vectors.",
        "model_fit_rows": json.dumps(model_rows, ensure_ascii=False),
    }


def build_distance_variants(
    centroids: pd.DataFrame,
    row_assignments: pd.DataFrame,
    transfer_predictions: pd.DataFrame,
    timestamp_assignments: pd.DataFrame,
    fit_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    centroid_cols = [col for col in centroids.columns if col != "latent_regime"]
    state_cols = [str(col) for col in fit_metadata.get("state_columns", []) if str(col) in centroid_cols]
    if not state_cols:
        state_cols = centroid_cols
    rd_cols = [str(col) for col in fit_metadata.get("rd_columns", []) if str(col) in centroid_cols]
    if not rd_cols:
        rd_cols = state_cols

    groups = infer_state_groups(state_cols, centroid_cols)
    variant_frames: list[pd.DataFrame] = []
    definition_rows: list[dict[str, Any]] = []

    def add_definition(
        variant: str,
        status: str,
        formula: str,
        input_artifact: str,
        columns: list[str] | str,
        skipped_reason: str = "",
        notes: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        used = columns if isinstance(columns, str) else ",".join(columns)
        row: dict[str, Any] = {
            "distance_variant": variant,
            "status": status,
            "formula": formula,
            "input_artifact": input_artifact,
            "columns_used": used,
            "n_columns": 0 if used == "" else len(used.split(",")),
            "skipped_reason": skipped_reason,
            "notes": notes,
        }
        if extra:
            row.update(extra)
        definition_rows.append(row)

    standard = pairwise_euclidean(centroids, rd_cols, "RD_standardized_euclidean")
    variant_frames.append(standard)
    add_definition(
        "RD_standardized_euclidean",
        "computed",
        "Euclidean distance between standardized latent-regime centroids over the original RD columns.",
        rel(TRANSFER_DIR / "regime_state_centroids_standardized.csv"),
        rd_cols,
        notes="Reuses saved h40 K=3 latent-regime centroids; no regime refit.",
    )

    return_vol_cols = [*groups["return_cols"], *groups["vol_cols"]]
    if return_vol_cols:
        variant_frames.append(pairwise_euclidean(centroids, return_vol_cols, "RD_return_volatility"))
        add_definition(
            "RD_return_volatility",
            "computed",
            "Euclidean distance between standardized centroids over return and volatility state columns.",
            rel(TRANSFER_DIR / "regime_state_centroids_standardized.csv"),
            return_vol_cols,
            notes="No regime refit.",
        )
    else:
        add_definition(
            "RD_return_volatility",
            "skipped_with_reason",
            "Euclidean distance over return and volatility state columns.",
            rel(TRANSFER_DIR / "regime_state_centroids_standardized.csv"),
            "",
            skipped_reason="no return/volatility state columns were available",
        )

    rvb_cols = [*return_vol_cols, *groups["breadth_cols"]]
    rvb_cols = list(dict.fromkeys(rvb_cols))
    if rvb_cols:
        variant_frames.append(pairwise_euclidean(centroids, rvb_cols, "RD_return_volatility_breadth"))
        add_definition(
            "RD_return_volatility_breadth",
            "computed",
            "Euclidean distance between standardized centroids over return, volatility, breadth, dispersion, and volume-shock proxies.",
            rel(TRANSFER_DIR / "regime_state_centroids_standardized.csv"),
            rvb_cols,
            notes="No regime refit.",
        )
    else:
        add_definition(
            "RD_return_volatility_breadth",
            "skipped_with_reason",
            "Euclidean distance over return, volatility, breadth, dispersion, and volume-shock proxies.",
            rel(TRANSFER_DIR / "regime_state_centroids_standardized.csv"),
            "",
            skipped_reason="no qualifying state columns were available",
        )

    maha_frame, maha_meta = mahalanobis_distances(row_assignments, centroids, state_cols, "RD_mahalanobis")
    if not maha_frame.empty:
        variant_frames.append(maha_frame)
    add_definition(
        "RD_mahalanobis",
        str(maha_meta.get("status", "skipped_with_reason")),
        "Mahalanobis distance between standardized centroids using train-window state covariance inversion.",
        rel(TRANSFER_DIR / "row_regime_assignments_h40.csv"),
        str(maha_meta.get("columns_used", "")),
        skipped_reason=str(maha_meta.get("skipped_reason", "")),
        notes="Duplicate/constant columns may be dropped before inversion; no latent-regime refit.",
        extra={key: value for key, value in maha_meta.items() if key not in {"status", "skipped_reason", "columns_used", "n_columns"}},
    )

    cosine_frame, cosine_reason = pairwise_cosine(centroids, state_cols, "RD_cosine")
    if not cosine_reason:
        variant_frames.append(cosine_frame)
    add_definition(
        "RD_cosine",
        "computed" if not cosine_reason else "skipped_with_reason",
        "Cosine distance, 1 - cosine similarity, between standardized latent-regime centroids.",
        rel(TRANSFER_DIR / "regime_state_centroids_standardized.csv"),
        state_cols,
        skipped_reason=cosine_reason,
        notes="No regime refit.",
    )

    coef_frame, coef_meta = coefficient_relationship_distances(timestamp_assignments, "FRD_coefficient_distance")
    if not coef_frame.empty:
        variant_frames.append(coef_frame)
    add_definition(
        "FRD_coefficient_distance",
        str(coef_meta.get("status", "skipped_with_reason")),
        "L2 distance between regime-specific Logistic L2 coefficient vectors.",
        "local feature builders plus saved timestamp_regime_assignments.csv",
        str(coef_meta.get("columns_used", "")),
        skipped_reason=str(coef_meta.get("skipped_reason", "")),
        notes=str(coef_meta.get("notes", "")) or "Computed only if local train-window model reconstruction is feasible.",
        extra={key: value for key, value in coef_meta.items() if key not in {"status", "skipped_reason", "columns_used", "n_columns", "notes"}},
    )

    prob_frame, prob_meta = probability_relationship_distances(transfer_predictions, "FRD_probability_distance")
    if not prob_frame.empty:
        variant_frames.append(prob_frame)
    add_definition(
        "FRD_probability_distance",
        str(prob_meta.get("status", "skipped_with_reason")),
        "Root-mean-square distance between regime-specific Logistic L2 probability vectors over common validation rows.",
        rel(TRANSFER_DIR / "transfer_predictions.csv"),
        str(prob_meta.get("columns_used", "")),
        skipped_reason=str(prob_meta.get("skipped_reason", "")),
        notes=str(prob_meta.get("notes", "")),
    )

    add_definition(
        "horizon_level_robustness",
        "skipped_with_reason",
        "Optional horizon-level extension using h20/h40/h60/h80 transfer artifacts.",
        rel(TRANSFER_DIR),
        "",
        skipped_reason="existing local regime-transfer artifacts contain h40 transfer outputs only; no fresh multi-horizon transfer run was performed",
        notes="Optional analysis skipped to keep the pack artifact-only and lightweight.",
    )

    distances = variant_frames[0]
    for frame in variant_frames[1:]:
        extra_cols = [col for col in frame.columns if col not in {"train_regime", "test_regime"} and not col.endswith("_common_rows")]
        distances = distances.merge(frame[["train_regime", "test_regime", *extra_cols]], on=["train_regime", "test_regime"], how="outer")
    return distances, pd.DataFrame(definition_rows)


def cell_metrics(transfer_predictions: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in transfer_predictions.groupby(group_cols, sort=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y = pd.to_numeric(group["y_true"], errors="coerce")
        prob = pd.to_numeric(group["probability"], errors="coerce")
        pred = pd.to_numeric(group["prediction"], errors="coerce")
        clean = pd.DataFrame({"y_true": y, "probability": prob, "prediction": pred}).dropna()
        row = {col: key for col, key in zip(group_cols, keys)}
        if clean.empty:
            row.update(
                {
                    "rows": 0,
                    "accuracy": math.nan,
                    "balanced_accuracy": math.nan,
                    "log_loss": math.nan,
                    "positive_ratio": math.nan,
                    "predicted_positive_ratio": math.nan,
                    "mean_probability": math.nan,
                }
            )
        else:
            y_arr = clean["y_true"].astype(int).to_numpy()
            prob_arr = np.clip(clean["probability"].to_numpy(dtype=float), EPS, 1.0 - EPS)
            pred_arr = clean["prediction"].astype(int).to_numpy()
            row.update(
                {
                    "rows": int(len(clean)),
                    "accuracy": float((y_arr == pred_arr).mean()),
                    "balanced_accuracy": balanced_accuracy_binary(y_arr, pred_arr),
                    "log_loss": safe_log_loss(y_arr, prob_arr),
                    "positive_ratio": float(np.mean(y_arr)),
                    "predicted_positive_ratio": float(np.mean(pred_arr)),
                    "mean_probability": float(np.mean(prob_arr)),
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def attach_transfer_losses(cells: pd.DataFrame, same_keys: list[str]) -> pd.DataFrame:
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
        }
    )
    keep_cols = [*same_keys, "same_rows", "same_accuracy", "same_balanced_accuracy", "same_log_loss"]
    out = cells.merge(same[keep_cols], on=same_keys, how="left")
    out = out.rename(
        columns={
            "rows": "cross_rows",
            "accuracy": "cross_accuracy",
            "balanced_accuracy": "balanced_accuracy_cross",
            "log_loss": "cross_log_loss",
        }
    )
    out["balanced_accuracy_same"] = out["same_balanced_accuracy"]
    out["TRR_accuracy"] = out["cross_accuracy"] / out["same_accuracy"].replace(0.0, np.nan)
    out["TG_accuracy"] = out["same_accuracy"] - out["cross_accuracy"]
    out["TRR_balanced_accuracy"] = out["balanced_accuracy_cross"] / out["balanced_accuracy_same"].replace(0.0, np.nan)
    out["TG_balanced_accuracy"] = out["balanced_accuracy_same"] - out["balanced_accuracy_cross"]
    out["logloss_gap"] = out["cross_log_loss"] - out["same_log_loss"]
    out["logloss_ratio"] = out["cross_log_loss"] / out["same_log_loss"].replace(0.0, np.nan)
    return out


def build_pair_and_ticker_cells(
    transfer_predictions: pd.DataFrame,
    distance_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    validation = transfer_predictions[transfer_predictions["split"].astype(str).eq("validation")].copy()
    validation["datetime"] = pd.to_datetime(validation["datetime"], errors="coerce")
    for col in ["y_true", "probability", "prediction"]:
        validation[col] = pd.to_numeric(validation[col], errors="coerce")
    validation = validation.dropna(subset=["datetime", "ticker", "train_regime", "test_regime", "y_true", "probability", "prediction"])

    pair = cell_metrics(validation, ["split", "train_regime", "test_regime"])
    pair = attach_transfer_losses(pair, ["split", "train_regime"])
    pair = pair[pair["cell_role"].eq("cross_regime")].copy()
    pair = pair.merge(distance_matrix, on=["train_regime", "test_regime"], how="left")

    ticker = cell_metrics(validation, ["split", "train_regime", "test_regime", "ticker"])
    ticker = attach_transfer_losses(ticker, ["split", "train_regime", "ticker"])
    ticker = ticker[ticker["cell_role"].eq("cross_regime")].copy()
    ticker = ticker.merge(distance_matrix, on=["train_regime", "test_regime"], how="left")

    feasible = not ticker.empty and ticker["ticker"].nunique() > 1
    meta = {
        "ticker_level_status": "computed" if feasible else "skipped_with_reason",
        "ticker_level_reason": "" if feasible else "ticker-level transfer cells could not be constructed from validation transfer predictions",
        "ticker_level_rows": int(len(ticker)),
        "ticker_count": int(ticker["ticker"].nunique()) if not ticker.empty else 0,
        "pair_level_rows": int(len(pair)),
    }
    return pair, ticker, meta


def correlation_value(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    pearson_corr = math.nan
    pearson_p = math.nan
    spearman_corr = math.nan
    spearman_p = math.nan
    if len(x) >= 2 and float(np.std(x)) > 0.0 and float(np.std(y)) > 0.0:
        if pearsonr is not None:
            pearson_corr, pearson_p = pearsonr(x, y)
        else:
            pearson_corr = float(np.corrcoef(x, y)[0, 1])
        if spearmanr is not None:
            spearman_corr, spearman_p = spearmanr(x, y)
        else:
            spearman_corr = float(pd.Series(x).rank().corr(pd.Series(y).rank()))
    return float(pearson_corr), float(pearson_p), float(spearman_corr), float(spearman_p)


def ols_value(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if len(x) < 2 or float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return {"ols_intercept": math.nan, "ols_slope": math.nan, "ols_r_squared": math.nan, "ols_p_value": math.nan}
    if linregress is not None:
        result = linregress(x, y)
        return {
            "ols_intercept": float(result.intercept),
            "ols_slope": float(result.slope),
            "ols_r_squared": float(result.rvalue**2),
            "ols_p_value": float(result.pvalue),
        }
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


def sign_matches(value: float, expected_direction: str) -> bool:
    if not math.isfinite(value):
        return False
    return value < 0.0 if expected_direction == "negative" else value > 0.0


def support_status(row: dict[str, Any]) -> str:
    n = int(row.get("n_observations", 0))
    if n < 6:
        return "inconclusive"
    slope = finite_float(row.get("ols_slope"))
    pearson_corr = finite_float(row.get("pearson_corr"))
    pearson_p = finite_float(row.get("pearson_p_value"))
    spearman_corr = finite_float(row.get("spearman_corr"))
    spearman_p = finite_float(row.get("spearman_p_value"))
    expected = str(row.get("expected_direction", ""))
    slope_ok = sign_matches(slope, expected)
    pearson_ok = sign_matches(pearson_corr, expected)
    spearman_ok = sign_matches(spearman_corr, expected)
    if not slope_ok:
        return "not supported"
    if pearson_ok and spearman_ok and math.isfinite(pearson_p) and math.isfinite(spearman_p) and pearson_p <= 0.05 and spearman_p <= 0.05:
        return "supported"
    if (pearson_ok and math.isfinite(pearson_p) and pearson_p <= 0.10) or (
        spearman_ok and math.isfinite(spearman_p) and spearman_p <= 0.10
    ):
        return "weak"
    return "not supported"


def expected_direction_for_metric(metric: str) -> str:
    if metric.startswith("TRR"):
        return "negative"
    return "positive"


def run_distance_metric_tests(
    pair_cells: pd.DataFrame,
    ticker_cells: pd.DataFrame,
    distance_definitions: pd.DataFrame,
) -> pd.DataFrame:
    distance_variants = distance_definitions[
        distance_definitions["status"].astype(str).eq("computed")
        & distance_definitions["distance_variant"].astype(str).ne("horizon_level_robustness")
    ]["distance_variant"].astype(str).tolist()
    transfer_metrics = [
        "TRR_accuracy",
        "TG_accuracy",
        "TRR_balanced_accuracy",
        "TG_balanced_accuracy",
        "logloss_gap",
        "logloss_ratio",
    ]
    level_frames = [("pair_level", pair_cells), ("ticker_level", ticker_cells)]
    rows: list[dict[str, Any]] = []
    for observation_level, frame in level_frames:
        for distance_variant in distance_variants:
            if distance_variant not in frame.columns:
                for metric in transfer_metrics:
                    rows.append(
                        {
                            "split": "validation",
                            "observation_level": observation_level,
                            "distance_variant": distance_variant,
                            "transfer_metric": metric,
                            "n_observations": 0,
                            "expected_direction": expected_direction_for_metric(metric),
                            "pearson_corr": math.nan,
                            "pearson_p_value": math.nan,
                            "spearman_corr": math.nan,
                            "spearman_p_value": math.nan,
                            "ols_intercept": math.nan,
                            "ols_slope": math.nan,
                            "ols_r_squared": math.nan,
                            "ols_p_value": math.nan,
                            "support_status": "inconclusive",
                            "reason": "distance variant not present on transfer cells",
                        }
                    )
                continue
            for metric in transfer_metrics:
                expected = expected_direction_for_metric(metric)
                if metric not in frame.columns:
                    rows.append(
                        {
                            "split": "validation",
                            "observation_level": observation_level,
                            "distance_variant": distance_variant,
                            "transfer_metric": metric,
                            "n_observations": 0,
                            "expected_direction": expected,
                            "pearson_corr": math.nan,
                            "pearson_p_value": math.nan,
                            "spearman_corr": math.nan,
                            "spearman_p_value": math.nan,
                            "ols_intercept": math.nan,
                            "ols_slope": math.nan,
                            "ols_r_squared": math.nan,
                            "ols_p_value": math.nan,
                            "support_status": "inconclusive",
                            "reason": "transfer metric not present on transfer cells",
                        }
                    )
                    continue
                work = frame[[distance_variant, metric]].replace([np.inf, -np.inf], np.nan).dropna()
                x = work[distance_variant].to_numpy(dtype=float)
                y = work[metric].to_numpy(dtype=float)
                pearson_corr, pearson_p, spearman_corr, spearman_p = correlation_value(x, y)
                row = {
                    "split": "validation",
                    "observation_level": observation_level,
                    "distance_variant": distance_variant,
                    "transfer_metric": metric,
                    "n_observations": int(len(work)),
                    "expected_direction": expected,
                    "pearson_corr": pearson_corr,
                    "pearson_p_value": pearson_p,
                    "spearman_corr": spearman_corr,
                    "spearman_p_value": spearman_p,
                    **ols_value(x, y),
                    "reason": "",
                }
                row["support_status"] = support_status(row)
                rows.append(row)
    return pd.DataFrame(rows)


def h4_verdict(metric_tests: pd.DataFrame) -> str:
    if metric_tests.empty:
        return "H4 remains not supported under the tested robustness variants."
    support_values = set(metric_tests["support_status"].astype(str))
    if support_values.intersection({"supported", "weak"}):
        return "H4 receives limited/metric-specific support, not general support."
    return "H4 remains not supported under the tested robustness variants."


def summary_frame(
    metric_tests: pd.DataFrame,
    distance_definitions: pd.DataFrame,
    transfer_meta: dict[str, Any],
    verdict: str,
) -> pd.DataFrame:
    computed_variants = distance_definitions[distance_definitions["status"].astype(str).eq("computed")]["distance_variant"].astype(str).tolist()
    tested_metrics = sorted(metric_tests["transfer_metric"].astype(str).unique()) if not metric_tests.empty else []
    rows: list[dict[str, Any]] = [
        {
            "scope": "overall_validation_h4_robustness",
            "current_h4_status": CURRENT_H4_STATUS,
            "distance_variants_computed": ",".join(computed_variants),
            "transfer_metrics_tested": ",".join(tested_metrics),
            "pair_level_observations": int(transfer_meta.get("pair_level_rows", 0)),
            "ticker_level_status": transfer_meta.get("ticker_level_status", ""),
            "ticker_level_observations": int(transfer_meta.get("ticker_level_rows", 0)),
            "ticker_count": int(transfer_meta.get("ticker_count", 0)),
            "supported_tests": int(metric_tests["support_status"].astype(str).eq("supported").sum()) if not metric_tests.empty else 0,
            "weak_tests": int(metric_tests["support_status"].astype(str).eq("weak").sum()) if not metric_tests.empty else 0,
            "not_supported_tests": int(metric_tests["support_status"].astype(str).eq("not supported").sum()) if not metric_tests.empty else 0,
            "inconclusive_tests": int(metric_tests["support_status"].astype(str).eq("inconclusive").sum()) if not metric_tests.empty else 0,
            "h4_robustness_verdict": verdict,
            "claim_note": "Validation diagnostics only; no causal, trading, profitability, live-deployment, final-window, final65, or generalization claim.",
        }
    ]
    if not metric_tests.empty:
        grouped = (
            metric_tests.groupby(["observation_level", "distance_variant", "support_status"], sort=True)
            .size()
            .reset_index(name="tests")
        )
        for (observation_level, distance_variant), group in grouped.groupby(["observation_level", "distance_variant"], sort=True):
            status_counts = {str(row["support_status"]): int(row["tests"]) for _, row in group.iterrows()}
            rows.append(
                {
                    "scope": f"{observation_level}:{distance_variant}",
                    "current_h4_status": CURRENT_H4_STATUS,
                    "distance_variants_computed": distance_variant,
                    "transfer_metrics_tested": ",".join(tested_metrics),
                    "pair_level_observations": int(transfer_meta.get("pair_level_rows", 0)) if observation_level == "pair_level" else 0,
                    "ticker_level_status": transfer_meta.get("ticker_level_status", "") if observation_level == "ticker_level" else "",
                    "ticker_level_observations": int(transfer_meta.get("ticker_level_rows", 0)) if observation_level == "ticker_level" else 0,
                    "ticker_count": int(transfer_meta.get("ticker_count", 0)) if observation_level == "ticker_level" else 0,
                    "supported_tests": status_counts.get("supported", 0),
                    "weak_tests": status_counts.get("weak", 0),
                    "not_supported_tests": status_counts.get("not supported", 0),
                    "inconclusive_tests": status_counts.get("inconclusive", 0),
                    "h4_robustness_verdict": verdict,
                    "claim_note": "Subsummary; ticker-level rows share regime-pair distances and should not be treated as independent proof.",
                }
            )
    return pd.DataFrame(rows)


def claim_boundary_frame(verdict: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "claim_area": "H4 distance-transfer mechanism",
                "boundary": verdict,
                "paper_action": "State the robustness verdict conservatively.",
            },
            {
                "claim_area": "Causality",
                "boundary": "Regime Distance is diagnostic and must not be described as causing transfer loss.",
                "paper_action": "Avoid causal wording.",
            },
            {
                "claim_area": "Primary window",
                "boundary": "Validation-window diagnostics govern the H4 verdict.",
                "paper_action": "Do not promote final-window results.",
            },
            {
                "claim_area": "Ticker-level expansion",
                "boundary": "Ticker-level cells increase diagnostic observations but share regime-pair distances and are not independent regime-pair evidence.",
                "paper_action": "Use as robustness diagnostics, not proof.",
            },
            {
                "claim_area": "Excluded claims",
                "boundary": "No trading, profitability, investment, live-deployment, final65, or generalization claim is supported.",
                "paper_action": "Keep these exclusions explicit.",
            },
        ]
    )


def audit_markdown(
    distance_definitions: pd.DataFrame,
    metric_tests: pd.DataFrame,
    summary: pd.DataFrame,
    claim_boundary: pd.DataFrame,
    transfer_meta: dict[str, Any],
    verdict: str,
) -> str:
    computed_definitions = distance_definitions[
        ["distance_variant", "status", "n_columns", "skipped_reason", "notes"]
    ].copy()
    test_columns = [
        "observation_level",
        "distance_variant",
        "transfer_metric",
        "n_observations",
        "expected_direction",
        "pearson_corr",
        "pearson_p_value",
        "spearman_corr",
        "spearman_p_value",
        "ols_slope",
        "ols_r_squared",
        "support_status",
    ]
    status_order = {"supported": 0, "weak": 1, "not supported": 2, "inconclusive": 3}
    shown_tests = metric_tests[test_columns].copy()
    shown_tests["_status_order"] = shown_tests["support_status"].map(status_order).fillna(9)
    shown_tests = shown_tests.sort_values(
        ["_status_order", "observation_level", "distance_variant", "transfer_metric"],
        ascending=[True, True, True, True],
    ).drop(columns=["_status_order"])
    return "\n".join(
        [
            "# H4 Distance Robustness Audit",
            "",
            "## Scope",
            "",
            "This audit creates a validation-governed robustness pack for H4 using existing local artifacts only. It reuses saved h40 K=3 latent-regime assignments and transfer predictions. It does not refit latent regimes, fetch market data, change provider behavior, edit DOCX/PDF files, edit the paper draft, commit, push, or tag.",
            "",
            "## Current H4 Status",
            "",
            f"Current H4 status: {CURRENT_H4_STATUS}. The existing validation RD tests use 6 cross-regime cells and do not support H4: TRR-vs-RD Pearson correlation = {CURRENT_VALIDATION_TRR_PEARSON:.7g}, p = {CURRENT_VALIDATION_TRR_P:.6g}; TG-vs-RD Pearson correlation = {CURRENT_VALIDATION_TG_PEARSON:.7g}, p = {CURRENT_VALIDATION_TG_P:.6g}.",
            "",
            "## Why n=6 Is Weak",
            "",
            "With three latent regimes, the directional cross-regime matrix has only six cells. That leaves very low power for correlation and slope diagnostics, makes results sensitive to one cell, and repeats only three underlying pairwise distances in opposite directions. These diagnostics are useful for discovery but weak as standalone evidence for a distance-transfer mechanism.",
            "",
            "## Alternative Regime Distance Definitions",
            "",
            markdown_table(computed_definitions, max_rows=20),
            "",
            "## Alternative Transfer Metrics",
            "",
            "- TRR_accuracy = cross_accuracy / same_accuracy; expected to decrease with distance.",
            "- TG_accuracy = same_accuracy - cross_accuracy; expected to increase with distance.",
            "- TRR_balanced_accuracy and TG_balanced_accuracy repeat the accuracy diagnostics with balanced accuracy where both classes are observed.",
            "- logloss_gap = cross_log_loss - same_log_loss; expected to increase with distance.",
            "- logloss_ratio = cross_log_loss / same_log_loss where finite; expected to increase with distance.",
            "",
            "## Ticker-Level Expansion",
            "",
            f"Ticker-level expansion status: {transfer_meta.get('ticker_level_status', '')}. Validation ticker-level cells: {transfer_meta.get('ticker_level_rows', 0)} across {transfer_meta.get('ticker_count', 0)} tickers. These rows increase diagnostic granularity, but they share regime-pair distances and should not be treated as independent proof.",
            "",
            "## Statistical Test Summary",
            "",
            "Rows marked weak or supported are shown first. Ticker-level p-values are diagnostic because ticker rows share repeated regime-pair distance values; they do not override the pair-level evidence.",
            "",
            markdown_table(shown_tests, max_rows=40),
            "",
            "## Robustness Summary",
            "",
            markdown_table(summary.head(12), max_rows=12),
            "",
            "## Final H4 Verdict",
            "",
            verdict,
            "",
            "The result remains diagnostic. RD is a latent-regime distance measure, not a causal driver of transfer loss.",
            "",
            "## Claim Boundary",
            "",
            markdown_table(claim_boundary, max_rows=20),
        ]
    )


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    required = [
        TRANSFER_DIR / "transfer_predictions.csv",
        TRANSFER_DIR / "regime_state_centroids_standardized.csv",
        TRANSFER_DIR / "row_regime_assignments_h40.csv",
        TRANSFER_DIR / "timestamp_regime_assignments.csv",
    ]
    for path in required:
        require_file(path)

    transfer_predictions = clean_regime_frame(pd.read_csv(TRANSFER_DIR / "transfer_predictions.csv", low_memory=False))
    centroids = pd.read_csv(TRANSFER_DIR / "regime_state_centroids_standardized.csv", low_memory=False)
    row_assignments = clean_regime_frame(pd.read_csv(TRANSFER_DIR / "row_regime_assignments_h40.csv", low_memory=False))
    timestamp_assignments = clean_regime_frame(pd.read_csv(TRANSFER_DIR / "timestamp_regime_assignments.csv", low_memory=False))
    fit_metadata = load_fit_metadata()

    distance_matrix, distance_definitions = build_distance_variants(
        centroids=centroids,
        row_assignments=row_assignments,
        transfer_predictions=transfer_predictions,
        timestamp_assignments=timestamp_assignments,
        fit_metadata=fit_metadata,
    )
    pair_cells, ticker_cells, transfer_meta = build_pair_and_ticker_cells(transfer_predictions, distance_matrix)
    metric_tests = run_distance_metric_tests(pair_cells, ticker_cells, distance_definitions)
    verdict = h4_verdict(metric_tests)
    summary = summary_frame(metric_tests, distance_definitions, transfer_meta, verdict)
    claim_boundary = claim_boundary_frame(verdict)

    write_frame(OUTPUT_DIR / "h4_distance_definitions.csv", distance_definitions)
    write_frame(OUTPUT_DIR / "h4_pair_level_transfer_cells.csv", pair_cells)
    if transfer_meta.get("ticker_level_status") == "computed":
        write_frame(OUTPUT_DIR / "h4_ticker_level_transfer_cells.csv", ticker_cells)
    else:
        write_frame(
            OUTPUT_DIR / "h4_ticker_level_transfer_cells_skipped.csv",
            pd.DataFrame([{"status": transfer_meta.get("ticker_level_status"), "reason": transfer_meta.get("ticker_level_reason")}]),
        )
    write_frame(OUTPUT_DIR / "h4_distance_metric_tests.csv", metric_tests)
    write_frame(OUTPUT_DIR / "h4_distance_robustness_summary.csv", summary)
    write_frame(OUTPUT_DIR / "h4_claim_boundary.csv", claim_boundary)
    write_json(
        OUTPUT_DIR / "h4_distance_robustness_manifest.json",
        {
            "status": "ok",
            "output_dir": rel(OUTPUT_DIR),
            "source_artifacts": [rel(path) for path in required],
            "horizon": HORIZON,
            "latent_regime_refit": False,
            "market_data_fetch": False,
            "provider_behavior_changed": False,
            "paper_draft_edited": False,
            "docx_pdf_edited": False,
            "qml_touched": False,
            "commit_push_tag": False,
            "h4_robustness_verdict": verdict,
        },
    )
    write_markdown(
        OUTPUT_DIR / "H4_DISTANCE_ROBUSTNESS_AUDIT.md",
        audit_markdown(distance_definitions, metric_tests, summary, claim_boundary, transfer_meta, verdict),
    )

    print(f"Wrote H4 distance robustness outputs to {rel(OUTPUT_DIR)}")
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
