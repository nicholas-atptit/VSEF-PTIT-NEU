"""Mine VN30 v3 promotion candidates and run offline strategy diagnostics.

This script does not tune a new grid. It consumes v3 promotion outputs,
freezes candidate definitions, reproduces fixed-parameter signals, and runs
offline long-only strategy simulations with explicit diagnostic claim labels.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_full_model_resurrection_index_pretrain import (  # noqa: E402
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    as_float,
    baseline_predictions,
    clean_feature_matrix,
    json_safe,
    now_utc,
    pct,
    pp,
    predict_probability,
    rel,
    split_indices,
    write_frame,
    write_json,
    write_markdown,
)
from scripts.research.run_vn30_full_model_tuning_v3 import (  # noqa: E402
    EXPLORATORY_BEST,
    CLAIMABLE_CHAMPION,
    baseline_frames_for_split,
    build_target_labels,
    build_v3_feature_frame,
    fit_regime_router,
    make_model,
    stock_forward_return_and_timestamp,
    target_variant_definition,
    predict_regime_router,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    load_index_data,
    target_timestamp_from_labels,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_v4_strategy"
V3_DIR = REPO_ROOT / "reports" / "generated" / "vn30_full_model_tuning_v3"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_V4_PROMOTION_QUEUE_AND_REAL_STRATEGY_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_V4_PROMOTION_QUEUE_AND_REAL_STRATEGY_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_V4_PROMOTION_QUEUE_AND_REAL_STRATEGY_CLAIM_BOUNDARY.md"

SEED = 42
MAX_POSITIONS = [3, 5, 10]
COST_BPS = [0, 5, 10, 20, 30]
SLIPPAGE_BPS = [0, 5, 10, 20]
STRATEGY_TEMPLATES = [
    "long_only_confidence",
    "long_only_market_regime_filter",
    "relative_strength_rotation",
    "cash_when_uncertain",
    "regime_gated_strategy",
]


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing required input: {rel(path)}")
    return pd.read_csv(path, low_memory=False)


def threshold_bin(value: Any, width: float = 0.025) -> float:
    number = as_float(value)
    if not math.isfinite(number):
        return math.nan
    return round(round(number / width) * width, 3)


def base_candidate_id(candidate_id: str) -> str:
    return re.sub(r"__t\d+p\d+$", "", str(candidate_id))


def cluster_key_columns() -> list[str]:
    return ["model_family", "target_variant", "feature_group", "horizon", "threshold_neighborhood"]


def mine_promotion_queue(
    promotion: pd.DataFrame,
    exploratory: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = promotion.copy()
    for col in ["threshold", "final_accuracy", "final_lift", "final_rows", "horizon", "validation_accuracy", "validation_lift"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    work["threshold_neighborhood"] = work["threshold"].apply(threshold_bin)
    work["base_candidate_id"] = work["candidate_id"].astype(str).map(base_candidate_id)
    work["beats_61_61"] = work["final_accuracy"] > CLAIMABLE_CHAMPION["final_accuracy"]
    work["beats_64_76"] = work["final_accuracy"] > EXPLORATORY_BEST["final_accuracy"]
    work["beats_65"] = work["final_accuracy"] > 0.65
    work["beats_plus_14pp_lift"] = work["final_lift"] > 0.14

    grouped = work.groupby(cluster_key_columns(), dropna=False)
    rows: list[dict[str, Any]] = []
    for key, group in grouped:
        key_payload = dict(zip(cluster_key_columns(), key))
        representative = group.sort_values(["final_accuracy", "final_lift"], ascending=[False, False]).iloc[0]
        threshold_span = as_float(group["threshold"].max()) - as_float(group["threshold"].min())
        acc_std = float(group["final_accuracy"].std(ddof=0)) if len(group) else math.nan
        stable = bool(len(group) >= 3 and threshold_span >= 0.01 and group["final_accuracy"].max() >= 0.62)
        isolated = bool(len(group) == 1 or (len(group) <= 2 and threshold_span < 0.01))
        rows.append(
            {
                **key_payload,
                "cluster_id": "__".join(str(key_payload[col]).replace(".", "p") for col in cluster_key_columns()),
                "candidate_count": int(len(group)),
                "representative_candidate_id": representative["candidate_id"],
                "representative_model_params": representative.get("model_params", ""),
                "min_threshold": float(group["threshold"].min()),
                "max_threshold": float(group["threshold"].max()),
                "threshold_span": threshold_span,
                "mean_final_accuracy": float(group["final_accuracy"].mean()),
                "max_final_accuracy": float(group["final_accuracy"].max()),
                "min_final_accuracy": float(group["final_accuracy"].min()),
                "std_final_accuracy": acc_std,
                "mean_final_lift": float(group["final_lift"].mean()),
                "max_final_lift": float(group["final_lift"].max()),
                "rows_min": int(group["final_rows"].min()) if group["final_rows"].notna().any() else 0,
                "rows_max": int(group["final_rows"].max()) if group["final_rows"].notna().any() else 0,
                "beats_61_61_count": int(group["beats_61_61"].sum()),
                "beats_64_76_count": int(group["beats_64_76"].sum()),
                "beats_65_count": int(group["beats_65"].sum()),
                "beats_plus_14pp_lift_count": int(group["beats_plus_14pp_lift"].sum()),
                "stable_cluster": stable,
                "isolated_one_off": isolated,
                "cluster_verdict": "stable_candidate_cluster" if stable else ("isolated_one_off" if isolated else "unstable_or_thin_cluster"),
            }
        )
    cluster_summary = pd.DataFrame(rows).sort_values(
        ["stable_cluster", "max_final_accuracy", "max_final_lift", "candidate_count"],
        ascending=[False, False, False, False],
    )

    top_accuracy = work.sort_values(["final_accuracy", "final_lift"], ascending=[False, False]).head(100)
    top_lift = work.sort_values(["final_lift", "final_accuracy"], ascending=[False, False]).head(100)

    threshold_rows: list[dict[str, Any]] = []
    for key, group in work.groupby(["model_family", "target_variant", "feature_group", "horizon", "base_candidate_id"], dropna=False):
        ordered = group.sort_values("threshold")
        threshold_rows.append(
            {
                "model_family": key[0],
                "target_variant": key[1],
                "feature_group": key[2],
                "horizon": key[3],
                "base_candidate_id": key[4],
                "threshold_count": int(len(ordered)),
                "min_threshold": float(ordered["threshold"].min()),
                "max_threshold": float(ordered["threshold"].max()),
                "threshold_span": float(ordered["threshold"].max() - ordered["threshold"].min()),
                "mean_final_accuracy": float(ordered["final_accuracy"].mean()),
                "max_final_accuracy": float(ordered["final_accuracy"].max()),
                "min_final_accuracy": float(ordered["final_accuracy"].min()),
                "std_final_accuracy": float(ordered["final_accuracy"].std(ddof=0)) if len(ordered) else math.nan,
                "mean_final_lift": float(ordered["final_lift"].mean()),
                "max_final_lift": float(ordered["final_lift"].max()),
                "stable_threshold_neighborhood": bool(len(ordered) >= 3 and ordered["threshold"].max() - ordered["threshold"].min() >= 0.01),
            }
        )
    threshold_stability = pd.DataFrame(threshold_rows).sort_values(["max_final_accuracy", "threshold_count"], ascending=[False, False])

    horizon_rows: list[dict[str, Any]] = []
    for key, group in work.groupby(["model_family", "target_variant", "feature_group", "threshold_neighborhood"], dropna=False):
        horizon_rows.append(
            {
                "model_family": key[0],
                "target_variant": key[1],
                "feature_group": key[2],
                "threshold_neighborhood": key[3],
                "horizon_count": int(group["horizon"].nunique()),
                "min_horizon": int(group["horizon"].min()),
                "max_horizon": int(group["horizon"].max()),
                "mean_final_accuracy": float(group["final_accuracy"].mean()),
                "max_final_accuracy": float(group["final_accuracy"].max()),
                "mean_final_lift": float(group["final_lift"].mean()),
                "max_final_lift": float(group["final_lift"].max()),
                "stable_horizon_neighborhood": bool(group["horizon"].nunique() >= 2 and group["final_accuracy"].max() >= 0.62),
            }
        )
    horizon_stability = pd.DataFrame(horizon_rows).sort_values(["max_final_accuracy", "horizon_count"], ascending=[False, False])

    return cluster_summary, top_accuracy, top_lift, threshold_stability, horizon_stability


def lookup_candidate_rows(promotion: pd.DataFrame, exploratory: pd.DataFrame) -> dict[str, dict[str, Any]]:
    combined = pd.concat([promotion, exploratory], ignore_index=True, sort=False)
    out: dict[str, dict[str, Any]] = {}
    for candidate_id, row in combined.drop_duplicates("candidate_id").set_index("candidate_id").to_dict("index").items():
        payload = dict(row)
        payload["candidate_id"] = candidate_id
        out[str(candidate_id)] = payload
    return out


def champion_candidate() -> dict[str, Any]:
    return {
        "frozen_candidate_id": "current_champion_l2_feature_set_C_closest_h40_t0p500",
        "source_candidate_id": CLAIMABLE_CHAMPION["candidate_id"],
        "freeze_role": "strict_replay_champion_61p61",
        "model_family": "logistic_regression",
        "model_params": compact_json(
            {
                "model_family": "logistic_regression",
                "penalty": "l2",
                "solver": "liblinear",
                "C": 0.3,
                "class_weight": "balanced",
                "l1_ratio": None,
            }
        ),
        "target_variant": "absolute_direction",
        "feature_group": "feature_set_C_closest",
        "horizon": 40,
        "threshold": 0.5,
        "reference_final_accuracy": CLAIMABLE_CHAMPION["final_accuracy"],
        "reference_final_lift": CLAIMABLE_CHAMPION["final_lift"],
        "claim_label": "baseline60_candidate",
    }


def freeze_candidate_from_row(row: dict[str, Any], role: str) -> dict[str, Any]:
    return {
        "frozen_candidate_id": str(row["candidate_id"]),
        "source_candidate_id": str(row["candidate_id"]),
        "freeze_role": role,
        "model_family": str(row["model_family"]),
        "model_params": str(row.get("model_params", "")),
        "target_variant": str(row["target_variant"]),
        "feature_group": str(row["feature_group"]),
        "horizon": int(as_float(row["horizon"])),
        "threshold": float(as_float(row["threshold"])),
        "reference_final_accuracy": as_float(row.get("final_accuracy")),
        "reference_final_lift": as_float(row.get("final_lift")),
        "claim_label": str(row.get("claim_label", "exploratory_not_claimable")),
    }


def build_frozen_candidate_set(
    promotion: pd.DataFrame,
    exploratory: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    lookup = lookup_candidate_rows(promotion, exploratory)
    frozen: dict[str, dict[str, Any]] = {}

    champion = champion_candidate()
    frozen[champion["frozen_candidate_id"]] = champion

    for candidate_id, role in [
        ("forced_exploratory_best_l1_compact_h50__t0p525", "exploratory_64p76_baseline"),
        ("grid_334693__t0p525", "exploratory_64p76_baseline"),
        ("forced_v3_calibrated_compact_h40__t0p540", "exploratory_65p51_best"),
    ]:
        if candidate_id in lookup:
            row = freeze_candidate_from_row(lookup[candidate_id], role)
            frozen[row["frozen_candidate_id"]] = row

    top_clusters = cluster_summary.head(5)
    for idx, cluster in top_clusters.iterrows():
        candidate_id = str(cluster["representative_candidate_id"])
        if candidate_id not in lookup:
            continue
        row = freeze_candidate_from_row(lookup[candidate_id], f"top_cluster_{idx + 1}")
        if row["frozen_candidate_id"] in frozen:
            frozen[row["frozen_candidate_id"]]["freeze_role"] += f";top_cluster_{idx + 1}"
        else:
            frozen[row["frozen_candidate_id"]] = row

    out = pd.DataFrame(frozen.values()).reset_index(drop=True)
    out["frozen_parameters_changed"] = False
    out["final_ranked_selection_claimable"] = False
    return out


def fit_candidate_payload(
    candidate: dict[str, Any],
    features: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    index_data: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    target_variant = str(candidate["target_variant"])
    horizon = int(candidate["horizon"])
    labels = build_target_labels(features, index_data, horizon, target_variant)
    splits = split_indices(features, labels)
    feature_cols = feature_groups.get(str(candidate["feature_group"]), [])
    train_idx = splits["train"]
    train_y = labels.reindex(train_idx).astype(int)
    if len(train_y) < 100 or train_y.nunique() < 2:
        raise ValueError("insufficient train labels")
    if not feature_cols:
        raise ValueError("empty feature group")
    model_family = str(candidate["model_family"])
    params = json.loads(str(candidate["model_params"])) if str(candidate.get("model_params", "")).strip() else {}
    if model_family == "regime_gated_ensemble":
        group_col = str(params.get("group_col", "market_direction_regime"))
        router = fit_regime_router(features, labels, splits, feature_cols, group_col)
        return {
            "payload_type": "regime_gate",
            "router": router,
            "candidate": candidate,
            "labels": labels,
            "splits": splits,
            "feature_cols": feature_cols,
        }
    model = make_model(model_family, params)
    if model is None:
        raise ValueError(f"unsupported model family: {model_family}")
    model.fit(clean_feature_matrix(features.loc[train_idx], feature_cols), train_y)
    return {
        "payload_type": "model",
        "model": model,
        "candidate": candidate,
        "labels": labels,
        "splits": splits,
        "feature_cols": feature_cols,
    }


def payload_prob(payload: dict[str, Any], features: pd.DataFrame, idx: pd.Index) -> np.ndarray:
    if payload["payload_type"] == "regime_gate":
        return predict_regime_router(payload["router"], features, idx)
    return predict_probability(payload["model"], clean_feature_matrix(features.loc[idx], payload["feature_cols"]))


def build_signal_rows(
    payload: dict[str, Any],
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
    split: str,
) -> pd.DataFrame:
    candidate = payload["candidate"]
    labels = payload["labels"]
    idx = payload["splits"][split]
    probability = payload_prob(payload, features, idx)
    threshold = float(candidate["threshold"])
    target_ts = target_timestamp_from_labels(labels).reindex(idx)
    abs_return, abs_target_ts = stock_forward_return_and_timestamp(features, int(candidate["horizon"]))
    frame = features.loc[idx, ["datetime", "ticker", "close", "feature_timestamp"]].copy()
    frame["target_timestamp"] = target_ts.to_numpy()
    frame["absolute_target_timestamp"] = abs_target_ts.reindex(idx).to_numpy()
    frame["y_true"] = labels.reindex(idx).astype(int).to_numpy()
    frame["y_score"] = np.asarray(probability, dtype=float)
    frame["threshold"] = threshold
    frame["y_pred"] = (frame["y_score"] >= threshold).astype(int)
    frame["split"] = split
    frame["frozen_candidate_id"] = candidate["frozen_candidate_id"]
    frame["source_candidate_id"] = candidate["source_candidate_id"]
    frame["freeze_role"] = candidate["freeze_role"]
    frame["model_family"] = candidate["model_family"]
    frame["target_variant"] = candidate["target_variant"]
    frame["feature_group"] = candidate["feature_group"]
    frame["horizon"] = int(candidate["horizon"])
    frame["stock_forward_return"] = abs_return.reindex(idx).to_numpy(dtype=float)
    for col in [
        "risk_on_risk_off_state",
        "market_momentum_20",
        "market_volatility_5",
        "market_volatility_20",
        "index_agreement_score",
        "cross_index_agreement_score",
        "relative_strength_vs_market_lag1",
        "relative_strength_vs_market_20",
        "momentum_20",
        "return_1_lag_1",
    ]:
        frame[col] = pd.to_numeric(features.loc[idx, col], errors="coerce") if col in features.columns else np.nan
    return frame.dropna(subset=["target_timestamp", "stock_forward_return"]).reset_index(drop=True)


def strongest_baseline_for_candidate(
    candidate: dict[str, Any],
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
    split: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    labels = build_target_labels(features, index_data, int(candidate["horizon"]), str(candidate["target_variant"]))
    splits = split_indices(features, labels)
    _baseline_df, strongest, frames = baseline_frames_for_split(
        features,
        labels,
        splits,
        split,
        int(candidate["horizon"]),
        str(candidate["target_variant"]),
    )
    return strongest, frames[str(strongest["baseline_name"])]


def rolling_origin_tables(
    signal_frames: dict[str, pd.DataFrame],
    frozen: pd.DataFrame,
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    confirmation_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for candidate in frozen.to_dict("records"):
        cid = str(candidate["frozen_candidate_id"])
        for split in ["validation", "final"]:
            frame = signal_frames.get(f"{cid}::{split}", pd.DataFrame())
            if frame.empty:
                continue
            strongest, baseline_frame = strongest_baseline_for_candidate(candidate, features, index_data, split)
            baseline_frame = baseline_frame[["datetime", "ticker", "correct"]].rename(columns={"correct": "baseline_correct"})
            work = frame.merge(baseline_frame, on=["datetime", "ticker"], how="left")
            work["correct"] = (work["y_true"].astype(int) == work["y_pred"].astype(int)).astype(int)
            work["quarter"] = pd.to_datetime(work["datetime"]).dt.to_period("Q").astype(str)
            work["month"] = pd.to_datetime(work["datetime"]).dt.to_period("M").astype(str)
            for origin_type, col in [("quarter", "quarter"), ("month", "month")]:
                for origin, group in work.groupby(col, sort=True):
                    ticker_acc = group.groupby("ticker")["correct"].mean()
                    confirmation_rows.append(
                        {
                            "frozen_candidate_id": cid,
                            "split": split,
                            "origin_type": origin_type,
                            "origin_period": origin,
                            "model_family": candidate["model_family"],
                            "target_variant": candidate["target_variant"],
                            "feature_group": candidate["feature_group"],
                            "horizon": candidate["horizon"],
                            "threshold": candidate["threshold"],
                            "accuracy": float(group["correct"].mean()),
                            "lift": float(group["correct"].mean() - group["baseline_correct"].mean()),
                            "strongest_baseline": strongest["baseline_name"],
                            "baseline_accuracy": float(group["baseline_correct"].mean()),
                            "rows": int(len(group)),
                            "prediction_up_ratio": float(group["y_pred"].mean()),
                            "ticker_median_accuracy": float(ticker_acc.median()) if len(ticker_acc) else math.nan,
                        }
                    )
            ticker_acc = work.groupby("ticker")["correct"].mean()
            quarter_acc = work.groupby("quarter")["correct"].mean()
            month_acc = work.groupby("month")["correct"].mean()
            summary_rows.append(
                {
                    "frozen_candidate_id": cid,
                    "split": split,
                    "model_family": candidate["model_family"],
                    "target_variant": candidate["target_variant"],
                    "feature_group": candidate["feature_group"],
                    "horizon": candidate["horizon"],
                    "threshold": candidate["threshold"],
                    "accuracy": float(work["correct"].mean()),
                    "lift": float(work["correct"].mean() - work["baseline_correct"].mean()),
                    "strongest_baseline": strongest["baseline_name"],
                    "baseline_accuracy": float(work["baseline_correct"].mean()),
                    "rows": int(len(work)),
                    "prediction_up_ratio": float(work["y_pred"].mean()),
                    "ticker_median_accuracy": float(ticker_acc.median()) if len(ticker_acc) else math.nan,
                    "quarter_min_accuracy": float(quarter_acc.min()) if len(quarter_acc) else math.nan,
                    "month_min_accuracy": float(month_acc.min()) if len(month_acc) else math.nan,
                    "positive_quarters": int((work.groupby("quarter")["correct"].mean() > work.groupby("quarter")["baseline_correct"].mean()).sum()),
                    "positive_months": int((work.groupby("month")["correct"].mean() > work.groupby("month")["baseline_correct"].mean()).sum()),
                    "rolling_origin_robust": bool(
                        work["correct"].mean() > work["baseline_correct"].mean()
                        and len(work) >= 3000
                        and (quarter_acc.min() if len(quarter_acc) else 0.0) >= 0.45
                    ),
                }
            )
    return pd.DataFrame(confirmation_rows), pd.DataFrame(summary_rows)


def strategy_rank_score(rows: pd.DataFrame, template: str, threshold: float) -> pd.Series:
    score = pd.to_numeric(rows["y_score"], errors="coerce").fillna(0.0)
    if template == "relative_strength_rotation":
        rel = pd.to_numeric(rows.get("relative_strength_vs_market_20", rows.get("relative_strength_vs_market_lag1")), errors="coerce")
        rel = rel.fillna(pd.to_numeric(rows.get("relative_strength_vs_market_lag1", 0.0), errors="coerce")).fillna(0.0)
        return score + rel.rank(pct=True).fillna(0.0) * 0.05
    if template == "cash_when_uncertain":
        return score
    if template == "regime_gated_strategy":
        risk = pd.to_numeric(rows.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
        agreement = pd.to_numeric(rows.get("index_agreement_score", rows.get("cross_index_agreement_score", 0.5)), errors="coerce").fillna(0.5)
        return score + 0.025 * risk + 0.025 * (agreement - 0.5)
    return score


def eligible_rows(rows: pd.DataFrame, template: str, threshold: float) -> pd.DataFrame:
    work = rows.copy()
    score = pd.to_numeric(work["y_score"], errors="coerce")
    if template == "long_only_confidence":
        mask = score >= threshold
    elif template == "long_only_market_regime_filter":
        risk = pd.to_numeric(work.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
        momentum = pd.to_numeric(work.get("market_momentum_20", 0.0), errors="coerce").fillna(0.0)
        mask = (score >= threshold) & (risk >= 0.0) & (momentum >= 0.0)
    elif template == "relative_strength_rotation":
        rel = pd.to_numeric(work.get("relative_strength_vs_market_20", work.get("relative_strength_vs_market_lag1")), errors="coerce")
        rel = rel.fillna(pd.to_numeric(work.get("relative_strength_vs_market_lag1", 0.0), errors="coerce")).fillna(0.0)
        mask = (score >= threshold) & (rel >= rel.median())
    elif template == "cash_when_uncertain":
        mask = score >= min(0.99, threshold + 0.05)
    elif template == "regime_gated_strategy":
        risk = pd.to_numeric(work.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
        vol5 = pd.to_numeric(work.get("market_volatility_5", 0.0), errors="coerce")
        vol20 = pd.to_numeric(work.get("market_volatility_20", 0.0), errors="coerce").replace(0.0, np.nan)
        high_vol = (vol5 / vol20) > 1.25
        mask = (score >= threshold) & (risk >= -0.1) & (~high_vol.fillna(False))
    else:
        mask = score >= threshold
    out = work[mask].copy()
    if out.empty:
        return out
    out["rank_score"] = strategy_rank_score(out, template, threshold)
    return out.sort_values(["rank_score", "y_score"], ascending=[False, False])


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running = equity.cummax()
    dd = equity / running - 1.0
    return float(dd.min())


def metrics_from_equity(
    equity_curve: pd.DataFrame,
    trade_log: pd.DataFrame,
    exposure_values: list[float],
    turnover_notional: float,
) -> dict[str, Any]:
    if equity_curve.empty:
        equity = pd.Series([1.0])
        dates = pd.Series(pd.to_datetime([pd.Timestamp.utcnow()]))
    else:
        equity = pd.to_numeric(equity_curve["equity"], errors="coerce").ffill().fillna(1.0)
        dates = pd.to_datetime(equity_curve["datetime"], errors="coerce")
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    total_return = float(equity.iloc[-1] - 1.0)
    days = max(1.0, float((dates.max() - dates.min()).days)) if len(dates.dropna()) >= 2 else 1.0
    annualized = float(equity.iloc[-1] ** (365.0 / days) - 1.0) if equity.iloc[-1] > 0 else -1.0
    sharpe = float((returns.mean() / returns.std(ddof=0)) * math.sqrt(252.0)) if len(returns) > 1 and returns.std(ddof=0) > 0 else 0.0
    downside = returns[returns < 0.0]
    sortino = float((returns.mean() / downside.std(ddof=0)) * math.sqrt(252.0)) if len(downside) > 1 and downside.std(ddof=0) > 0 else 0.0
    mdd = max_drawdown(equity)
    calmar = float(annualized / abs(mdd)) if mdd < 0 else 0.0
    if trade_log.empty:
        win_rate = 0.0
        profit_factor = 0.0
        average_trade_return = 0.0
        trade_count = 0
    else:
        tr = pd.to_numeric(trade_log["net_return"], errors="coerce").fillna(0.0)
        wins = tr[tr > 0.0]
        losses = tr[tr < 0.0]
        win_rate = float((tr > 0.0).mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else (math.inf if wins.sum() > 0 else 0.0)
        average_trade_return = float(tr.mean())
        trade_count = int(len(trade_log))
    return {
        "total_return": total_return,
        "annualized_return": annualized,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "average_trade_return": average_trade_return,
        "trade_count": trade_count,
        "exposure_ratio": float(np.mean(exposure_values)) if exposure_values else 0.0,
        "turnover": float(turnover_notional),
        "cost_adjusted_return": total_return,
    }


def simulate_strategy(
    signal_rows: pd.DataFrame,
    candidate_id: str,
    template: str,
    split: str,
    max_positions: int,
    cost_bps: float,
    slippage_bps: float,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if signal_rows.empty:
        equity_curve = pd.DataFrame([{"datetime": pd.NaT, "equity": 1.0, "cash": 1.0, "invested": 0.0, "open_positions": 0}])
        metrics = metrics_from_equity(equity_curve, pd.DataFrame(), [], 0.0)
        return metrics, equity_curve, pd.DataFrame()

    rows = signal_rows.copy()
    rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce")
    rows["target_timestamp"] = pd.to_datetime(rows["target_timestamp"], errors="coerce")
    rows = rows.dropna(subset=["datetime", "target_timestamp", "stock_forward_return"])
    threshold = float(rows["threshold"].iloc[0]) if "threshold" in rows.columns and len(rows) else 0.5
    cash = 1.0
    open_positions: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exposure_values: list[float] = []
    turnover_notional = 0.0
    round_trip_drag = 2.0 * (float(cost_bps) + float(slippage_bps)) / 10000.0

    for timestamp, group in rows.groupby("datetime", sort=True):
        still_open: list[dict[str, Any]] = []
        for pos in open_positions:
            if pos["exit_time"] <= timestamp:
                exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
                cash += exit_value
                turnover_notional += exit_value
                trade_rows.append({**pos, "exit_value": exit_value, "realized_at": timestamp})
            else:
                still_open.append(pos)
        open_positions = still_open

        open_tickers = {pos["ticker"] for pos in open_positions}
        slots = max(0, int(max_positions) - len(open_positions))
        if slots > 0 and cash > 1e-12:
            eligible = eligible_rows(group, template, threshold)
            if not eligible.empty:
                eligible = eligible[~eligible["ticker"].isin(open_tickers)].head(slots)
                if not eligible.empty:
                    current_equity = cash + sum(pos["entry_value"] for pos in open_positions)
                    slot_value = min(cash / len(eligible), current_equity / float(max_positions))
                    for _, row in eligible.iterrows():
                        if cash <= 1e-12:
                            break
                        entry_value = min(slot_value, cash)
                        if entry_value <= 1e-12:
                            continue
                        cash -= entry_value
                        turnover_notional += entry_value
                        raw_return = float(row["stock_forward_return"])
                        net_return = raw_return - round_trip_drag
                        open_positions.append(
                            {
                                "strategy_id": "",
                                "candidate_id": candidate_id,
                                "template": template,
                                "split": split,
                                "ticker": row["ticker"],
                                "entry_time": timestamp,
                                "exit_time": row["target_timestamp"],
                                "entry_value": entry_value,
                                "raw_return": raw_return,
                                "net_return": net_return,
                                "score": row.get("y_score", math.nan),
                                "threshold": threshold,
                                "cost_bps": cost_bps,
                                "slippage_bps": slippage_bps,
                            }
                        )

        invested = sum(pos["entry_value"] for pos in open_positions)
        equity = cash + invested
        exposure_values.append(invested / equity if equity > 0 else 0.0)
        curve_rows.append({"datetime": timestamp, "equity": equity, "cash": cash, "invested": invested, "open_positions": len(open_positions)})

    for pos in sorted(open_positions, key=lambda item: item["exit_time"]):
        exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
        cash += exit_value
        turnover_notional += exit_value
        trade_rows.append({**pos, "exit_value": exit_value, "realized_at": pos["exit_time"]})
        curve_rows.append({"datetime": pos["exit_time"], "equity": cash, "cash": cash, "invested": 0.0, "open_positions": 0})

    equity_curve = pd.DataFrame(curve_rows).sort_values("datetime").reset_index(drop=True)
    trade_log = pd.DataFrame(trade_rows)
    metrics = metrics_from_equity(equity_curve, trade_log, exposure_values, turnover_notional)
    return metrics, equity_curve, trade_log


def baseline_buy_hold_index(index_data: dict[str, pd.DataFrame], split: str, cost_bps: float, slippage_bps: float) -> dict[str, Any]:
    frame = index_data.get("VN30", pd.DataFrame()).copy()
    if frame.empty:
        return {"baseline_name": "buy_and_hold_vn30_index", "split": split, "total_return": 0.0}
    frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    start = VAL_START if split == "validation" else FINAL_START
    end = VAL_END if split == "validation" else frame["datetime"].max()
    work = frame[frame["datetime"].between(start, end)].dropna(subset=["datetime", "close"]).sort_values("datetime")
    if len(work) < 2:
        total = 0.0
    else:
        total = float(work["close"].iloc[-1] / work["close"].iloc[0] - 1.0 - 2.0 * (cost_bps + slippage_bps) / 10000.0)
    return {"baseline_name": "buy_and_hold_vn30_index", "split": split, "total_return": total}


def baseline_equal_weight_basket(features: pd.DataFrame, split: str, cost_bps: float, slippage_bps: float) -> dict[str, Any]:
    start = VAL_START if split == "validation" else FINAL_START
    end = VAL_END if split == "validation" else features["datetime"].max()
    work = features[pd.to_datetime(features["datetime"]).between(start, end)].copy()
    returns: list[float] = []
    for _ticker, group in work.groupby("ticker", sort=True):
        group = group.sort_values("datetime")
        if len(group) >= 2:
            returns.append(float(group["close"].iloc[-1] / group["close"].iloc[0] - 1.0))
    total = float(np.mean(returns) - 2.0 * (cost_bps + slippage_bps) / 10000.0) if returns else 0.0
    return {"baseline_name": "equal_weight_vn30_stock_basket", "split": split, "total_return": total}


def build_baseline_signal_rows(features: pd.DataFrame, index_data: dict[str, pd.DataFrame], split: str, baseline_name: str) -> pd.DataFrame:
    horizon = 40
    labels = build_target_labels(features, index_data, horizon, "absolute_direction")
    splits = split_indices(features, labels)
    idx = splits[split]
    abs_return, abs_target_ts = stock_forward_return_and_timestamp(features, horizon)
    rows = features.loc[idx, ["datetime", "ticker", "close", "feature_timestamp"]].copy()
    rows["target_timestamp"] = abs_target_ts.reindex(idx).to_numpy()
    rows["stock_forward_return"] = abs_return.reindex(idx).to_numpy(dtype=float)
    if baseline_name == "simple_momentum":
        rows["y_score"] = pd.to_numeric(features.loc[idx, "momentum_20"], errors="coerce").rank(pct=True).to_numpy()
    elif baseline_name == "simple_relative_strength":
        col = "relative_strength_vs_market_20" if "relative_strength_vs_market_20" in features.columns else "relative_strength_vs_market_lag1"
        rows["y_score"] = pd.to_numeric(features.loc[idx, col], errors="coerce").rank(pct=True).to_numpy()
    elif baseline_name == "random_signal_same_turnover":
        rng = np.random.default_rng(SEED)
        rows["y_score"] = rng.random(len(rows))
    else:
        rows["y_score"] = 1.0
    rows["threshold"] = -math.inf
    rows["y_pred"] = 1
    rows["split"] = split
    rows["frozen_candidate_id"] = baseline_name
    rows["source_candidate_id"] = baseline_name
    rows["freeze_role"] = "baseline_strategy"
    rows["model_family"] = baseline_name
    rows["target_variant"] = "absolute_direction"
    rows["feature_group"] = "baseline_features"
    rows["horizon"] = horizon
    for col in [
        "risk_on_risk_off_state",
        "market_momentum_20",
        "market_volatility_5",
        "market_volatility_20",
        "index_agreement_score",
        "cross_index_agreement_score",
        "relative_strength_vs_market_lag1",
        "relative_strength_vs_market_20",
        "momentum_20",
        "return_1_lag_1",
    ]:
        rows[col] = pd.to_numeric(features.loc[idx, col], errors="coerce") if col in features.columns else np.nan
    return rows.dropna(subset=["target_timestamp", "stock_forward_return"]).reset_index(drop=True)


def run_strategy_grid(
    signal_frames: dict[str, pd.DataFrame],
    frozen: pd.DataFrame,
    features: pd.DataFrame,
    index_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    grid_rows: list[dict[str, Any]] = []
    validation_results: list[dict[str, Any]] = []
    final_results: list[dict[str, Any]] = []
    equity_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    baseline_rows: list[dict[str, Any]] = []

    strategy_seq = 0
    for split in ["validation", "final"]:
        baseline_signal_cache = {
            name: build_baseline_signal_rows(features, index_data, split, name)
            for name in ["simple_momentum", "simple_relative_strength", "random_signal_same_turnover", "always_up_directional_baseline"]
        }
        for cost_bps in COST_BPS:
            for slippage_bps in SLIPPAGE_BPS:
                baseline_rows.append({"baseline_name": "cash_no_trade", "split": split, "cost_bps": cost_bps, "slippage_bps": slippage_bps, "total_return": 0.0})
                baseline_rows.append({**baseline_buy_hold_index(index_data, split, cost_bps, slippage_bps), "cost_bps": cost_bps, "slippage_bps": slippage_bps})
                baseline_rows.append({**baseline_equal_weight_basket(features, split, cost_bps, slippage_bps), "cost_bps": cost_bps, "slippage_bps": slippage_bps})
                for max_positions in MAX_POSITIONS:
                    for baseline_name, signal_rows in baseline_signal_cache.items():
                        metrics, _curve, _trades = simulate_strategy(signal_rows, baseline_name, "long_only_confidence", split, max_positions, cost_bps, slippage_bps)
                        baseline_rows.append(
                            {
                                "baseline_name": baseline_name,
                                "split": split,
                                "max_positions": max_positions,
                                "cost_bps": cost_bps,
                                "slippage_bps": slippage_bps,
                                **metrics,
                            }
                        )

        for candidate in frozen.to_dict("records"):
            cid = str(candidate["frozen_candidate_id"])
            signal_rows = signal_frames.get(f"{cid}::{split}", pd.DataFrame())
            if signal_rows.empty:
                continue
            for template in STRATEGY_TEMPLATES:
                for max_positions in MAX_POSITIONS:
                    for cost_bps in COST_BPS:
                        for slippage_bps in SLIPPAGE_BPS:
                            strategy_seq += 1
                            strategy_id = f"v4strat_{strategy_seq:06d}"
                            grid_rows.append(
                                {
                                    "strategy_id": strategy_id,
                                    "split": split,
                                    "frozen_candidate_id": cid,
                                    "template": template,
                                    "max_positions": max_positions,
                                    "cost_bps": cost_bps,
                                    "slippage_bps": slippage_bps,
                                    "horizon": candidate["horizon"],
                                    "threshold": candidate["threshold"],
                                }
                            )
                            metrics, curve, trades = simulate_strategy(signal_rows, cid, template, split, max_positions, cost_bps, slippage_bps)
                            result = {
                                "strategy_id": strategy_id,
                                "split": split,
                                "frozen_candidate_id": cid,
                                "source_candidate_id": candidate["source_candidate_id"],
                                "template": template,
                                "model_family": candidate["model_family"],
                                "target_variant": candidate["target_variant"],
                                "feature_group": candidate["feature_group"],
                                "horizon": candidate["horizon"],
                                "threshold": candidate["threshold"],
                                "max_positions": max_positions,
                                "cost_bps": cost_bps,
                                "slippage_bps": slippage_bps,
                                **metrics,
                            }
                            result["claim_label"] = (
                                "strategy_positive_after_cost"
                                if result["total_return"] > 0 and (cost_bps + slippage_bps) > 0
                                else "diagnostic_only"
                            )
                            if split == "validation":
                                validation_results.append(result)
                            else:
                                final_results.append(result)
                                curve = curve.copy()
                                curve.insert(0, "strategy_id", strategy_id)
                                equity_rows.append(curve)
                                if not trades.empty:
                                    trades = trades.copy()
                                    trades["strategy_id"] = strategy_id
                                    trade_rows.append(trades)

    validation_df = pd.DataFrame(validation_results)
    final_df = pd.DataFrame(final_results)
    baseline_df = pd.DataFrame(baseline_rows)
    if not final_df.empty and not baseline_df.empty:
        best_baseline = (
            baseline_df[baseline_df["split"].eq("final")]
            .groupby(["cost_bps", "slippage_bps"], dropna=False)["total_return"]
            .max()
            .reset_index()
            .rename(columns={"total_return": "best_baseline_total_return"})
        )
        final_df = final_df.merge(best_baseline, on=["cost_bps", "slippage_bps"], how="left")
        final_df["baseline_delta"] = final_df["total_return"] - final_df["best_baseline_total_return"]
        final_df.loc[final_df["baseline_delta"] > 0, "claim_label"] = "strategy_outperforms_baseline"
    if not validation_df.empty and not baseline_df.empty:
        best_baseline = (
            baseline_df[baseline_df["split"].eq("validation")]
            .groupby(["cost_bps", "slippage_bps"], dropna=False)["total_return"]
            .max()
            .reset_index()
            .rename(columns={"total_return": "best_baseline_total_return"})
        )
        validation_df = validation_df.merge(best_baseline, on=["cost_bps", "slippage_bps"], how="left")
        validation_df["baseline_delta"] = validation_df["total_return"] - validation_df["best_baseline_total_return"]
    equity_df = pd.concat(equity_rows, ignore_index=True) if equity_rows else pd.DataFrame()
    trade_df = pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame()
    exposure = final_df[
        [
            "strategy_id",
            "frozen_candidate_id",
            "template",
            "max_positions",
            "cost_bps",
            "slippage_bps",
            "exposure_ratio",
            "turnover",
            "trade_count",
        ]
    ].copy() if not final_df.empty else pd.DataFrame()
    drawdown = final_df[
        [
            "strategy_id",
            "frozen_candidate_id",
            "template",
            "max_positions",
            "cost_bps",
            "slippage_bps",
            "max_drawdown",
            "calmar",
            "total_return",
        ]
    ].copy() if not final_df.empty else pd.DataFrame()
    cost_sensitivity = (
        final_df.groupby(["frozen_candidate_id", "template", "max_positions", "cost_bps", "slippage_bps"], dropna=False)
        .agg(total_return=("total_return", "max"), sharpe=("sharpe", "max"), max_drawdown=("max_drawdown", "min"), trade_count=("trade_count", "max"))
        .reset_index()
        if not final_df.empty
        else pd.DataFrame()
    )
    return (
        pd.DataFrame(grid_rows),
        validation_df,
        final_df,
        equity_df,
        trade_df,
        baseline_df,
        cost_sensitivity,
        exposure,
        drawdown,
    )


def write_reports(
    cluster_summary: pd.DataFrame,
    rolling_summary: pd.DataFrame,
    final_strategy: pd.DataFrame,
    baseline_strategy: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> None:
    candidate_6551 = cluster_summary[
        cluster_summary["representative_candidate_id"].astype(str).eq("forced_v3_calibrated_compact_h40__t0p540")
    ]
    if candidate_6551.empty:
        candidate_6551 = cluster_summary[
            cluster_summary["cluster_id"].astype(str).str.contains("calibrated_logistic__absolute_direction__compact_stable_features__40", regex=False)
        ]
    cluster_verdict = candidate_6551.iloc[0]["cluster_verdict"] if not candidate_6551.empty else "not_found"

    robust = rolling_summary[rolling_summary["rolling_origin_robust"].astype(bool)] if not rolling_summary.empty else pd.DataFrame()
    rolling_pool = robust if not robust.empty else rolling_summary
    best_rolling = rolling_pool.sort_values(["lift", "accuracy"], ascending=[False, False]).iloc[0].to_dict() if not rolling_pool.empty else {}
    after_cost = final_strategy[(final_strategy["cost_bps"] > 0) | (final_strategy["slippage_bps"] > 0)].copy() if not final_strategy.empty else pd.DataFrame()
    best_after_cost = after_cost.sort_values(["total_return", "baseline_delta", "sharpe"], ascending=[False, False, False]).iloc[0].to_dict() if not after_cost.empty else {}

    final_baselines = baseline_strategy[baseline_strategy["split"].eq("final")].copy() if not baseline_strategy.empty else pd.DataFrame()
    buy_hold = final_baselines[final_baselines["baseline_name"].eq("buy_and_hold_vn30_index")]["total_return"].max() if not final_baselines.empty else math.nan
    equal_weight = final_baselines[final_baselines["baseline_name"].eq("equal_weight_vn30_stock_basket")]["total_return"].max() if not final_baselines.empty else math.nan
    random_signal = final_baselines[final_baselines["baseline_name"].eq("random_signal_same_turnover")]["total_return"].max() if not final_baselines.empty else math.nan
    best_return = as_float(best_after_cost.get("total_return"))
    beats_buy_hold = bool(math.isfinite(best_return) and math.isfinite(as_float(buy_hold)) and best_return > as_float(buy_hold))
    beats_equal_weight = bool(math.isfinite(best_return) and math.isfinite(as_float(equal_weight)) and best_return > as_float(equal_weight))
    beats_random = bool(math.isfinite(best_return) and math.isfinite(as_float(random_signal)) and best_return > as_float(random_signal))
    cost_range = (
        cost_sensitivity.groupby(["cost_bps", "slippage_bps"])["total_return"].max().reset_index().sort_values("total_return", ascending=False)
        if not cost_sensitivity.empty
        else pd.DataFrame()
    )
    best_cost = cost_range.iloc[0].to_dict() if not cost_range.empty else {}
    worst_cost = cost_range.iloc[-1].to_dict() if not cost_range.empty else {}

    protocol = f"""# VN30 V4 Promotion Queue And Real Strategy Protocol

## Scope

- VN30 stock hourly diagnostic benchmark only.
- V3 promotion queue is mined before any simulation.
- Frozen candidates are evaluated without changing model parameters, features, horizons, or thresholds.
- Index data is used only as lagged market-context features or baseline VN30 index buy-and-hold comparison.
- Strategy results are offline diagnostics only.
- Out of scope: trading, profitability claim, BUY/SELL, recommendation, live deployment, DOCX, push, merge, tag, VN100.

## Split Discipline

- Train rows require feature_timestamp <= `{TRAIN_END}` and target_timestamp <= `{TRAIN_END}`.
- Validation rows require feature_timestamp and target_timestamp from `{VAL_START}` through `{VAL_END}`.
- Final rows require feature_timestamp and target_timestamp >= `{FINAL_START}`.
- Final-ranked v3 candidates remain exploratory unless re-locked or future-blind confirmed.

## Strategy Assumptions

- Initial capital is normalized to 1.0.
- No leverage and no shorting.
- Max positions: {MAX_POSITIONS}.
- Equal-weight entries, fixed-horizon exits, cash earns 0.
- Transaction cost bps: {COST_BPS}.
- Slippage bps: {SLIPPAGE_BPS}.
- Templates: {", ".join(STRATEGY_TEMPLATES)}.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    result = f"""# VN30 V4 Promotion Queue And Real Strategy Result Summary

## Promotion Queue Verdict

- 65.51 candidate cluster verdict: {cluster_verdict}.
- Stable clusters found: {int(cluster_summary["stable_cluster"].sum()) if not cluster_summary.empty else 0}.
- Isolated one-off clusters found: {int(cluster_summary["isolated_one_off"].sum()) if not cluster_summary.empty else 0}.

## Rolling-Origin Confirmation

- Best rolling-origin candidate: `{best_rolling.get("frozen_candidate_id", "")}`.
- Best rolling-origin split: {best_rolling.get("split", "")}.
- Accuracy/lift: {pct(best_rolling.get("accuracy", math.nan))} / {pp(best_rolling.get("lift", math.nan))}.
- Rolling-origin robust candidates: {len(robust)}.

## Strategy Diagnostics

- Best after-cost strategy: `{best_after_cost.get("strategy_id", "")}`.
- Candidate/template: `{best_after_cost.get("frozen_candidate_id", "")}` / {best_after_cost.get("template", "")}.
- Cost/slippage/max positions: {best_after_cost.get("cost_bps", "")} bps / {best_after_cost.get("slippage_bps", "")} bps / {best_after_cost.get("max_positions", "")}.
- Total return: {pct(best_after_cost.get("total_return", math.nan))}.
- Sharpe: {best_after_cost.get("sharpe", "")}.
- Max drawdown: {pct(best_after_cost.get("max_drawdown", math.nan))}.
- Exposure/turnover/trades: {best_after_cost.get("exposure_ratio", "")} / {best_after_cost.get("turnover", "")} / {best_after_cost.get("trade_count", "")}.

## Required Answers

1. 65.51 candidate stable cluster or one-off: {cluster_verdict}.
2. Any frozen candidate show rolling-origin robustness: {str(len(robust) > 0).lower()}.
3. Best real strategy after costs: `{best_after_cost.get("strategy_id", "")}`.
4. Any strategy beat buy-and-hold VN30: {str(beats_buy_hold).lower()}.
5. Any strategy beat equal-weight VN30: {str(beats_equal_weight).lower()}.
6. Any strategy beat random same-turnover: {str(beats_random).lower()}.
7. Cost/slippage sensitivity: best grid {best_cost}; worst grid {worst_cost}.
8. Any result becomes claimable: false.
9. Exploratory results: promotion queue clusters, frozen final-ranked candidates, rolling-origin confirmations, and all strategy diagnostics.
10. Paper-safe claim boundary: offline diagnostic only; no trading, profitability, recommendation, live deployment, or claimable champion replacement is made.

Paper-safe wording:

> VN30 V4 mined the v3 promotion queue and evaluated frozen VN30 hourly candidate signals in offline fixed-horizon long-only strategy diagnostics with explicit cost and slippage assumptions. The current 61.61% strict-replay directional champion is not replaced. Final-ranked and strategy results remain exploratory diagnostics requiring re-lock or future-blind confirmation before any claim.
"""
    write_markdown(RESULT_PATH, result)

    claim = f"""# VN30 V4 Promotion Queue And Real Strategy Claim Boundary

- Claimable scope: VN30 stock hourly diagnostic benchmark only.
- Current claimable directional champion remains the 61.61% L2 Logistic baseline60_candidate.
- The 65.51% candidate remains exploratory and requires re-lock or future-blind confirmation.
- Strategy simulations are offline diagnostic simulations only.
- Best after-cost strategy is not a trading, profitability, BUY/SELL, recommendation, or deployment claim.
- Index data is used only as lagged market-context features or explicit VN30 index baseline comparison.
- No index result is claimed as stock directional accuracy.
- No result becomes claimable in V4.
- Claim labels are diagnostic_only, exploratory_not_claimable, strategy_positive_after_cost, strategy_outperforms_baseline, future_blind_required, or not_claimable as applicable.
- No DOCX, paper, push, merge, tag, live deployment, or VN100 claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    promotion = read_csv(V3_DIR / "promotion_candidate_queue.csv")
    exploratory = read_csv(V3_DIR / "exploratory_final_leaderboard.csv")
    champion_comparison = read_csv(V3_DIR / "champion_comparison.csv")

    cluster_summary, top_accuracy, top_lift, threshold_stability, horizon_stability = mine_promotion_queue(promotion, exploratory)
    frozen = build_frozen_candidate_set(promotion, exploratory, cluster_summary)

    write_frame(OUTPUT_DIR / "promotion_queue_cluster_summary.csv", cluster_summary)
    write_frame(OUTPUT_DIR / "top_accuracy_candidates.csv", top_accuracy)
    write_frame(OUTPUT_DIR / "top_lift_candidates.csv", top_lift)
    write_frame(OUTPUT_DIR / "threshold_neighborhood_stability.csv", threshold_stability)
    write_frame(OUTPUT_DIR / "horizon_neighborhood_stability.csv", horizon_stability)
    write_frame(OUTPUT_DIR / "frozen_candidate_set.csv", frozen)

    print("Building features and frozen candidate signals...", flush=True)
    features, feature_groups, _index_audit, _index_manifest, feature_manifest = build_v3_feature_frame()
    index_data = load_index_data()
    signal_frames: dict[str, pd.DataFrame] = {}
    signal_failures: list[dict[str, Any]] = []
    for candidate in frozen.to_dict("records"):
        cid = str(candidate["frozen_candidate_id"])
        try:
            payload = fit_candidate_payload(candidate, features, feature_groups, index_data)
            for split in ["validation", "final"]:
                signal_frames[f"{cid}::{split}"] = build_signal_rows(payload, features, index_data, split)
        except Exception as exc:
            signal_failures.append({"frozen_candidate_id": cid, "failure_reason": str(exc)[:500], "status": "failed"})
    if signal_failures:
        write_frame(OUTPUT_DIR / "frozen_signal_failures.csv", pd.DataFrame(signal_failures))

    rolling_confirmation, rolling_summary = rolling_origin_tables(signal_frames, frozen, features, index_data)
    write_frame(OUTPUT_DIR / "rolling_origin_confirmation.csv", rolling_confirmation)
    write_frame(OUTPUT_DIR / "rolling_origin_summary.csv", rolling_summary)

    print("Running offline strategy diagnostics...", flush=True)
    (
        strategy_grid,
        validation_strategy,
        final_strategy,
        final_equity,
        final_trade_log,
        baseline_strategy,
        cost_sensitivity,
        exposure_summary,
        drawdown_summary,
    ) = run_strategy_grid(signal_frames, frozen, features, index_data)

    write_frame(OUTPUT_DIR / "strategy_grid.csv", strategy_grid)
    write_frame(OUTPUT_DIR / "validation_strategy_results.csv", validation_strategy)
    write_frame(OUTPUT_DIR / "final_strategy_results.csv", final_strategy)
    write_frame(OUTPUT_DIR / "final_equity_curves.csv", final_equity)
    write_frame(OUTPUT_DIR / "final_trade_log.csv", final_trade_log)
    write_frame(OUTPUT_DIR / "baseline_strategy_comparison.csv", baseline_strategy)
    write_frame(OUTPUT_DIR / "cost_slippage_sensitivity.csv", cost_sensitivity)
    write_frame(OUTPUT_DIR / "exposure_turnover_summary.csv", exposure_summary)
    write_frame(OUTPUT_DIR / "drawdown_summary.csv", drawdown_summary)

    manifest = {
        "created_at_utc": now_utc(),
        "scope": "VN30 V4 promotion queue mining and offline fixed-horizon long-only strategy diagnostics",
        "v3_inputs": {
            "promotion_candidate_queue": rel(V3_DIR / "promotion_candidate_queue.csv"),
            "exploratory_final_leaderboard": rel(V3_DIR / "exploratory_final_leaderboard.csv"),
            "champion_comparison": rel(V3_DIR / "champion_comparison.csv"),
        },
        "split_rules": {
            "train_end": str(TRAIN_END),
            "validation_start": str(VAL_START),
            "validation_end": str(VAL_END),
            "final_start": str(FINAL_START),
        },
        "strategy_templates": STRATEGY_TEMPLATES,
        "max_positions": MAX_POSITIONS,
        "cost_bps": COST_BPS,
        "slippage_bps": SLIPPAGE_BPS,
        "frozen_candidate_count": int(len(frozen)),
        "signal_failure_count": int(len(signal_failures)),
        "cluster_count": int(len(cluster_summary)),
        "stable_cluster_count": int(cluster_summary["stable_cluster"].sum()) if not cluster_summary.empty else 0,
        "strategy_grid_rows": int(len(strategy_grid)),
        "validation_strategy_rows": int(len(validation_strategy)),
        "final_strategy_rows": int(len(final_strategy)),
        "feature_manifest": feature_manifest,
        "champion_comparison_rows_loaded": int(len(champion_comparison)),
        "final_ranked_candidates_claimable": False,
        "strategy_results_claimable": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
    }
    write_json(OUTPUT_DIR / "strategy_manifest.json", manifest)
    write_reports(cluster_summary, rolling_summary, final_strategy, baseline_strategy, cost_sensitivity)

    best_strategy = final_strategy[(final_strategy["cost_bps"] > 0) | (final_strategy["slippage_bps"] > 0)].sort_values(
        ["total_return", "baseline_delta", "sharpe"],
        ascending=[False, False, False],
    ).head(1)
    best_id = best_strategy.iloc[0]["strategy_id"] if not best_strategy.empty else ""
    print(f"VN30 V4 strategy diagnostics complete: {rel(OUTPUT_DIR)}", flush=True)
    print(f"Frozen candidates: {len(frozen)}", flush=True)
    print(f"Stable clusters: {manifest['stable_cluster_count']}", flush=True)
    print(f"Best after-cost strategy: {best_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
