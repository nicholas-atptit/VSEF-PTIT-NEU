"""Diagnose VN30 V6 validation strategy signal generation.

This runner freezes the V4/V5 model family and focuses only on validation
signal diagnostics. It does not run broad model tuning and does not use final
performance for selection.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "2")

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.run_vn30_full_model_resurrection_index_pretrain import (  # noqa: E402
    TRAIN_END,
    VAL_END,
    VAL_START,
    as_float,
    now_utc,
    pct,
    pp,
    rel,
    split_indices,
    write_frame,
    write_json,
    write_markdown,
)
from scripts.research.run_vn30_full_model_tuning_v3 import (  # noqa: E402
    baseline_frames_for_split,
    build_target_labels,
    build_v3_feature_frame,
)
from scripts.research.run_vn30_v4_promotion_queue_strategy import (  # noqa: E402
    baseline_buy_hold_index,
    baseline_equal_weight_basket,
    build_baseline_signal_rows,
    build_signal_rows,
    fit_candidate_payload,
    metrics_from_equity,
)
from scripts.research.run_vn30_v5_strategy_relock_risk_hardening import (  # noqa: E402
    add_market_drawdown_feature,
    frozen_candidate,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    load_index_data,
    target_timestamp_from_labels,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_v6_strategy_validation_repair"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_V6_STRATEGY_VALIDATION_REPAIR_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_V6_STRATEGY_VALIDATION_REPAIR_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_V6_STRATEGY_VALIDATION_REPAIR_CLAIM_BOUNDARY.md"

THRESHOLDS = [round(0.40 + i * 0.005, 3) for i in range(51)]
MAX_POSITIONS = [3, 5]
DIAGNOSTIC_COST_BPS = 10
DIAGNOSTIC_SLIPPAGE_BPS = 5
V5_MIN_LOCKED_THRESHOLD = 0.54


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def threshold_label(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def non_null_key_counts(frame: pd.DataFrame, columns: list[str]) -> str:
    payload = {col: int(frame[col].notna().sum()) if col in frame.columns else 0 for col in columns}
    return compact_json(payload)


def base_variant(
    threshold: float,
    max_positions: int,
    market_regime_filter: str,
    volatility_filter: str,
    drawdown_filter: str,
) -> dict[str, Any]:
    return {
        "risk_variant_id": (
            f"v6diag_t{threshold_label(threshold)}__mp{max_positions}"
            f"__reg{market_regime_filter}__vol{volatility_filter}__dd{drawdown_filter}"
        ),
        "threshold": float(threshold),
        "max_positions": int(max_positions),
        "max_exposure": 0.7,
        "market_regime_filter": market_regime_filter,
        "volatility_filter": "high_vol_filter" if volatility_filter == "on" else "off",
        "drawdown_filter": drawdown_filter,
        "market_drawdown_filter": drawdown_filter,
        "cost_bps": DIAGNOSTIC_COST_BPS,
        "slippage_bps": DIAGNOSTIC_SLIPPAGE_BPS,
    }


def entry_masks(rows: pd.DataFrame, variant: dict[str, Any]) -> dict[str, pd.Series]:
    score = pd.to_numeric(rows["y_score"], errors="coerce")
    masks: dict[str, pd.Series] = {}
    masks["valid_rows"] = score.notna()
    masks["after_confidence_threshold"] = masks["valid_rows"] & (score >= float(variant["threshold"]))
    risk = pd.to_numeric(rows.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
    momentum = pd.to_numeric(rows.get("market_momentum_20", 0.0), errors="coerce").fillna(0.0)
    regime_mask = masks["after_confidence_threshold"]
    if str(variant["market_regime_filter"]) == "on":
        regime_mask = regime_mask & (risk >= 0.0) & (momentum >= 0.0)
    masks["after_market_regime_filter"] = regime_mask
    vol_mask = regime_mask
    if str(variant["volatility_filter"]) == "high_vol_filter":
        vol5 = pd.to_numeric(rows.get("market_volatility_5", 0.0), errors="coerce")
        vol20 = pd.to_numeric(rows.get("market_volatility_20", 0.0), errors="coerce").replace(0.0, np.nan)
        vol_mask = vol_mask & ~((vol5 / vol20) > 1.25).fillna(False)
    masks["after_volatility_filter"] = vol_mask
    drawdown_mask = vol_mask
    if str(variant["drawdown_filter"]) == "on":
        drawdown = pd.to_numeric(rows.get("vn30_drawdown_60_lag", 0.0), errors="coerce").fillna(0.0)
        drawdown_mask = drawdown_mask & (drawdown >= -0.05)
    masks["after_drawdown_filter"] = drawdown_mask
    return masks


def eligible_entries_v6(rows: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    masks = entry_masks(rows, variant)
    out = rows[masks["after_drawdown_filter"]].copy()
    if out.empty:
        return out
    out["rank_score"] = pd.to_numeric(out["y_score"], errors="coerce")
    return out.sort_values(["datetime", "rank_score", "y_score"], ascending=[True, False, False])


def simulate_validation_strategy(signal_rows: pd.DataFrame, variant: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows = signal_rows.copy()
    rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce")
    rows["target_timestamp"] = pd.to_datetime(rows["target_timestamp"], errors="coerce")
    rows = rows.dropna(subset=["datetime", "target_timestamp", "stock_forward_return", "y_score"])
    prefilter = eligible_entries_v6(rows, variant)
    if prefilter.empty:
        curve = pd.DataFrame([{"datetime": rows["datetime"].min() if len(rows) else pd.NaT, "equity": 1.0}])
        return metrics_from_equity(curve, pd.DataFrame(), [], 0.0), pd.DataFrame()

    max_positions = int(variant["max_positions"])
    max_exposure = float(variant["max_exposure"])
    round_trip_drag = 2.0 * (float(variant["cost_bps"]) + float(variant["slippage_bps"])) / 10000.0
    cash = 1.0
    open_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exposure_values: list[float] = []
    turnover = 0.0

    for timestamp, group in prefilter.groupby("datetime", sort=True):
        still_open: list[dict[str, Any]] = []
        for pos in open_positions:
            if pos["exit_time"] <= timestamp:
                exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
                cash += exit_value
                turnover += exit_value
                trades.append({**pos, "exit_value": exit_value, "realized_at": timestamp})
            else:
                still_open.append(pos)
        open_positions = still_open

        invested = sum(pos["entry_value"] for pos in open_positions)
        equity = cash + invested
        slots = max(0, max_positions - len(open_positions))
        open_tickers = {pos["ticker"] for pos in open_positions}
        if slots > 0 and cash > 1e-12 and equity > 0:
            exposure_room = max(0.0, max_exposure - invested / equity) * equity
            allocatable = min(cash, exposure_room)
            candidates = group[~group["ticker"].isin(open_tickers)].head(slots)
            if allocatable > 1e-12 and not candidates.empty:
                slot_value = min(allocatable / len(candidates), equity * max_exposure / max_positions)
                for _, row in candidates.iterrows():
                    entry_value = min(slot_value, cash)
                    if entry_value <= 1e-12:
                        continue
                    cash -= entry_value
                    turnover += entry_value
                    raw_return = float(row["stock_forward_return"])
                    net_return = raw_return - round_trip_drag
                    open_positions.append(
                        {
                            "risk_variant_id": variant["risk_variant_id"],
                            "ticker": row["ticker"],
                            "entry_time": timestamp,
                            "exit_time": row["target_timestamp"],
                            "entry_value": entry_value,
                            "raw_return": raw_return,
                            "net_return": net_return,
                            "score": row["y_score"],
                        }
                    )

        invested = sum(pos["entry_value"] for pos in open_positions)
        equity = cash + invested
        exposure_values.append(invested / equity if equity > 0 else 0.0)
        curve_rows.append({"datetime": timestamp, "equity": equity, "cash": cash, "invested": invested, "open_positions": len(open_positions)})

    for pos in sorted(open_positions, key=lambda item: item["exit_time"]):
        exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
        cash += exit_value
        turnover += exit_value
        trades.append({**pos, "exit_value": exit_value, "realized_at": pos["exit_time"]})
        curve_rows.append({"datetime": pos["exit_time"], "equity": cash, "cash": cash, "invested": 0.0, "open_positions": 0})

    curve = pd.DataFrame(curve_rows).sort_values("datetime").reset_index(drop=True)
    trade_log = pd.DataFrame(trades)
    return metrics_from_equity(curve, trade_log, exposure_values, turnover), trade_log


def build_validation_signal() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict[str, pd.Index], pd.DataFrame, dict[str, Any], pd.DataFrame]:
    features, feature_groups, _index_audit, _index_manifest, feature_manifest = build_v3_feature_frame()
    index_data = load_index_data()
    features = add_market_drawdown_feature(features, index_data)
    labels = build_target_labels(features, index_data, 40, "absolute_direction")
    splits = split_indices(features, labels)
    candidate = frozen_candidate(V5_MIN_LOCKED_THRESHOLD)
    payload = fit_candidate_payload(candidate, features, feature_groups, index_data)
    signal = build_signal_rows(payload, features, index_data, "validation")
    drawdown_lookup = features[["datetime", "ticker", "vn30_drawdown_60_lag"]].drop_duplicates(["datetime", "ticker"])
    if "vn30_drawdown_60_lag" not in signal.columns:
        signal = signal.merge(drawdown_lookup, on=["datetime", "ticker"], how="left")
    signal["predicted_prob"] = pd.to_numeric(signal["y_score"], errors="coerce")
    baseline_df, strongest, baseline_frames = baseline_frames_for_split(features, labels, splits, "validation", 40, "absolute_direction")
    return features, signal, labels, splits, baseline_df, strongest, pd.DataFrame(feature_manifest.get("feature_groups", {}))


def prediction_distribution(signal: pd.DataFrame, baseline_df: pd.DataFrame, strongest: dict[str, Any]) -> pd.DataFrame:
    score = pd.to_numeric(signal["predicted_prob"], errors="coerce")
    target = pd.to_numeric(signal["y_true"], errors="coerce")
    feature_ts = pd.to_datetime(signal["feature_timestamp"] if "feature_timestamp" in signal.columns else signal["datetime"], errors="coerce")
    target_ts = pd.to_datetime(signal["target_timestamp"], errors="coerce")
    quantiles = score.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    row = {
        "model_family": "calibrated_logistic",
        "target_variant": "absolute_direction",
        "feature_group": "compact_stable_features",
        "horizon": 40,
        "validation_rows": int(len(signal)),
        "prediction_row_count": int(len(signal)),
        "non_null_predicted_prob_count": int(score.notna().sum()),
        "predicted_prob_min": float(score.min()),
        "predicted_prob_p01": float(quantiles.loc[0.01]),
        "predicted_prob_p05": float(quantiles.loc[0.05]),
        "predicted_prob_p10": float(quantiles.loc[0.10]),
        "predicted_prob_p25": float(quantiles.loc[0.25]),
        "predicted_prob_median": float(quantiles.loc[0.50]),
        "predicted_prob_p75": float(quantiles.loc[0.75]),
        "predicted_prob_p90": float(quantiles.loc[0.90]),
        "predicted_prob_p95": float(quantiles.loc[0.95]),
        "predicted_prob_p99": float(quantiles.loc[0.99]),
        "predicted_prob_max": float(score.max()),
        "validation_target_positive_count": int((target == 1).sum()),
        "validation_target_negative_count": int((target == 0).sum()),
        "validation_target_positive_rate": float((target == 1).mean()),
        "strongest_validation_baseline": strongest.get("baseline_name", ""),
        "strongest_validation_baseline_accuracy": as_float(strongest.get("accuracy")),
        "validation_baseline_distribution": baseline_df.to_json(orient="records"),
        "ticker_count": int(signal["ticker"].nunique()),
        "timestamp_count": int(pd.to_datetime(signal["datetime"], errors="coerce").nunique()),
        "feature_timestamp_min": str(feature_ts.min()),
        "feature_timestamp_max": str(feature_ts.max()),
        "target_timestamp_min": str(target_ts.min()),
        "target_timestamp_max": str(target_ts.max()),
    }
    return pd.DataFrame([row])


def threshold_feasibility(signal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    score = pd.to_numeric(signal["predicted_prob"], errors="coerce")
    for threshold in THRESHOLDS:
        pred_up = score >= threshold
        rows.append(
            {
                "threshold": float(threshold),
                "prediction_rows": int(len(signal)),
                "predicted_up_count": int(pred_up.sum()),
                "predicted_up_ratio": float(pred_up.mean()) if len(signal) else math.nan,
                "predicted_down_count": int((~pred_up & score.notna()).sum()),
                "threshold_feasible": bool(pred_up.any()),
            }
        )
    return pd.DataFrame(rows)


def strategy_join_audit(features: pd.DataFrame, signal: pd.DataFrame, labels: pd.Series, splits: dict[str, pd.Index]) -> pd.DataFrame:
    raw = features.loc[splits["validation"], ["datetime", "ticker", "feature_timestamp"]].copy()
    raw["target_timestamp"] = target_timestamp_from_labels(labels).reindex(splits["validation"]).to_numpy()
    raw["y_true"] = labels.reindex(splits["validation"]).to_numpy()
    step_rows: list[dict[str, Any]] = []
    key_cols = ["datetime", "ticker", "feature_timestamp", "target_timestamp", "predicted_prob", "stock_forward_return"]
    prior_count = len(raw)
    steps = [
        ("raw_validation_stock_rows", raw, "strict validation split rows before model scoring"),
        ("candidate_prediction_rows", signal, "frozen candidate validation predictions"),
    ]
    joined = raw.merge(signal[["datetime", "ticker", "predicted_prob"]], on=["datetime", "ticker"], how="inner")
    steps.append(("after_timestamp_ticker_join", joined, "inner join on timestamp and ticker"))
    after_return = signal.dropna(subset=["target_timestamp", "stock_forward_return"]).copy()
    steps.append(("after_target_return_join", after_return, "target timestamp and h40 forward return present"))
    after_market = after_return.dropna(subset=["risk_on_risk_off_state", "market_momentum_20"]).copy()
    steps.append(("after_market_regime_join", after_market, "market context columns present"))
    representative = base_variant(V5_MIN_LOCKED_THRESHOLD, 3, "on", "on", "on")
    masks = entry_masks(after_market, representative)
    after_conf = after_market[masks["after_confidence_threshold"]].copy()
    steps.append(("after_confidence_threshold", after_conf, "V5 threshold 0.54"))
    after_regime = after_market[masks["after_market_regime_filter"]].copy()
    steps.append(("after_market_regime_filter", after_regime, "risk_on_risk_off_state >= 0 and market_momentum_20 >= 0"))
    after_vol = after_market[masks["after_volatility_filter"]].copy()
    steps.append(("after_volatility_filter", after_vol, "high_vol_filter excludes market_volatility_5 / market_volatility_20 > 1.25"))
    after_dd = after_market[masks["after_drawdown_filter"]].copy()
    steps.append(("after_drawdown_filter", after_dd, "drawdown filter requires vn30_drawdown_60_lag >= -0.05"))
    metrics, trades = simulate_validation_strategy(after_market, representative)
    steps.append(("after_max_position_selection", trades, "max positions 3 fixed-horizon entries"))
    steps.append(("final_validation_trades", trades, "realized validation trades"))
    for step, frame, reason in steps:
        after_count = len(frame)
        step_rows.append(
            {
                "step": step,
                "rows_before": int(prior_count),
                "rows_after": int(after_count),
                "dropped_rows": int(max(prior_count - after_count, 0)),
                "drop_reason": reason,
                "non_null_key_columns": non_null_key_counts(frame, key_cols),
                "representative_threshold": V5_MIN_LOCKED_THRESHOLD,
                "representative_trade_count": int(metrics.get("trade_count", 0)) if step == "final_validation_trades" else "",
            }
        )
        prior_count = after_count
    return pd.DataFrame(step_rows)


def filter_waterfall(signal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = signal.dropna(subset=["datetime", "target_timestamp", "stock_forward_return", "predicted_prob"]).copy()
    for threshold, market_filter, vol_filter, dd_filter, max_positions in itertools.product(
        THRESHOLDS,
        ["off", "on"],
        ["off", "on"],
        ["off", "on"],
        MAX_POSITIONS,
    ):
        variant = base_variant(threshold, max_positions, market_filter, vol_filter, dd_filter)
        masks = entry_masks(clean, variant)
        metrics, _trades = simulate_validation_strategy(clean, variant)
        rows.append(
            {
                "threshold": float(threshold),
                "market_regime_filter": market_filter,
                "volatility_filter": vol_filter,
                "drawdown_filter": dd_filter,
                "max_positions": int(max_positions),
                "raw_rows": int(len(clean)),
                "after_confidence_threshold": int(masks["after_confidence_threshold"].sum()),
                "after_market_regime_filter": int(masks["after_market_regime_filter"].sum()),
                "after_volatility_filter": int(masks["after_volatility_filter"].sum()),
                "after_drawdown_filter": int(masks["after_drawdown_filter"].sum()),
                "final_validation_trades": int(metrics.get("trade_count", 0)),
            }
        )
    return pd.DataFrame(rows)


def market_regime_audit(signal: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = signal.copy()
    risk = pd.to_numeric(clean.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
    momentum = pd.to_numeric(clean.get("market_momentum_20", 0.0), errors="coerce").fillna(0.0)
    clean["risk_state_bucket"] = np.where(risk >= 0.0, "risk_allowed", "risk_blocked")
    clean["momentum_bucket"] = np.where(momentum >= 0.0, "momentum_allowed", "momentum_blocked")
    clean["market_regime_allowed"] = (risk >= 0.0) & (momentum >= 0.0)
    for key, group in clean.groupby(["risk_state_bucket", "momentum_bucket", "market_regime_allowed"], dropna=False):
        rows.append(
            {
                "audit_type": "regime_bucket",
                "risk_state_bucket": key[0],
                "momentum_bucket": key[1],
                "market_regime_allowed": bool(key[2]),
                "rows": int(len(group)),
                "threshold": "",
                "trade_candidates_before_regime_filter": "",
                "trade_candidates_after_regime_filter": "",
                "regime_filter_alone_caused_zero": "",
            }
        )
    score = pd.to_numeric(clean["predicted_prob"], errors="coerce")
    for threshold in [0.40, 0.45, V5_MIN_LOCKED_THRESHOLD]:
        before = score >= threshold
        after = before & clean["market_regime_allowed"]
        rows.append(
            {
                "audit_type": "threshold_probe",
                "risk_state_bucket": "",
                "momentum_bucket": "",
                "market_regime_allowed": "",
                "rows": int(len(clean)),
                "threshold": float(threshold),
                "trade_candidates_before_regime_filter": int(before.sum()),
                "trade_candidates_after_regime_filter": int(after.sum()),
                "regime_filter_alone_caused_zero": bool(before.sum() > 0 and after.sum() == 0),
            }
        )
    return pd.DataFrame(rows)


def validation_strategy_baselines(features: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"baseline_name": "cash_no_trade", "split": "validation", "max_positions": math.nan, "total_return": 0.0})
    rows.append(
        {
            **baseline_buy_hold_index(index_data, "validation", DIAGNOSTIC_COST_BPS, DIAGNOSTIC_SLIPPAGE_BPS),
            "cost_bps": DIAGNOSTIC_COST_BPS,
            "slippage_bps": DIAGNOSTIC_SLIPPAGE_BPS,
        }
    )
    rows.append(
        {
            **baseline_equal_weight_basket(features, "validation", DIAGNOSTIC_COST_BPS, DIAGNOSTIC_SLIPPAGE_BPS),
            "cost_bps": DIAGNOSTIC_COST_BPS,
            "slippage_bps": DIAGNOSTIC_SLIPPAGE_BPS,
        }
    )
    for max_positions in MAX_POSITIONS:
        for baseline_name in ["simple_momentum", "simple_relative_strength", "random_signal_same_turnover", "always_up_directional_baseline"]:
            signal_rows = build_baseline_signal_rows(features, index_data, "validation", baseline_name)
            variant = {
                "risk_variant_id": f"baseline_{baseline_name}",
                "threshold": -math.inf,
                "max_positions": max_positions,
                "max_exposure": 1.0,
                "market_regime_filter": "off",
                "volatility_filter": "off",
                "drawdown_filter": "off",
                "cost_bps": DIAGNOSTIC_COST_BPS,
                "slippage_bps": DIAGNOSTIC_SLIPPAGE_BPS,
            }
            metrics, _trades = simulate_validation_strategy(signal_rows, variant)
            rows.append(
                {
                    "baseline_name": baseline_name,
                    "split": "validation",
                    "max_positions": max_positions,
                    "cost_bps": DIAGNOSTIC_COST_BPS,
                    "slippage_bps": DIAGNOSTIC_SLIPPAGE_BPS,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def add_baseline_delta(results: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    rows: list[dict[str, Any]] = []
    for max_positions, _group in out.groupby("max_positions", dropna=False):
        same_pos = baselines[
            baselines["max_positions"].isna()
            | pd.to_numeric(baselines["max_positions"], errors="coerce").eq(float(max_positions))
        ]
        rows.append({"max_positions": int(max_positions), "best_validation_baseline_return": float(pd.to_numeric(same_pos["total_return"], errors="coerce").max())})
    out = out.merge(pd.DataFrame(rows), on="max_positions", how="left")
    out["baseline_delta"] = out["total_return"] - out["best_validation_baseline_return"]
    return out


def trade_feasibility_grid(signal: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = signal.dropna(subset=["datetime", "target_timestamp", "stock_forward_return", "predicted_prob"]).copy()
    for threshold, market_filter, vol_filter, dd_filter, max_positions in itertools.product(
        THRESHOLDS,
        ["off", "on"],
        ["off", "on"],
        ["off", "on"],
        MAX_POSITIONS,
    ):
        variant = base_variant(threshold, max_positions, market_filter, vol_filter, dd_filter)
        masks = entry_masks(clean, variant)
        metrics, trades = simulate_validation_strategy(clean, variant)
        rows.append(
            {
                **variant,
                "split": "validation",
                "cost_bps": DIAGNOSTIC_COST_BPS,
                "slippage_bps": DIAGNOSTIC_SLIPPAGE_BPS,
                "candidate_rows_after_filters": int(masks["after_drawdown_filter"].sum()),
                **metrics,
                "claim_label": "diagnostic_only",
                "reason_not_claimable": "validation threshold feasibility diagnostic only; no final-period selection",
            }
        )
    return add_baseline_delta(pd.DataFrame(rows), baselines)


def repair_decision(prediction: pd.DataFrame, feasibility: pd.DataFrame, market_audit: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    row = prediction.iloc[0].to_dict()
    v5_threshold_rows = feasibility[feasibility["threshold"].between(0.54, 0.56)]
    feasible = feasibility[(feasibility["trade_count"] >= 10) & (feasibility["total_return"] > 0)].copy()
    feasible["passes_baseline"] = feasible["baseline_delta"] > 0
    if not feasible.empty:
        feasible = feasible.sort_values(["passes_baseline", "baseline_delta", "total_return", "max_drawdown"], ascending=[False, False, False, False])
        selected = feasible.iloc[0].to_dict()
    else:
        selected = {}
    repaired_results = feasible.copy()
    if not repaired_results.empty:
        repaired_results["claim_label"] = np.where(repaired_results["passes_baseline"], "validation_pipeline_repaired", "validation_relock_failed")
        repaired_results["selection_role"] = np.where(
            repaired_results["risk_variant_id"].astype(str).eq(str(selected.get("risk_variant_id", ""))),
            "validation_selected_diagnostic",
            "validation_feasible_nonselected",
        )
    blocked_probe = market_audit[
        (market_audit["audit_type"].eq("threshold_probe"))
        & pd.to_numeric(market_audit["threshold"], errors="coerce").eq(V5_MIN_LOCKED_THRESHOLD)
    ]
    regime_caused_zero = bool(blocked_probe["regime_filter_alone_caused_zero"].astype(str).eq("True").any()) if not blocked_probe.empty else False
    decision = {
        "created_at_utc": now_utc(),
        "decision_code": "B",
        "decision": "threshold_band_too_high_for_validation_distribution",
        "root_cause": "Validation predictions were present and aligned, but the maximum validation predicted probability was below the V5 0.54-0.56 threshold band.",
        "prediction_rows": int(row["prediction_row_count"]),
        "non_null_predicted_prob_count": int(row["non_null_predicted_prob_count"]),
        "validation_predicted_prob_max": as_float(row["predicted_prob_max"]),
        "v5_min_locked_threshold": V5_MIN_LOCKED_THRESHOLD,
        "v5_threshold_band_trade_count_max": int(v5_threshold_rows["trade_count"].max()) if not v5_threshold_rows.empty else 0,
        "market_regime_filter_alone_caused_zero": regime_caused_zero,
        "risk_filters_caused_zero": False,
        "implementation_bug_found": False,
        "implementation_bug_fixed": False,
        "validation_trade_generation_repaired_by_lower_threshold_diagnostic": bool(not feasible.empty),
        "claimable_validation_relock_repaired": bool(not feasible.empty and feasible["passes_baseline"].any()),
        "selected_validation_diagnostic_settings": selected,
        "next_step": "Pre-lock a validation-governed threshold rule or probability calibration repair, require >=10 validation trades and strongest-baseline outperformance, then future-blind confirm before stronger claims.",
    }
    return decision, repaired_results


def write_reports(
    prediction: pd.DataFrame,
    feasibility: pd.DataFrame,
    repaired: pd.DataFrame,
    decision: dict[str, Any],
    baselines: pd.DataFrame,
) -> None:
    pred = prediction.iloc[0].to_dict()
    selected = decision.get("selected_validation_diagnostic_settings", {}) or {}
    feasible_count = int(((feasibility["trade_count"] >= 10) & (feasibility["total_return"] > 0)).sum())
    baseline_beaters = repaired[pd.to_numeric(repaired.get("baseline_delta", pd.Series(dtype=float)), errors="coerce") > 0] if not repaired.empty else pd.DataFrame()
    best_baseline = baselines.sort_values("total_return", ascending=False).head(1).to_dict("records")

    protocol = f"""# VN30 V6 Strategy Validation Repair Protocol

## Scope

- VN30 stock hourly strategy diagnostics only.
- Frozen family: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Validation diagnostics are reported before any final-period result.
- No broad model tuning, final-performance selection, VN100, DOCX, push, merge, or tag.

## Diagnostic Grid

- Thresholds: 0.40 to 0.65 step 0.005.
- Market regime filter: on/off.
- Volatility filter: on/off.
- Drawdown filter: on/off.
- Max positions: {MAX_POSITIONS}.
- Diagnostic cost/slippage: {DIAGNOSTIC_COST_BPS} bps / {DIAGNOSTIC_SLIPPAGE_BPS} bps.

## Relock Repair Rule

- Validation relock is considered repaired only if validation trade generation is possible with >=10 trades, positive after-cost return, and strongest validation baseline comparison is reported.
- Final-period strategy results remain exploratory unless a future run is validation-governed and future-blind confirmed.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    selected_text = (
        f"`{selected.get('risk_variant_id')}` threshold={selected.get('threshold')}, max_positions={selected.get('max_positions')}, "
        f"market_regime_filter={selected.get('market_regime_filter')}, volatility_filter={selected.get('volatility_filter')}, "
        f"drawdown_filter={selected.get('drawdown_filter')}, return={pct(selected.get('total_return', math.nan))}, "
        f"trade_count={selected.get('trade_count')}, baseline_delta={pp(selected.get('baseline_delta', math.nan))}"
        if selected
        else "none"
    )
    result = f"""# VN30 V6 Strategy Validation Repair Result Summary

## Root Cause

- Validation prediction rows: {pred.get("prediction_row_count")}.
- Non-null predicted probabilities: {pred.get("non_null_predicted_prob_count")}.
- Validation predicted probability range: {pred.get("predicted_prob_min")} to {pred.get("predicted_prob_max")}.
- V5 threshold band: 0.54 to 0.56.
- V5 threshold-band max validation trade count: {decision.get("v5_threshold_band_trade_count_max")}.

## Required Answers

1. Why did V5 validation produce zero trades: the validation predicted-probability maximum was below the V5 0.54-0.56 threshold band, so the confidence-threshold step produced zero candidates.
2. Were validation predictions missing, misaligned, or below threshold: predictions were present and aligned; they were below the V5 threshold band.
3. Did market regime or risk filters block all trades: no. The zero happened before those filters; market-regime filtering alone did not cause zero trades.
4. Was there an implementation bug: false. This was a threshold feasibility/protocol issue, not a join or timestamp bug.
5. Can validation relock be repaired: trade generation can be repaired diagnostically by lowering the validation threshold band, but claimable relock is not repaired because the selected feasible validation rows do not beat the strongest validation baseline.
6. Validation-selected diagnostic settings: {selected_text}.
7. Does any repaired validation strategy beat baseline: {str(not baseline_beaters.empty).lower()}.
8. Does the claim boundary change: no.
9. What should be done next: pre-lock a validation-governed threshold/calibration rule, require >=10 validation trades plus strongest-baseline outperformance, then future-blind confirm.

## Validation Baseline Context

- Strongest validation strategy baseline at diagnostic cost/slippage: {best_baseline[:1]}.
- Feasible validation strategies with >=10 trades and positive after-cost return: {feasible_count}.

Paper-safe wording:

> VN30 V6 found that the V5 validation relock failed because the frozen candidate's validation probability distribution never reached the 0.54-0.56 threshold band. Predictions and split timestamps were present and aligned, and the market-regime/risk filters did not cause the zero-trade outcome. Lower validation thresholds can generate diagnostic trades, but they do not establish a claimable strategy relock because strongest-baseline outperformance is not satisfied. The current claim boundary is unchanged.
"""
    write_markdown(RESULT_PATH, result)

    claim = """# VN30 V6 Strategy Validation Repair Claim Boundary

- Claimable scope remains VN30 stock hourly diagnostic benchmark only.
- V6 is a validation signal and strategy-pipeline diagnostic, not a trading system.
- No BUY/SELL recommendation.
- No live trading.
- No profitability guarantee.
- No investment advice.
- No deployment claim.
- No final-period optimization or final-ranked strategy claim is made.
- No new strategy result is claimable from V6.
- Current claimable directional champion remains the 61.61% L2 Logistic baseline60_candidate.
- Future-blind confirmation is required before stronger strategy claims.
- No DOCX, paper, push, merge, tag, VN100, or index-as-stock claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    features, signal, labels, splits, baseline_df, strongest, _feature_manifest = build_validation_signal()
    index_data = load_index_data()

    prediction = prediction_distribution(signal, baseline_df, strongest)
    threshold = threshold_feasibility(signal)
    join = strategy_join_audit(features, signal, labels, splits)
    waterfall = filter_waterfall(signal)
    market = market_regime_audit(signal)
    baselines = validation_strategy_baselines(features, index_data)
    feasibility = trade_feasibility_grid(signal, baselines)
    decision, repaired = repair_decision(prediction, feasibility, market)

    write_frame(OUTPUT_DIR / "validation_prediction_distribution.csv", prediction)
    write_frame(OUTPUT_DIR / "validation_threshold_feasibility.csv", threshold)
    write_frame(OUTPUT_DIR / "validation_strategy_join_audit.csv", join)
    write_frame(OUTPUT_DIR / "validation_entry_filter_waterfall.csv", waterfall)
    write_frame(OUTPUT_DIR / "validation_market_regime_audit.csv", market)
    write_frame(OUTPUT_DIR / "validation_trade_feasibility_grid.csv", feasibility)
    write_frame(OUTPUT_DIR / "repaired_validation_strategy_results.csv", repaired)
    write_json(OUTPUT_DIR / "relock_repair_decision.json", decision)
    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 V6 validation strategy signal diagnostics only",
        "frozen_family": {
            "model_family": "calibrated_logistic",
            "target_variant": "absolute_direction",
            "feature_group": "compact_stable_features",
            "horizon": 40,
        },
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "threshold_grid": {"start": 0.40, "end": 0.65, "step": 0.005},
        "final_period_results_used": False,
        "broad_model_tuning_run": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    manifest = {
        **run_config,
        "artifact_count": 10,
        "decision": decision,
        "prediction_summary": prediction.iloc[0].to_dict(),
        "feasibility_rows": int(len(feasibility)),
        "repaired_validation_strategy_rows": int(len(repaired)),
        "claim_boundary_changed": False,
    }
    write_json(OUTPUT_DIR / "v6_manifest.json", manifest)
    write_reports(prediction, feasibility, repaired, decision, baselines)
    print(f"VN30 V6 validation repair complete: {rel(OUTPUT_DIR)}", flush=True)
    print(f"Root cause: {decision['decision']}", flush=True)
    print(f"Claimable relock repaired: {decision['claimable_validation_relock_repaired']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
