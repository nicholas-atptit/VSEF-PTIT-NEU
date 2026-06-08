"""Run VN30 V5 strategy relock and risk-hardening diagnostics.

This runner freezes the V4 strategy family and does not perform broad model
tuning. Validation results drive the relock audit. Final-period strategy
rankings remain exploratory diagnostics.
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
    FINAL_START,
    TRAIN_END,
    VAL_END,
    VAL_START,
    as_float,
    json_safe,
    now_utc,
    pct,
    pp,
    rel,
    write_frame,
    write_json,
    write_markdown,
)
from scripts.research.run_vn30_v4_promotion_queue_strategy import (  # noqa: E402
    baseline_buy_hold_index,
    baseline_equal_weight_basket,
    build_baseline_signal_rows,
    build_signal_rows,
    fit_candidate_payload,
    max_drawdown,
    metrics_from_equity,
)
from scripts.research.run_vn30_full_model_tuning_v3 import (  # noqa: E402
    build_v3_feature_frame,
)
from scripts.research.vn30_hourly_dual_track_common import (  # noqa: E402
    REPO_ROOT,
    load_index_data,
)

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_v5_strategy_relock"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_V5_STRATEGY_RELOCK_RISK_HARDENING_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_V5_STRATEGY_RELOCK_RISK_HARDENING_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_V5_STRATEGY_RELOCK_RISK_HARDENING_CLAIM_BOUNDARY.md"

SEED = 42
THRESHOLDS = [0.54, 0.545, 0.55, 0.555, 0.56]
MAX_POSITIONS = [3, 5]
MAX_EXPOSURE = [0.5, 0.7, 1.0]
VOLATILITY_FILTERS = ["off", "high_vol_filter"]
MARKET_DRAWDOWN_FILTERS = ["off", "on"]
STOP_LOSS = [None, -0.03, -0.05, -0.07]
TAKE_PROFIT = [None, 0.05, 0.08, 0.10]
COOLDOWN = [0, 1, 2]
COST_BPS = [0, 5, 10, 20, 30]
SLIPPAGE_BPS = [0, 5, 10, 20]
V4_DRAWDOWN = -0.38416040635827775
CACHE_SCHEMA_VERSION = "v5_cost_family_expanded_20260525"

FROZEN_MODEL_PARAMS = {
    "model_family": "calibrated_logistic",
    "penalty": "l2",
    "C": 0.003,
    "class_weight": None,
    "solver": "liblinear",
    "calibration": "sigmoid_cv3",
}


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":"))


def stop_label(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def take_label(value: float | None) -> str:
    return "none" if value is None else f"{value:.2f}"


def add_market_drawdown_feature(features: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = features.copy()
    idx = index_data.get("VN30", pd.DataFrame()).copy()
    if idx.empty:
        out["vn30_drawdown_60_lag"] = np.nan
        return out
    idx["datetime"] = pd.to_datetime(idx["datetime"], errors="coerce")
    idx["close"] = pd.to_numeric(idx["close"], errors="coerce")
    idx = idx.dropna(subset=["datetime", "close"]).sort_values("datetime")
    rolling_high = idx["close"].rolling(60, min_periods=20).max()
    idx["vn30_drawdown_60_lag"] = (idx["close"] / rolling_high - 1.0).shift(1)
    query = pd.DataFrame({"_row": out.index, "datetime": pd.to_datetime(out["datetime"], errors="coerce")}).dropna(subset=["datetime"]).sort_values("datetime")
    merged = pd.merge_asof(query, idx[["datetime", "vn30_drawdown_60_lag"]].dropna().sort_values("datetime"), on="datetime", direction="backward")
    out["vn30_drawdown_60_lag"] = merged.set_index("_row")["vn30_drawdown_60_lag"].reindex(out.index).astype(float)
    return out


def frozen_candidate(threshold: float) -> dict[str, Any]:
    return {
        "frozen_candidate_id": f"v5_calibrated_compact_h40__t{threshold:.3f}".replace(".", "p"),
        "source_candidate_id": "forced_v3_calibrated_compact_h40__t0p540",
        "freeze_role": "v5_frozen_family_threshold_candidate",
        "model_family": "calibrated_logistic",
        "model_params": compact_json(FROZEN_MODEL_PARAMS),
        "target_variant": "absolute_direction",
        "feature_group": "compact_stable_features",
        "horizon": 40,
        "threshold": float(threshold),
        "claim_label": "exploratory_not_claimable",
    }


def build_threshold_signals(features: pd.DataFrame, feature_groups: dict[str, list[str]], index_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    base_candidate = frozen_candidate(0.54)
    payload = fit_candidate_payload(base_candidate, features, feature_groups, index_data)
    drawdown_lookup = pd.DataFrame()
    if "vn30_drawdown_60_lag" in features.columns:
        drawdown_lookup = features[["datetime", "ticker", "vn30_drawdown_60_lag"]].drop_duplicates(["datetime", "ticker"]).copy()
    signals: dict[str, pd.DataFrame] = {}
    for split in ["validation", "final"]:
        base = build_signal_rows(payload, features, index_data, split)
        if not drawdown_lookup.empty and "vn30_drawdown_60_lag" not in base.columns:
            base = base.merge(drawdown_lookup, on=["datetime", "ticker"], how="left")
        for threshold in THRESHOLDS:
            cid = frozen_candidate(threshold)["frozen_candidate_id"]
            frame = base.copy()
            frame["threshold"] = float(threshold)
            frame["y_pred"] = (pd.to_numeric(frame["y_score"], errors="coerce") >= float(threshold)).astype(int)
            frame["frozen_candidate_id"] = cid
            frame["source_candidate_id"] = base_candidate["source_candidate_id"]
            frame["horizon"] = 40
            signals[f"{cid}::{split}"] = frame
    return signals


def risk_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = 0
    for threshold, max_positions, max_exposure, vol_filter, dd_filter, stop_loss, take_profit, cooldown, cost_bps, slippage_bps in itertools.product(
        THRESHOLDS,
        MAX_POSITIONS,
        MAX_EXPOSURE,
        VOLATILITY_FILTERS,
        MARKET_DRAWDOWN_FILTERS,
        STOP_LOSS,
        TAKE_PROFIT,
        COOLDOWN,
        COST_BPS,
        SLIPPAGE_BPS,
    ):
        seq += 1
        rows.append(
            {
                "risk_variant_id": f"v5risk_{seq:05d}",
                "model_family": "calibrated_logistic",
                "target_variant": "absolute_direction",
                "feature_group": "compact_stable_features",
                "horizon": 40,
                "threshold": float(threshold),
                "strategy_template": "long_only_market_regime_filter",
                "max_positions": int(max_positions),
                "max_exposure": float(max_exposure),
                "volatility_filter": vol_filter,
                "market_drawdown_filter": dd_filter,
                "stop_loss_proxy": stop_label(stop_loss),
                "take_profit_proxy": take_label(take_profit),
                "cooldown_after_loss": int(cooldown),
                "cost_bps": int(cost_bps),
                "slippage_bps": int(slippage_bps),
                "selection_stage": "validation_risk_hardening_grid",
            }
        )
    return pd.DataFrame(rows)


def parse_optional_float(value: Any) -> float | None:
    text = str(value)
    if text == "none" or text.strip() == "":
        return None
    return float(text)


def eligible_entries(rows: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    threshold = float(variant["threshold"])
    score = pd.to_numeric(rows["y_score"], errors="coerce")
    risk = pd.to_numeric(rows.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
    momentum = pd.to_numeric(rows.get("market_momentum_20", 0.0), errors="coerce").fillna(0.0)
    mask = (score >= threshold) & (risk >= 0.0) & (momentum >= 0.0)
    if str(variant["volatility_filter"]) == "high_vol_filter":
        vol5 = pd.to_numeric(rows.get("market_volatility_5", 0.0), errors="coerce")
        vol20 = pd.to_numeric(rows.get("market_volatility_20", 0.0), errors="coerce").replace(0.0, np.nan)
        mask &= ~((vol5 / vol20) > 1.25).fillna(False)
    if str(variant["market_drawdown_filter"]) == "on":
        drawdown = pd.to_numeric(rows.get("vn30_drawdown_60_lag", 0.0), errors="coerce").fillna(0.0)
        mask &= drawdown >= -0.05
    out = rows[mask].copy()
    if out.empty:
        return out
    out["rank_score"] = score.loc[out.index]
    return out.sort_values(["rank_score", "y_score"], ascending=[False, False])


def simulate_risk_variant(signal_rows: pd.DataFrame, variant: dict[str, Any], split: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    if signal_rows.empty:
        curve = pd.DataFrame([{"datetime": pd.NaT, "equity": 1.0, "cash": 1.0, "invested": 0.0, "open_positions": 0}])
        return metrics_from_equity(curve, pd.DataFrame(), [], 0.0), curve, pd.DataFrame()
    rows = signal_rows.copy()
    rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce")
    rows["target_timestamp"] = pd.to_datetime(rows["target_timestamp"], errors="coerce")
    rows = rows.dropna(subset=["datetime", "target_timestamp", "stock_forward_return"])
    prefilter = eligible_entries(rows, variant)
    if prefilter.empty:
        curve = pd.DataFrame([{"datetime": rows["datetime"].min() if len(rows) else pd.NaT, "equity": 1.0, "cash": 1.0, "invested": 0.0, "open_positions": 0}])
        return metrics_from_equity(curve, pd.DataFrame(), [], 0.0), curve, pd.DataFrame()

    max_positions = int(variant["max_positions"])
    max_exposure = float(variant["max_exposure"])
    cost_bps = float(variant["cost_bps"])
    slippage_bps = float(variant["slippage_bps"])
    stop_loss = parse_optional_float(variant["stop_loss_proxy"])
    take_profit = parse_optional_float(variant["take_profit_proxy"])
    cooldown_periods = int(variant["cooldown_after_loss"])
    round_trip_drag = 2.0 * (cost_bps + slippage_bps) / 10000.0
    cash = 1.0
    open_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exposure_values: list[float] = []
    turnover = 0.0
    cooldown_remaining = 0

    for timestamp, group in prefilter.groupby("datetime", sort=True):
        loss_realized = False
        still_open: list[dict[str, Any]] = []
        for pos in open_positions:
            if pos["exit_time"] <= timestamp:
                exit_value = pos["entry_value"] * (1.0 + pos["net_return"])
                cash += exit_value
                turnover += exit_value
                loss_realized = loss_realized or pos["net_return"] < 0.0
                trades.append({**pos, "exit_value": exit_value, "realized_at": timestamp})
            else:
                still_open.append(pos)
        open_positions = still_open
        if loss_realized and cooldown_periods > 0:
            cooldown_remaining = max(cooldown_remaining, cooldown_periods)

        invested = sum(pos["entry_value"] for pos in open_positions)
        equity = cash + invested
        if cooldown_remaining <= 0:
            slots = max(0, max_positions - len(open_positions))
            open_tickers = {pos["ticker"] for pos in open_positions}
            if slots > 0 and cash > 1e-12 and equity > 0:
                current_exposure = invested / equity
                exposure_room = max(0.0, max_exposure - current_exposure) * equity
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
                        capped_return = raw_return
                        if stop_loss is not None:
                            capped_return = max(capped_return, stop_loss)
                        if take_profit is not None:
                            capped_return = min(capped_return, take_profit)
                        net_return = capped_return - round_trip_drag
                        open_positions.append(
                            {
                                "risk_variant_id": variant["risk_variant_id"],
                                "candidate_id": row["frozen_candidate_id"],
                                "split": split,
                                "ticker": row["ticker"],
                                "entry_time": timestamp,
                                "exit_time": row["target_timestamp"],
                                "entry_value": entry_value,
                                "raw_return": raw_return,
                                "capped_return": capped_return,
                                "net_return": net_return,
                                "score": row.get("y_score", math.nan),
                                "threshold": variant["threshold"],
                                "max_positions": max_positions,
                                "max_exposure": max_exposure,
                                "volatility_filter": variant["volatility_filter"],
                                "market_drawdown_filter": variant["market_drawdown_filter"],
                                "stop_loss_proxy": variant["stop_loss_proxy"],
                                "take_profit_proxy": variant["take_profit_proxy"],
                                "cooldown_after_loss": cooldown_periods,
                                "cost_bps": cost_bps,
                                "slippage_bps": slippage_bps,
                            }
                        )
        else:
            cooldown_remaining -= 1
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
    metrics = metrics_from_equity(curve, trade_log, exposure_values, turnover)
    return metrics, curve, trade_log


def build_baselines(features: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["validation", "final"]:
        baseline_signal_cache = {
            name: build_baseline_signal_rows(features, index_data, split, name)
            for name in ["simple_momentum", "simple_relative_strength", "random_signal_same_turnover", "always_up_directional_baseline"]
        }
        for cost_bps, slippage_bps in itertools.product(COST_BPS, SLIPPAGE_BPS):
            rows.append({"baseline_name": "cash_no_trade", "split": split, "cost_bps": cost_bps, "slippage_bps": slippage_bps, "total_return": 0.0})
            rows.append({**baseline_buy_hold_index(index_data, split, cost_bps, slippage_bps), "cost_bps": cost_bps, "slippage_bps": slippage_bps})
            rows.append({**baseline_equal_weight_basket(features, split, cost_bps, slippage_bps), "cost_bps": cost_bps, "slippage_bps": slippage_bps})
            for max_positions in MAX_POSITIONS:
                for baseline_name, signal_rows in baseline_signal_cache.items():
                    variant = {
                        "risk_variant_id": f"baseline_{baseline_name}",
                        "threshold": -math.inf,
                        "max_positions": max_positions,
                        "max_exposure": 1.0,
                        "volatility_filter": "off",
                        "market_drawdown_filter": "off",
                        "stop_loss_proxy": "none",
                        "take_profit_proxy": "none",
                        "cooldown_after_loss": 0,
                        "cost_bps": cost_bps,
                        "slippage_bps": slippage_bps,
                    }
                    metrics, _curve, _trades = simulate_risk_variant(signal_rows, variant, split)
                    rows.append(
                        {
                            "baseline_name": baseline_name,
                            "split": split,
                            "max_positions": max_positions,
                            "cost_bps": cost_bps,
                            "slippage_bps": slippage_bps,
                            **metrics,
                        }
                    )
    return pd.DataFrame(rows)


def add_baseline_delta(results: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    out = results.copy()
    if out.empty:
        return out
    baseline_rows = []
    for (split, cost_bps, slippage_bps, max_positions), group in out.groupby(["split", "cost_bps", "slippage_bps", "max_positions"], dropna=False):
        subset = baselines[
            baselines["split"].eq(split)
            & pd.to_numeric(baselines["cost_bps"], errors="coerce").eq(float(cost_bps))
            & pd.to_numeric(baselines["slippage_bps"], errors="coerce").eq(float(slippage_bps))
        ].copy()
        same_pos = subset[pd.to_numeric(subset.get("max_positions", np.nan), errors="coerce").eq(float(max_positions))]
        subset = pd.concat([subset[subset.get("max_positions", pd.Series(index=subset.index)).isna()], same_pos], ignore_index=True, sort=False)
        best = float(pd.to_numeric(subset["total_return"], errors="coerce").max()) if not subset.empty else 0.0
        baseline_rows.append({"split": split, "cost_bps": cost_bps, "slippage_bps": slippage_bps, "max_positions": max_positions, "best_baseline_total_return": best})
    out = out.merge(pd.DataFrame(baseline_rows), on=["split", "cost_bps", "slippage_bps", "max_positions"], how="left")
    out["baseline_delta"] = out["total_return"] - out["best_baseline_total_return"]
    return out


def run_grid(
    grid: pd.DataFrame,
    signals: dict[str, pd.DataFrame],
    split: str,
    keep_curves: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    for row in grid.to_dict("records"):
        cid = frozen_candidate(float(row["threshold"]))["frozen_candidate_id"]
        signal_rows = signals[f"{cid}::{split}"]
        metrics, curve, trades = simulate_risk_variant(signal_rows, row, split)
        result = {
            **row,
            "split": split,
            "frozen_candidate_id": cid,
            "source_candidate_id": "forced_v3_calibrated_compact_h40__t0p540",
            **metrics,
            "claim_label": "diagnostic_only",
            "reason_not_claimable": "offline diagnostic; final-period strategy ranking is exploratory",
        }
        if result["total_return"] > 0 and (float(row["cost_bps"]) + float(row["slippage_bps"]) > 0):
            result["claim_label"] = "strategy_positive_after_cost"
        result_rows.append(result)
        if keep_curves:
            curve = curve.copy()
            curve.insert(0, "risk_variant_id", row["risk_variant_id"])
            curve.insert(1, "frozen_candidate_id", cid)
            curve_rows.append(curve)
            if not trades.empty:
                trades = trades.copy()
                trades["risk_variant_id"] = row["risk_variant_id"]
                trade_rows.append(trades)
    return (
        pd.DataFrame(result_rows),
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(),
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
    )


def select_locked_strategy(validation_results: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    audit = validation_results.copy()
    audit["passes_min_trade_count"] = audit["trade_count"].astype(float) >= 3
    audit["passes_positive_after_cost"] = audit["total_return"].astype(float) > 0
    audit["passes_baseline"] = audit["baseline_delta"].astype(float) > 0
    audit["selection_composite"] = (
        audit["total_return"].astype(float)
        + audit["sharpe"].astype(float) * 0.10
        + audit["baseline_delta"].astype(float) * 0.25
        + (audit["max_drawdown"].astype(float) + 1.0) * 0.10
        - audit["turnover"].astype(float) * 0.005
    )
    eligible = audit[audit["passes_min_trade_count"] & audit["passes_positive_after_cost"] & audit["passes_baseline"]].copy()
    if eligible.empty:
        locked = {
            "selection_status": "no_validation_trade_or_positive_baseline_variant",
            "selected_by": "predeclared_lowest_threshold_moderate_risk_fallback",
            "threshold": 0.54,
            "max_positions": 3,
            "max_exposure": 0.7,
            "volatility_filter": "high_vol_filter",
            "market_drawdown_filter": "on",
            "stop_loss_proxy": "-0.05",
            "take_profit_proxy": "0.08",
            "cooldown_after_loss": 1,
            "cost_bps": 10,
            "slippage_bps": 5,
            "claimable": False,
            "final_performance_used": False,
        }
    else:
        selected = eligible.sort_values(["selection_composite", "total_return", "max_drawdown"], ascending=[False, False, False]).iloc[0].to_dict()
        locked = {
            "selection_status": "validation_selected",
            "selected_by": "validation_only_composite",
            "threshold": as_float(selected["threshold"]),
            "max_positions": int(as_float(selected["max_positions"])),
            "max_exposure": as_float(selected["max_exposure"]),
            "volatility_filter": selected["volatility_filter"],
            "market_drawdown_filter": selected["market_drawdown_filter"],
            "stop_loss_proxy": selected["stop_loss_proxy"],
            "take_profit_proxy": selected["take_profit_proxy"],
            "cooldown_after_loss": int(as_float(selected["cooldown_after_loss"])),
            "cost_bps": int(as_float(selected["cost_bps"])),
            "slippage_bps": int(as_float(selected["slippage_bps"])),
            "claimable": False,
            "final_performance_used": False,
        }
    locked["locked_at_utc"] = now_utc()
    locked["model_family"] = "calibrated_logistic"
    locked["target_variant"] = "absolute_direction"
    locked["feature_group"] = "compact_stable_features"
    locked["horizon"] = 40
    locked["strategy_template"] = "long_only_market_regime_filter"
    return locked, audit


def variant_filter(grid: pd.DataFrame, selected: dict[str, Any]) -> pd.DataFrame:
    mask = pd.Series(True, index=grid.index)
    for col in ["threshold", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "cooldown_after_loss"]:
        mask &= pd.to_numeric(grid[col], errors="coerce").eq(float(selected[col]))
    for col in ["volatility_filter", "market_drawdown_filter", "stop_loss_proxy", "take_profit_proxy"]:
        mask &= grid[col].astype(str).eq(str(selected[col]))
    return grid[mask].copy()


def choose_final_grid(grid: pd.DataFrame, validation_results: pd.DataFrame, locked: dict[str, Any]) -> pd.DataFrame:
    locked_grid = variant_filter(grid, locked)
    ranked = validation_results.sort_values(["passes_min_trade_count", "selection_composite", "total_return"], ascending=[False, False, False]).head(50)
    top_ids = set(ranked["risk_variant_id"].astype(str))
    top_grid = grid[grid["risk_variant_id"].astype(str).isin(top_ids)].copy()
    stress = grid[
        grid["threshold"].isin([0.54, 0.545, 0.55])
        & grid["max_positions"].isin([3, 5])
        & grid["max_exposure"].isin([0.5, 0.7])
        & grid["volatility_filter"].eq("high_vol_filter")
        & grid["market_drawdown_filter"].eq("on")
        & grid["stop_loss_proxy"].isin(["-0.03", "-0.05", "-0.07"])
        & grid["take_profit_proxy"].isin(["0.05", "0.08", "0.10"])
        & grid["cooldown_after_loss"].isin([1, 2])
    ].copy()
    out = pd.concat([locked_grid, top_grid, stress], ignore_index=True, sort=False).drop_duplicates("risk_variant_id")
    family_cols = [
        "threshold",
        "max_positions",
        "max_exposure",
        "volatility_filter",
        "market_drawdown_filter",
        "stop_loss_proxy",
        "take_profit_proxy",
        "cooldown_after_loss",
    ]
    families = out[family_cols].drop_duplicates()
    sibling_grid = grid.merge(families, on=family_cols, how="inner")
    out = pd.concat([out, sibling_grid], ignore_index=True, sort=False).drop_duplicates("risk_variant_id")
    out["selection_stage"] = "final_locked_or_validation_shortlist_or_stress_grid"
    return out.reset_index(drop=True)


def robustness_tables(
    final_results: pd.DataFrame,
    final_trade_log: pd.DataFrame,
    final_equity: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    after_cost = final_results[(final_results["cost_bps"].astype(float) + final_results["slippage_bps"].astype(float)) > 0].copy()
    selected_ids = set(after_cost.sort_values(["total_return", "baseline_delta"], ascending=[False, False]).head(5)["risk_variant_id"].astype(str))
    selected_ids.update(final_results.sort_values(["max_drawdown", "total_return"], ascending=[False, False]).head(3)["risk_variant_id"].astype(str))
    trades = final_trade_log[final_trade_log["risk_variant_id"].astype(str).isin(selected_ids)].copy()
    loo_rows: list[dict[str, Any]] = []
    if not trades.empty:
        result_lookup = final_results.set_index("risk_variant_id").to_dict("index")
        for risk_id, group in trades.groupby("risk_variant_id", sort=True):
            original = result_lookup.get(risk_id, {})
            original_equity = 1.0 + as_float(original.get("total_return", 0.0))
            for i, trade in group.reset_index(drop=True).iterrows():
                profit = as_float(trade.get("exit_value")) - as_float(trade.get("entry_value"))
                removed_equity = original_equity - profit
                remaining = group.drop(group.index[i])
                remaining_returns = pd.to_numeric(remaining["net_return"], errors="coerce").dropna()
                sharpe = float((remaining_returns.mean() / remaining_returns.std(ddof=0)) * math.sqrt(max(len(remaining_returns), 1))) if len(remaining_returns) > 1 and remaining_returns.std(ddof=0) > 0 else 0.0
                loo_rows.append(
                    {
                        "risk_variant_id": risk_id,
                        "removed_trade_index": int(i),
                        "removed_ticker": trade.get("ticker", ""),
                        "removed_entry_time": trade.get("entry_time", ""),
                        "removed_exit_time": trade.get("exit_time", ""),
                        "removed_net_return": trade.get("net_return", math.nan),
                        "removed_trade_profit": profit,
                        "return_after_removal": removed_equity - 1.0,
                        "sharpe_after_removal": sharpe,
                        "remaining_trade_count": int(len(remaining)),
                    }
                )
    loo = pd.DataFrame(loo_rows)
    loo_return = loo.copy()
    loo_sharpe = loo[["risk_variant_id", "removed_trade_index", "sharpe_after_removal", "remaining_trade_count"]].copy() if not loo.empty else pd.DataFrame()
    best_removed = (
        loo.sort_values("removed_trade_profit", ascending=False).groupby("risk_variant_id", as_index=False).head(1)
        if not loo.empty
        else pd.DataFrame()
    )
    worst_removed = (
        loo.sort_values("removed_trade_profit", ascending=True).groupby("risk_variant_id", as_index=False).head(1)
        if not loo.empty
        else pd.DataFrame()
    )

    boot_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(SEED)
    for risk_id, group in trades.groupby("risk_variant_id", sort=True):
        profits = (pd.to_numeric(group["exit_value"], errors="coerce") - pd.to_numeric(group["entry_value"], errors="coerce")).dropna().to_numpy(dtype=float)
        if len(profits) == 0:
            continue
        samples = [float(rng.choice(profits, size=len(profits), replace=True).sum()) for _ in range(500)]
        boot_rows.append(
            {
                "risk_variant_id": risk_id,
                "bootstrap_iterations": 500,
                "mean_total_return": float(np.mean(samples)),
                "median_total_return": float(np.median(samples)),
                "p05_total_return": float(np.quantile(samples, 0.05)),
                "p95_total_return": float(np.quantile(samples, 0.95)),
                "positive_rate": float((np.asarray(samples) > 0.0).mean()),
            }
        )
    bootstrap = pd.DataFrame(boot_rows)

    curve = final_equity[final_equity["risk_variant_id"].astype(str).isin(selected_ids)].copy()
    period_rows: list[dict[str, Any]] = []
    if not curve.empty:
        curve["datetime"] = pd.to_datetime(curve["datetime"], errors="coerce")
        curve["equity"] = pd.to_numeric(curve["equity"], errors="coerce")
        for risk_id, group in curve.dropna(subset=["datetime", "equity"]).groupby("risk_variant_id", sort=True):
            group = group.sort_values("datetime")
            for freq, period_alias, label in [("ME", "M", "month"), ("QE", "Q", "quarter")]:
                period = group.set_index("datetime")["equity"].resample(freq).last().dropna()
                returns = period.pct_change().dropna()
                for timestamp, value in returns.items():
                    period_rows.append({"risk_variant_id": risk_id, "period_type": label, "period": str(timestamp.to_period(period_alias)), "period_return": float(value)})
    period = pd.DataFrame(period_rows)
    monthly = period[period["period_type"].eq("month")].drop(columns=["period_type"]) if not period.empty else pd.DataFrame()
    quarterly = period[period["period_type"].eq("quarter")].drop(columns=["period_type"]) if not period.empty else pd.DataFrame()
    return loo_return, loo_sharpe, best_removed, worst_removed, bootstrap, monthly, quarterly


def write_reports(
    locked: dict[str, Any],
    validation_results: pd.DataFrame,
    final_results: pd.DataFrame,
    baselines: pd.DataFrame,
    best_removed: pd.DataFrame,
    cost_sensitivity: pd.DataFrame,
) -> None:
    after_cost = final_results[(final_results["cost_bps"].astype(float) + final_results["slippage_bps"].astype(float)) > 0].copy()
    best = after_cost.sort_values(["total_return", "baseline_delta", "max_drawdown"], ascending=[False, False, False]).iloc[0].to_dict() if not after_cost.empty else {}
    locked_result = final_results[final_results["risk_variant_id"].eq(locked.get("risk_variant_id", ""))]
    locked_row = locked_result.iloc[0].to_dict() if not locked_result.empty else {}
    best_cost = as_float(best.get("cost_bps"))
    best_slippage = as_float(best.get("slippage_bps"))
    best_max_positions = as_float(best.get("max_positions"))
    final_baselines = baselines[baselines["split"].eq("final")].copy() if not baselines.empty else pd.DataFrame()
    matching_cost = final_baselines[
        pd.to_numeric(final_baselines.get("cost_bps", np.nan), errors="coerce").eq(best_cost)
        & pd.to_numeric(final_baselines.get("slippage_bps", np.nan), errors="coerce").eq(best_slippage)
    ].copy()
    buy_hold = (
        matching_cost[matching_cost["baseline_name"].eq("buy_and_hold_vn30_index")]["total_return"].astype(float).max()
        if not matching_cost.empty
        else math.nan
    )
    random_match = matching_cost[
        matching_cost["baseline_name"].eq("random_signal_same_turnover")
        & pd.to_numeric(matching_cost.get("max_positions", np.nan), errors="coerce").eq(best_max_positions)
    ]
    random_ret = random_match["total_return"].astype(float).max() if not random_match.empty else math.nan
    beat_buy_hold = bool(as_float(best.get("total_return")) > as_float(buy_hold))
    baseline_outperformers = after_cost[pd.to_numeric(after_cost["baseline_delta"], errors="coerce") > 0.0].copy()
    beat_random = bool(not baseline_outperformers.empty)
    best_outperformer = (
        baseline_outperformers.sort_values(["total_return", "baseline_delta"], ascending=[False, False]).iloc[0].to_dict()
        if not baseline_outperformers.empty
        else {}
    )
    outperformer_text = (
        f"`{best_outperformer.get('risk_variant_id')}` at {best_outperformer.get('cost_bps')} bps / {best_outperformer.get('slippage_bps')} bps, "
        f"baseline_delta={pp(best_outperformer.get('baseline_delta', math.nan))}"
        if best_outperformer
        else "none"
    )
    dd_improved = bool(as_float(best.get("max_drawdown")) > V4_DRAWDOWN)
    best_removed_row = best_removed[best_removed["risk_variant_id"].astype(str).eq(str(best.get("risk_variant_id", "")))].head(1)
    best_removed_payload = best_removed_row.iloc[0].to_dict() if not best_removed_row.empty else {}
    survives_best_trade = bool(as_float(best_removed_payload.get("return_after_removal")) > 0.0)
    sharpe_survives = bool(as_float(best_removed_payload.get("sharpe_after_removal")) > 0.0)
    concentration = "yes" if not survives_best_trade else "no"
    cost_best = cost_sensitivity.sort_values("total_return", ascending=False).head(1).to_dict("records")
    cost_worst = cost_sensitivity.sort_values("total_return", ascending=True).head(1).to_dict("records")

    protocol = f"""# VN30 V5 Strategy Relock Risk Hardening Protocol

## Scope

- VN30 stock hourly strategy diagnostic only.
- Frozen family: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Threshold candidates: {THRESHOLDS}.
- Strategy template: long_only_market_regime_filter.
- Max positions: {MAX_POSITIONS}.
- No broad model tuning or final-period claimable ranking.

## Relock Rule

- Selection uses validation-period strategy diagnostics only.
- If no validation variant has sufficient trades and positive baseline-relative return, the run records a non-claimable predeclared fallback and keeps final-period risk rankings exploratory.
- Final-period optimization is not claimable.

## Risk Grid

- max_exposure: {MAX_EXPOSURE}.
- volatility_filter: {VOLATILITY_FILTERS}.
- market_drawdown_filter: {MARKET_DRAWDOWN_FILTERS}.
- stop_loss_proxy: none, -3%, -5%, -7%.
- take_profit_proxy: none, +5%, +8%, +10%.
- cooldown_after_loss: {COOLDOWN}.
- cost_bps: {COST_BPS}.
- slippage_bps: {SLIPPAGE_BPS}.

## Claim Boundary

- Offline diagnostic simulation only.
- No trading, profitability, BUY/SELL, recommendation, live deployment, DOCX, push, merge, or tag.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    result = f"""# VN30 V5 Strategy Relock Risk Hardening Result Summary

## Relock

- Relock status: {locked.get("selection_status", "")}.
- Selected threshold: {locked.get("threshold", "")}.
- Selected max positions: {locked.get("max_positions", "")}.
- Selected reporting cost/slippage: {locked.get("cost_bps", "")} bps / {locked.get("slippage_bps", "")} bps.
- Selected risk controls: max_exposure={locked.get("max_exposure", "")}, volatility_filter={locked.get("volatility_filter", "")}, market_drawdown_filter={locked.get("market_drawdown_filter", "")}, stop_loss={locked.get("stop_loss_proxy", "")}, take_profit={locked.get("take_profit_proxy", "")}, cooldown={locked.get("cooldown_after_loss", "")}.

## Best Risk-Hardened Final Diagnostic

- Risk variant: `{best.get("risk_variant_id", "")}`.
- Threshold/max positions/max exposure: {best.get("threshold", "")} / {best.get("max_positions", "")} / {best.get("max_exposure", "")}.
- Controls: volatility={best.get("volatility_filter", "")}, market_drawdown={best.get("market_drawdown_filter", "")}, stop_loss={best.get("stop_loss_proxy", "")}, take_profit={best.get("take_profit_proxy", "")}, cooldown={best.get("cooldown_after_loss", "")}.
- Cost/slippage: {best.get("cost_bps", "")} bps / {best.get("slippage_bps", "")} bps.
- Return/Sharpe/drawdown: {pct(best.get("total_return", math.nan))} / {best.get("sharpe", "")} / {pct(best.get("max_drawdown", math.nan))}.
- Trade count/exposure/turnover: {best.get("trade_count", "")} / {best.get("exposure_ratio", "")} / {best.get("turnover", "")}.

## Required Answers

1. Threshold selected by relock logic: {locked.get("threshold", "")}.
2. Did risk hardening reduce max drawdown below V4 -38.42%: {str(dd_improved).lower()}.
3. Any variant kept return above buy-and-hold VN30 after costs: {str(beat_buy_hold).lower()} (matching cost/slippage baseline={pct(buy_hold)}).
4. Any variant beat random same-turnover after costs: {str(beat_random).lower()} (best total-return variant matching random baseline={pct(random_ret)}; strongest matching-baseline outperformer={outperformer_text}).
5. Cost/slippage sensitivity: best={cost_best[:1]}, worst={cost_worst[:1]}.
6. Performance survives best-trade removal: return_positive={str(survives_best_trade).lower()}, sharpe_positive={str(sharpe_survives).lower()}.
7. Performance depends on a small number of trades: {concentration}.
8. Claimable or exploratory: exploratory_not_claimable; future_blind_required.
9. Paper-safe wording: offline diagnostic only; no trading, profitability, recommendation, live deployment, or claimable strategy claim.

Paper-safe wording:

> VN30 V5 froze the V4 calibrated-logistic compact-stable h40 strategy family and ran validation-only relock auditing plus final-period risk-hardening diagnostics. Because final-period strategy ranking remains exploratory and validation relock did not establish a claimable strategy, no strategy result is claimable. Future-blind confirmation is required before any stronger statement.
"""
    write_markdown(RESULT_PATH, result)

    claim = f"""# VN30 V5 Strategy Relock Risk Hardening Claim Boundary

- Strategy simulation is offline diagnostic only.
- No BUY/SELL recommendation.
- No live trading.
- No profitability guarantee.
- No investment advice.
- No deployment claim.
- Final-period strategy rankings and risk-hardening winners are exploratory_not_claimable.
- Future-blind confirmation is required before stronger claims.
- Current claimable directional champion remains the 61.61% L2 Logistic baseline60_candidate.
- No DOCX, paper, push, merge, tag, VN100, or index-as-stock claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def load_json_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    validation_cached_paths = [
        OUTPUT_DIR / "risk_hardening_grid.csv",
        OUTPUT_DIR / "baseline_strategy_comparison.csv",
        OUTPUT_DIR / "validation_strategy_results.csv",
        OUTPUT_DIR / "relock_selection_audit.csv",
        OUTPUT_DIR / "locked_strategy_family.json",
    ]
    complete_cached_paths = [
        *validation_cached_paths,
        OUTPUT_DIR / "final_strategy_results.csv",
        OUTPUT_DIR / "final_equity_curves.csv",
        OUTPUT_DIR / "final_trade_log.csv",
    ]
    cached_manifest = OUTPUT_DIR / "strategy_manifest.json"
    cached_schema_ok = (
        cached_manifest.exists()
        and load_json_file(cached_manifest).get("cache_schema_version") == CACHE_SCHEMA_VERSION
    )
    if all(path.exists() for path in complete_cached_paths) and cached_schema_ok:
        print("Reusing completed V5 grid artifacts; writing robustness and reports", flush=True)
        grid = pd.read_csv(OUTPUT_DIR / "risk_hardening_grid.csv")
        baselines = pd.read_csv(OUTPUT_DIR / "baseline_strategy_comparison.csv")
        validation_results = pd.read_csv(OUTPUT_DIR / "validation_strategy_results.csv")
        final_results = pd.read_csv(OUTPUT_DIR / "final_strategy_results.csv")
        final_equity = pd.read_csv(OUTPUT_DIR / "final_equity_curves.csv")
        final_trade_log = pd.read_csv(OUTPUT_DIR / "final_trade_log.csv")
        locked = load_json_file(OUTPUT_DIR / "locked_strategy_family.json")
        feature_manifest = {
            "source": "cached_completed_v5_grid_artifacts",
            "note": "Core validation/final grids were generated by this runner before the robustness resample compatibility fix.",
        }
    elif all(path.exists() for path in validation_cached_paths):
        print("Reusing validation relock artifacts; recomputing expanded final diagnostic grid", flush=True)
        grid = pd.read_csv(OUTPUT_DIR / "risk_hardening_grid.csv")
        baselines = pd.read_csv(OUTPUT_DIR / "baseline_strategy_comparison.csv")
        validation_results = pd.read_csv(OUTPUT_DIR / "validation_strategy_results.csv")
        audit = pd.read_csv(OUTPUT_DIR / "relock_selection_audit.csv")
        locked = load_json_file(OUTPUT_DIR / "locked_strategy_family.json")
        features, feature_groups, _audit, _manifest, feature_manifest = build_v3_feature_frame()
        index_data = load_index_data()
        features = add_market_drawdown_feature(features, index_data)
        signals = build_threshold_signals(features, feature_groups, index_data)
        final_grid = choose_final_grid(grid, audit, locked)
        print(f"Running expanded final diagnostic grid: {len(final_grid)} variants", flush=True)
        final_results, final_equity, final_trade_log = run_grid(final_grid, signals, "final", keep_curves=True)
        final_results = add_baseline_delta(final_results, baselines)
        final_results.loc[final_results["baseline_delta"] > 0, "claim_label"] = "strategy_outperforms_baseline"
        final_results.loc[
            (final_results["baseline_delta"] > 0)
            & (final_results["max_drawdown"].astype(float) > V4_DRAWDOWN)
            & ((final_results["cost_bps"].astype(float) + final_results["slippage_bps"].astype(float)) > 0),
            "claim_label",
        ] = "risk_hardened_strategy_candidate"
        write_frame(OUTPUT_DIR / "final_strategy_results.csv", final_results)
        write_frame(OUTPUT_DIR / "final_equity_curves.csv", final_equity)
        write_frame(OUTPUT_DIR / "final_trade_log.csv", final_trade_log)
    else:
        features, feature_groups, _audit, _manifest, feature_manifest = build_v3_feature_frame()
        index_data = load_index_data()
        features = add_market_drawdown_feature(features, index_data)
        signals = build_threshold_signals(features, feature_groups, index_data)
        grid = risk_grid()
        write_frame(OUTPUT_DIR / "risk_hardening_grid.csv", grid)

        baselines = build_baselines(features, index_data)
        write_frame(OUTPUT_DIR / "baseline_strategy_comparison.csv", baselines)

        print(f"Running validation risk grid: {len(grid)} variants", flush=True)
        validation_results, _validation_curves, _validation_trades = run_grid(grid, signals, "validation", keep_curves=False)
        validation_results = add_baseline_delta(validation_results, baselines)
        locked, audit = select_locked_strategy(validation_results)
        locked_grid_match = variant_filter(grid, locked)
        locked["risk_variant_id"] = locked_grid_match.iloc[0]["risk_variant_id"] if not locked_grid_match.empty else ""
        write_frame(OUTPUT_DIR / "validation_strategy_results.csv", validation_results)
        write_frame(OUTPUT_DIR / "relock_selection_audit.csv", audit)
        write_json(OUTPUT_DIR / "locked_strategy_family.json", locked)

        final_grid = choose_final_grid(grid, audit, locked)
        print(f"Running final diagnostic grid: {len(final_grid)} variants", flush=True)
        final_results, final_equity, final_trade_log = run_grid(final_grid, signals, "final", keep_curves=True)
        final_results = add_baseline_delta(final_results, baselines)
        final_results.loc[final_results["baseline_delta"] > 0, "claim_label"] = "strategy_outperforms_baseline"
        final_results.loc[
            (final_results["baseline_delta"] > 0)
            & (final_results["max_drawdown"].astype(float) > V4_DRAWDOWN)
            & ((final_results["cost_bps"].astype(float) + final_results["slippage_bps"].astype(float)) > 0),
            "claim_label",
        ] = "risk_hardened_strategy_candidate"
        write_frame(OUTPUT_DIR / "final_strategy_results.csv", final_results)
        write_frame(OUTPUT_DIR / "final_equity_curves.csv", final_equity)
        write_frame(OUTPUT_DIR / "final_trade_log.csv", final_trade_log)

    cost_sensitivity = (
        final_results.groupby(["threshold", "max_positions", "max_exposure", "volatility_filter", "market_drawdown_filter", "stop_loss_proxy", "take_profit_proxy", "cooldown_after_loss", "cost_bps", "slippage_bps"], dropna=False)
        .agg(total_return=("total_return", "max"), sharpe=("sharpe", "max"), max_drawdown=("max_drawdown", "max"), trade_count=("trade_count", "max"))
        .reset_index()
    )
    exposure = final_results[
        ["risk_variant_id", "threshold", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "exposure_ratio", "turnover", "trade_count"]
    ].copy()
    drawdown = final_results[
        ["risk_variant_id", "threshold", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "max_drawdown", "calmar", "total_return"]
    ].copy()
    write_frame(OUTPUT_DIR / "cost_slippage_sensitivity.csv", cost_sensitivity)
    write_frame(OUTPUT_DIR / "exposure_turnover_summary.csv", exposure)
    write_frame(OUTPUT_DIR / "drawdown_summary.csv", drawdown)

    loo_return, loo_sharpe, best_removed, worst_removed, bootstrap, monthly, quarterly = robustness_tables(final_results, final_trade_log, final_equity)
    write_frame(OUTPUT_DIR / "leave_one_trade_out_return.csv", loo_return)
    write_frame(OUTPUT_DIR / "leave_one_trade_out_sharpe.csv", loo_sharpe)
    write_frame(OUTPUT_DIR / "best_trade_removed_result.csv", best_removed)
    write_frame(OUTPUT_DIR / "worst_trade_removed_result.csv", worst_removed)
    write_frame(OUTPUT_DIR / "trade_bootstrap_summary.csv", bootstrap)
    write_frame(OUTPUT_DIR / "monthly_return_summary.csv", monthly)
    write_frame(OUTPUT_DIR / "quarterly_return_summary.csv", quarterly)

    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 V5 strategy relock and risk-hardening diagnostics",
        "frozen_family": {
            "model_family": "calibrated_logistic",
            "target_variant": "absolute_direction",
            "feature_group": "compact_stable_features",
            "horizon": 40,
            "threshold_candidates": THRESHOLDS,
            "strategy_template": "long_only_market_regime_filter",
        },
        "split_rules": {
            "train_end": str(TRAIN_END),
            "validation_start": str(VAL_START),
            "validation_end": str(VAL_END),
            "final_start": str(FINAL_START),
        },
        "risk_grid_rows": int(len(grid)),
        "validation_rows": int(len(validation_results)),
        "final_rows": int(len(final_results)),
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "feature_manifest": feature_manifest,
        "final_performance_used_for_claimable_selection": False,
        "strategy_results_claimable": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    manifest = {
        **run_config,
        "locked_strategy_family": locked,
        "best_final_diagnostic": final_results.sort_values(["total_return", "baseline_delta"], ascending=[False, False]).head(1).to_dict("records"),
        "robustness_tables": {
            "leave_one_trade_out_rows": int(len(loo_return)),
            "bootstrap_rows": int(len(bootstrap)),
            "monthly_rows": int(len(monthly)),
            "quarterly_rows": int(len(quarterly)),
        },
    }
    write_json(OUTPUT_DIR / "strategy_manifest.json", manifest)
    write_reports(locked, validation_results, final_results, baselines, best_removed, cost_sensitivity)
    print(f"VN30 V5 strategy relock complete: {rel(OUTPUT_DIR)}", flush=True)
    print(f"Locked status: {locked['selection_status']} threshold={locked['threshold']}", flush=True)
    print(f"Final diagnostic rows: {len(final_results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
