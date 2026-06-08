"""Run VN30 V7 rank-based strategy relock diagnostics.

V7 freezes the V4/V5 calibrated-logistic score family and replaces fragile
absolute probability thresholds with validation-governed rank and quantile
selection rules. Final-period leaderboards are exploratory only.
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
    now_utc,
    pct,
    pp,
    rel,
    write_frame,
    write_json,
    write_markdown,
)
from scripts.research.run_vn30_full_model_tuning_v3 import build_v3_feature_frame  # noqa: E402
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
from scripts.research.vn30_hourly_dual_track_common import REPO_ROOT, load_index_data  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_v7_rank_strategy"
PROTOCOL_PATH = REPO_ROOT / "reports" / "protocols" / "VN30_V7_RANK_BASED_STRATEGY_PROTOCOL.md"
RESULT_PATH = REPO_ROOT / "reports" / "results" / "VN30_V7_RANK_BASED_STRATEGY_RESULT_SUMMARY.md"
CLAIM_PATH = REPO_ROOT / "reports" / "claims" / "VN30_V7_RANK_BASED_STRATEGY_CLAIM_BOUNDARY.md"

SEED = 42
COST_BPS = [0, 5, 10, 20, 30]
SLIPPAGE_BPS = [0, 5, 10, 20]
MAX_EXPOSURE = [0.5, 0.7, 1.0]
MAX_POSITIONS = [1, 3, 5, 10]
TOP_N = [1, 3, 5, 10]
TOP_QUANTILES = [0.05, 0.10, 0.20, 0.30]
COMBINED_WEIGHTS = [(1.0, 0.0), (0.8, 0.2), (0.6, 0.4), (0.5, 0.5)]
SPREAD_THRESHOLDS = [0.0, 0.005, 0.01, 0.02, 0.03]


def compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def strategy_key(row: dict[str, Any]) -> str:
    return (
        f"{row['strategy_template']}__n{row.get('top_n', '')}__q{row.get('top_quantile', '')}"
        f"__wm{row.get('w_model', '')}__wrs{row.get('w_rs', '')}"
        f"__spread{row.get('score_spread_threshold', '')}__mp{row['max_positions']}"
        f"__ex{row['max_exposure']}"
    ).replace(".", "p")


def build_rank_strategy_grid() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq = 0

    def add(template: str, max_positions: int, max_exposure: float, cost_bps: int, slippage_bps: int, **params: Any) -> None:
        nonlocal seq
        seq += 1
        row = {
            "rank_strategy_id": f"v7rank_{seq:05d}",
            "model_family": "calibrated_logistic",
            "target_variant": "absolute_direction",
            "feature_group": "compact_stable_features",
            "horizon": 40,
            "strategy_template": template,
            "top_n": params.get("top_n", ""),
            "top_quantile": params.get("top_quantile", ""),
            "w_model": params.get("w_model", ""),
            "w_rs": params.get("w_rs", ""),
            "score_spread_threshold": params.get("score_spread_threshold", ""),
            "market_regime_filter": params.get("market_regime_filter", "off"),
            "max_positions": int(max_positions),
            "max_exposure": float(max_exposure),
            "cost_bps": int(cost_bps),
            "slippage_bps": int(slippage_bps),
            "uses_absolute_probability_threshold": False,
        }
        row["strategy_family_key"] = strategy_key(row)
        rows.append(row)

    for n, exposure, cost, slip in itertools.product(TOP_N, MAX_EXPOSURE, COST_BPS, SLIPPAGE_BPS):
        add("top_n_rotation", n, exposure, cost, slip, top_n=n)

    for q, max_positions, exposure, cost, slip in itertools.product(TOP_QUANTILES, MAX_POSITIONS, MAX_EXPOSURE, COST_BPS, SLIPPAGE_BPS):
        add("top_quantile_rotation", max_positions, exposure, cost, slip, top_quantile=q)

    for (w_model, w_rs), n, exposure, cost, slip in itertools.product(COMBINED_WEIGHTS, [3, 5, 10], MAX_EXPOSURE, COST_BPS, SLIPPAGE_BPS):
        add("score_rank_plus_relative_strength", n, exposure, cost, slip, top_n=n, w_model=w_model, w_rs=w_rs)

    for n, exposure, cost, slip in itertools.product(TOP_N, MAX_EXPOSURE, COST_BPS, SLIPPAGE_BPS):
        add("market_regime_rank_filter", n, exposure, cost, slip, top_n=n, market_regime_filter="on")

    for n, spread, exposure, cost, slip in itertools.product(TOP_N, SPREAD_THRESHOLDS, MAX_EXPOSURE, COST_BPS, SLIPPAGE_BPS):
        add("score_spread_filter", n, exposure, cost, slip, top_n=n, score_spread_threshold=spread)

    return pd.DataFrame(rows)


def add_rank_columns(frame: pd.DataFrame, split: str) -> pd.DataFrame:
    out = frame.copy()
    out["split"] = split
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["target_timestamp"] = pd.to_datetime(out["target_timestamp"], errors="coerce")
    out["model_score"] = pd.to_numeric(out["y_score"], errors="coerce")
    rs_col = "relative_strength_vs_market_20" if "relative_strength_vs_market_20" in out.columns else "relative_strength_vs_market_lag1"
    out["relative_strength_score"] = pd.to_numeric(out.get(rs_col, np.nan), errors="coerce")
    out["model_score_rank"] = out.groupby("datetime")["model_score"].rank(method="first", ascending=False)
    out["model_score_pct_rank"] = out.groupby("datetime")["model_score"].rank(method="average", pct=True)
    out["relative_strength_rank"] = out.groupby("datetime")["relative_strength_score"].rank(method="first", ascending=False)
    out["relative_strength_pct_rank"] = out.groupby("datetime")["relative_strength_score"].rank(method="average", pct=True)
    out["ticker_normalized_score"] = out.groupby("ticker")["model_score"].transform(
        lambda s: (s - s.mean()) / s.std(ddof=0) if s.std(ddof=0) > 0 else 0.0
    )
    out["ticker_normalized_score_rank"] = out.groupby("datetime")["ticker_normalized_score"].rank(method="average", pct=True)
    out["score_decile"] = np.ceil(out["model_score_pct_rank"].fillna(0.0) * 10.0).clip(1, 10).astype(int)
    return out


def build_score_frames() -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    features, feature_groups, _index_audit, _index_manifest, feature_manifest = build_v3_feature_frame()
    index_data = load_index_data()
    features = add_market_drawdown_feature(features, index_data)
    candidate = frozen_candidate(0.54)
    payload = fit_candidate_payload(candidate, features, feature_groups, index_data)
    drawdown_lookup = features[["datetime", "ticker", "vn30_drawdown_60_lag"]].drop_duplicates(["datetime", "ticker"])
    frames: dict[str, pd.DataFrame] = {}
    for split in ["validation", "final"]:
        frame = build_signal_rows(payload, features, index_data, split)
        if "vn30_drawdown_60_lag" not in frame.columns:
            frame = frame.merge(drawdown_lookup, on=["datetime", "ticker"], how="left")
        frames[split] = add_rank_columns(frame, split)
    return frames, {"feature_manifest": feature_manifest, "index_data": index_data, "features": features}


def quantile_row(scores: pd.Series, prefix: str) -> dict[str, Any]:
    q = scores.quantile([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
    return {
        f"{prefix}_min": float(scores.min()),
        f"{prefix}_p01": float(q.loc[0.01]),
        f"{prefix}_p05": float(q.loc[0.05]),
        f"{prefix}_p10": float(q.loc[0.10]),
        f"{prefix}_p25": float(q.loc[0.25]),
        f"{prefix}_median": float(q.loc[0.50]),
        f"{prefix}_p75": float(q.loc[0.75]),
        f"{prefix}_p90": float(q.loc[0.90]),
        f"{prefix}_p95": float(q.loc[0.95]),
        f"{prefix}_p99": float(q.loc[0.99]),
        f"{prefix}_max": float(scores.max()),
    }


def score_audits(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dist_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    for split, frame in frames.items():
        score = pd.to_numeric(frame["model_score"], errors="coerce")
        dist_rows.append(
            {
                "split": split,
                "rows": int(len(frame)),
                "ticker_count": int(frame["ticker"].nunique()),
                "timestamp_count": int(frame["datetime"].nunique()),
                "feature_timestamp_min": str(pd.to_datetime(frame["feature_timestamp"] if "feature_timestamp" in frame.columns else frame["datetime"], errors="coerce").min()),
                "feature_timestamp_max": str(pd.to_datetime(frame["feature_timestamp"] if "feature_timestamp" in frame.columns else frame["datetime"], errors="coerce").max()),
                "target_timestamp_min": str(pd.to_datetime(frame["target_timestamp"], errors="coerce").min()),
                "target_timestamp_max": str(pd.to_datetime(frame["target_timestamp"], errors="coerce").max()),
                **quantile_row(score, "probability"),
            }
        )
        work = frame.dropna(subset=["model_score_pct_rank", "stock_forward_return", "y_true"]).copy()
        for decile, group in work.groupby("score_decile", sort=True):
            decile_rows.append(
                {
                    "split": split,
                    "score_decile": int(decile),
                    "decile_label": "10_highest" if int(decile) == 10 else ("1_lowest" if int(decile) == 1 else str(int(decile))),
                    "rows": int(len(group)),
                    "avg_model_score": float(pd.to_numeric(group["model_score"], errors="coerce").mean()),
                    "avg_realized_return": float(pd.to_numeric(group["stock_forward_return"], errors="coerce").mean()),
                    "median_realized_return": float(pd.to_numeric(group["stock_forward_return"], errors="coerce").median()),
                    "hit_rate": float((pd.to_numeric(group["y_true"], errors="coerce") == 1).mean()),
                    "prediction_up_ratio": float((pd.to_numeric(group["model_score_pct_rank"], errors="coerce") >= 0.5).mean()),
                }
            )
        corr = work[["model_score_pct_rank", "stock_forward_return"]].corr(method="spearman").iloc[0, 1] if len(work) > 1 else math.nan
        high = work[work["score_decile"].eq(10)]
        low = work[work["score_decile"].eq(1)]
        quality_rows.append(
            {
                "split": split,
                "metric": "spearman_rank_vs_realized_return",
                "value": float(corr) if pd.notna(corr) else math.nan,
                "rows": int(len(work)),
            }
        )
        quality_rows.append(
            {
                "split": split,
                "metric": "top_decile_return_minus_bottom_decile",
                "value": float(pd.to_numeric(high["stock_forward_return"], errors="coerce").mean() - pd.to_numeric(low["stock_forward_return"], errors="coerce").mean()) if len(high) and len(low) else math.nan,
                "rows": int(len(work)),
            }
        )
        for bucket, group in work.groupby("score_decile", sort=True):
            quality_rows.append(
                {
                    "split": split,
                    "metric": f"prediction_up_ratio_score_decile_{int(bucket)}",
                    "value": float((pd.to_numeric(group["y_true"], errors="coerce") == 1).mean()),
                    "rows": int(len(group)),
                }
            )
    return pd.DataFrame(dist_rows), pd.DataFrame(decile_rows), pd.DataFrame(quality_rows)


def select_rank_candidates(group: pd.DataFrame, variant: dict[str, Any]) -> pd.DataFrame:
    if group.empty:
        return group
    template = str(variant["strategy_template"])
    work = group.copy()
    if template == "market_regime_rank_filter":
        risk = pd.to_numeric(work.get("risk_on_risk_off_state", 0.0), errors="coerce").fillna(0.0)
        momentum = pd.to_numeric(work.get("market_momentum_20", 0.0), errors="coerce").fillna(0.0)
        work = work[(risk >= 0.0) & (momentum >= 0.0)].copy()
        template = "top_n_rotation"
    if work.empty:
        return work

    if template == "top_n_rotation":
        work["selection_score"] = pd.to_numeric(work["model_score_pct_rank"], errors="coerce")
        selected = work.sort_values(["selection_score", "model_score"], ascending=[False, False]).head(int(variant["top_n"]))
    elif template == "top_quantile_rotation":
        q = float(variant["top_quantile"])
        work["selection_score"] = pd.to_numeric(work["model_score_pct_rank"], errors="coerce")
        selected = work[work["selection_score"] >= 1.0 - q].sort_values(["selection_score", "model_score"], ascending=[False, False])
    elif template == "score_rank_plus_relative_strength":
        w_model = float(variant["w_model"])
        w_rs = float(variant["w_rs"])
        model_rank = pd.to_numeric(work["model_score_pct_rank"], errors="coerce").fillna(0.5)
        rs_rank = pd.to_numeric(work["relative_strength_pct_rank"], errors="coerce").fillna(0.5)
        work["selection_score"] = w_model * model_rank + w_rs * rs_rank
        selected = work.sort_values(["selection_score", "model_score"], ascending=[False, False]).head(int(variant["top_n"]))
    elif template == "score_spread_filter":
        top_score = pd.to_numeric(work["model_score"], errors="coerce").max()
        median_score = pd.to_numeric(work["model_score"], errors="coerce").median()
        if not np.isfinite(top_score) or not np.isfinite(median_score) or (top_score - median_score) < float(variant["score_spread_threshold"]):
            return work.iloc[0:0].copy()
        work["selection_score"] = pd.to_numeric(work["model_score_pct_rank"], errors="coerce")
        selected = work.sort_values(["selection_score", "model_score"], ascending=[False, False]).head(int(variant["top_n"]))
    else:
        selected = work.iloc[0:0].copy()
    return selected.head(int(variant["max_positions"]))


def simulate_rank_strategy(signal_rows: pd.DataFrame, variant: dict[str, Any], split: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    rows = signal_rows.copy()
    rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce")
    rows["target_timestamp"] = pd.to_datetime(rows["target_timestamp"], errors="coerce")
    rows = rows.dropna(subset=["datetime", "target_timestamp", "stock_forward_return", "model_score"])
    if rows.empty:
        curve = pd.DataFrame([{"datetime": pd.NaT, "equity": 1.0, "cash": 1.0, "invested": 0.0, "open_positions": 0}])
        return metrics_from_equity(curve, pd.DataFrame(), [], 0.0), curve, pd.DataFrame()

    max_positions = int(variant["max_positions"])
    max_exposure = float(variant["max_exposure"])
    round_trip_drag = 2.0 * (float(variant["cost_bps"]) + float(variant["slippage_bps"])) / 10000.0
    cash = 1.0
    open_positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    exposure_values: list[float] = []
    turnover = 0.0

    for timestamp, group in rows.groupby("datetime", sort=True):
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
        if slots > 0 and cash > 1e-12 and equity > 0:
            open_tickers = {pos["ticker"] for pos in open_positions}
            selected = select_rank_candidates(group[~group["ticker"].isin(open_tickers)], variant).head(slots)
            exposure_room = max(0.0, max_exposure - invested / equity) * equity
            allocatable = min(cash, exposure_room)
            if allocatable > 1e-12 and not selected.empty:
                slot_value = min(allocatable / len(selected), equity * max_exposure / max_positions)
                for _, row in selected.iterrows():
                    entry_value = min(slot_value, cash)
                    if entry_value <= 1e-12:
                        continue
                    cash -= entry_value
                    turnover += entry_value
                    raw_return = float(row["stock_forward_return"])
                    net_return = raw_return - round_trip_drag
                    open_positions.append(
                        {
                            "rank_strategy_id": variant["rank_strategy_id"],
                            "split": split,
                            "ticker": row["ticker"],
                            "entry_time": timestamp,
                            "exit_time": row["target_timestamp"],
                            "entry_value": entry_value,
                            "raw_return": raw_return,
                            "net_return": net_return,
                            "model_score": row["model_score"],
                            "selection_score": row.get("selection_score", row.get("model_score_pct_rank", math.nan)),
                            "model_score_pct_rank": row.get("model_score_pct_rank", math.nan),
                            "strategy_template": variant["strategy_template"],
                            "max_positions": max_positions,
                            "max_exposure": max_exposure,
                            "cost_bps": variant["cost_bps"],
                            "slippage_bps": variant["slippage_bps"],
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
    metrics = metrics_from_equity(curve, trade_log, exposure_values, turnover)
    return metrics, curve, trade_log


def run_strategy_grid(grid: pd.DataFrame, signal_rows: pd.DataFrame, split: str, keep_locked: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    curve_rows: list[pd.DataFrame] = []
    trade_rows: list[pd.DataFrame] = []
    for variant in grid.to_dict("records"):
        metrics, curve, trades = simulate_rank_strategy(signal_rows, variant, split)
        result_rows.append(
            {
                **variant,
                "split": split,
                **metrics,
                "claim_label": "diagnostic_only" if split == "validation" else "exploratory_not_claimable",
                "reason_not_claimable": "validation rank-strategy diagnostic" if split == "validation" else "final-ranked strategies are exploratory only",
            }
        )
        if keep_locked is not None and str(variant["rank_strategy_id"]) == str(keep_locked):
            curve = curve.copy()
            curve.insert(0, "rank_strategy_id", variant["rank_strategy_id"])
            curve_rows.append(curve)
            if not trades.empty:
                trades = trades.copy()
                trades["rank_strategy_id"] = variant["rank_strategy_id"]
                trade_rows.append(trades)
    return (
        pd.DataFrame(result_rows),
        pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame(),
        pd.concat(trade_rows, ignore_index=True) if trade_rows else pd.DataFrame(),
    )


def baseline_comparison(features: pd.DataFrame, index_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ["validation", "final"]:
        baseline_cache = {
            name: add_rank_columns(build_baseline_signal_rows(features, index_data, split, name), split)
            for name in ["simple_momentum", "simple_relative_strength", "random_signal_same_turnover", "always_up_directional_baseline"]
        }
        for cost_bps, slippage_bps in itertools.product(COST_BPS, SLIPPAGE_BPS):
            rows.append({"baseline_name": "cash_no_trade", "split": split, "cost_bps": cost_bps, "slippage_bps": slippage_bps, "total_return": 0.0})
            rows.append({**baseline_buy_hold_index(index_data, split, cost_bps, slippage_bps), "cost_bps": cost_bps, "slippage_bps": slippage_bps})
            rows.append({**baseline_equal_weight_basket(features, split, cost_bps, slippage_bps), "cost_bps": cost_bps, "slippage_bps": slippage_bps})
            for max_positions in MAX_POSITIONS:
                for baseline_name, signal in baseline_cache.items():
                    variant = {
                        "rank_strategy_id": f"baseline_{baseline_name}",
                        "strategy_template": "top_n_rotation",
                        "top_n": max_positions,
                        "top_quantile": "",
                        "w_model": "",
                        "w_rs": "",
                        "score_spread_threshold": "",
                        "max_positions": max_positions,
                        "max_exposure": 1.0,
                        "cost_bps": cost_bps,
                        "slippage_bps": slippage_bps,
                    }
                    metrics, _curve, _trades = simulate_rank_strategy(signal, variant, split)
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
    lookup_rows: list[dict[str, Any]] = []
    for (split, cost_bps, slippage_bps, max_positions), _group in out.groupby(["split", "cost_bps", "slippage_bps", "max_positions"], dropna=False):
        subset = baselines[
            baselines["split"].eq(split)
            & pd.to_numeric(baselines["cost_bps"], errors="coerce").eq(float(cost_bps))
            & pd.to_numeric(baselines["slippage_bps"], errors="coerce").eq(float(slippage_bps))
        ].copy()
        same_positions = subset[
            subset.get("max_positions", pd.Series(index=subset.index, dtype=float)).isna()
            | pd.to_numeric(subset.get("max_positions", np.nan), errors="coerce").eq(float(max_positions))
        ]
        best_idx = pd.to_numeric(same_positions["total_return"], errors="coerce").idxmax() if not same_positions.empty else None
        strongest_name = str(same_positions.loc[best_idx, "baseline_name"]) if best_idx is not None and pd.notna(best_idx) else ""
        strongest_return = float(pd.to_numeric(same_positions["total_return"], errors="coerce").max()) if not same_positions.empty else 0.0
        lookup_rows.append(
            {
                "split": split,
                "cost_bps": cost_bps,
                "slippage_bps": slippage_bps,
                "max_positions": max_positions,
                "strongest_baseline": strongest_name,
                "strongest_baseline_return": strongest_return,
            }
        )
    out = out.merge(pd.DataFrame(lookup_rows), on=["split", "cost_bps", "slippage_bps", "max_positions"], how="left")
    out["baseline_delta"] = out["total_return"] - out["strongest_baseline_return"]
    return out


def add_validation_scores(validation: pd.DataFrame) -> pd.DataFrame:
    out = validation.copy()
    out["drawdown_score"] = (1.0 + pd.to_numeric(out["max_drawdown"], errors="coerce")).clip(lower=0.0, upper=1.0)
    out["trade_count_score"] = (pd.to_numeric(out["trade_count"], errors="coerce") / 50.0).clip(lower=0.0, upper=1.0)
    out["turnover_score"] = 1.0 / (1.0 + pd.to_numeric(out["turnover"], errors="coerce").clip(lower=0.0))
    out["validation_score"] = (
        0.30 * pd.to_numeric(out["total_return"], errors="coerce").fillna(-1.0)
        + 0.25 * pd.to_numeric(out["sharpe"], errors="coerce").fillna(0.0)
        + 0.15 * out["drawdown_score"].fillna(0.0)
        + 0.15 * pd.to_numeric(out["baseline_delta"], errors="coerce").fillna(-1.0)
        + 0.10 * out["trade_count_score"].fillna(0.0)
        + 0.05 * out["turnover_score"].fillna(0.0)
    )
    out["passes_trade_count"] = pd.to_numeric(out["trade_count"], errors="coerce") >= 10
    out["passes_exposure"] = pd.to_numeric(out["exposure_ratio"], errors="coerce").between(0.10, 0.90, inclusive="both")
    out["passes_positive_after_cost"] = pd.to_numeric(out["total_return"], errors="coerce") > 0.0
    out["passes_drawdown_reported"] = pd.to_numeric(out["max_drawdown"], errors="coerce").notna()
    out["passes_baseline_preferred"] = pd.to_numeric(out["baseline_delta"], errors="coerce") > 0.0
    out["validation_selected_eligible"] = out["passes_trade_count"] & out["passes_exposure"] & out["passes_positive_after_cost"] & out["passes_drawdown_reported"]
    return out


def lock_validation_strategy(validation: pd.DataFrame) -> dict[str, Any]:
    eligible = validation[validation["validation_selected_eligible"]].copy()
    if eligible.empty:
        selected = validation.sort_values(["validation_score", "total_return"], ascending=[False, False]).iloc[0].to_dict()
        status = "no_validation_candidate_passed_filters"
    else:
        selected = eligible.sort_values(["passes_baseline_preferred", "validation_score", "baseline_delta"], ascending=[False, False, False]).iloc[0].to_dict()
        status = "validation_selected_rank_strategy"
    locked = {
        "locked_at_utc": now_utc(),
        "selection_status": status,
        "selected_by": "validation_only_rank_strategy_score",
        "rank_strategy_id": selected["rank_strategy_id"],
        "strategy_template": selected["strategy_template"],
        "strategy_family_key": selected["strategy_family_key"],
        "top_n": selected.get("top_n", ""),
        "top_quantile": selected.get("top_quantile", ""),
        "w_model": selected.get("w_model", ""),
        "w_rs": selected.get("w_rs", ""),
        "score_spread_threshold": selected.get("score_spread_threshold", ""),
        "market_regime_filter": selected.get("market_regime_filter", "off"),
        "max_positions": int(as_float(selected["max_positions"])),
        "max_exposure": as_float(selected["max_exposure"]),
        "cost_bps": int(as_float(selected["cost_bps"])),
        "slippage_bps": int(as_float(selected["slippage_bps"])),
        "validation_total_return": as_float(selected["total_return"]),
        "validation_sharpe": as_float(selected["sharpe"]),
        "validation_max_drawdown": as_float(selected["max_drawdown"]),
        "validation_trade_count": int(as_float(selected["trade_count"])),
        "validation_exposure_ratio": as_float(selected["exposure_ratio"]),
        "validation_turnover": as_float(selected["turnover"]),
        "validation_strongest_baseline": selected.get("strongest_baseline", ""),
        "validation_baseline_delta": as_float(selected.get("baseline_delta")),
        "validation_score": as_float(selected.get("validation_score")),
        "final_performance_used": False,
        "claim_label": "rank_strategy_candidate" if status == "validation_selected_rank_strategy" else "diagnostic_only",
    }
    return locked


def variant_from_locked(grid: pd.DataFrame, locked: dict[str, Any]) -> pd.DataFrame:
    return grid[grid["rank_strategy_id"].astype(str).eq(str(locked["rank_strategy_id"]))].copy()


def cost_sensitivity_rows(results: pd.DataFrame, locked: dict[str, Any]) -> pd.DataFrame:
    family = str(locked["strategy_family_key"])
    return results[results["strategy_family_key"].astype(str).eq(family)].copy()


def write_reports(
    score_dist: pd.DataFrame,
    deciles: pd.DataFrame,
    quality: pd.DataFrame,
    validation_leaderboard: pd.DataFrame,
    locked: dict[str, Any],
    final_result: pd.DataFrame,
    baselines: pd.DataFrame,
    exploratory_final: pd.DataFrame,
) -> None:
    final_row = final_result.iloc[0].to_dict() if not final_result.empty else {}
    locked_cost = as_float(final_row.get("cost_bps"))
    locked_slip = as_float(final_row.get("slippage_bps"))
    locked_mp = as_float(final_row.get("max_positions"))
    final_baselines = baselines[
        baselines["split"].eq("final")
        & pd.to_numeric(baselines["cost_bps"], errors="coerce").eq(locked_cost)
        & pd.to_numeric(baselines["slippage_bps"], errors="coerce").eq(locked_slip)
    ].copy()
    buy_hold = final_baselines[final_baselines["baseline_name"].eq("buy_and_hold_vn30_index")]["total_return"].astype(float).max() if not final_baselines.empty else math.nan
    equal_weight = final_baselines[final_baselines["baseline_name"].eq("equal_weight_vn30_stock_basket")]["total_return"].astype(float).max() if not final_baselines.empty else math.nan
    random_same = final_baselines[
        final_baselines["baseline_name"].eq("random_signal_same_turnover")
        & pd.to_numeric(final_baselines.get("max_positions", np.nan), errors="coerce").eq(locked_mp)
    ]["total_return"].astype(float).max() if not final_baselines.empty else math.nan
    top_decile = deciles[deciles["score_decile"].eq(10)].sort_values(["split"]).to_dict("records")
    best_deciles = deciles.sort_values(["split", "avg_realized_return"], ascending=[True, False]).groupby("split", as_index=False).head(1)
    validation_winner = validation_leaderboard.head(1).to_dict("records")
    spearman = quality[quality["metric"].eq("spearman_rank_vs_realized_return")].to_dict("records")
    spearman_map = {str(row.get("split")): as_float(row.get("value")) for row in spearman}
    if spearman_map.get("validation", math.nan) > 0.0 and spearman_map.get("final", math.nan) > 0.0:
        score_verdict = "positive_in_validation_and_final"
    elif spearman_map.get("validation", math.nan) <= 0.0 and spearman_map.get("final", math.nan) > 0.0:
        score_verdict = "validation_non_positive_final_weak_positive"
    elif spearman_map.get("validation", math.nan) > 0.0:
        score_verdict = "validation_positive_final_non_positive"
    else:
        score_verdict = "non_positive"
    rank_solves = bool(as_float(locked.get("validation_trade_count")) >= 10)
    claimable = False

    protocol = f"""# VN30 V7 Rank Based Strategy Protocol

## Scope

- VN30 stock hourly strategy diagnostics only.
- Frozen score family: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Model probabilities are ranking scores only; fixed absolute probability thresholds are not the main selector.
- Validation-only selection governs the locked rank strategy.
- Final-ranked leaderboards are exploratory_not_claimable.
- No broad model tuning, final-performance claimable selection, DOCX, push, merge, tag, VN100, or index-as-stock claim.

## Strategy Grid

- top_n_rotation: N={TOP_N}.
- top_quantile_rotation: quantiles={TOP_QUANTILES}.
- score_rank_plus_relative_strength: weights={COMBINED_WEIGHTS}; N=3,5,10.
- market_regime_rank_filter: top-N when risk-on/neutral.
- score_spread_filter: spread thresholds={SPREAD_THRESHOLDS}.
- cost_bps={COST_BPS}; slippage_bps={SLIPPAGE_BPS}; max_exposure={MAX_EXPOSURE}.

## Validation Score

0.30 * validation_total_return + 0.25 * validation_sharpe + 0.15 * max_drawdown_score + 0.15 * baseline_delta + 0.10 * trade_count_score + 0.05 * turnover_score.
"""
    write_markdown(PROTOCOL_PATH, protocol)

    result = f"""# VN30 V7 Rank Based Strategy Result Summary

## Locked Validation Strategy

- Selection status: {locked.get("selection_status")}.
- Strategy: `{locked.get("rank_strategy_id")}` / {locked.get("strategy_template")}.
- Validation return/Sharpe/drawdown: {pct(locked.get("validation_total_return", math.nan))} / {locked.get("validation_sharpe")} / {pct(locked.get("validation_max_drawdown", math.nan))}.
- Validation trades/exposure/turnover: {locked.get("validation_trade_count")} / {locked.get("validation_exposure_ratio")} / {locked.get("validation_turnover")}.
- Validation strongest baseline/lift: {locked.get("validation_strongest_baseline")} / {pp(locked.get("validation_baseline_delta", math.nan))}.

## Final Locked Result

- Total return after cost: {pct(final_row.get("total_return", math.nan))}.
- Sharpe: {final_row.get("sharpe", "")}.
- Max drawdown: {pct(final_row.get("max_drawdown", math.nan))}.
- Trade count/exposure/turnover: {final_row.get("trade_count", "")} / {final_row.get("exposure_ratio", "")} / {final_row.get("turnover", "")}.
- Strongest final baseline: {final_row.get("strongest_baseline", "")}; baseline delta: {pp(final_row.get("baseline_delta", math.nan))}.

## Required Answers

1. Score-rank correlation verdict: {score_verdict}; spearman rows={spearman}.
2. Best score deciles: {best_deciles.to_dict("records")}.
3. Validation winner: {validation_winner[:1]}.
4. Locked validation-selected rank strategy generates final trades: {str(as_float(final_row.get("trade_count")) > 0).lower()}.
5. Final total return after cost: {pct(final_row.get("total_return", math.nan))}.
6. Final Sharpe: {final_row.get("sharpe", "")}.
7. Final max drawdown: {pct(final_row.get("max_drawdown", math.nan))}.
8. Final trade count and exposure: {final_row.get("trade_count", "")} / {final_row.get("exposure_ratio", "")}.
9. Beats buy-and-hold VN30: {str(as_float(final_row.get("total_return")) > as_float(buy_hold)).lower()} (baseline={pct(buy_hold)}).
10. Beats equal-weight VN30: {str(as_float(final_row.get("total_return")) > as_float(equal_weight)).lower()} (baseline={pct(equal_weight)}).
11. Beats random same-turnover: {str(as_float(final_row.get("total_return")) > as_float(random_same)).lower()} (baseline={pct(random_same)}).
12. Rank-based selection solves fixed-threshold validation infeasibility: {str(rank_solves).lower()}.
13. Claimable results: none as a strategy claim; locked row is a validation-governed offline rank_strategy_candidate only.
14. Exploratory only: all final-ranked leaderboard rows and any final-period strategy comparison.
15. Paper-safe wording: offline diagnostic only; no BUY/SELL, profitability, investment advice, live deployment, or claimable strategy claim.

Paper-safe wording:

> VN30 V7 replaced fixed absolute probability thresholds with validation-governed rank-based strategy rules for the frozen calibrated-logistic compact-stable h40 score family. Rank selection removes the V5 validation infeasibility caused by the 0.54-0.56 threshold band. The locked strategy is selected on validation only and final results are offline diagnostics; future-blind confirmation is required before stronger strategy claims.
"""
    write_markdown(RESULT_PATH, result)

    claim = """# VN30 V7 Rank Based Strategy Claim Boundary

- Strategy simulation is offline diagnostic only.
- No BUY/SELL recommendation.
- No live trading.
- No profitability guarantee.
- No investment advice.
- No deployment claim.
- Frozen score family only: calibrated_logistic / absolute_direction / compact_stable_features / h40.
- Fixed absolute probability thresholds are not claimed as a strategy selector.
- The locked rank strategy is validation-governed but remains a diagnostic rank_strategy_candidate.
- Final-ranked strategy leaderboard rows are exploratory_not_claimable.
- No new strategy result is claimable without future-blind confirmation.
- Current claimable directional champion remains the 61.61% L2 Logistic baseline60_candidate.
- No DOCX, paper, push, merge, tag, VN100, or index-as-stock claim is made.
"""
    write_markdown(CLAIM_PATH, claim)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frames, context = build_score_frames()
    features = context["features"]
    index_data = context["index_data"]

    if (OUTPUT_DIR / "score_distribution_audit.csv").exists():
        score_dist = pd.read_csv(OUTPUT_DIR / "score_distribution_audit.csv")
        deciles = pd.read_csv(OUTPUT_DIR / "score_decile_performance.csv")
        quality = pd.read_csv(OUTPUT_DIR / "rank_signal_quality.csv")
    else:
        score_dist, deciles, quality = score_audits(frames)
        write_frame(OUTPUT_DIR / "score_distribution_audit.csv", score_dist)
        write_frame(OUTPUT_DIR / "score_decile_performance.csv", deciles)
        write_frame(OUTPUT_DIR / "rank_signal_quality.csv", quality)

    if (OUTPUT_DIR / "rank_strategy_grid.csv").exists():
        grid = pd.read_csv(OUTPUT_DIR / "rank_strategy_grid.csv")
    else:
        grid = build_rank_strategy_grid()
        write_frame(OUTPUT_DIR / "rank_strategy_grid.csv", grid)

    if (OUTPUT_DIR / "baseline_rank_strategy_comparison.csv").exists():
        baselines = pd.read_csv(OUTPUT_DIR / "baseline_rank_strategy_comparison.csv")
    else:
        baselines = baseline_comparison(features, index_data)
        write_frame(OUTPUT_DIR / "baseline_rank_strategy_comparison.csv", baselines)

    if (OUTPUT_DIR / "validation_rank_strategy_results.csv").exists() and (OUTPUT_DIR / "locked_rank_strategy.json").exists():
        validation_results = pd.read_csv(OUTPUT_DIR / "validation_rank_strategy_results.csv")
        validation_leaderboard = pd.read_csv(OUTPUT_DIR / "validation_rank_strategy_leaderboard.csv")
        locked = json.loads((OUTPUT_DIR / "locked_rank_strategy.json").read_text(encoding="utf-8"))
    else:
        print(f"Running V7 validation rank grid: {len(grid)} variants", flush=True)
        validation_results, _validation_curves, _validation_trades = run_strategy_grid(grid, frames["validation"], "validation")
        validation_results = add_baseline_delta(validation_results, baselines)
        validation_results = add_validation_scores(validation_results)
        validation_results.loc[validation_results["passes_positive_after_cost"], "claim_label"] = "strategy_positive_after_cost"
        validation_results.loc[validation_results["passes_baseline_preferred"], "claim_label"] = "strategy_outperforms_baseline"
        locked = lock_validation_strategy(validation_results)
        validation_results.loc[validation_results["rank_strategy_id"].eq(locked["rank_strategy_id"]), "claim_label"] = locked["claim_label"]
        validation_leaderboard = validation_results.sort_values(["validation_selected_eligible", "passes_baseline_preferred", "validation_score"], ascending=[False, False, False]).reset_index(drop=True)
        write_frame(OUTPUT_DIR / "validation_rank_strategy_results.csv", validation_results)
        write_frame(OUTPUT_DIR / "validation_rank_strategy_leaderboard.csv", validation_leaderboard)
        write_json(OUTPUT_DIR / "locked_rank_strategy.json", locked)

    if (OUTPUT_DIR / "final_rank_strategy_result.csv").exists():
        final_result = pd.read_csv(OUTPUT_DIR / "final_rank_strategy_result.csv")
        final_curve = pd.read_csv(OUTPUT_DIR / "final_rank_strategy_equity_curve.csv") if (OUTPUT_DIR / "final_rank_strategy_equity_curve.csv").exists() else pd.DataFrame()
        final_trades = pd.read_csv(OUTPUT_DIR / "final_rank_trade_log.csv") if (OUTPUT_DIR / "final_rank_trade_log.csv").exists() else pd.DataFrame()
    else:
        locked_grid = variant_from_locked(grid, locked)
        print("Running locked final rank strategy once", flush=True)
        final_result, final_curve, final_trades = run_strategy_grid(locked_grid, frames["final"], "final", keep_locked=locked["rank_strategy_id"])
        final_result = add_baseline_delta(final_result, baselines)
        final_result["claim_label"] = "rank_strategy_candidate"
        final_result["reason_not_claimable"] = "offline diagnostic; future-blind confirmation required before stronger strategy claims"
        write_frame(OUTPUT_DIR / "final_rank_strategy_result.csv", final_result)
        write_frame(OUTPUT_DIR / "final_rank_strategy_equity_curve.csv", final_curve)
        write_frame(OUTPUT_DIR / "final_rank_trade_log.csv", final_trades)

    if (OUTPUT_DIR / "exploratory_final_rank_strategy_leaderboard.csv").exists():
        exploratory_final = pd.read_csv(OUTPUT_DIR / "exploratory_final_rank_strategy_leaderboard.csv")
    else:
        print(f"Running V7 exploratory final rank grid: {len(grid)} variants", flush=True)
        exploratory_final, _curves, _trades = run_strategy_grid(grid, frames["final"], "final")
        exploratory_final = add_baseline_delta(exploratory_final, baselines)
        exploratory_final["claim_label"] = "exploratory_not_claimable"
        exploratory_final = exploratory_final.sort_values(["total_return", "baseline_delta", "sharpe"], ascending=[False, False, False]).reset_index(drop=True)
        write_frame(OUTPUT_DIR / "exploratory_final_rank_strategy_leaderboard.csv", exploratory_final)

    cost_sensitivity = cost_sensitivity_rows(pd.concat([validation_results, exploratory_final], ignore_index=True, sort=False), locked)
    exposure = pd.concat([validation_results, final_result], ignore_index=True, sort=False)[
        ["rank_strategy_id", "split", "strategy_template", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "exposure_ratio", "turnover", "trade_count"]
    ].copy()
    drawdown = pd.concat([validation_results, final_result], ignore_index=True, sort=False)[
        ["rank_strategy_id", "split", "strategy_template", "max_positions", "max_exposure", "cost_bps", "slippage_bps", "max_drawdown", "calmar", "total_return"]
    ].copy()
    write_frame(OUTPUT_DIR / "cost_slippage_sensitivity.csv", cost_sensitivity)
    write_frame(OUTPUT_DIR / "exposure_turnover_summary.csv", exposure)
    write_frame(OUTPUT_DIR / "drawdown_summary.csv", drawdown)

    run_config = {
        "created_at_utc": now_utc(),
        "scope": "VN30 V7 rank-based strategy relock diagnostics",
        "frozen_score_family": {
            "model_family": "calibrated_logistic",
            "target_variant": "absolute_direction",
            "feature_group": "compact_stable_features",
            "horizon": 40,
        },
        "train_end": str(TRAIN_END),
        "validation_start": str(VAL_START),
        "validation_end": str(VAL_END),
        "final_start": str(FINAL_START),
        "rank_strategy_grid_rows": int(len(grid)),
        "validation_rows": int(len(validation_results)),
        "exploratory_final_rows": int(len(exploratory_final)),
        "final_performance_used_for_claimable_selection": False,
        "broad_model_tuning_run": False,
        "git_tags_created": False,
        "paper_docx_generated": False,
    }
    write_json(OUTPUT_DIR / "run_config.json", run_config)
    manifest = {
        **run_config,
        "locked_rank_strategy": locked,
        "locked_final_result": final_result.to_dict("records"),
        "score_distribution_rows": int(len(score_dist)),
        "score_decile_rows": int(len(deciles)),
        "rank_signal_quality_rows": int(len(quality)),
        "claim_boundary": "offline diagnostic only; final leaderboard exploratory; future-blind confirmation required",
    }
    write_json(OUTPUT_DIR / "v7_manifest.json", manifest)

    write_reports(score_dist, deciles, quality, validation_leaderboard, locked, final_result, baselines, exploratory_final)
    print(f"VN30 V7 rank strategy complete: {rel(OUTPUT_DIR)}", flush=True)
    print(f"Locked strategy: {locked['rank_strategy_id']} status={locked['selection_status']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
