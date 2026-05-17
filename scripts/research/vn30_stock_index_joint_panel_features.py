"""Feature builder for the VN30 stock + supported-index joint panel."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
if str(REPO_ROOT_BOOTSTRAP) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_BOOTSTRAP))

from scripts.research.index_benchmark_common import INDEX_CODES, read_index_frame
from scripts.research.vn30_hourly_common import REPO_ROOT, VN30_TICKERS, standardize_hourly_frame


REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_stock_index_joint_panel"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_stock_index_joint_panel_training_v1"
STOCK_CACHE = REPO_ROOT / "data" / "hourly_market_split_data"
INDEX_CACHE = REPO_ROOT / "data" / "market_cache" / "vnstock_data" / "indices" / "hourly_2015"
SUPPORTED_INDICES = ("VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VN100")
HORIZONS = (40, 60, 80, 100, 120)
OHLCV = ("open", "high", "low", "close", "volume")
TRAIN_SPLIT = 0.60
VALIDATION_SPLIT = 0.20
BASELINE_STOCK_RF_H60_REFERENCE = 0.6031


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[dict[str, Any]], max_rows: int | None = None) -> str:
    display = rows if max_rows is None else rows[:max_rows]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in display:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def pct(value: Any) -> str:
    try:
        number = float(value)
        if not math.isfinite(number):
            return ""
        return f"{number * 100:.2f}%"
    except (TypeError, ValueError):
        return ""


def load_stock_frame(ticker: str) -> pd.DataFrame:
    path = STOCK_CACHE / f"{ticker}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["datetime", "instrument_code", "instrument_type", *OHLCV])
    raw = pd.read_csv(path, low_memory=False)
    frame = standardize_hourly_frame(raw, ticker)
    if frame.empty:
        return pd.DataFrame(columns=["datetime", "instrument_code", "instrument_type", *OHLCV])
    frame = frame.rename(columns={"ticker": "instrument_code"})
    frame["instrument_type"] = "stock"
    return frame[["datetime", "instrument_code", "instrument_type", *OHLCV]].copy()


def load_index_panel_frame(code: str) -> pd.DataFrame:
    path = INDEX_CACHE / f"{code}.csv"
    frame = read_index_frame(path, code=code, frequency="1H")
    if frame.empty:
        return pd.DataFrame(columns=["datetime", "instrument_code", "instrument_type", *OHLCV])
    frame = frame.rename(columns={"index_code": "instrument_code"})
    frame["instrument_type"] = "index"
    return frame[["datetime", "instrument_code", "instrument_type", *OHLCV]].copy()


def load_joint_ohlcv_panel() -> pd.DataFrame:
    frames = [load_stock_frame(ticker) for ticker in VN30_TICKERS]
    frames.extend(load_index_panel_frame(code) for code in SUPPORTED_INDICES)
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["datetime", "instrument_code", "instrument_type", *OHLCV])
    panel = pd.concat(frames, ignore_index=True)
    panel["datetime"] = pd.to_datetime(panel["datetime"], errors="coerce")
    for column in OHLCV:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel = panel.dropna(subset=["datetime", "instrument_code", "instrument_type", *OHLCV])
    panel["instrument_code"] = panel["instrument_code"].astype(str).str.upper().str.strip()
    panel["instrument_type"] = panel["instrument_type"].astype(str).str.lower().str.strip()
    panel = panel.sort_values(["instrument_code", "datetime"]).drop_duplicates(["instrument_code", "datetime"], keep="last")
    return panel.reset_index(drop=True)


def add_own_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.sort_values(["instrument_code", "datetime"]).copy()
    grouped = out.groupby("instrument_code", group_keys=False)
    out["return_1"] = grouped["close"].pct_change(fill_method=None)
    out["volume_change_1"] = grouped["volume"].pct_change(fill_method=None)
    out["high_low_range"] = (out["high"] - out["low"]) / out["close"].replace(0.0, np.nan)
    out["open_close_spread"] = (out["close"] - out["open"]) / out["open"].replace(0.0, np.nan)
    for lag in (1, 2, 3, 5, 10):
        out[f"return_1_lag_{lag}"] = grouped["return_1"].shift(lag)
    for window in (3, 5, 10, 20):
        min_periods = max(2, window // 2)
        out[f"rolling_return_{window}"] = grouped["close"].pct_change(window, fill_method=None)
        out[f"rolling_volatility_{window}"] = grouped["return_1"].rolling(window, min_periods=min_periods).std().reset_index(level=0, drop=True)
        out[f"momentum_{window}"] = out["close"] / grouped["close"].shift(window) - 1.0
        out[f"range_shock_{window}"] = out["high_low_range"] / grouped["high_low_range"].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True) - 1.0
        out[f"volume_shock_{window}"] = out["volume"] / grouped["volume"].rolling(window, min_periods=min_periods).mean().reset_index(level=0, drop=True) - 1.0
    return out


def build_index_context(panel: pd.DataFrame) -> pd.DataFrame:
    index_panel = panel[panel["instrument_code"].isin(SUPPORTED_INDICES)].copy()
    if index_panel.empty:
        return pd.DataFrame(columns=["datetime"])
    pieces: list[pd.DataFrame] = []
    for code, group in index_panel.groupby("instrument_code"):
        group = group.sort_values("datetime").copy()
        returns = group["close"].pct_change(fill_method=None)
        context = pd.DataFrame({"datetime": group["datetime"]})
        context[f"{code}_lag_return_1"] = returns.shift(1)
        context[f"{code}_lag_return_3"] = group["close"].pct_change(3, fill_method=None).shift(1)
        context[f"{code}_lag_volatility_10"] = returns.rolling(10, min_periods=5).std().shift(1)
        context[f"{code}_lag_trend_10"] = (group["close"] / group["close"].rolling(10, min_periods=5).mean() - 1.0).shift(1)
        pieces.append(context)
    context = pieces[0]
    for piece in pieces[1:]:
        context = pd.merge(context, piece, on="datetime", how="outer")
    return context.sort_values("datetime")


def merge_market_context(panel: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    if context.empty:
        return panel.copy()
    left = panel.sort_values("datetime").copy()
    right = context.sort_values("datetime").copy()
    merged = pd.merge_asof(left, right, on="datetime", direction="backward", allow_exact_matches=False)
    return merged.sort_values(["instrument_code", "datetime"]).reset_index(drop=True)


def add_relative_and_panel_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["instrument_type_flag"] = (out["instrument_type"] == "index").astype(int)
    codes = {code: index for index, code in enumerate(sorted(out["instrument_code"].unique()))}
    out["instrument_code_id"] = out["instrument_code"].map(codes).astype(float)
    out["return_minus_vnindex"] = out["return_1_lag_1"] - out.get("VNINDEX_lag_return_1", np.nan)
    out["return_minus_vn30"] = out["return_1_lag_1"] - out.get("VN30_lag_return_1", np.nan)
    out["vol_minus_vnindex"] = out["rolling_volatility_10"] - out.get("VNINDEX_lag_volatility_10", np.nan)
    out["vol_minus_vn30"] = out["rolling_volatility_10"] - out.get("VN30_lag_volatility_10", np.nan)
    out["hnxindex_minus_vnindex"] = out.get("HNXINDEX_lag_return_1", np.nan) - out.get("VNINDEX_lag_return_1", np.nan)
    out["upcomindex_minus_vnindex"] = out.get("UPCOMINDEX_lag_return_1", np.nan) - out.get("VNINDEX_lag_return_1", np.nan)
    out["vn30_minus_vnindex"] = out.get("VN30_lag_return_1", np.nan) - out.get("VNINDEX_lag_return_1", np.nan)

    sort_cols = ["datetime", "instrument_code"]
    ranked = out.sort_values(sort_cols).copy()
    ranked["prior_return_for_rank"] = ranked.groupby("instrument_code")["return_1"].shift(1)
    ranked["prior_vol_for_rank"] = ranked.groupby("instrument_code")["rolling_volatility_10"].shift(1)
    ranked["prior_volume_shock_for_rank"] = ranked.groupby("instrument_code")["volume_shock_10"].shift(1)
    ranked["cross_sectional_rank_return"] = ranked.groupby("datetime")["prior_return_for_rank"].rank(pct=True)
    ranked["cross_sectional_rank_volatility"] = ranked.groupby("datetime")["prior_vol_for_rank"].rank(pct=True)
    ranked["cross_sectional_rank_volume_shock"] = ranked.groupby("datetime")["prior_volume_shock_for_rank"].rank(pct=True)
    direction = (ranked["prior_return_for_rank"] > 0).astype(float)
    ranked["market_breadth_proxy"] = direction.groupby(ranked["datetime"]).transform("mean")
    return ranked.drop(columns=["prior_return_for_rank", "prior_vol_for_rank", "prior_volume_shock_for_rank"])


def add_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    out = frame.sort_values(["instrument_code", "datetime"]).copy()
    grouped = out.groupby("instrument_code", group_keys=False)
    out["target_horizon"] = horizon
    out["future_close"] = grouped["close"].shift(-horizon)
    out["target_timestamp"] = grouped["datetime"].shift(-horizon)
    out["target_direction"] = (out["future_close"] > out["close"]).astype(float)
    out.loc[out["future_close"].isna(), "target_direction"] = np.nan
    out["previous_direction"] = (grouped["close"].diff() > 0).astype(float)
    out["previous_direction"] = out.groupby("instrument_code")["previous_direction"].shift(1)
    return out


OWN_FEATURES = [
    "return_1_lag_1",
    "return_1_lag_2",
    "return_1_lag_3",
    "return_1_lag_5",
    "return_1_lag_10",
    "rolling_return_3",
    "rolling_return_5",
    "rolling_return_10",
    "rolling_return_20",
    "rolling_volatility_3",
    "rolling_volatility_5",
    "rolling_volatility_10",
    "rolling_volatility_20",
    "momentum_3",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "high_low_range",
    "volume_change_1",
    "range_shock_5",
    "range_shock_10",
    "volume_shock_5",
    "volume_shock_10",
]

MARKET_CONTEXT_FEATURES = [
    f"{code}_{suffix}"
    for code in SUPPORTED_INDICES
    for suffix in ("lag_return_1", "lag_return_3", "lag_volatility_10", "lag_trend_10")
]

RELATIVE_FEATURES = [
    "return_minus_vnindex",
    "return_minus_vn30",
    "vol_minus_vnindex",
    "vol_minus_vn30",
    "hnxindex_minus_vnindex",
    "upcomindex_minus_vnindex",
    "vn30_minus_vnindex",
]

PANEL_FEATURES = [
    "instrument_type_flag",
    "instrument_code_id",
    "cross_sectional_rank_return",
    "cross_sectional_rank_volatility",
    "cross_sectional_rank_volume_shock",
    "market_breadth_proxy",
]

FEATURE_SETS = {
    "own_instrument_features": OWN_FEATURES,
    "own_plus_market_context": OWN_FEATURES + MARKET_CONTEXT_FEATURES,
    "own_plus_relative_features": OWN_FEATURES + MARKET_CONTEXT_FEATURES + RELATIVE_FEATURES,
    "own_plus_panel_features": OWN_FEATURES + PANEL_FEATURES,
    "combined_joint_v1": OWN_FEATURES + MARKET_CONTEXT_FEATURES + RELATIVE_FEATURES + PANEL_FEATURES,
    "combined_joint_regime_v1": OWN_FEATURES + MARKET_CONTEXT_FEATURES + RELATIVE_FEATURES + PANEL_FEATURES,
}


def build_base_feature_panel() -> pd.DataFrame:
    panel = load_joint_ohlcv_panel()
    own = add_own_features(panel)
    context = build_index_context(own)
    merged = merge_market_context(own, context)
    return add_relative_and_panel_features(merged)


def build_model_dataset(horizon: int, feature_set: str) -> tuple[pd.DataFrame, list[str]]:
    base = build_base_feature_panel()
    labeled = add_targets(base, horizon)
    feature_cols = [col for col in FEATURE_SETS[feature_set] if col in labeled.columns]
    required = [
        "datetime",
        "instrument_code",
        "instrument_type",
        "target_horizon",
        "target_timestamp",
        "target_direction",
        "previous_direction",
        *feature_cols,
    ]
    data = labeled[required].copy()
    data = data.dropna(subset=["target_direction"])
    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.reset_index(drop=True), feature_cols


def add_time_splits(data: pd.DataFrame) -> pd.DataFrame:
    out = data.sort_values(["instrument_code", "datetime"]).copy()
    split_values: list[str] = []
    for _code, group in out.groupby("instrument_code", sort=False):
        n = len(group)
        train_end = max(1, int(n * TRAIN_SPLIT))
        validation_end = max(train_end + 1, int(n * (TRAIN_SPLIT + VALIDATION_SPLIT)))
        validation_end = min(validation_end, max(train_end + 1, n - 1)) if n >= 3 else n
        labels = ["train"] * n
        for pos in range(train_end, validation_end):
            if pos < n:
                labels[pos] = "validation"
        for pos in range(validation_end, n):
            labels[pos] = "final"
        split_values.extend(labels)
    out["split"] = split_values
    return out.reset_index(drop=True)


def metric_slice(frame: pd.DataFrame, y_col: str = "target_direction", pred_col: str = "prediction") -> dict[str, Any]:
    if frame.empty:
        return {"accuracy": math.nan, "rows": 0, "correct": 0, "coverage": 0.0, "instrument_count": 0}
    valid = frame[[y_col, pred_col]].dropna()
    rows = int(len(valid))
    correct = int((valid[y_col].astype(int) == valid[pred_col].astype(int)).sum()) if rows else 0
    return {
        "accuracy": correct / rows if rows else math.nan,
        "rows": rows,
        "correct": correct,
        "coverage": rows / len(frame) if len(frame) else 0.0,
        "instrument_count": int(frame.loc[valid.index, "instrument_code"].nunique()) if rows else 0,
    }


def split_metrics(scored: pd.DataFrame, split: str) -> dict[str, Any]:
    selected = scored[scored["split"].eq(split)].copy()
    combined = metric_slice(selected)
    stock = metric_slice(selected[selected["instrument_type"].eq("stock")])
    index = metric_slice(selected[selected["instrument_type"].eq("index")])
    return {
        f"{split}_combined_accuracy": combined["accuracy"],
        f"{split}_stock_only_accuracy": stock["accuracy"],
        f"{split}_index_only_accuracy": index["accuracy"],
        f"{split}_rows_combined": combined["rows"],
        f"{split}_rows_stock_only": stock["rows"],
        f"{split}_rows_index_only": index["rows"],
        f"{split}_coverage_combined": combined["coverage"],
        f"{split}_stock_instrument_count": stock["instrument_count"],
        f"{split}_index_instrument_count": index["instrument_count"],
        f"{split}_total_instrument_count": combined["instrument_count"],
    }


def baseline_prediction(frame: pd.DataFrame, method: str) -> pd.Series:
    if method == "always_up":
        return pd.Series(1, index=frame.index, dtype=float)
    if method == "majority_class":
        train = frame[frame["split"].eq("train")]
        majority = 1 if train["target_direction"].mean() >= 0.5 else 0
        return pd.Series(majority, index=frame.index, dtype=float)
    if method == "previous_direction":
        return frame["previous_direction"].fillna(1).astype(float)
    if method == "moving_average_signal":
        return (frame["rolling_return_5"].fillna(0) > 0).astype(float)
    raise ValueError(f"Unknown baseline: {method}")


def write_feature_inventory_and_leakage_audit() -> None:
    rows: list[dict[str, Any]] = []
    families = {
        "own_instrument_features": OWN_FEATURES,
        "market_context_features": MARKET_CONTEXT_FEATURES,
        "relative_features": RELATIVE_FEATURES,
        "panel_features": PANEL_FEATURES,
    }
    for family, features in families.items():
        for feature in features:
            rows.append(
                {
                    "feature": feature,
                    "family": family,
                    "uses_future_information": False,
                    "uses_target_direction": False,
                    "uses_target_timestamp": False,
                    "notes": "lagged or contemporaneous row metadata only; target columns excluded from features",
                }
            )
    write_csv(REPORT_DIR / "joint_feature_inventory.csv", rows)
    content = [
        "# Joint Panel Feature Leakage Audit",
        "",
        "- Future own returns: not used.",
        "- Future index returns: not used.",
        "- Future market regime: not used.",
        "- Target direction leakage: not used.",
        "- Target timestamp leakage: not used.",
        "- Final-period hand-crafted filters: not used.",
        "- Final accuracy in model selection: forbidden by protocol; training script selection uses validation metrics only.",
        "",
        "Index context features are lagged before as-of joining into instrument rows. Cross-sectional features use prior returns/volatility/volume shock at each timestamp.",
    ]
    (REPORT_DIR / "joint_feature_leakage_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "joint_feature_leakage_audit.md").write_text("\n".join(content) + "\n", encoding="utf-8")


if __name__ == "__main__":
    write_feature_inventory_and_leakage_audit()
    print(f"feature_inventory={rel(REPORT_DIR / 'joint_feature_inventory.csv')}")
