"""Cost/slippage proxy diagnostics for VN30 hourly 2015 benchmark signals."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

PREDICTIONS_PATH = REPO_ROOT / "outputs" / "vn30_hourly_2015_jan2025_benchmark" / "hourly" / "predicted_vs_actual.csv"
OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark" / "cost_slippage"
SUMMARY_PATH = OUTPUT_DIR / "vn30_cost_slippage_proxy_summary.csv"
REPORT_PATH = OUTPUT_DIR / "vn30_cost_slippage_proxy_report.md"

TRANSACTION_COST_BPS_GRID = [0, 5, 10, 20]
SLIPPAGE_BPS_GRID = [0, 5, 10, 20]
STRATEGIES = ["long_flat", "direction_following_long_short"]
SUMMARY_COLUMNS = [
    "model",
    "horizon",
    "strategy",
    "transaction_cost_bps",
    "slippage_bps",
    "total_cost_bps",
    "row_count",
    "ticker_count",
    "start_timestamp",
    "end_timestamp",
    "gross_return",
    "net_return",
    "turnover",
    "max_drawdown",
    "win_rate",
    "profit_factor",
    "trade_count",
    "exposure",
    "diagnostic_only",
    "claim_boundary",
]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100:.2f}%"


def fmt_num(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.4g}"


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int = 20) -> str:
    if not rows:
        return "No rows available."
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows[:max_rows]:
        values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    if len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def load_predictions(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark predictions: {rel(path)}")
    frame = pd.read_csv(path, low_memory=False)
    required = {"timestamp", "ticker", "model", "horizon", "frequency", "actual_return", "predicted_direction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")
    frame = frame[frame["frequency"].astype(str).str.lower().eq("hourly")].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="coerce")
    frame["actual_return"] = pd.to_numeric(frame["actual_return"], errors="coerce")
    frame["predicted_direction"] = pd.to_numeric(frame["predicted_direction"], errors="coerce")
    frame = frame.dropna(subset=["timestamp", "ticker", "model", "horizon", "actual_return", "predicted_direction"])
    frame = frame[frame["predicted_direction"].isin([0, 1])].copy()
    frame["horizon"] = frame["horizon"].astype(int)
    frame["predicted_direction"] = frame["predicted_direction"].astype(int)
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    return frame


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def profit_factor(returns: pd.Series) -> float:
    positive = float(returns[returns > 0.0].sum())
    negative = float(returns[returns < 0.0].sum())
    if negative == 0.0:
        return float("inf") if positive > 0.0 else float("nan")
    return positive / abs(negative)


def add_strategy_columns(group: pd.DataFrame, strategy: str, total_cost_bps: float) -> pd.DataFrame:
    prepared = group.sort_values(["ticker", "timestamp"]).copy()
    if strategy == "long_flat":
        prepared["position"] = prepared["predicted_direction"].astype(float)
    elif strategy == "direction_following_long_short":
        prepared["position"] = np.where(prepared["predicted_direction"].astype(int).eq(1), 1.0, -1.0)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    previous = prepared.groupby("ticker", sort=False)["position"].shift(1).fillna(0.0)
    prepared["turnover_step"] = (prepared["position"] - previous).abs()
    cost_rate = float(total_cost_bps) / 10000.0
    prepared["gross_row_return"] = prepared["position"] * prepared["actual_return"]
    prepared["net_row_return"] = prepared["gross_row_return"] - prepared["turnover_step"] * cost_rate
    return prepared


def summarize_group(group: pd.DataFrame, *, strategy: str, transaction_cost_bps: int, slippage_bps: int) -> dict[str, Any]:
    total_cost_bps = int(transaction_cost_bps) + int(slippage_bps)
    prepared = add_strategy_columns(group, strategy, total_cost_bps)
    by_timestamp = (
        prepared.groupby("timestamp", sort=True)[["gross_row_return", "net_row_return"]]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    gross_equity = (1.0 + by_timestamp["gross_row_return"]).cumprod()
    net_equity = (1.0 + by_timestamp["net_row_return"]).cumprod()
    row_returns = prepared["net_row_return"].replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "model": str(group["model"].iloc[0]),
        "horizon": int(group["horizon"].iloc[0]),
        "strategy": strategy,
        "transaction_cost_bps": int(transaction_cost_bps),
        "slippage_bps": int(slippage_bps),
        "total_cost_bps": int(total_cost_bps),
        "row_count": int(len(prepared)),
        "ticker_count": int(prepared["ticker"].nunique()),
        "start_timestamp": prepared["timestamp"].min().strftime("%Y-%m-%d %H:%M:%S"),
        "end_timestamp": prepared["timestamp"].max().strftime("%Y-%m-%d %H:%M:%S"),
        "gross_return": float(gross_equity.iloc[-1] - 1.0) if not gross_equity.empty else float("nan"),
        "net_return": float(net_equity.iloc[-1] - 1.0) if not net_equity.empty else float("nan"),
        "turnover": float(prepared["turnover_step"].mean()),
        "max_drawdown": max_drawdown(net_equity),
        "win_rate": float((row_returns > 0.0).mean()) if not row_returns.empty else float("nan"),
        "profit_factor": profit_factor(row_returns),
        "trade_count": int((prepared["turnover_step"] > 0.0).sum()),
        "exposure": float(prepared["position"].abs().mean()),
        "diagnostic_only": True,
        "claim_boundary": "proxy_only_not_executable_trading_performance",
    }


def build_summary(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if predictions.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)
    for (_model, _horizon), group in predictions.groupby(["model", "horizon"], sort=True):
        for strategy in STRATEGIES:
            for transaction_cost_bps in TRANSACTION_COST_BPS_GRID:
                for slippage_bps in SLIPPAGE_BPS_GRID:
                    rows.append(
                        summarize_group(
                            group,
                            strategy=strategy,
                            transaction_cost_bps=transaction_cost_bps,
                            slippage_bps=slippage_bps,
                        )
                    )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def top_rows(summary: pd.DataFrame) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    working = summary.copy()
    working["net_return"] = pd.to_numeric(working["net_return"], errors="coerce")
    working = working.sort_values(["net_return", "row_count"], ascending=[False, False]).head(20)
    rows: list[dict[str, Any]] = []
    for row in working.itertuples(index=False):
        rows.append(
            {
                "model": row.model,
                "horizon": int(row.horizon),
                "strategy": row.strategy,
                "cost_bps": int(row.transaction_cost_bps),
                "slippage_bps": int(row.slippage_bps),
                "net_return": fmt_pct(row.net_return),
                "max_drawdown": fmt_pct(row.max_drawdown),
                "win_rate": fmt_pct(row.win_rate),
                "profit_factor": fmt_num(row.profit_factor),
                "trade_count": int(row.trade_count),
                "exposure": fmt_pct(row.exposure),
            }
        )
    return rows


def best_standard_cost_row(summary: pd.DataFrame) -> dict[str, Any] | None:
    if summary.empty:
        return None
    standard = summary[(summary["transaction_cost_bps"] == 10) & (summary["slippage_bps"] == 10)].copy()
    if standard.empty:
        return None
    standard["net_return"] = pd.to_numeric(standard["net_return"], errors="coerce")
    return standard.sort_values(["net_return", "row_count"], ascending=[False, False]).iloc[0].to_dict()


def write_report(summary: pd.DataFrame) -> None:
    standard = best_standard_cost_row(summary)
    content = [
        "# VN30 Hourly 2015 Cost/Slippage Proxy Diagnostics",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(PREDICTIONS_PATH)}`.",
        "- Frequency: hourly only.",
        "- Signal proxy: model predicted direction applied to realized benchmark prediction returns.",
        f"- Strategies: {', '.join(STRATEGIES)}.",
        f"- Transaction cost grid bps: {', '.join(str(item) for item in TRANSACTION_COST_BPS_GRID)}.",
        f"- Slippage grid bps: {', '.join(str(item) for item in SLIPPAGE_BPS_GRID)}.",
        "",
        "## Top Proxy Rows",
        "",
        markdown_table(
            [
                "model",
                "horizon",
                "strategy",
                "cost_bps",
                "slippage_bps",
                "net_return",
                "max_drawdown",
                "win_rate",
                "profit_factor",
                "trade_count",
                "exposure",
            ],
            top_rows(summary),
        ),
        "",
        "## Standard 10 bps Cost + 10 bps Slippage Diagnostic",
        "",
    ]
    if standard is None:
        content.append("No standard-cost diagnostic row is available.")
    else:
        content.extend(
            [
                f"- Best row: {standard['model']} h={int(standard['horizon'])} {standard['strategy']}.",
                f"- Net return proxy: {fmt_pct(standard['net_return'])}.",
                f"- Max drawdown proxy: {fmt_pct(standard['max_drawdown'])}.",
                f"- Win rate: {fmt_pct(standard['win_rate'])}.",
                f"- Trade count: {int(standard['trade_count'])}.",
                f"- Exposure: {fmt_pct(standard['exposure'])}.",
            ]
        )
    content.extend(
        [
            "",
            "## Boundary",
            "",
            "- This is a signal diagnostic proxy, not an executable backtest.",
            "- It does not model order book depth, fill probability, liquidity filters, position sizing, borrow costs, taxes, or real execution constraints.",
            "- No trading-readiness or profitability claim is made.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(content), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly 2015 cost/slippage proxy diagnostics.")
    parser.add_argument("--predictions", type=Path, default=PREDICTIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    global PREDICTIONS_PATH, OUTPUT_DIR, SUMMARY_PATH, REPORT_PATH
    PREDICTIONS_PATH = args.predictions
    OUTPUT_DIR = args.output_dir
    SUMMARY_PATH = OUTPUT_DIR / "vn30_cost_slippage_proxy_summary.csv"
    REPORT_PATH = OUTPUT_DIR / "vn30_cost_slippage_proxy_report.md"

    predictions = load_predictions(PREDICTIONS_PATH)
    summary = build_summary(predictions)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    write_report(summary)
    print(f"VN30 cost/slippage proxy diagnostics complete: rows={len(summary)} report={rel(REPORT_PATH)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
