"""Run proxy cost/slippage diagnostics for frozen VN30 hourly candidates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.vn30_hourly_common import (  # noqa: E402
    BENCHMARK_OUTPUT_DIR,
    REPORT_ROOT,
    as_bool,
    as_float,
    load_hourly_predictions,
    markdown_table,
    read_csv_rows,
    rel,
    save_placeholder_figure,
)


DEFAULT_OUTPUT_DIR = REPORT_ROOT / "cost_slippage"
DEFAULT_CONFIDENCE_PATH = REPORT_ROOT / "confidence" / "vn30_hourly_confidence_sweep_summary.csv"
DEFAULT_REGIME_PATH = REPORT_ROOT / "regime" / "vn30_hourly_exante_regime_accuracy_summary.csv"
TRANSACTION_COST_BPS = (5, 10, 15, 20)
SLIPPAGE_BPS = (5, 10, 15, 20)
BASE_COST_BPS = 10
BASE_SLIPPAGE_BPS = 10
BASELINES = (
    "model_signal",
    "buy_and_hold",
    "flat_no_trade",
    "always_up",
    "moving_average_signal",
    "previous_direction_signal",
)
SUMMARY_COLUMNS = [
    "slice_name",
    "candidate_source",
    "baseline",
    "transaction_cost_bps",
    "slippage_bps",
    "row_count",
    "gross_return",
    "net_return",
    "turnover",
    "max_drawdown",
    "profit_factor",
    "win_rate",
    "trade_count",
    "average_trade_return",
    "exposure",
    "benchmark_comparison",
    "status",
]
TRADE_COLUMNS = [
    "slice_name",
    "baseline",
    "timestamp",
    "ticker",
    "position",
    "previous_position",
    "turnover_event",
    "gross_event_return",
    "net_event_return",
]
EQUITY_COLUMNS = [
    "slice_name",
    "baseline",
    "timestamp",
    "gross_period_return",
    "net_period_return",
    "turnover_period",
    "exposure",
    "gross_equity",
    "net_equity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN30 hourly proxy cost/slippage diagnostics.")
    parser.add_argument("--artifact-dir", type=Path, default=BENCHMARK_OUTPUT_DIR)
    parser.add_argument("--confidence-summary", type=Path, default=DEFAULT_CONFIDENCE_PATH)
    parser.add_argument("--regime-summary", type=Path, default=DEFAULT_REGIME_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def selected_confidence_candidates(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    candidates: list[dict[str, Any]] = []
    selected_flags = [
        "selected_at_50pct_floor",
        "selected_at_40pct_floor",
        "selected_at_30pct_floor",
        "selected_at_20pct_floor",
    ]
    for row in rows:
        if not any(as_bool(row.get(flag)) for flag in selected_flags):
            continue
        threshold = as_float(row.get("threshold"))
        horizon = as_float(row.get("horizon"))
        if threshold is None or horizon is None:
            continue
        candidates.append(
            {
                "slice_name": f"confidence_{row.get('model')}_h{int(horizon)}_t{threshold:.2f}",
                "candidate_source": "confidence_sweep",
                "model": str(row.get("model")),
                "horizon": int(horizon),
                "confidence_threshold": threshold,
                "regime": "",
            }
        )
    unique: dict[tuple[str, int, float], dict[str, Any]] = {}
    for item in candidates:
        unique[(item["model"], item["horizon"], item["confidence_threshold"])] = item
    return list(unique.values())[:8]


def selected_regime_candidates(path: Path) -> list[dict[str, Any]]:
    rows = read_csv_rows(path)
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("regime_source", "")) != "exante_proxy":
            continue
        if not as_bool(row.get("passed_60pct")):
            continue
        horizon = as_float(row.get("horizon"))
        if horizon is None:
            continue
        candidates.append(
            {
                "slice_name": f"regime_{row.get('model')}_h{int(horizon)}_{row.get('regime')}_{row.get('ticker')}",
                "candidate_source": "exante_regime",
                "model": str(row.get("model")),
                "horizon": int(horizon),
                "confidence_threshold": None,
                "regime": str(row.get("regime")),
                "ticker": str(row.get("ticker")),
            }
        )
    return candidates[:8]


def candidate_rows(data: pd.DataFrame, candidate: dict[str, Any]) -> pd.DataFrame:
    selected = data[
        (data["frequency"].astype(str).str.lower() == "hourly")
        & (data["model"].astype(str) == candidate["model"])
        & (pd.to_numeric(data["horizon"], errors="coerce") == int(candidate["horizon"]))
    ].copy()
    threshold = candidate.get("confidence_threshold")
    if threshold is not None and "confidence" in selected.columns:
        selected = selected[selected["confidence"] >= float(threshold)]
    if candidate.get("ticker"):
        selected = selected[selected["ticker"].astype(str) == candidate["ticker"]]
    if candidate.get("regime"):
        if "regime" in selected.columns:
            selected = selected[selected["regime"].astype(str) == str(candidate["regime"])]
    return selected.sort_values(["ticker", "timestamp_sort"])


def add_positions(rows: pd.DataFrame, baseline: str) -> pd.DataFrame:
    frame = rows.sort_values(["ticker", "timestamp_sort"]).copy()
    if frame.empty:
        return frame
    if baseline == "model_signal":
        frame["position"] = (pd.to_numeric(frame["predicted_direction"], errors="coerce") == 1).astype(int)
    elif baseline in {"buy_and_hold", "always_up"}:
        frame["position"] = 1
    elif baseline == "flat_no_trade":
        frame["position"] = 0
    elif baseline == "previous_direction_signal":
        prior_direction = frame.groupby("ticker")["actual_direction"].shift(1)
        frame["position"] = (prior_direction == 1).astype(int)
    elif baseline == "moving_average_signal":
        prior_return = pd.to_numeric(frame["actual_return"], errors="coerce").groupby(frame["ticker"]).shift(1)
        trailing = prior_return.groupby(frame["ticker"]).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
        frame["position"] = (trailing > 0).astype(int)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    frame["previous_position"] = frame.groupby("ticker")["position"].shift(1).fillna(0)
    frame["turnover_event"] = (frame["position"] - frame["previous_position"]).abs()
    frame["gross_event_return"] = frame["position"] * pd.to_numeric(frame["actual_return"], errors="coerce").fillna(0.0)
    return frame


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def simulate(
    rows: pd.DataFrame,
    candidate: dict[str, Any],
    baseline: str,
    transaction_cost_bps: int,
    slippage_bps: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    positioned = add_positions(rows, baseline)
    if positioned.empty:
        summary = {
            "slice_name": candidate["slice_name"],
            "candidate_source": candidate["candidate_source"],
            "baseline": baseline,
            "transaction_cost_bps": transaction_cost_bps,
            "slippage_bps": slippage_bps,
            "row_count": 0,
            "gross_return": "",
            "net_return": "",
            "turnover": "",
            "max_drawdown": "",
            "profit_factor": "",
            "win_rate": "",
            "trade_count": 0,
            "average_trade_return": "",
            "exposure": "",
            "benchmark_comparison": "",
            "status": "missing_rows",
        }
        return summary, pd.DataFrame(columns=TRADE_COLUMNS), pd.DataFrame(columns=EQUITY_COLUMNS)
    cost_rate = (int(transaction_cost_bps) + int(slippage_bps)) / 10000.0
    positioned["net_event_return"] = positioned["gross_event_return"] - positioned["turnover_event"] * cost_rate
    by_time = (
        positioned.groupby("timestamp_sort")
        .agg(
            gross_period_return=("gross_event_return", "mean"),
            net_period_return=("net_event_return", "mean"),
            turnover_period=("turnover_event", "mean"),
            exposure=("position", "mean"),
        )
        .reset_index()
        .sort_values("timestamp_sort")
    )
    by_time["gross_equity"] = (1.0 + by_time["gross_period_return"]).cumprod()
    by_time["net_equity"] = (1.0 + by_time["net_period_return"]).cumprod()
    gross_return = float(by_time["gross_equity"].iloc[-1] - 1.0)
    net_return = float(by_time["net_equity"].iloc[-1] - 1.0)
    active_returns = positioned.loc[positioned["position"] > 0, "net_event_return"]
    wins = active_returns[active_returns > 0]
    losses = active_returns[active_returns < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if float(abs(losses.sum())) > 0 else None
    trade_count = int((positioned["turnover_event"] > 0).sum())
    summary = {
        "slice_name": candidate["slice_name"],
        "candidate_source": candidate["candidate_source"],
        "baseline": baseline,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "row_count": int(len(positioned)),
        "gross_return": gross_return,
        "net_return": net_return,
        "turnover": float(positioned["turnover_event"].mean()),
        "max_drawdown": max_drawdown(by_time["net_equity"]),
        "profit_factor": profit_factor,
        "win_rate": float((active_returns > 0).mean()) if len(active_returns) else 0.0,
        "trade_count": trade_count,
        "average_trade_return": float(active_returns.mean()) if len(active_returns) else 0.0,
        "exposure": float(positioned["position"].mean()),
        "benchmark_comparison": "",
        "status": "proxy_diagnostic",
    }
    trade_list = positioned.loc[positioned["turnover_event"] > 0].copy()
    if not trade_list.empty:
        trade_list = trade_list.assign(
            slice_name=candidate["slice_name"],
            baseline=baseline,
            timestamp=trade_list["timestamp_sort"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        )[TRADE_COLUMNS]
    equity_curve = by_time.assign(
        slice_name=candidate["slice_name"],
        baseline=baseline,
        timestamp=by_time["timestamp_sort"].dt.strftime("%Y-%m-%d %H:%M:%S"),
    )[EQUITY_COLUMNS]
    return summary, trade_list, equity_curve


def add_benchmark_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    index = {
        (row["slice_name"], row["transaction_cost_bps"], row["slippage_bps"]): row
        for row in rows
        if row.get("baseline") == "buy_and_hold" and isinstance(row.get("net_return"), float)
    }
    for row in rows:
        benchmark = index.get((row["slice_name"], row["transaction_cost_bps"], row["slippage_bps"]))
        if benchmark and isinstance(row.get("net_return"), float):
            row["benchmark_comparison"] = float(row["net_return"] - benchmark["net_return"])
    return rows


def run_diagnostics(data: pd.DataFrame, confidence_path: Path, regime_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if data.empty:
        return (
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            pd.DataFrame(columns=TRADE_COLUMNS),
            pd.DataFrame(columns=EQUITY_COLUMNS),
        )
    candidates = selected_confidence_candidates(confidence_path) + selected_regime_candidates(regime_path)
    if not candidates:
        return (
            pd.DataFrame(columns=SUMMARY_COLUMNS),
            pd.DataFrame(columns=TRADE_COLUMNS),
            pd.DataFrame(columns=EQUITY_COLUMNS),
        )
    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    equity_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        rows = candidate_rows(data, candidate)
        for baseline in BASELINES:
            for cost_bps in TRANSACTION_COST_BPS:
                for slippage_bps in SLIPPAGE_BPS:
                    summary, trades, equity = simulate(rows, candidate, baseline, cost_bps, slippage_bps)
                    summary_rows.append(summary)
                    if cost_bps == BASE_COST_BPS and slippage_bps == BASE_SLIPPAGE_BPS:
                        if not trades.empty:
                            trade_frames.append(trades)
                        if not equity.empty:
                            equity_frames.append(equity)
    summary_rows = add_benchmark_comparisons(summary_rows)
    trade_list = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=TRADE_COLUMNS)
    equity_curve = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame(columns=EQUITY_COLUMNS)
    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS), trade_list, equity_curve


def write_report(path: Path, artifact_dir: Path, summary: pd.DataFrame) -> None:
    if summary.empty:
        top_rows: list[dict[str, Any]] = []
    else:
        top_rows = (
            summary.sort_values(["net_return", "row_count"], ascending=[False, False])
            .head(20)
            .to_dict("records")
        )
    content = [
        "# VN30 Hourly Cost/Slippage Proxy Validation Report",
        "",
        "## Source",
        "",
        f"- Prediction artifact: `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        "- Frequency: hourly only.",
        "- Cost grid: transaction cost bps 5, 10, 15, 20; slippage bps 5, 10, 15, 20.",
        "- Baselines: buy-and-hold, flat/no-trade, always-up, moving-average signal, previous-direction signal.",
        "",
        "## Top Proxy Rows",
        "",
        markdown_table(
            [
                "slice_name",
                "baseline",
                "transaction_cost_bps",
                "slippage_bps",
                "row_count",
                "gross_return",
                "net_return",
                "turnover",
                "max_drawdown",
                "profit_factor",
                "win_rate",
                "trade_count",
                "exposure",
                "benchmark_comparison",
            ],
            top_rows,
        )
        if top_rows
        else "No cost/slippage proxy diagnostics are available.",
        "",
        "## Boundary",
        "",
        "This remains proxy diagnostics. It does not establish live trading readiness because real entry/exit execution prices, liquidity filters, and fill assumptions are not implemented.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(content), encoding="utf-8")


def write_plot(path: Path, equity_curve: pd.DataFrame) -> str:
    if equity_curve.empty:
        return save_placeholder_figure(
            path,
            "VN30 hourly proxy equity curve",
            "No official hourly predictions or selected candidates were available for cost/slippage diagnostics.",
        )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        working = equity_curve.copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
        groups = working[["slice_name", "baseline"]].drop_duplicates().head(8)
        fig, ax = plt.subplots(figsize=(9, 5))
        for row in groups.itertuples(index=False):
            group = working[(working["slice_name"] == row.slice_name) & (working["baseline"] == row.baseline)].sort_values("timestamp")
            ax.plot(group["timestamp"], group["net_equity"], linewidth=1, label=f"{row.slice_name} {row.baseline}")
        ax.set_title("VN30 hourly cost/slippage proxy equity curve")
        ax.set_ylabel("Net equity")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        return "rendered"
    except Exception as exc:
        return save_placeholder_figure(path, "VN30 hourly proxy equity curve", f"Plot rendering failed: {exc}")


def main() -> int:
    args = parse_args()
    data = load_hourly_predictions(args.artifact_dir)
    summary, trade_list, equity_curve = run_diagnostics(data, args.confidence_summary, args.regime_summary)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "vn30_hourly_cost_slippage_summary.csv"
    report_path = args.output_dir / "vn30_hourly_cost_slippage_validation_report.md"
    trade_path = args.output_dir / "vn30_hourly_trade_list.csv"
    equity_path = args.output_dir / "vn30_hourly_equity_curve.csv"
    figure_path = args.output_dir / "vn30_hourly_equity_curve.png"
    summary.to_csv(summary_path, index=False)
    trade_list.to_csv(trade_path, index=False)
    equity_curve.to_csv(equity_path, index=False)
    write_report(report_path, args.artifact_dir, summary)
    figure_status = write_plot(figure_path, equity_curve)
    print(
        "VN30 hourly cost/slippage diagnostics complete: "
        f"rows={len(summary)} report={rel(report_path)} figure={rel(figure_path)} status={figure_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
