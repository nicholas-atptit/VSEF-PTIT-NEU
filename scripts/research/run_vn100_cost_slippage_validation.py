"""Run cost/slippage-aware VN100 selected-signal diagnostics.

This script maps existing prediction rows to simple long/flat signal proxies.
It does not claim executable trading readiness because official prediction rows
do not contain full entry/exit execution prices, liquidity, or fill assumptions.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "generated" / "evidence_gap_closure"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vn100_cost_slippage_validation"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VN100 cost/slippage diagnostics from existing artifacts.")
    parser.add_argument("--artifact-dir", type=Path, default=OFFICIAL_ARTIFACT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_predictions(artifact_dir: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for frequency in ("daily", "hourly"):
        path = artifact_dir / frequency / "predicted_vs_actual.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "frequency" not in frame.columns:
            frame["frequency"] = frequency
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    data["timestamp_sort"] = pd.to_datetime(
        data["timestamp"] if "timestamp" in data.columns else data["date"],
        errors="coerce",
    )
    for column in ("horizon", "confidence", "actual_return", "actual_direction", "predicted_direction"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["timestamp_sort", "horizon", "actual_return", "predicted_direction"])
    data["frequency"] = data["frequency"].astype(str)
    data["model"] = data["model"].astype(str)
    data["ticker"] = data["ticker"].astype(str)
    if "regime" not in data.columns:
        data["regime"] = ""
    data["regime"] = data["regime"].astype(str)
    return data


def selected_signal_definitions(report_dir: Path) -> list[dict[str, Any]]:
    definitions: list[dict[str, Any]] = [
        {
            "slice_name": "hourly_stacking_h1_confidence_0.57",
            "frequency": "hourly",
            "model": "stacking",
            "horizon": 1,
            "confidence_threshold": 0.57,
            "regime": "",
            "source": "official selected confidence slice",
        },
        {
            "slice_name": "daily_lightgbm_h20_posthoc_bear",
            "frequency": "daily",
            "model": "lightgbm",
            "horizon": 20,
            "confidence_threshold": None,
            "regime": "bear",
            "source": "official post-hoc regime diagnostic",
        },
        {
            "slice_name": "daily_xgboost_h20_posthoc_bear",
            "frequency": "daily",
            "model": "xgboost",
            "horizon": 20,
            "confidence_threshold": None,
            "regime": "bear",
            "source": "official post-hoc regime diagnostic",
        },
    ]

    sweep_path = report_dir / "vn100_full_confidence_sweep_summary.csv"
    if not sweep_path.exists():
        return definitions
    sweep = pd.read_csv(sweep_path)
    if sweep.empty:
        return definitions
    for column in ("filtered_accuracy", "coverage_ratio", "threshold", "horizon"):
        sweep[column] = pd.to_numeric(sweep[column], errors="coerce")
    pass_rows = sweep[(sweep["filtered_accuracy"] >= 0.60) & (sweep["coverage_ratio"] >= 0.20)].copy()
    if pass_rows.empty:
        return definitions
    pass_rows = pass_rows.sort_values(["filtered_accuracy", "coverage_ratio"], ascending=[False, False])
    existing_keys = {(item["frequency"], item["model"], item["horizon"], item.get("confidence_threshold")) for item in definitions}
    added = 0
    for _, row in pass_rows.iterrows():
        key = (row["frequency"], row["model"], int(row["horizon"]), round(float(row["threshold"]), 2))
        if key in existing_keys:
            continue
        definitions.append(
            {
                "slice_name": f"full_sweep_{row['frequency']}_{row['model']}_h{int(row['horizon'])}_t{float(row['threshold']):.2f}",
                "frequency": row["frequency"],
                "model": row["model"],
                "horizon": int(row["horizon"]),
                "confidence_threshold": float(row["threshold"]),
                "regime": "",
                "source": "derived full confidence sweep candidate",
            }
        )
        added += 1
        if added >= 3:
            break
    return definitions


def slice_rows(data: pd.DataFrame, definition: dict[str, Any]) -> pd.DataFrame:
    selected = data[
        (data["frequency"] == definition["frequency"])
        & (data["model"] == definition["model"])
        & (data["horizon"] == definition["horizon"])
    ].copy()
    threshold = definition.get("confidence_threshold")
    if threshold is not None:
        selected = selected[selected["confidence"] >= float(threshold)]
    regime = definition.get("regime")
    if regime:
        selected = selected[selected["regime"] == regime]
    return selected.sort_values(["ticker", "timestamp_sort"]).copy()


def add_positions(rows: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if rows.empty:
        return rows.copy()
    frame = rows.sort_values(["ticker", "timestamp_sort"]).copy()
    if baseline == "model_signal":
        frame["position"] = (frame["predicted_direction"] == 1).astype(int)
    elif baseline in {"buy_and_hold", "always_up"}:
        frame["position"] = 1
    elif baseline == "flat_no_trade":
        frame["position"] = 0
    elif baseline == "previous_direction_signal":
        prior_direction = frame.groupby("ticker")["actual_direction"].shift(1)
        frame["position"] = (prior_direction == 1).astype(int)
    elif baseline == "moving_average_signal":
        prior_return = frame.groupby("ticker")["actual_return"].shift(1)
        trailing_return = (
            prior_return.groupby(frame["ticker"]).rolling(5, min_periods=3).mean().reset_index(level=0, drop=True)
        )
        frame["position"] = (trailing_return > 0).astype(int)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    frame["previous_position"] = frame.groupby("ticker")["position"].shift(1).fillna(0)
    frame["turnover_event"] = (frame["position"] - frame["previous_position"]).abs()
    horizon = frame["horizon"].replace(0, 1)
    frame["period_return_proxy"] = frame["actual_return"] / horizon
    frame["gross_event_return"] = frame["position"] * frame["period_return_proxy"]
    return frame


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def simulate(
    rows: pd.DataFrame,
    definition: dict[str, Any],
    baseline: str,
    transaction_cost_bps: int,
    slippage_bps: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    positioned = add_positions(rows, baseline)
    if positioned.empty:
        summary = {
            "slice_name": definition["slice_name"],
            "baseline": baseline,
            "transaction_cost_bps": transaction_cost_bps,
            "slippage_bps": slippage_bps,
            "row_count": 0,
            "gross_return": "",
            "net_return": "",
            "cost_adjusted_return": "",
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
        return summary, pd.DataFrame(), pd.DataFrame()

    cost_rate = (transaction_cost_bps + slippage_bps) / 10000.0
    positioned["cost_event_return"] = positioned["turnover_event"] * cost_rate
    positioned["net_event_return"] = positioned["gross_event_return"] - positioned["cost_event_return"]
    positioned["trade_event"] = positioned["turnover_event"] > 0
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
    positive_sum = float(positioned.loc[positioned["net_event_return"] > 0, "net_event_return"].sum())
    negative_sum = float(positioned.loc[positioned["net_event_return"] < 0, "net_event_return"].sum())
    profit_factor = positive_sum / abs(negative_sum) if negative_sum < 0 else ""
    summary = {
        "slice_name": definition["slice_name"],
        "source": definition["source"],
        "frequency": definition["frequency"],
        "model": definition["model"],
        "horizon": definition["horizon"],
        "confidence_threshold": definition.get("confidence_threshold") or "",
        "regime": definition.get("regime") or "",
        "baseline": baseline,
        "transaction_cost_bps": transaction_cost_bps,
        "slippage_bps": slippage_bps,
        "row_count": int(len(positioned)),
        "gross_return": gross_return,
        "net_return": net_return,
        "cost_adjusted_return": net_return,
        "cost_drag": net_return - gross_return,
        "turnover": float(positioned["turnover_event"].mean()),
        "max_drawdown": max_drawdown(by_time["net_equity"]),
        "profit_factor": profit_factor,
        "win_rate": float((active_returns > 0).mean()) if len(active_returns) else "",
        "trade_count": int(positioned["trade_event"].sum()),
        "average_trade_return": float(active_returns.mean()) if len(active_returns) else "",
        "exposure": float(positioned["position"].mean()),
        "benchmark_comparison": "",
        "status": "diagnostic_proxy",
    }
    by_time["slice_name"] = definition["slice_name"]
    by_time["baseline"] = baseline
    by_time["transaction_cost_bps"] = transaction_cost_bps
    by_time["slippage_bps"] = slippage_bps
    trade_rows = positioned[positioned["trade_event"]].copy()
    trade_rows["slice_name"] = definition["slice_name"]
    trade_rows["baseline"] = baseline
    trade_rows["transaction_cost_bps"] = transaction_cost_bps
    trade_rows["slippage_bps"] = slippage_bps
    return summary, by_time, trade_rows


def add_benchmark_comparisons(summary_rows: list[dict[str, Any]]) -> None:
    benchmark_returns = {
        (row["slice_name"], row["transaction_cost_bps"], row["slippage_bps"]): row["net_return"]
        for row in summary_rows
        if row["baseline"] == "buy_and_hold" and row["net_return"] != ""
    }
    for row in summary_rows:
        benchmark = benchmark_returns.get((row["slice_name"], row["transaction_cost_bps"], row["slippage_bps"]))
        if benchmark is None or row["net_return"] == "":
            row["benchmark_comparison"] = ""
        else:
            row["benchmark_comparison"] = float(row["net_return"]) - float(benchmark)


def write_plot(equity_rows: pd.DataFrame, path: Path) -> str:
    if equity_rows.empty:
        return "missing: no equity rows"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local optional package
        path.with_suffix(".md").write_text(
            "# VN100 Equity Curve Figure Missing\n\n"
            f"Matplotlib was unavailable, so the PNG figure was not rendered.\n\nError: `{exc}`.\n",
            encoding="utf-8",
        )
        return f"missing: matplotlib unavailable ({exc})"

    plot_rows = equity_rows[equity_rows["baseline"] == "model_signal"].copy()
    if plot_rows.empty:
        return "missing: no model-signal equity rows"
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for slice_name, group in plot_rows.groupby("slice_name"):
        group = group.sort_values("timestamp_sort")
        ax.plot(group["timestamp_sort"], group["net_equity"], linewidth=1, label=slice_name)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_title("VN100 selected-signal net equity proxy (10 bps cost + 10 bps slippage)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Net equity")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return "ready"


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    displayed = rows if max_rows is None else rows[:max_rows]
    for row in displayed:
        lines.append("| " + " | ".join(format_value(row.get(header, ""), header) for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def format_value(value: Any, header: str = "") -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if header in {
            "gross_return",
            "net_return",
            "cost_adjusted_return",
            "cost_drag",
            "turnover",
            "max_drawdown",
            "win_rate",
            "average_trade_return",
            "exposure",
            "benchmark_comparison",
        }:
            return f"{value * 100:.2f}%"
        if header == "profit_factor":
            return f"{value:.4g}"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_report(path: Path, summary_rows: list[dict[str, Any]], figure_status: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_rows = [
        row
        for row in summary_rows
        if row["baseline"] == "model_signal"
        and row["transaction_cost_bps"] == BASE_COST_BPS
        and row["slippage_bps"] == BASE_SLIPPAGE_BPS
    ]
    worst_cost_rows = [
        row
        for row in summary_rows
        if row["baseline"] == "model_signal" and row["transaction_cost_bps"] == 20 and row["slippage_bps"] == 20
    ]
    positive_base = [row for row in base_rows if row["net_return"] != "" and float(row["net_return"]) > 0]
    content = [
        "# VN100 Cost and Slippage Validation Report",
        "",
        "## Source and Method",
        "",
        "- Prediction inputs: official `daily/predicted_vs_actual.csv` and `hourly/predicted_vs_actual.csv`.",
        "- Signal mapping: long when the selected signal predicts upward direction; flat otherwise.",
        "- Return proxy: official target return divided by horizon, so overlapping h-step returns are not compounded as one-period returns.",
        "- Baselines: buy-and-hold, flat/no-trade, always-up, moving-average signal, and previous-direction signal.",
        "- Cost grid: transaction cost bps 5/10/15/20 crossed with slippage bps 5/10/15/20.",
        "- This is a diagnostic proxy using official target returns, not an executable trade simulator with entry/exit prices.",
        "",
        "## Model-Signal Results at 10 bps Cost + 10 bps Slippage",
        "",
        markdown_table(
            [
                "slice_name",
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
            base_rows,
        ),
        "",
        "## Model-Signal Results at 20 bps Cost + 20 bps Slippage",
        "",
        markdown_table(
            [
                "slice_name",
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
            worst_cost_rows,
        ),
        "",
        "## Readiness Interpretation",
        "",
        f"- Figure status: {figure_status}.",
        f"- Positive model-signal net return at the 10/10 bps diagnostic grid: {len(positive_base)} of {len(base_rows)} slices.",
        "- Practical trading readiness is not established because official artifacts still lack execution prices, liquidity filters, fills, and deployment constraints.",
        "- Weak or cost-sensitive rows should be treated as evidence against broad trading-readiness claims.",
        "",
        "## Output Artifacts",
        "",
        f"- Summary CSV: `{rel(path.parent / 'vn100_cost_slippage_summary.csv')}`.",
        f"- Trade list CSV: `{rel(path.parent / 'vn100_trade_list.csv')}`.",
        f"- Equity curve CSV: `{rel(path.parent / 'vn100_equity_curve.csv')}`.",
        f"- Equity curve figure: `{rel(path.parent / 'vn100_equity_curve.png')}`.",
        "",
        "## Claim Boundary",
        "",
        "This report adds cost/slippage-aware diagnostic artifacts. It does not justify profitability, investment suitability,",
        "or live trading readiness claims.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_output_readme(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "# VN100 Cost/Slippage Validation Output\n\n"
        "Detailed committed summaries are under `reports/generated/evidence_gap_closure/`. This ignored output\n"
        "directory marks the requested experiment-output location without storing large generated artifacts in git.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    data = load_predictions(args.artifact_dir)
    definitions = selected_signal_definitions(args.report_dir)
    summary_rows: list[dict[str, Any]] = []
    equity_frames: list[pd.DataFrame] = []
    trade_frames: list[pd.DataFrame] = []

    for definition in definitions:
        rows = slice_rows(data, definition)
        for cost_bps in TRANSACTION_COST_BPS:
            for slippage_bps in SLIPPAGE_BPS:
                for baseline in BASELINES:
                    summary, equity, trades = simulate(rows, definition, baseline, cost_bps, slippage_bps)
                    summary_rows.append(summary)
                    if cost_bps == BASE_COST_BPS and slippage_bps == BASE_SLIPPAGE_BPS:
                        if not equity.empty:
                            equity_frames.append(equity)
                        if not trades.empty:
                            trade_frames.append(trades)

    add_benchmark_comparisons(summary_rows)
    summary_path = args.report_dir / "vn100_cost_slippage_summary.csv"
    trade_path = args.report_dir / "vn100_trade_list.csv"
    equity_path = args.report_dir / "vn100_equity_curve.csv"
    figure_path = args.report_dir / "vn100_equity_curve.png"
    report_path = args.report_dir / "vn100_cost_slippage_validation_report.md"
    write_csv(summary_path, summary_rows)

    trade_columns = [
        "slice_name",
        "baseline",
        "transaction_cost_bps",
        "slippage_bps",
        "timestamp",
        "date",
        "ticker",
        "position",
        "previous_position",
        "turnover_event",
        "actual_return",
        "period_return_proxy",
        "gross_event_return",
        "cost_event_return",
        "net_event_return",
    ]
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=trade_columns)
    trades[[column for column in trade_columns if column in trades.columns]].to_csv(trade_path, index=False)

    equity = pd.concat(equity_frames, ignore_index=True) if equity_frames else pd.DataFrame()
    figure_status = write_plot(equity, figure_path)
    equity_for_csv = equity.copy()
    if not equity_for_csv.empty:
        equity_for_csv["timestamp_sort"] = equity_for_csv["timestamp_sort"].astype(str)
    equity_for_csv.to_csv(equity_path, index=False)
    write_report(report_path, summary_rows, figure_status)
    write_output_readme(args.output_root)
    print(f"Wrote {rel(summary_path)}")
    print(f"Wrote {rel(trade_path)}")
    print(f"Wrote {rel(equity_path)}")
    print(f"Figure status: {figure_status}")
    print(f"Prepared {rel(args.output_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
