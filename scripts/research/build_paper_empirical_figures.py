"""Build paper-ready empirical figures from generated paper tables.

This script uses matplotlib only and consumes existing repository-derived
tables. It does not fetch data, train models, run benchmarks, or regenerate
row-level predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE_DIR = REPO_ROOT / "reports" / "generated" / "paper_tables_current"
TARGET62_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_target62_paper_ready_stability"
FIG_DIR = REPO_ROOT / "reports" / "generated" / "paper_figures_current"
ROW_LEVEL_TODO = REPO_ROOT / "reports" / "PAPER_ROW_LEVEL_FIGURE_TODO.md"


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def read_table(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / name)


def percent_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, stem: str, report_rows: list[dict[str, Any]], source: str, note: str = "") -> None:
    png = FIG_DIR / f"{stem}.png"
    svg = FIG_DIR / f"{stem}.svg"
    fig.tight_layout()
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    report_rows.append({"figure": png.name, "svg": svg.name, "source": source, "note": note})


def add_reference_lines(ax: plt.Axes, include_60: bool = True, include_62: bool = False) -> None:
    if include_60:
        ax.axhline(60, color="#6b7280", linewidth=1.1, linestyle="--", label="60% reference")
    if include_62:
        ax.axhline(62, color="#9f1239", linewidth=1.1, linestyle=":", label="62% target")


def annotate_bars(ax: plt.Axes, values: list[float], fmt: str = "{:.2f}%") -> None:
    for patch, value in zip(ax.patches, values):
        if patch.get_width() > patch.get_height():
            ax.text(
                patch.get_width() + 0.4,
                patch.get_y() + patch.get_height() / 2,
                fmt.format(value),
                va="center",
                ha="left",
                fontsize=8,
            )
        else:
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 0.6,
                fmt.format(value),
                va="bottom",
                ha="center",
                fontsize=8,
            )


def fig01_research_design_flow(report_rows: list[dict[str, Any]]) -> None:
    labels = [
        "Data",
        "Readiness",
        "Feature\nEngineering",
        "Validation\nSelection",
        "Final\nScoring",
        "Stability\nAudit",
        "Claim\nBoundary",
    ]
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.axis("off")
    x_positions = [i / (len(labels) - 1) for i in range(len(labels))]
    for idx, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x,
            0.55,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#334155", "linewidth": 1.2},
            transform=ax.transAxes,
        )
        if idx < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[idx + 1] - 0.065, 0.55),
                xytext=(x + 0.065, 0.55),
                xycoords=ax.transAxes,
                arrowprops={"arrowstyle": "->", "color": "#334155", "linewidth": 1.4},
            )
    ax.set_title("Research Design Flow", pad=16)
    save_figure(fig, "fig01_research_design_flow", report_rows, "protocol-derived flow from target62 audit artifacts")


def fig02_main_accuracy_comparison(report_rows: list[dict[str, Any]]) -> None:
    main = read_table("table_main_result_summary.csv").iloc[0]
    labels = ["Majority\nbaseline", "RF h60\nreference", "Logistic h40\nbaseline", "Selected\nL2 Logistic h40"]
    values = [
        float(main["majority_baseline_pct"]),
        float(main["rf_h60_reference_pct"]),
        float(main["logistic_baseline_pct"]),
        float(main["final_accuracy_pct"]),
    ]
    colors = ["#94a3b8", "#64748b", "#475569", "#0f766e"]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(labels, values, color=colors)
    add_reference_lines(ax, include_60=True, include_62=True)
    annotate_bars(ax, values)
    ax.set_ylim(0, 72)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_title("Final Accuracy Compared With Baseline References")
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "fig02_main_accuracy_comparison", report_rows, "table_main_result_summary.csv")


def fig03_baseline_lift(report_rows: list[dict[str, Any]]) -> None:
    main = read_table("table_main_result_summary.csv").iloc[0]
    labels = ["vs majority", "vs Logistic h40", "vs RF h60"]
    values = [
        float(main["lift_vs_majority_pp"]),
        float(main["lift_vs_logistic_baseline_pp"]),
        float(main["lift_vs_rf_h60_pp"]),
    ]
    colors = ["#0f766e", "#2563eb", "#7c3aed"]
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#111827", linewidth=1)
    annotate_bars(ax, values, fmt="{:+.2f} pp")
    ax.set_ylabel("Lift (percentage points)")
    ax.set_title("Lift Over Baseline References")
    save_figure(fig, "fig03_baseline_lift", report_rows, "table_main_result_summary.csv")


def fig04_ticker_accuracy(report_rows: list[dict[str, Any]]) -> None:
    ticker = read_table("table_ticker_accuracy.csv").sort_values("accuracy_pct")
    fig_height = max(7.0, 0.24 * len(ticker) + 1.5)
    fig, ax = plt.subplots(figsize=(8.8, fig_height))
    values = percent_series(ticker["accuracy_pct"]).tolist()
    colors = ["#0f766e" if value >= 60 else "#9ca3af" for value in values]
    ax.barh(ticker["ticker"], values, color=colors)
    ax.axvline(60, color="#6b7280", linestyle="--", linewidth=1.1, label="60% reference")
    ax.axvline(62, color="#9f1239", linestyle=":", linewidth=1.1, label="62% target")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Directional accuracy (%)")
    ax.set_title("Ticker-Level Final Accuracy")
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, "fig04_ticker_accuracy", report_rows, "table_ticker_accuracy.csv")


def fig05_monthly_accuracy(report_rows: list[dict[str, Any]]) -> None:
    month = read_table("table_monthly_accuracy.csv").sort_values("month")
    values = percent_series(month["accuracy_pct"]).tolist()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(month["month"], values, marker="o", color="#0f766e", linewidth=2)
    add_reference_lines(ax, include_60=True, include_62=True)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_xlabel("Month")
    ax.set_title("Monthly Final Accuracy")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "fig05_monthly_accuracy", report_rows, "table_monthly_accuracy.csv")


def fig06_quarterly_accuracy(report_rows: list[dict[str, Any]]) -> None:
    quarter = read_table("table_quarterly_accuracy.csv").sort_values("quarter")
    values = percent_series(quarter["accuracy_pct"]).tolist()
    colors = ["#0f766e" if value >= 60 else "#9ca3af" for value in values]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar(quarter["quarter"], values, color=colors)
    add_reference_lines(ax, include_60=True, include_62=True)
    annotate_bars(ax, values)
    ax.set_ylim(0, 90)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_title("Quarterly Final Accuracy")
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "fig06_quarterly_accuracy", report_rows, "table_quarterly_accuracy.csv")


def fig07_regime_accuracy(report_rows: list[dict[str, Any]]) -> None:
    regime = read_table("table_regime_summary.csv").sort_values("regime")
    values = percent_series(regime["accuracy_pct"]).tolist()
    colors = ["#0f766e" if value >= 60 else "#9ca3af" for value in values]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(regime["regime"], values, color=colors)
    add_reference_lines(ax, include_60=True, include_62=True)
    annotate_bars(ax, values)
    ax.set_ylim(0, 82)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_title("Regime Slice Accuracy")
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "fig07_regime_accuracy", report_rows, "table_regime_summary.csv")


def fig08_bootstrap_ci(report_rows: list[dict[str, Any]]) -> None:
    main = read_table("table_main_result_summary.csv").iloc[0]
    bootstrap = pd.read_csv(TARGET62_DIR / "bootstrap_ci.csv").iloc[0]
    point = float(main["final_accuracy_pct"])
    low = float(bootstrap["ci_low"]) * 100.0
    high = float(bootstrap["ci_high"]) * 100.0
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.errorbar([point], [0], xerr=[[point - low], [high - point]], fmt="o", color="#0f766e", ecolor="#334155", elinewidth=2.2, capsize=6)
    ax.axvline(60, color="#6b7280", linestyle="--", linewidth=1.1, label="60% reference")
    ax.axvline(62, color="#9f1239", linestyle=":", linewidth=1.1, label="62% target")
    ax.set_xlim(50, 72)
    ax.set_yticks([])
    ax.set_xlabel("Directional accuracy (%)")
    ax.set_title("Bootstrap Confidence Interval")
    ax.text(point, 0.08, f"{point:.2f}%\nCI {low:.2f}-{high:.2f}%", ha="center", va="bottom")
    ax.legend(loc="upper left", frameon=False)
    save_figure(fig, "fig08_bootstrap_ci", report_rows, "bootstrap_ci.csv and table_main_result_summary.csv")


def fig09_validation_final_gap(report_rows: list[dict[str, Any]]) -> None:
    mismatch = pd.read_csv(TARGET62_DIR / "validation_final_mismatch.csv").iloc[0]
    validation = float(mismatch["validation_accuracy"]) * 100.0
    final = float(mismatch["final_accuracy"]) * 100.0
    gap = float(mismatch["validation_final_gap"]) * 100.0
    labels = ["Validation", "Final"]
    values = [validation, final]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.bar(labels, values, color=["#64748b", "#0f766e"])
    annotate_bars(ax, values)
    ax.annotate(
        f"+{gap:.2f} pp",
        xy=(1, final),
        xytext=(0.5, max(values) + 8),
        ha="center",
        arrowprops={"arrowstyle": "->", "color": "#334155", "linewidth": 1.4},
    )
    ax.set_ylim(0, max(values) + 18)
    ax.set_ylabel("Directional accuracy (%)")
    ax.set_title("Validation-Final Accuracy Gap")
    save_figure(fig, "fig09_validation_final_gap", report_rows, "validation_final_mismatch.csv")


def fig10_data_scope_timeline(report_rows: list[dict[str, Any]]) -> None:
    scope = read_table("table_data_scope_summary.csv")
    rows = []
    for _, row in scope.iterrows():
        start = pd.to_datetime(row["earliest_timestamp"], errors="coerce")
        end = pd.to_datetime(row["latest_timestamp"], errors="coerce")
        if pd.isna(start) or pd.isna(end):
            continue
        rows.append((row["track"], start, end))
    base = pd.Timestamp("2015-01-01")
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for idx, (label, start, end) in enumerate(rows):
        left = (start - base).days
        width = max((end - start).days, 1)
        ax.barh(idx, width, left=left, height=0.48, color="#0f766e" if "hourly" in label.lower() else "#2563eb")
        ax.text(left + width + 40, idx, f"{start.date()} to {end.date()}", va="center", fontsize=8)
    ticks = [pd.Timestamp(f"{year}-01-01") for year in range(2015, 2027)]
    ax.set_xticks([(tick - base).days for tick in ticks])
    ax.set_xticklabels([str(tick.year) for tick in ticks], rotation=45)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([row[0] for row in rows])
    ax.set_xlabel("Calendar year")
    ax.set_title("Data Scope and Coverage Timeline")
    ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)
    save_figure(fig, "fig10_data_scope_timeline", report_rows, "table_data_scope_summary.csv")


def write_row_level_todo() -> None:
    lines = [
        "# Paper Row-Level Figure TODO",
        "",
        "Target62 row-level prediction files were not saved for the selected L2 Logistic h40 candidate. The existing target62 paper-ready audit marks rolling 250/500/1000 row accuracy as unavailable and states that it did not regenerate predictions.",
        "",
        "Unsupported optional figures:",
        "",
        "- Rolling 250/500/1000 row accuracy.",
        "- Prediction correctness over time.",
        "- Error streak plots.",
        "",
        "Do not generate these figures until a protocol-approved artifact provides row-level target62 predictions or an approved rerun creates them.",
        "",
    ]
    ROW_LEVEL_TODO.write_text("\n".join(lines), encoding="utf-8")


def write_build_report(report_rows: list[dict[str, Any]]) -> None:
    report_path = FIG_DIR / "PAPER_FIGURES_BUILD_REPORT.md"
    columns = ["figure", "svg", "source", "note"]
    lines = [
        "# Paper Figures Build Report",
        "",
        "- Data fetch run: no.",
        "- Benchmark run: no.",
        "- Model training run: no.",
        "- Matplotlib only: yes.",
        "- PNG resolution: 300 DPI.",
        "- SVG exported: yes.",
        "- Row-level optional figures generated: no.",
        "",
        "## Generated Figures",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in report_rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    report_rows: list[dict[str, Any]] = []

    fig01_research_design_flow(report_rows)
    fig02_main_accuracy_comparison(report_rows)
    fig03_baseline_lift(report_rows)
    fig04_ticker_accuracy(report_rows)
    fig05_monthly_accuracy(report_rows)
    fig06_quarterly_accuracy(report_rows)
    fig07_regime_accuracy(report_rows)
    fig08_bootstrap_ci(report_rows)
    fig09_validation_final_gap(report_rows)
    fig10_data_scope_timeline(report_rows)
    write_row_level_todo()
    write_build_report(report_rows)

    print(f"paper_figures_generated={len(report_rows)} output_dir={rel(FIG_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
