"""Build a full VN100 confidence-threshold sweep from existing predictions.

The script does not train models or rerun benchmarks. It reads official
predicted-vs-actual rows and derives coverage/accuracy diagnostics for every
available frequency/model/horizon combination.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
EXPANDED_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_expanded_cache"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "evidence_gap_closure"
COVERAGE_FLOORS = (0.50, 0.40, 0.30, 0.20)
THRESHOLDS = [round(value / 100, 2) for value in range(50, 91)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive VN100 confidence-threshold sweep diagnostics.")
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def choose_artifact_dir(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        return explicit, "explicit"
    if (EXPANDED_ARTIFACT_DIR / "daily" / "predicted_vs_actual.csv").exists() or (
        EXPANDED_ARTIFACT_DIR / "hourly" / "predicted_vs_actual.csv"
    ).exists():
        return EXPANDED_ARTIFACT_DIR, "expanded"
    return OFFICIAL_ARTIFACT_DIR, "current_official"


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
    for column in ("horizon", "confidence", "is_correct"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data["model"] = data["model"].astype(str)
    data["frequency"] = data["frequency"].astype(str)
    data["ticker"] = data["ticker"].astype(str)
    return data.dropna(subset=["frequency", "model", "horizon", "confidence", "is_correct"])


def build_sweep(data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["frequency", "model", "horizon"]
    for (frequency, model, horizon), group in data.groupby(group_cols, sort=True):
        total_rows = int(len(group))
        for threshold in THRESHOLDS:
            selected = group[group["confidence"] >= threshold]
            evaluated_rows = int(len(selected))
            coverage = evaluated_rows / total_rows if total_rows else 0.0
            accuracy = float(selected["is_correct"].mean()) if evaluated_rows else None
            row: dict[str, Any] = {
                "frequency": frequency,
                "model": model,
                "horizon": int(horizon),
                "threshold": threshold,
                "total_rows": total_rows,
                "evaluated_rows": evaluated_rows,
                "coverage_ratio": coverage,
                "filtered_accuracy": accuracy,
                "passed_60pct": bool(accuracy is not None and accuracy >= 0.60),
            }
            for floor in COVERAGE_FLOORS:
                row[f"coverage_ok_{int(floor * 100)}pct"] = bool(coverage >= floor)
                row[f"selected_at_{int(floor * 100)}pct_floor"] = False
            rows.append(row)

    sweep = pd.DataFrame(rows)
    if sweep.empty:
        return sweep

    for floor in COVERAGE_FLOORS:
        flag = f"selected_at_{int(floor * 100)}pct_floor"
        eligible = sweep[(sweep["coverage_ratio"] >= floor) & sweep["filtered_accuracy"].notna()].copy()
        if eligible.empty:
            continue
        eligible = eligible.sort_values(
            ["filtered_accuracy", "coverage_ratio", "evaluated_rows", "threshold"],
            ascending=[False, False, False, True],
        )
        best_index = eligible.index[0]
        sweep.loc[best_index, flag] = True
    return sweep


def concentration_for_candidate(data: pd.DataFrame, candidate: pd.Series, floor: float) -> dict[str, Any]:
    selected = data[
        (data["frequency"] == candidate["frequency"])
        & (data["model"] == candidate["model"])
        & (data["horizon"] == candidate["horizon"])
        & (data["confidence"] >= candidate["threshold"])
    ]
    if selected.empty:
        return {
            "coverage_floor": f"{int(floor * 100)}%",
            "candidate": "",
            "ticker_count": 0,
            "top_ticker": "",
            "top1_prediction_share": "",
            "top3_prediction_share": "",
            "assessment": "missing",
        }
    counts = selected.groupby("ticker").size().sort_values(ascending=False)
    total = float(counts.sum())
    top1 = float(counts.iloc[0] / total)
    top3 = float(counts.head(3).sum() / total)
    if top1 >= 0.35 or top3 >= 0.70:
        assessment = "high"
    elif top1 >= 0.25 or top3 >= 0.60:
        assessment = "moderate"
    else:
        assessment = "low"
    return {
        "coverage_floor": f"{int(floor * 100)}%",
        "candidate": candidate_label(candidate),
        "ticker_count": int(counts.shape[0]),
        "top_ticker": str(counts.index[0]),
        "top1_prediction_share": top1,
        "top3_prediction_share": top3,
        "assessment": assessment,
    }


def candidate_label(row: pd.Series | dict[str, Any]) -> str:
    return f"{row['frequency']} {row['model']} h={int(row['horizon'])} threshold {float(row['threshold']):.2f}"


def best_candidates_by_floor(sweep: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if sweep.empty:
        return rows
    for floor in COVERAGE_FLOORS:
        flag = f"selected_at_{int(floor * 100)}pct_floor"
        selected = sweep[sweep[flag] == True]  # noqa: E712
        if selected.empty:
            rows.append(
                {
                    "coverage_floor": f">= {int(floor * 100)}%",
                    "candidate": "missing",
                    "evaluated_rows": "",
                    "coverage_ratio": "",
                    "filtered_accuracy": "",
                    "passed_60pct": "",
                }
            )
            continue
        row = selected.iloc[0]
        rows.append(
            {
                "coverage_floor": f">= {int(floor * 100)}%",
                "candidate": candidate_label(row),
                "evaluated_rows": int(row["evaluated_rows"]),
                "coverage_ratio": float(row["coverage_ratio"]),
                "filtered_accuracy": float(row["filtered_accuracy"]),
                "passed_60pct": bool(row["passed_60pct"]),
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    displayed = rows if max_rows is None else rows[:max_rows]
    for row in displayed:
        lines.append("| " + " | ".join(format_value(row.get(header, "")) for header in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + ["" for _ in headers[1:]]) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if 0 <= value <= 1:
            return f"{value * 100:.2f}%"
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def write_plot(sweep: pd.DataFrame, output_path: Path) -> str:
    if sweep.empty:
        return "missing: no sweep rows"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on local optional package
        note_path = output_path.with_suffix(".md")
        note_path.write_text(
            "# VN100 Confidence Threshold Figure Missing\n\n"
            f"Matplotlib was unavailable, so the PNG figure was not rendered.\n\nError: `{exc}`.\n",
            encoding="utf-8",
        )
        return f"missing: matplotlib unavailable ({exc})"

    selected_groups = sweep[
        sweep[[f"selected_at_{int(floor * 100)}pct_floor" for floor in COVERAGE_FLOORS]].any(axis=1)
    ][["frequency", "model", "horizon"]].drop_duplicates()
    if selected_groups.empty:
        selected_groups = sweep[["frequency", "model", "horizon"]].drop_duplicates().head(4)

    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for _, group_row in selected_groups.head(4).iterrows():
        group = sweep[
            (sweep["frequency"] == group_row["frequency"])
            & (sweep["model"] == group_row["model"])
            & (sweep["horizon"] == group_row["horizon"])
        ].sort_values("threshold")
        label = f"{group_row['frequency']} {group_row['model']} h={int(group_row['horizon'])}"
        axes[0].plot(group["threshold"], group["filtered_accuracy"], marker="o", markersize=3, linewidth=1, label=label)
        axes[1].plot(group["threshold"], group["coverage_ratio"], marker="o", markersize=3, linewidth=1, label=label)
    axes[0].axhline(0.60, color="black", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("VN100 confidence threshold sweep")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.30, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Confidence threshold")
    axes[1].set_ylabel("Coverage")
    axes[1].grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return "ready"


def write_report(
    path: Path,
    artifact_dir: Path,
    artifact_mode: str,
    data: pd.DataFrame,
    sweep: pd.DataFrame,
    figure_status: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    daily_official_rows = pd.read_csv(artifact_dir / "daily" / "confidence_threshold_sweep_summary.csv").shape[0] if (
        artifact_dir / "daily" / "confidence_threshold_sweep_summary.csv"
    ).exists() else 0
    combos = data.groupby(["frequency", "model", "horizon"]).size().reset_index(name="rows") if not data.empty else pd.DataFrame()
    best_rows = best_candidates_by_floor(sweep)
    pass_rows = sweep[(sweep["passed_60pct"] == True) & (sweep["coverage_ratio"] >= 0.20)].copy() if not sweep.empty else pd.DataFrame()  # noqa: E712
    if not pass_rows.empty:
        pass_rows = pass_rows.sort_values(["coverage_ratio", "filtered_accuracy"], ascending=[False, False])
    concentration_rows = []
    for floor in COVERAGE_FLOORS:
        flag = f"selected_at_{int(floor * 100)}pct_floor"
        selected = sweep[sweep[flag] == True] if not sweep.empty else pd.DataFrame()  # noqa: E712
        if not selected.empty:
            concentration_rows.append(concentration_for_candidate(data, selected.iloc[0], floor))

    content = [
        "# VN100 Full Confidence Sweep Report",
        "",
        "## Source",
        "",
        f"- Artifact directory: `{rel(artifact_dir)}`.",
        f"- Artifact mode: `{artifact_mode}`.",
        "- The sweep is derived from existing `predicted_vs_actual.csv` files; no model training or benchmark rerun was performed.",
        "",
        "## Coverage",
        "",
        f"- Prediction rows used: {len(data):,}.",
        f"- Available frequency/model/horizon combinations swept: {len(combos)}.",
        f"- Thresholds swept: {THRESHOLDS[0]:.2f} to {THRESHOLDS[-1]:.2f} in 0.01 increments.",
        f"- Daily confidence sweep rows now exist in this v2 derived artifact: {'yes' if not sweep[sweep['frequency'] == 'daily'].empty else 'no'}.",
        f"- Official daily threshold-sweep source rows remain: {daily_official_rows} data rows.",
        f"- Figure status: {figure_status}.",
        "",
        "## Best Candidates by Coverage Floor",
        "",
        markdown_table(
            ["coverage_floor", "candidate", "evaluated_rows", "coverage_ratio", "filtered_accuracy", "passed_60pct"],
            best_rows,
        ),
        "",
        "## Rows Passing 60% at >=20% Coverage",
        "",
    ]
    pass_display = [
        {
            "frequency": row["frequency"],
            "model": row["model"],
            "horizon": int(row["horizon"]),
            "threshold": float(row["threshold"]),
            "evaluated_rows": int(row["evaluated_rows"]),
            "coverage_ratio": float(row["coverage_ratio"]),
            "filtered_accuracy": float(row["filtered_accuracy"]),
        }
        for _, row in pass_rows.iterrows()
    ]
    content.append(
        markdown_table(
            ["frequency", "model", "horizon", "threshold", "evaluated_rows", "coverage_ratio", "filtered_accuracy"],
            pass_display,
            max_rows=40,
        )
        if pass_display
        else "No derived sweep row passes 60% while retaining at least 20% coverage."
    )
    content.extend(
        [
            "",
            "## Selected Candidate Concentration",
            "",
            markdown_table(
                [
                    "coverage_floor",
                    "candidate",
                    "ticker_count",
                    "top_ticker",
                    "top1_prediction_share",
                    "top3_prediction_share",
                    "assessment",
                ],
                concentration_rows,
            )
            if concentration_rows
            else "No selected candidates were available for concentration review.",
            "",
            "## Interpretation",
            "",
            "The v2 sweep closes the missing daily/model/horizon threshold-row gap at the diagnostic level because",
            "it derives thresholds from official prediction rows. It remains a derived analysis, not a fresh official",
            "benchmark rerun. Single-window and seven-ticker limitations still apply.",
            "",
            "## Claim Boundary",
            "",
            "- A row passing 60% after confidence filtering is a conditional diagnostic, not a global benchmark pass.",
            "- Coverage below broad-market levels should be described as selective signal coverage.",
            "- This report does not establish trading readiness or full VN100 representativeness.",
            "",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> int:
    args = parse_args()
    artifact_dir, artifact_mode = choose_artifact_dir(args.artifact_dir)
    output_dir = args.output_dir
    data = load_predictions(artifact_dir)
    sweep = build_sweep(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "vn100_full_confidence_sweep_summary.csv"
    report_path = output_dir / "vn100_full_confidence_sweep_report.md"
    figure_path = output_dir / "vn100_confidence_threshold_coverage_accuracy_v2.png"
    sweep.to_csv(csv_path, index=False)
    figure_status = write_plot(sweep, figure_path)
    write_report(report_path, artifact_dir, artifact_mode, data, sweep, figure_status)
    print(f"Wrote {rel(csv_path)}")
    print(f"Wrote {rel(report_path)}")
    print(f"Figure status: {figure_status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
