"""Create VN100 multi-window validation evidence from available artifacts.

This script intentionally does not rerun the benchmark. It records which
requested windows have official prediction artifacts available and emits
missing-evidence rows for unavailable years.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_2025_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "vn100_multiwindow_walkforward"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "generated" / "evidence_gap_closure"
WINDOWS = [
    {"eval_year": 2022, "train_cutoff": "2021-12-31", "eval_start": "2022-01-01", "eval_end": "2022-12-31"},
    {"eval_year": 2023, "train_cutoff": "2022-12-31", "eval_start": "2023-01-01", "eval_end": "2023-12-31"},
    {"eval_year": 2024, "train_cutoff": "2023-12-31", "eval_start": "2024-01-01", "eval_end": "2024-12-31"},
    {"eval_year": 2025, "train_cutoff": "2024-12-31", "eval_start": "2025-01-01", "eval_end": "2025-12-31"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize VN100 multi-window validation availability.")
    parser.add_argument("--official-2025-dir", type=Path, default=OFFICIAL_2025_DIR)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


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


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def best_confidence_row(official_dir: Path) -> dict[str, Any]:
    rows = read_csv(official_dir / "confidence_threshold_sweep_summary.csv")
    selected = [row for row in rows if str(row.get("selected_candidate", "")).lower() == "true"]
    if not selected:
        return {}
    row = selected[0]
    return {
        "frequency": row.get("frequency", ""),
        "model": row.get("model", ""),
        "horizon": row.get("horizon", ""),
        "threshold": row.get("threshold", ""),
        "accuracy": as_float(row.get("filtered_accuracy")),
        "coverage": as_float(row.get("coverage_ratio")),
        "evaluated_rows": row.get("evaluated_rows", ""),
    }


def regime_slice(official_dir: Path, frequency: str, model: str, horizon: int, regime: str) -> dict[str, Any]:
    rows = read_csv(official_dir / frequency / "regime_accuracy_summary.csv")
    for row in rows:
        if (
            row.get("model") == model
            and int(float(row.get("horizon", 0))) == horizon
            and row.get("regime") == regime
        ):
            return row
    return {}


def build_accuracy_summary(official_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in WINDOWS:
        year = int(window["eval_year"])
        if year != 2025:
            for frequency in ("daily", "hourly"):
                rows.append(
                    {
                        **window,
                        "frequency": frequency,
                        "status": "unavailable",
                        "artifact_dir": "",
                        "evaluated_ticker_count": "",
                        "prediction_count": "",
                        "overall_accuracy": "",
                        "global_passed_60pct": "",
                        "selected_confidence_accuracy": "",
                        "selected_confidence_coverage": "",
                        "limitation": "No official prediction artifacts for this window; heavy rerun not performed.",
                    }
                )
            continue

        confidence = best_confidence_row(official_dir)
        for frequency in ("daily", "hourly"):
            summary = read_json(official_dir / frequency / "benchmark_summary.json")
            rows.append(
                {
                    **window,
                    "frequency": frequency,
                    "status": "available_official_2025",
                    "artifact_dir": rel(official_dir),
                    "evaluated_ticker_count": len(summary.get("evaluated_tickers", [])),
                    "prediction_count": summary.get("n_predictions", ""),
                    "overall_accuracy": summary.get("overall_accuracy", ""),
                    "global_passed_60pct": summary.get("passed", ""),
                    "selected_confidence_accuracy": confidence.get("accuracy", "") if frequency == "hourly" else "",
                    "selected_confidence_coverage": confidence.get("coverage", "") if frequency == "hourly" else "",
                    "limitation": "Single official window; not evidence of multi-window stability.",
                }
            )
    return rows


def build_stability_matrix(official_dir: Path) -> list[dict[str, Any]]:
    daily_summary = read_json(official_dir / "daily" / "benchmark_summary.json")
    hourly_summary = read_json(official_dir / "hourly" / "benchmark_summary.json")
    confidence = best_confidence_row(official_dir)
    lgbm_bear = regime_slice(official_dir, "daily", "lightgbm", 20, "bear")
    xgb_bear = regime_slice(official_dir, "daily", "xgboost", 20, "bear")
    signals = [
        {
            "signal": "global_daily",
            "2025_accuracy": daily_summary.get("overall_accuracy", ""),
            "2025_status": "available",
            "2025_passed_60pct": daily_summary.get("passed", ""),
        },
        {
            "signal": "global_hourly",
            "2025_accuracy": hourly_summary.get("overall_accuracy", ""),
            "2025_status": "available",
            "2025_passed_60pct": hourly_summary.get("passed", ""),
        },
        {
            "signal": "hourly_stacking_h1_confidence_0.57",
            "2025_accuracy": confidence.get("accuracy", ""),
            "2025_status": "available" if confidence else "missing",
            "2025_passed_60pct": bool(confidence and confidence.get("accuracy", 0) >= 0.60),
        },
        {
            "signal": "daily_lightgbm_h20_posthoc_bear",
            "2025_accuracy": lgbm_bear.get("accuracy", ""),
            "2025_status": "available" if lgbm_bear else "missing",
            "2025_passed_60pct": lgbm_bear.get("passed_60pct", ""),
        },
        {
            "signal": "daily_xgboost_h20_posthoc_bear",
            "2025_accuracy": xgb_bear.get("accuracy", ""),
            "2025_status": "available" if xgb_bear else "missing",
            "2025_passed_60pct": xgb_bear.get("passed_60pct", ""),
        },
    ]

    rows: list[dict[str, Any]] = []
    for signal in signals:
        rows.append(
            {
                "signal": signal["signal"],
                "2022_status": "unavailable",
                "2022_accuracy": "",
                "2023_status": "unavailable",
                "2023_accuracy": "",
                "2024_status": "unavailable",
                "2024_accuracy": "",
                "2025_status": signal["2025_status"],
                "2025_accuracy": signal["2025_accuracy"],
                "2025_passed_60pct": signal["2025_passed_60pct"],
                "stable_across_windows": "not_established",
                "interpretation": "Only 2025 has official artifacts in this phase.",
            }
        )
    return rows


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")).replace("|", "\\|") for header in headers) + " |")
    return "\n".join(lines)


def write_report(path: Path, accuracy_rows: list[dict[str, Any]], stability_rows: list[dict[str, Any]], official_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    yearly = [
        {
            "eval_year": year,
            "available": any(row["status"] != "unavailable" for row in accuracy_rows if row["eval_year"] == year),
            "note": "official artifact available" if year == 2025 else "missing official prediction artifacts",
        }
        for year in (2022, 2023, 2024, 2025)
    ]
    content = [
        "# VN100 Multi-Window Validation Report",
        "",
        "## Source",
        "",
        f"- Official 2025 artifact directory: `{rel(official_dir)}`.",
        "- Requested windows: 2022, 2023, 2024, and 2025.",
        "- No heavy benchmark rerun was performed.",
        "",
        "## Window Availability",
        "",
        markdown_table(["eval_year", "available", "note"], yearly),
        "",
        "## Stability Matrix",
        "",
        markdown_table(
            [
                "signal",
                "2022_status",
                "2023_status",
                "2024_status",
                "2025_status",
                "2025_accuracy",
                "stable_across_windows",
            ],
            stability_rows,
        ),
        "",
        "## Required Answers",
        "",
        "- Stable signals across windows: not established; only the 2025 official window has prediction artifacts.",
        "- Selected confidence result persistence: not established beyond 2025.",
        "- Bear-regime diagnostic persistence: not established beyond 2025.",
        "- Global benchmark pass in any available window: no, the available 2025 daily and hourly global summaries do not pass 60%.",
        "- Claim-boundary impact: unchanged. Multi-window evidence remains a major missing-evidence item.",
        "",
        "## Missing Evidence",
        "",
        "- Official 2022 prediction and benchmark artifacts with train_cutoff=2021-12-31.",
        "- Official 2023 prediction and benchmark artifacts with train_cutoff=2022-12-31.",
        "- Official 2024 prediction and benchmark artifacts with train_cutoff=2023-12-31.",
        "- Recomputed confidence, regime, baseline, significance, and concentration diagnostics for each window.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_output_readme(output_root: Path) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "README.md").write_text(
        "# VN100 Multi-Window Walk-Forward Output\n\n"
        "This directory records the evidence-gap-closure phase output location. Heavy benchmark reruns were not\n"
        "performed, so 2022-2024 official prediction artifacts are unavailable. Committed paper-readable summaries\n"
        "are under `reports/generated/evidence_gap_closure/`.\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    accuracy_rows = build_accuracy_summary(args.official_2025_dir)
    stability_rows = build_stability_matrix(args.official_2025_dir)
    write_output_readme(args.output_root)
    accuracy_path = args.report_dir / "vn100_multiwindow_accuracy_summary.csv"
    stability_path = args.report_dir / "vn100_multiwindow_stability_matrix.csv"
    report_path = args.report_dir / "vn100_multiwindow_validation_report.md"
    write_csv(accuracy_path, accuracy_rows)
    write_csv(stability_path, stability_rows)
    write_report(report_path, accuracy_rows, stability_rows, args.official_2025_dir)
    print(f"Wrote {rel(accuracy_path)}")
    print(f"Wrote {rel(stability_path)}")
    print(f"Wrote {rel(report_path)}")
    print(f"Prepared {rel(args.output_root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
