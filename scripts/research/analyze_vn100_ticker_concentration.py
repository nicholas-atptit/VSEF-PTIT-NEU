"""Analyze ticker concentration in the official VN100 benchmark artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "outputs" / "vn100_hybrid_official_2025_confidence_sweep_traincutoff"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "generated"
SUMMARY_COLUMNS = [
    "scope",
    "frequency",
    "model",
    "horizon",
    "threshold",
    "regime_type",
    "regime_value",
    "ticker",
    "prediction_count",
    "correct_count",
    "accuracy",
    "contribution_share",
    "correct_prediction_share",
    "excess_correct_vs_50pct",
    "positive_excess_share",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create ticker concentration diagnostics from official VN100 benchmark artifacts."
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> int:
    parsed = as_float(value)
    return int(parsed) if parsed is not None else 0


def as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def filter_prediction_rows(
    rows: list[dict[str, str]],
    *,
    model: str | None = None,
    horizon: int | None = None,
    confidence_threshold: float | None = None,
    regime_type: str | None = None,
    regime_value: str | None = None,
) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for row in rows:
        if model is not None and str(row.get("model", "")) != model:
            continue
        if horizon is not None and as_int(row.get("horizon")) != horizon:
            continue
        if confidence_threshold is not None:
            confidence = as_float(row.get("confidence"))
            if confidence is None or confidence < confidence_threshold:
                continue
        if regime_type is not None and regime_value is not None and str(row.get(regime_type, "")) != regime_value:
            continue
        filtered.append(row)
    return filtered


def summarize_scope(
    *,
    scope: str,
    frequency: str,
    rows: list[dict[str, str]],
    model: str = "",
    horizon: str = "",
    threshold: str = "",
    regime_type: str = "",
    regime_value: str = "",
) -> list[dict[str, Any]]:
    by_ticker: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0.0, "correct": 0.0})
    for row in rows:
        ticker = str(row.get("ticker", "")).strip().upper()
        if not ticker:
            continue
        by_ticker[ticker]["count"] += 1.0
        by_ticker[ticker]["correct"] += float(as_int(row.get("is_correct")))

    total_count = sum(item["count"] for item in by_ticker.values())
    total_correct = sum(item["correct"] for item in by_ticker.values())
    positive_excess_total = sum(max(0.0, item["correct"] - (0.5 * item["count"])) for item in by_ticker.values())

    output_rows: list[dict[str, Any]] = []
    for ticker, values in sorted(by_ticker.items()):
        count = values["count"]
        correct = values["correct"]
        positive_excess = max(0.0, correct - (0.5 * count))
        output_rows.append(
            {
                "scope": scope,
                "frequency": frequency,
                "model": model,
                "horizon": horizon,
                "threshold": threshold,
                "regime_type": regime_type,
                "regime_value": regime_value,
                "ticker": ticker,
                "prediction_count": int(count),
                "correct_count": int(correct),
                "accuracy": correct / count if count else "",
                "contribution_share": count / total_count if total_count else "",
                "correct_prediction_share": correct / total_correct if total_correct else "",
                "excess_correct_vs_50pct": correct - (0.5 * count),
                "positive_excess_share": positive_excess / positive_excess_total if positive_excess_total else "",
            }
        )
    return output_rows


def concentration_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "tickers": 0,
            "predictions": 0,
            "accuracy": "",
            "top_ticker": "",
            "top1_share": "",
            "top3_share": "",
            "positive_edge_top_ticker": "",
            "positive_edge_top1_share": "",
            "positive_edge_top3_share": "",
            "assessment": "missing",
        }
    total_predictions = sum(int(row["prediction_count"]) for row in rows)
    total_correct = sum(int(row["correct_count"]) for row in rows)
    by_share = sorted(rows, key=lambda item: float(item["contribution_share"]), reverse=True)
    top1_share = float(by_share[0]["contribution_share"])
    top3_share = sum(float(row["contribution_share"]) for row in by_share[:3])

    edge_rows = [row for row in rows if row["positive_excess_share"] != ""]
    edge_rows = sorted(edge_rows, key=lambda item: float(item["positive_excess_share"]), reverse=True)
    edge_top1_share = float(edge_rows[0]["positive_excess_share"]) if edge_rows else ""
    edge_top3_share = sum(float(row["positive_excess_share"]) for row in edge_rows[:3]) if edge_rows else ""

    assessment = "low"
    if top1_share >= 0.35 or top3_share >= 0.70:
        assessment = "high"
    elif top1_share >= 0.25 or top3_share >= 0.60:
        assessment = "moderate"

    return {
        "tickers": len(rows),
        "predictions": total_predictions,
        "accuracy": total_correct / total_predictions if total_predictions else "",
        "top_ticker": by_share[0]["ticker"],
        "top1_share": top1_share,
        "top3_share": top3_share,
        "positive_edge_top_ticker": edge_rows[0]["ticker"] if edge_rows else "",
        "positive_edge_top1_share": edge_top1_share,
        "positive_edge_top3_share": edge_top3_share,
        "assessment": assessment,
    }


def format_pct(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value) * 100:.2f}%"


def format_float(value: Any) -> str:
    if value == "" or value is None:
        return "n/a"
    return f"{float(value):.6f}"


def selected_confidence_scopes(artifact_dir: Path, predictions_by_frequency: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    rows = read_csv_rows(artifact_dir / "confidence_threshold_sweep_summary.csv")
    scopes: list[dict[str, Any]] = []
    for row in rows:
        if not as_bool(row.get("selected_candidate")):
            continue
        frequency = str(row.get("frequency", "")).strip()
        model = str(row.get("model", "")).strip()
        horizon = as_int(row.get("horizon"))
        threshold = as_float(row.get("threshold"))
        if not frequency or not model or threshold is None:
            continue
        prediction_rows = filter_prediction_rows(
            predictions_by_frequency.get(frequency, []),
            model=model,
            horizon=horizon,
            confidence_threshold=threshold,
        )
        scopes.append(
            {
                "scope": f"selected_confidence_{frequency}_{model}_h{horizon}_t{threshold:g}",
                "frequency": frequency,
                "model": model,
                "horizon": str(horizon),
                "threshold": f"{threshold:g}",
                "regime_type": "",
                "regime_value": "",
                "rows": prediction_rows,
            }
        )
    return scopes


def best_regime_scopes(artifact_dir: Path, predictions_by_frequency: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    for frequency, prediction_rows in predictions_by_frequency.items():
        summary = read_json(artifact_dir / frequency / "benchmark_summary.json")
        model = str(summary.get("best_regime_model", "")).strip()
        horizon = as_int(summary.get("best_regime_horizon"))
        regime_value = str(summary.get("best_regime", "")).strip()
        if not model or not horizon or not regime_value:
            continue

        regime_type = ""
        if any(str(row.get("regime", "")) == regime_value for row in prediction_rows):
            regime_type = "regime"
        elif any(str(row.get("volatility_regime", "")) == regime_value for row in prediction_rows):
            regime_type = "volatility_regime"
        if not regime_type:
            continue

        scoped_rows = filter_prediction_rows(
            prediction_rows,
            model=model,
            horizon=horizon,
            regime_type=regime_type,
            regime_value=regime_value,
        )
        scopes.append(
            {
                "scope": f"best_regime_{frequency}_{model}_h{horizon}_{regime_value}",
                "frequency": frequency,
                "model": model,
                "horizon": str(horizon),
                "threshold": "",
                "regime_type": regime_type,
                "regime_value": regime_value,
                "rows": scoped_rows,
            }
        )
    return scopes


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(
    *,
    artifact_dir: Path,
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    evaluated_tickers: list[str],
) -> None:
    scopes = sorted({row["scope"] for row in summary_rows})
    concentration_rows: list[list[str]] = []
    finding_lines: list[str] = []

    for scope in scopes:
        rows = [row for row in summary_rows if row["scope"] == scope]
        metrics = concentration_metrics(rows)
        concentration_rows.append(
            [
                scope,
                str(metrics["tickers"]),
                str(metrics["predictions"]),
                format_pct(metrics["accuracy"]),
                str(metrics["top_ticker"]),
                format_pct(metrics["top1_share"]),
                format_pct(metrics["top3_share"]),
                str(metrics["positive_edge_top_ticker"]),
                format_pct(metrics["positive_edge_top1_share"]),
                format_pct(metrics["positive_edge_top3_share"]),
                str(metrics["assessment"]),
            ]
        )
        if scope.startswith("selected_confidence_") or scope.startswith("best_regime_"):
            finding_lines.append(
                "- "
                + scope
                + ": prediction-count concentration is "
                + str(metrics["assessment"])
                + f" (top ticker {metrics['top_ticker']} at {format_pct(metrics['top1_share'])}; "
                + f"top three at {format_pct(metrics['top3_share'])})."
            )

    top_ticker_rows = sorted(
        summary_rows,
        key=lambda row: (row["scope"], -float(row["contribution_share"]), row["ticker"]),
    )
    top_ticker_rows = [
        row
        for row in top_ticker_rows
        if row["scope"].startswith("selected_confidence_") or row["scope"].startswith("best_regime_")
    ][:30]
    top_rows = [
        [
            str(row["scope"]),
            str(row["ticker"]),
            str(row["prediction_count"]),
            format_pct(row["accuracy"]),
            format_pct(row["contribution_share"]),
            format_float(row["excess_correct_vs_50pct"]),
            format_pct(row["positive_excess_share"]),
        ]
        for row in top_ticker_rows
    ]

    content = [
        "# VN100 Ticker Concentration Summary",
        "",
        "## Source",
        "",
        f"- Official artifact directory: `{rel(artifact_dir)}`.",
        f"- Prediction inputs: `{rel(artifact_dir / 'daily' / 'predicted_vs_actual.csv')}` and `{rel(artifact_dir / 'hourly' / 'predicted_vs_actual.csv')}`.",
        f"- Selected confidence candidate source: `{rel(artifact_dir / 'confidence_threshold_sweep_summary.csv')}`.",
        "- Contribution share is the share of prediction rows within each diagnostic scope.",
        "- Positive excess share is the share of correct predictions above a 50% null, counted only where a ticker has positive excess.",
        "",
        "## Evaluated Tickers",
        "",
        ", ".join(evaluated_tickers) if evaluated_tickers else "No evaluated tickers were found in benchmark summaries.",
        "",
        "## Scope Concentration",
        "",
        markdown_table(
            [
                "scope",
                "tickers",
                "predictions",
                "accuracy",
                "top ticker",
                "top1 prediction share",
                "top3 prediction share",
                "top positive-edge ticker",
                "top1 positive-edge share",
                "top3 positive-edge share",
                "prediction-count assessment",
            ],
            concentration_rows,
        ),
        "",
        "## Selected Finding Assessment",
        "",
        "\n".join(finding_lines) if finding_lines else "No selected confidence or best-regime scope was available.",
        "",
        "## Selected Scope Ticker Rows",
        "",
        markdown_table(
            [
                "scope",
                "ticker",
                "predictions",
                "accuracy",
                "contribution share",
                "excess correct vs 50%",
                "positive excess share",
            ],
            top_rows,
        )
        if top_rows
        else "No selected scope ticker rows were available.",
        "",
        "## Interpretation Boundary",
        "",
        "These diagnostics address concentration of the official prediction rows and positive directional edge. They do not establish trading profitability, cost-adjusted returns, or stability beyond the official artifact window.",
        "",
    ]
    output_path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_by_frequency = {
        frequency: read_csv_rows(artifact_dir / frequency / "predicted_vs_actual.csv")
        for frequency in ("daily", "hourly")
    }

    evaluated_tickers = sorted(
        {
            str(ticker)
            for frequency in ("daily", "hourly")
            for ticker in read_json(artifact_dir / frequency / "benchmark_summary.json").get("evaluated_tickers", [])
        }
    )

    scopes: list[dict[str, Any]] = []
    for frequency, rows in predictions_by_frequency.items():
        if rows:
            scopes.append(
                {
                    "scope": f"global_{frequency}",
                    "frequency": frequency,
                    "model": "",
                    "horizon": "",
                    "threshold": "",
                    "regime_type": "",
                    "regime_value": "",
                    "rows": rows,
                }
            )
    scopes.extend(selected_confidence_scopes(artifact_dir, predictions_by_frequency))
    scopes.extend(best_regime_scopes(artifact_dir, predictions_by_frequency))

    summary_rows: list[dict[str, Any]] = []
    for scope in scopes:
        summary_rows.extend(
            summarize_scope(
                scope=scope["scope"],
                frequency=scope["frequency"],
                rows=scope["rows"],
                model=scope["model"],
                horizon=scope["horizon"],
                threshold=scope["threshold"],
                regime_type=scope["regime_type"],
                regime_value=scope["regime_value"],
            )
        )

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (row["scope"], -float(row["contribution_share"]), row["ticker"]),
    )

    csv_path = output_dir / "vn100_ticker_concentration_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        writer.writerows(summary_rows)

    write_markdown(
        artifact_dir=artifact_dir,
        output_path=output_dir / "vn100_ticker_concentration_summary.md",
        summary_rows=summary_rows,
        evaluated_tickers=evaluated_tickers,
    )

    print(f"wrote {rel(csv_path)}")
    print(f"wrote {rel(output_dir / 'vn100_ticker_concentration_summary.md')}")


if __name__ == "__main__":
    main()
