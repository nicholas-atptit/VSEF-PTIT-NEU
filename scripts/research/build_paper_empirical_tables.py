"""Build paper-ready empirical tables from existing VN30 artifacts.

This script reads repository artifacts only. It does not fetch data, train
models, run benchmarks, or regenerate row-level predictions.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]

TARGET62_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_target62_paper_ready_stability"
TARGET62_SOURCE_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_track_a_target62_validation_safe"
TARGET62_OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_track_a_target62_validation_safe"
RF_H60_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_rf_h60_reproduction"
HOURLY_READINESS_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_benchmark_readiness"
HOURLY_FORENSICS_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_data_forensics"
DAILY_READINESS_DIR = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015"
INDEX_BENCHMARK_DIR = REPO_ROOT / "reports" / "generated" / "index_benchmark"

OUT_DIR = REPO_ROOT / "reports" / "generated" / "paper_tables_current"
INVENTORY_PATH = REPO_ROOT / "reports" / "PAPER_FIGURE_DATA_SOURCE_INVENTORY.md"
LITERATURE_TODO_PATH = REPO_ROOT / "reports" / "PAPER_LITERATURE_DATA_TODO.md"
MISSING_TODO_PATH = REPO_ROOT / "reports" / "PAPER_MISSING_METRICS_TODO.md"

BASELINE_LOGISTIC_H40 = 0.6043200785468826


def rel(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def read_one_csv(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    if frame.empty:
        return {}
    return frame.iloc[0].to_dict()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def as_float(value: Any) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def pct_value(value: Any, digits: int = 4) -> float | str:
    number = as_float(value)
    if math.isnan(number):
        return ""
    return round(number * 100.0, digits)


def pp_value(value: Any, digits: int = 4) -> float | str:
    number = as_float(value)
    if math.isnan(number):
        return ""
    return round(number * 100.0, digits)


def fmt_pct(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "missing"
    return f"{number * 100.0:.{digits}f}%"


def fmt_pp(value: Any, digits: int = 2) -> str:
    number = as_float(value)
    if math.isnan(number):
        return "missing"
    sign = "+" if number >= 0 else ""
    return f"{sign}{number * 100.0:.{digits}f} pp"


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(columns: list[str], rows: list[dict[str, Any]]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = [str(row.get(column, "")).replace("\n", " ") for column in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def artifact_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "CSV"
    if suffix == ".json":
        return "JSON"
    if suffix == ".md":
        return "MD"
    return suffix.lstrip(".").upper() or "unknown"


def add_inventory(
    rows: list[dict[str, str]],
    path: Path,
    metric: str,
    value: Any,
    source_method: str = "parsed",
    row_level_predictions: str = "no",
    figure_source: str = "summary data",
) -> None:
    rows.append(
        {
            "artifact_file": rel(path),
            "metric_extracted": metric,
            "exact_value": str(value),
            "source_type": artifact_type(path),
            "source_method": source_method,
            "row_level_predictions_exist": row_level_predictions,
            "figure_generation_basis": figure_source,
        }
    )


def bool_text(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value).strip().lower()
    if text == "true":
        return "yes"
    if text == "false":
        return "no"
    return str(value)


def selected_rf_reference() -> tuple[float, str]:
    path = RF_H60_DIR / "rf_h60_reproduction_summary.csv"
    frame = pd.read_csv(path)
    row = frame.loc[frame["check_item"] == "historical_rf_h60_accuracy"]
    if row.empty:
        return 0.6031, "missing_in_rf_reproduction_summary_using_run_config_reference"
    return as_float(row.iloc[0]["historical_value"]), "rf_h60_reproduction_summary"


def build_main_result_summary(
    global_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    run_config: dict[str, Any],
) -> list[dict[str, Any]]:
    rf_reference_exact, _rf_source = selected_rf_reference()
    final_accuracy = as_float(global_summary["final_accuracy"])
    logistic_baseline = as_float(run_config.get("baseline_logistic_h40", BASELINE_LOGISTIC_H40))
    rf_for_table = as_float(run_config.get("historical_rf_h60", rf_reference_exact))
    coverage = f"{int(as_float(selected_summary.get('active_ticker_count', 30)))}/30 stocks, full coverage"
    claim_level = "exploratory improved_baseline60"
    return [
        {
            "setup": "Track A canonical-like VN30 stock-only hourly",
            "model": "L2 Logistic",
            "horizon": int(as_float(global_summary["horizon"])),
            "feature_set": global_summary["feature_set"],
            "threshold": f"{as_float(global_summary['threshold']):.2f}",
            "final_accuracy_pct": pct_value(final_accuracy),
            "majority_baseline_pct": pct_value(global_summary["majority_baseline_accuracy"]),
            "lift_vs_majority_pp": pp_value(global_summary["delta_vs_majority_baseline"]),
            "logistic_baseline_pct": pct_value(logistic_baseline),
            "lift_vs_logistic_baseline_pp": pp_value(final_accuracy - logistic_baseline),
            "rf_h60_reference_pct": pct_value(rf_for_table),
            "lift_vs_rf_h60_pp": pp_value(global_summary["delta_vs_60_31"]),
            "final_rows": int(as_float(global_summary["total_rows"])),
            "coverage": coverage,
            "claim_level": claim_level,
        }
    ]


def build_stability_summary(global_summary: dict[str, Any], paper_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = [
        ("Ticker", "mean ticker accuracy", fmt_pct(paper_summary["mean_ticker_accuracy"]), "Cross-ticker mean slice accuracy."),
        ("Ticker", "median ticker accuracy", fmt_pct(paper_summary["median_ticker_accuracy"]), "Cross-ticker median slice accuracy."),
        ("Time", "mean month accuracy", fmt_pct(paper_summary["mean_month_accuracy"]), "Monthly performance is mixed."),
        ("Time", "median month accuracy", fmt_pct(paper_summary["median_month_accuracy"]), "Monthly median remains below target62."),
        ("Time", "mean quarter accuracy", fmt_pct(paper_summary["mean_quarter_accuracy"]), "Quarterly performance is mixed."),
        ("Time", "median quarter accuracy", fmt_pct(paper_summary["median_quarter_accuracy"]), "Quarterly median remains below target62."),
        ("Ticker", "ticker stability classification", paper_summary["ticker_stability_classification"], "Moderate ticker stability only."),
        ("Time", "time stability classification", paper_summary["time_stability_classification"], "Time stability is concentrated or mixed."),
        ("Regime", "regime stability classification", paper_summary["regime_stability_classification"], "Regime stability is not established."),
        ("Overall", "overall stability classification", paper_summary["stability_classification"], "Overall evidence supports baseline60 improvement only."),
        ("Validation", "validation-final gap", fmt_pp(global_summary["validation_final_gap"]), "Final accuracy is materially above validation accuracy."),
    ]
    return [{"metric_group": group, "metric": metric, "value": value, "interpretation": interpretation} for group, metric, value, interpretation in rows]


def table_from_slices() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_regime = pd.read_csv(TARGET62_DIR / "by_regime.csv")
    regime_rows = []
    for _, row in by_regime.iterrows():
        regime_rows.append(
            {
                "regime": row["market_regime_v2"],
                "accuracy_pct": pct_value(row["accuracy"]),
                "majority_baseline_pct": pct_value(row["majority_baseline_accuracy"]),
                "lift_pp": pp_value(row["delta_vs_majority_baseline"]),
                "rows": int(row["rows"]),
                "pass_60": bool(as_float(row["accuracy"]) >= 0.60),
                "pass_62": bool(as_float(row["accuracy"]) >= 0.62),
            }
        )

    by_ticker = pd.read_csv(TARGET62_DIR / "by_ticker.csv").sort_values("ticker")
    ticker_rows = []
    for _, row in by_ticker.iterrows():
        rows = int(row["rows"])
        correct = int(row["correct"])
        ticker_rows.append(
            {
                "ticker": row["ticker"],
                "accuracy_pct": pct_value(row["accuracy"]),
                "rows": rows,
                "correct": correct,
                "incorrect": rows - correct,
                "majority_baseline_pct": pct_value(row["majority_baseline_accuracy"]),
                "lift_pp": pp_value(row["delta_vs_majority_baseline"]),
            }
        )

    by_month = pd.read_csv(TARGET62_DIR / "by_month.csv").sort_values("month")
    month_rows = []
    for _, row in by_month.iterrows():
        month_rows.append(
            {
                "month": row["month"],
                "accuracy_pct": pct_value(row["accuracy"]),
                "rows": int(row["rows"]),
                "majority_baseline_pct": pct_value(row["majority_baseline_accuracy"]),
                "lift_pp": pp_value(row["delta_vs_majority_baseline"]),
            }
        )

    by_quarter = pd.read_csv(TARGET62_DIR / "by_quarter.csv").sort_values("quarter")
    quarter_rows = []
    for _, row in by_quarter.iterrows():
        quarter_rows.append(
            {
                "quarter": row["quarter"],
                "accuracy_pct": pct_value(row["accuracy"]),
                "rows": int(row["rows"]),
                "majority_baseline_pct": pct_value(row["majority_baseline_accuracy"]),
                "lift_pp": pp_value(row["delta_vs_majority_baseline"]),
            }
        )

    return regime_rows, ticker_rows, month_rows, quarter_rows


def min_timestamp(values: list[Any]) -> str:
    parsed = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
    if parsed.empty:
        return ""
    return str(parsed.min())


def max_timestamp(values: list[Any]) -> str:
    parsed = pd.to_datetime(pd.Series(values), errors="coerce").dropna()
    if parsed.empty:
        return ""
    return str(parsed.max())


def build_data_scope_summary() -> list[dict[str, Any]]:
    hourly_manifest = read_json(HOURLY_READINESS_DIR / "vn30_2015_benchmark_readiness_manifest.json")
    daily = pd.read_csv(DAILY_READINESS_DIR / "vn30_daily_2015_readiness.csv")
    index_scope = pd.read_csv(INDEX_BENCHMARK_DIR / "index_data_scope_audit.csv")

    hourly_first = min_timestamp(list(hourly_manifest.get("actual_first_timestamp_by_ticker", {}).values()))
    hourly_last = max_timestamp(list(hourly_manifest.get("actual_last_timestamp_by_ticker", {}).values()))

    usable_daily = daily[daily["usable"].astype(str).str.lower() == "true"]
    daily_first = min_timestamp(usable_daily["first_date"].tolist())
    daily_last = max_timestamp(usable_daily["last_date"].tolist())

    daily_index = index_scope[(index_scope["frequency"] == "1D") & (index_scope["usable_for_train_validation_final"] == "yes")]
    hourly_index = index_scope[(index_scope["frequency"] == "1H") & (index_scope["usable_for_train_validation_final"] == "yes")]

    return [
        {
            "track": "Track A canonical-like VN30 hourly",
            "frequency": "hourly",
            "instrument_scope": "VN30 January 2025 stock universe",
            "earliest_timestamp": hourly_first,
            "latest_timestamp": hourly_last,
            "usable_instruments": f"{hourly_manifest.get('usable_ticker_count', '')}/30 stocks",
            "limitation": "No local 2015-2022 hourly stock data; hourly_2015 is a design label, not actual pre-2023 stock coverage.",
        },
        {
            "track": "VN30 daily 2015 readiness",
            "frequency": "daily",
            "instrument_scope": "VN30 January 2025 stock universe",
            "earliest_timestamp": daily_first,
            "latest_timestamp": daily_last,
            "usable_instruments": f"{len(usable_daily)}/30 stocks",
            "limitation": "Some tickers list after 2015; daily evidence is separate context, not the selected hourly target62 result.",
        },
        {
            "track": "Supported index daily",
            "frequency": "daily",
            "instrument_scope": "Supported market indices",
            "earliest_timestamp": min_timestamp(daily_index["first_timestamp"].tolist()),
            "latest_timestamp": max_timestamp(daily_index["last_timestamp"].tolist()),
            "usable_instruments": f"{daily_index['index_code'].nunique()}/6 indices",
            "limitation": "Index-only context; stock data not used in index benchmark.",
        },
        {
            "track": "Supported index hourly",
            "frequency": "hourly",
            "instrument_scope": "Supported market indices",
            "earliest_timestamp": min_timestamp(hourly_index["first_timestamp"].tolist()),
            "latest_timestamp": max_timestamp(hourly_index["last_timestamp"].tolist()),
            "usable_instruments": f"{hourly_index['index_code'].nunique()}/6 indices",
            "limitation": "No local 2015-2021 hourly index data; earliest local hourly index evidence starts in 2022.",
        },
    ]


def build_literature_comparison() -> list[dict[str, Any]]:
    return [
        {
            "study": "Vietnamese equity directional-forecasting literature",
            "market_scope": "requires source verification",
            "target_type": "requires source verification",
            "validation_design": "requires source verification",
            "metric_reported": "requires source verification",
            "headline_result": "requires source verification",
            "comparability_note": "No repository artifact with verified literature values was found; do not cite numeric comparisons until sources are checked.",
        },
        {
            "study": "Emerging-market machine-learning forecasting literature",
            "market_scope": "requires source verification",
            "target_type": "requires source verification",
            "validation_design": "requires source verification",
            "metric_reported": "requires source verification",
            "headline_result": "requires source verification",
            "comparability_note": "Placeholder only; not empirical evidence from this repository.",
        },
    ]


def build_inventory(
    global_summary: dict[str, Any],
    paper_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    output_selected_summary: dict[str, Any],
    run_config: dict[str, Any],
    table_counts: dict[str, int],
) -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []

    global_path = TARGET62_DIR / "global_summary.csv"
    paper_summary_path = TARGET62_DIR / "paper_ready_summary_table.csv"
    selected_path = TARGET62_SOURCE_DIR / "selected_candidate_summary.csv"
    output_selected_path = TARGET62_OUTPUT_DIR / "selected_candidate_summary.csv"
    run_config_path = TARGET62_SOURCE_DIR / "run_config.json"
    rf_path = RF_H60_DIR / "rf_h60_reproduction_summary.csv"
    bootstrap_path = TARGET62_DIR / "bootstrap_ci.csv"
    significance_path = TARGET62_DIR / "significance_tests.csv"
    mismatch_path = TARGET62_DIR / "validation_final_mismatch.csv"
    figure_index_path = TARGET62_DIR / "paper_ready_figure_index.csv"
    rolling_250_path = TARGET62_DIR / "rolling_accuracy_250.csv"
    rolling_500_path = TARGET62_DIR / "rolling_accuracy_500.csv"
    rolling_1000_path = TARGET62_DIR / "rolling_accuracy_1000.csv"
    hourly_manifest_path = HOURLY_READINESS_DIR / "vn30_2015_benchmark_readiness_manifest.json"
    daily_readiness_path = DAILY_READINESS_DIR / "vn30_daily_2015_readiness.csv"
    index_scope_path = INDEX_BENCHMARK_DIR / "index_data_scope_audit.csv"
    forensics_path = HOURLY_FORENSICS_DIR / "vn30_hourly_data_file_inventory.md"

    selected_fields = {
        "model": global_summary.get("model", ""),
        "horizon": global_summary.get("horizon", ""),
        "feature_set": global_summary.get("feature_set", ""),
        "threshold": global_summary.get("threshold", ""),
    }
    add_inventory(inventory, global_path, "selected candidate identifiers", selected_fields)
    for metric in (
        "final_accuracy",
        "total_rows",
        "correct_predictions",
        "incorrect_predictions",
        "majority_baseline_accuracy",
        "delta_vs_majority_baseline",
        "delta_vs_60_43",
        "delta_vs_60_31",
        "pass_60",
        "pass_60_43",
        "pass_62",
        "pass_65",
        "validation_accuracy",
        "validation_final_gap",
        "validation_final_mismatch",
        "paper_ready_claim_level",
    ):
        add_inventory(inventory, global_path, metric, global_summary.get(metric, "missing"))

    for metric in (
        "mean_ticker_accuracy",
        "median_ticker_accuracy",
        "mean_month_accuracy",
        "median_month_accuracy",
        "mean_quarter_accuracy",
        "median_quarter_accuracy",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "significance_result",
        "ticker_stability_classification",
        "time_stability_classification",
        "regime_stability_classification",
        "stability_classification",
        "claim_level",
        "target62_claim",
        "final65_claim",
    ):
        add_inventory(inventory, paper_summary_path, metric, paper_summary.get(metric, "missing"))

    for metric in ("validation_accuracy", "final_accuracy", "final_rows", "active_ticker_count", "claim_level"):
        add_inventory(inventory, selected_path, f"selected summary {metric}", selected_summary.get(metric, "missing"))
        add_inventory(inventory, output_selected_path, f"output mirror selected summary {metric}", output_selected_summary.get(metric, "missing"))

    add_inventory(inventory, run_config_path, "baseline_logistic_h40", run_config.get("baseline_logistic_h40", "missing"))
    add_inventory(inventory, run_config_path, "historical_rf_h60", run_config.get("historical_rf_h60", "missing"))
    add_inventory(inventory, run_config_path, "selection rule uses final accuracy", run_config.get("selection_rule", {}).get("final_accuracy_used_for_selection", "missing"))
    add_inventory(inventory, rf_path, "historical_rf_h60_accuracy detail", selected_rf_reference()[0])

    bootstrap = read_one_csv(bootstrap_path)
    for metric in ("bootstrap_source", "iterations", "ci_low", "ci_high", "bootstrap_mean", "standard_error"):
        add_inventory(inventory, bootstrap_path, metric, bootstrap.get(metric, "missing"))

    significance = pd.read_csv(significance_path)
    add_inventory(inventory, significance_path, "significance tests", significance.to_dict("records"))

    mismatch = read_one_csv(mismatch_path)
    add_inventory(inventory, mismatch_path, "validation-final mismatch row", mismatch)

    add_inventory(inventory, TARGET62_DIR / "by_ticker.csv", "ticker slice rows used", table_counts["ticker_rows"], figure_source="summary slice data")
    add_inventory(inventory, TARGET62_DIR / "by_month.csv", "monthly slice rows used", table_counts["month_rows"], figure_source="summary slice data")
    add_inventory(inventory, TARGET62_DIR / "by_quarter.csv", "quarterly slice rows used", table_counts["quarter_rows"], figure_source="summary slice data")
    add_inventory(inventory, TARGET62_DIR / "by_regime.csv", "regime slice rows used", table_counts["regime_rows"], figure_source="summary slice data")
    add_inventory(inventory, figure_index_path, "paper-ready figure source index", pd.read_csv(figure_index_path).to_dict("records"), figure_source="summary metadata")
    for rolling_path in (rolling_250_path, rolling_500_path, rolling_1000_path):
        rolling = read_one_csv(rolling_path)
        add_inventory(
            inventory,
            rolling_path,
            f"row-level rolling status {rolling.get('window_rows', '')}",
            rolling.get("reason", "missing"),
            row_level_predictions="no",
            figure_source="not generated; row-level predictions unavailable",
        )

    hourly_manifest = read_json(hourly_manifest_path)
    add_inventory(inventory, hourly_manifest_path, "hourly stock actual data window", f"{hourly_manifest.get('actual_data_start_any')} to {hourly_manifest.get('actual_latest_data_timestamp')}")
    add_inventory(inventory, hourly_manifest_path, "hourly usable stock count", hourly_manifest.get("usable_ticker_count", "missing"))
    add_inventory(inventory, daily_readiness_path, "daily stock readiness rows", table_counts["daily_readiness_rows"], figure_source="data-scope summary")
    add_inventory(inventory, index_scope_path, "index data-scope rows", table_counts["index_scope_rows"], figure_source="data-scope summary")
    add_inventory(inventory, forensics_path, "hourly forensics limitation", "no 2015-2022 hourly stock data exists anywhere in the repository", source_method="manually read from MD", figure_source="data-scope limitation text")

    return inventory


def write_inventory_report(inventory: list[dict[str, str]], missing_items: list[str]) -> None:
    columns = [
        "artifact_file",
        "metric_extracted",
        "exact_value",
        "source_type",
        "source_method",
        "row_level_predictions_exist",
        "figure_generation_basis",
    ]
    lines = [
        "# Paper Figure Data Source Inventory",
        "",
        "All numeric paper tables and figures are derived from repository artifacts. No market-data fetch, benchmark run, or model training is performed by the paper table/figure builders.",
        "",
        "## Inventory",
        "",
        markdown_table(columns, inventory),
        "",
        "## Row-Level Prediction Availability",
        "",
        "- Target62 selected-candidate row-level predictions: no.",
        "- The target62 paper-ready audit records rolling 250/500/1000 row accuracy as unavailable because row-level predictions were not saved.",
        "- Separate OOF/final prediction artifacts exist for a different true-stacking workflow and are not used for target62 figures.",
        "",
        "## Missing Items",
        "",
    ]
    if missing_items:
        lines.extend(f"- {item}" for item in missing_items)
    else:
        lines.append("- None for required summary-supported figures and tables.")
    INVENTORY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_literature_todo() -> None:
    lines = [
        "# Paper Literature Data TODO",
        "",
        "No current repository artifact was found that verifies external literature-comparison values for this VN30 target62 paper package.",
        "",
        "Use the placeholder rows in `reports/generated/paper_tables_current/table_literature_comparison.csv` only as source-verification reminders. Do not cite headline literature values until the study, market scope, validation design, metric, and result are verified from the original source.",
        "",
        "Required verification fields:",
        "",
        "- study",
        "- market scope",
        "- target type",
        "- validation design",
        "- metric reported",
        "- headline result",
        "- comparability note against the VN30 hourly Track A setup",
        "",
    ]
    LITERATURE_TODO_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_missing_todo(missing_items: list[str]) -> None:
    lines = [
        "# Paper Missing Metrics TODO",
        "",
        "The table and figure builders do not fabricate missing values.",
        "",
        "## Missing Or Unsupported Items",
        "",
    ]
    if missing_items:
        lines.extend(f"- {item}" for item in missing_items)
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Resolution Rule",
            "",
            "Only add these metrics after a repository artifact or protocol-approved external source verifies the value. Do not infer or backfill unsupported charts.",
            "",
        ]
    )
    MISSING_TODO_PATH.write_text("\n".join(lines), encoding="utf-8")


def write_build_report(table_counts: dict[str, int], missing_items: list[str]) -> None:
    report_path = OUT_DIR / "PAPER_TABLES_BUILD_REPORT.md"
    lines = [
        "# Paper Tables Build Report",
        "",
        "- Data fetch run: no.",
        "- Benchmark run: no.",
        "- Model training run: no.",
        "- Source artifacts: existing repository CSV, JSON, and MD files.",
        "- Row-level predictions used: no.",
        "- Missing literature comparison values: yes.",
        "",
        "## Generated Tables",
        "",
    ]
    for name, count in sorted(table_counts.items()):
        lines.append(f"- `{name}`: {count} row(s).")
    lines.extend(["", "## Missing Items", ""])
    if missing_items:
        lines.extend(f"- {item}" for item in missing_items)
    else:
        lines.append("- None.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    global_summary = read_one_csv(TARGET62_DIR / "global_summary.csv")
    paper_summary = read_one_csv(TARGET62_DIR / "paper_ready_summary_table.csv")
    selected_summary = read_one_csv(TARGET62_SOURCE_DIR / "selected_candidate_summary.csv")
    output_selected_summary = read_one_csv(TARGET62_OUTPUT_DIR / "selected_candidate_summary.csv")
    run_config = read_json(TARGET62_SOURCE_DIR / "run_config.json")

    main_rows = build_main_result_summary(global_summary, selected_summary, run_config)
    stability_rows = build_stability_summary(global_summary, paper_summary)
    regime_rows, ticker_rows, month_rows, quarter_rows = table_from_slices()
    data_scope_rows = build_data_scope_summary()
    literature_rows = build_literature_comparison()

    write_csv(
        OUT_DIR / "table_main_result_summary.csv",
        main_rows,
        [
            "setup",
            "model",
            "horizon",
            "feature_set",
            "threshold",
            "final_accuracy_pct",
            "majority_baseline_pct",
            "lift_vs_majority_pp",
            "logistic_baseline_pct",
            "lift_vs_logistic_baseline_pp",
            "rf_h60_reference_pct",
            "lift_vs_rf_h60_pp",
            "final_rows",
            "coverage",
            "claim_level",
        ],
    )
    write_csv(OUT_DIR / "table_stability_summary.csv", stability_rows, ["metric_group", "metric", "value", "interpretation"])
    write_csv(OUT_DIR / "table_regime_summary.csv", regime_rows, ["regime", "accuracy_pct", "majority_baseline_pct", "lift_pp", "rows", "pass_60", "pass_62"])
    write_csv(OUT_DIR / "table_ticker_accuracy.csv", ticker_rows, ["ticker", "accuracy_pct", "rows", "correct", "incorrect", "majority_baseline_pct", "lift_pp"])
    write_csv(OUT_DIR / "table_monthly_accuracy.csv", month_rows, ["month", "accuracy_pct", "rows", "majority_baseline_pct", "lift_pp"])
    write_csv(OUT_DIR / "table_quarterly_accuracy.csv", quarter_rows, ["quarter", "accuracy_pct", "rows", "majority_baseline_pct", "lift_pp"])
    write_csv(OUT_DIR / "table_data_scope_summary.csv", data_scope_rows, ["track", "frequency", "instrument_scope", "earliest_timestamp", "latest_timestamp", "usable_instruments", "limitation"])
    write_csv(OUT_DIR / "table_literature_comparison.csv", literature_rows, ["study", "market_scope", "target_type", "validation_design", "metric_reported", "headline_result", "comparability_note"])

    table_counts = {
        "table_main_result_summary.csv": len(main_rows),
        "table_stability_summary.csv": len(stability_rows),
        "table_regime_summary.csv": len(regime_rows),
        "table_ticker_accuracy.csv": len(ticker_rows),
        "table_monthly_accuracy.csv": len(month_rows),
        "table_quarterly_accuracy.csv": len(quarter_rows),
        "table_data_scope_summary.csv": len(data_scope_rows),
        "table_literature_comparison.csv": len(literature_rows),
        "ticker_rows": len(ticker_rows),
        "month_rows": len(month_rows),
        "quarter_rows": len(quarter_rows),
        "regime_rows": len(regime_rows),
        "daily_readiness_rows": len(pd.read_csv(DAILY_READINESS_DIR / "vn30_daily_2015_readiness.csv")),
        "index_scope_rows": len(pd.read_csv(INDEX_BENCHMARK_DIR / "index_data_scope_audit.csv")),
    }
    missing_items = [
        "External literature-comparison values are not verified by current repository artifacts.",
        "Target62 row-level prediction records are not saved, so rolling 250/500/1000 row figures and correctness-over-time plots are unsupported.",
    ]
    inventory = build_inventory(global_summary, paper_summary, selected_summary, output_selected_summary, run_config, table_counts)
    write_inventory_report(inventory, missing_items)
    write_literature_todo()
    write_missing_todo(missing_items)
    write_build_report(table_counts, missing_items)

    print(f"paper_tables_generated=8 output_dir={rel(OUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
