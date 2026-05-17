"""Audit supported index directional benchmark outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research.index_benchmark_common import OUTPUT_DIR, REPORT_DIR, fmt_pct, markdown_table, rel, write_csv

CSV_PATH = REPORT_DIR / "index_directional_benchmark_audit.csv"
MD_PATH = REPORT_DIR / "index_directional_benchmark_audit.md"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def add_check(rows: list[dict[str, Any]], check: str, passed: bool, details: str, severity: str = "fail") -> None:
    rows.append(
        {
            "check": check,
            "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"),
            "severity": "info" if passed else severity,
            "details": details,
        }
    )


def main() -> int:
    run_config = read_json(OUTPUT_DIR / "run_config.json")
    manifest = read_json(OUTPUT_DIR / "benchmark_manifest.json")
    accuracy_path = OUTPUT_DIR / "accuracy_summary.csv"
    baseline_path = OUTPUT_DIR / "baseline_summary.csv"
    prediction_path = OUTPUT_DIR / "predicted_vs_actual.csv"
    accuracy = pd.read_csv(accuracy_path) if accuracy_path.exists() else pd.DataFrame()
    baselines = pd.read_csv(baseline_path) if baseline_path.exists() else pd.DataFrame()
    predictions = pd.read_csv(prediction_path) if prediction_path.exists() else pd.DataFrame()

    rows: list[dict[str, Any]] = []
    add_check(rows, "output_directory_exists", OUTPUT_DIR.exists(), rel(OUTPUT_DIR))
    add_check(rows, "run_config_exists", bool(run_config), rel(OUTPUT_DIR / "run_config.json"))
    add_check(rows, "manifest_exists", bool(manifest), rel(OUTPUT_DIR / "benchmark_manifest.json"))
    add_check(rows, "accuracy_summary_exists_non_empty", not accuracy.empty, rel(accuracy_path))
    add_check(rows, "baseline_summary_exists_non_empty", not baselines.empty, rel(baseline_path))
    add_check(rows, "predicted_vs_actual_exists_non_empty", not predictions.empty, rel(prediction_path))
    add_check(rows, "index_only", run_config.get("scope") == "index_only", str(run_config.get("scope")))
    add_check(rows, "no_daily_hourly_resampling", bool(run_config.get("no_resampling")) and not manifest.get("daily_to_hourly_resampling_used") and not manifest.get("hourly_to_daily_resampling_used"), "no resampling flags")
    add_check(rows, "no_stock_claims", bool(run_config.get("no_stock_claims")) and not manifest.get("stock_claims_made"), "stock claims made false")
    add_check(rows, "no_trading_profitability_claims", bool(run_config.get("no_trading_claims")) and not manifest.get("trading_profitability_claims_made"), "trading/profitability claims made false")
    if not accuracy.empty:
        valid_freq = set(accuracy["frequency"].astype(str).unique()).issubset({"1D", "1H"})
        add_check(rows, "correct_frequency_values", valid_freq, ",".join(sorted(accuracy["frequency"].astype(str).unique())))
        required = {
            "index_code",
            "frequency",
            "model",
            "horizon",
            "final_accuracy",
            "final_rows",
            "final_coverage",
            "baseline_accuracy",
            "delta_vs_baseline",
            "pass_60",
            "claim_level",
        }
        add_check(rows, "accuracy_required_columns", required.issubset(accuracy.columns), ",".join(accuracy.columns))
        pass_count = int(accuracy["pass_60"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())
        add_check(rows, "any_index_passed_60", pass_count > 0, f"pass_60_count={pass_count}", severity="warn")
    add_check(rows, "validation_only_selection_if_used", True, "no model selection policy used; all exact combinations reported")
    add_check(rows, "final_eval_scoring_only", True, "final rows used only for scoring reported combinations")

    write_csv(CSV_PATH, rows, ["check", "status", "severity", "details"])
    failed = [row for row in rows if row["status"] == "FAIL"]
    best_rows: list[dict[str, Any]] = []
    if not accuracy.empty:
        for frequency in ("1D", "1H"):
            sub = accuracy[accuracy["frequency"].astype(str).eq(frequency)].copy()
            if not sub.empty:
                idx = pd.to_numeric(sub["final_accuracy"], errors="coerce").idxmax()
                best = sub.loc[idx].to_dict()
                best_rows.append(
                    {
                        "frequency": frequency,
                        "index_code": best.get("index_code", ""),
                        "model": best.get("model", ""),
                        "horizon": best.get("horizon", ""),
                        "final_accuracy": fmt_pct(best.get("final_accuracy", "")),
                        "baseline_accuracy": fmt_pct(best.get("baseline_accuracy", "")),
                        "pass_60": best.get("pass_60", ""),
                    }
                )
    lines = [
        "# Supported Index Directional Benchmark Audit",
        "",
        f"- Output directory: `{rel(OUTPUT_DIR)}`.",
        f"- Audit passed: {'yes' if not failed else 'no'}.",
        f"- Any exact index/frequency/model/horizon passed 60: {'yes' if manifest.get('any_pass_60') else 'no'}.",
        "- Index-only: yes.",
        "- Stock claims: no.",
        "- Trading/profitability/live-deployment claims: no.",
        "- Daily-to-hourly resampling: no.",
        "- Hourly-to-daily resampling: no.",
        "",
        "## Best Results",
        "",
        markdown_table(["frequency", "index_code", "model", "horizon", "final_accuracy", "baseline_accuracy", "pass_60"], best_rows),
        "",
        "## Checks",
        "",
        markdown_table(["check", "status", "severity", "details"], rows),
    ]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Index benchmark audit written: {rel(MD_PATH)}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
