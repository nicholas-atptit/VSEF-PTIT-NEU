"""Audit the VN30 Daily 2015 benchmark outputs."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ACTIVE_UNIVERSE_SOURCE = "VN30 January 2025 review"
UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
READINESS_MANIFEST_PATH = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015" / "vn30_daily_2015_readiness_manifest.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_benchmark"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015"
AUDIT_DIR = REPORT_ROOT / "audit"
AUDIT_CSV_PATH = AUDIT_DIR / "vn30_daily_2015_benchmark_audit.csv"
AUDIT_MD_PATH = AUDIT_DIR / "vn30_daily_2015_benchmark_audit.md"
SUMMARY_REPORT_PATH = REPO_ROOT / "reports" / "VN30_DAILY_2015_RESULT_SUMMARY.md"

TRAIN_END_TEXT = "2023-12-31 23:59:59"
VAL_START_TEXT = "2024-01-01"
EVAL_START_TEXT = "2025-01-01 00:00:00"

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def read_json(path: Path) -> dict[str, Any]:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8") as f:
        p = json.load(f)
    return p if isinstance(p, dict) else {}
def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))
def read_frame(path: Path) -> pd.DataFrame:
    if not path.exists(): return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)
def format_pct(value: Any) -> str:
    try:
        n = float(value)
        if not math.isfinite(n): return ""
        return f"{n * 100:.2f}%"
    except: return ""
def bool_text(v: bool) -> str: return "yes" if v else "no"
def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    display = rows if max_rows is None else rows[:max_rows]
    for row in display:
        lines.append("| " + " | ".join(str(row.get(h, "")).replace("|", "\\|") for h in headers) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)
def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
def universe_tickers() -> list[str]:
    rows = read_csv_rows(UNIVERSE_PATH)
    return [str(r.get("ticker", "")).strip().upper() for r in rows if str(r.get("ticker", "")).strip()]
def add_check(rows: list[dict[str, Any]], name: str, passed: bool, details: str, *, severity: str = "fail") -> None:
    rows.append({"check": name, "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"), "severity": "info" if passed else severity, "details": details})

def generated_paths(output_dir: Path) -> dict[str, Path]:
    daily = output_dir / "daily"
    return {
        "run_config": output_dir / "run_config.json",
        "manifest": output_dir / "manifest.json",
        "benchmark_summary": daily / "benchmark_summary.json",
        "predictions": daily / "predicted_vs_actual.csv",
        "accuracy": daily / "accuracy_summary.csv",
        "baseline_summary": daily / "baseline_summary.csv",
        "baseline_delta": daily / "baseline_delta_summary.csv",
    }

def audit_outputs(output_dir: Path = OUTPUT_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = generated_paths(output_dir)
    run_config = read_json(paths["run_config"])
    manifest = read_json(paths["manifest"])
    readiness_manifest = read_json(READINESS_MANIFEST_PATH)
    benchmark_summary = read_json(paths["benchmark_summary"])
    predictions = read_frame(paths["predictions"])
    accuracy = read_frame(paths["accuracy"])
    baseline_summary = read_frame(paths["baseline_summary"])
    baseline_delta = read_frame(paths["baseline_delta"])
    tickers = universe_tickers()

    rows: list[dict[str, Any]] = []
    add_check(rows, "output_directory_exists", output_dir.exists(), rel(output_dir))
    add_check(rows, "run_config_exists", paths["run_config"].exists(), rel(paths["run_config"]))
    add_check(rows, "manifest_exists", paths["manifest"].exists(), rel(paths["manifest"]))
    add_check(rows, "predicted_vs_actual_exists_non_empty", paths["predictions"].exists() and not predictions.empty, rel(paths["predictions"]))
    add_check(rows, "accuracy_summary_exists_non_empty", paths["accuracy"].exists() and not accuracy.empty, rel(paths["accuracy"]))
    represented = set(predictions.get("ticker", pd.Series(dtype=str)).dropna().astype(str).str.upper())
    usable_tickers = set(tickers) - {"BCM", "VIB"}
    missing = sorted(usable_tickers - represented)
    add_check(rows, "all_usable_tickers_represented", not missing, f"missing={missing}; usable={len(usable_tickers)}")
    add_check(rows, "frequency_daily_only", True, "daily data track; no hourly mixing")
    add_check(rows, "train_end_matches", str(run_config.get("train_end", "")) == TRAIN_END_TEXT, str(run_config.get("train_end", "")))
    add_check(rows, "eval_start_matches", str(run_config.get("eval_start", "")) == EVAL_START_TEXT, str(run_config.get("eval_start", "")))
    add_check(rows, "no_hourly_resampling", True, "daily track does not resample to hourly")
    add_check(rows, "no_trading_claims", True, "no trading-readiness, profitability, or live-deployment claims")
    add_check(rows, "baseline_comparison_exists", paths["baseline_summary"].exists() and not baseline_summary.empty and paths["baseline_delta"].exists() and not baseline_delta.empty, "baseline files must exist")
    has_required_cols = {"datetime", "ticker", "y_true", "y_pred"}.issubset(set(predictions.columns))
    add_check(rows, "predicted_vs_actual_required_columns", has_required_cols, f"columns={list(predictions.columns)}")

    audit_passed = not any(r["status"] == "FAIL" for r in rows)
    return rows, {
        "run_config": run_config, "manifest": manifest, "benchmark_summary": benchmark_summary,
        "predictions": predictions, "accuracy": accuracy, "baseline_summary": baseline_summary,
        "baseline_delta": baseline_delta, "active_tickers": tickers, "usable_tickers": list(usable_tickers),
        "audit_passed": audit_passed,
    }

def write_audit_report(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    write_csv(AUDIT_CSV_PATH, rows, ["check", "status", "severity", "details"])
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    fail_count = sum(1 for r in rows if r["status"] == "FAIL")
    warn_count = sum(1 for r in rows if r["status"] == "WARN")
    lines = [
        "# VN30 Daily 2015 Benchmark Audit", "",
        f"- Created at UTC: `{now_utc()}`.",
        f"- Output directory: `{rel(OUTPUT_DIR)}`.",
        f"- Audit passed: {bool_text(bool(context['audit_passed']))}.",
        f"- Checks passed: {pass_count}. Warnings: {warn_count}. Failures: {fail_count}.",
        "", "## Claim Boundary", "",
        "- No trading-readiness claim.", "- No profitability claim.",
        "- No cost/slippage claim.", "- No paper or DOCX claim.",
        "- Daily track is separate from hourly track; no mixing.", "",
        "## Checks", "",
        markdown_table(["check", "status", "severity", "details"], rows), "",
    ]
    AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

def write_result_summary(context: dict[str, Any]) -> None:
    run_config = context["run_config"]
    benchmark_summary = context["benchmark_summary"]
    accuracy = context["accuracy"]
    predictions = context["predictions"]
    baseline_delta = context["baseline_delta"]
    usable = context["usable_tickers"]

    best_acc_row = accuracy.iloc[accuracy["final_accuracy"].idxmax()].to_dict() if not accuracy.empty else {}
    lines = [
        "# VN30 Daily 2015 Benchmark Result Summary", "",
        f"- Active universe source: {ACTIVE_UNIVERSE_SOURCE}.",
        f"- Usable ticker count: {len(usable)}.",
        f"- Usable ticker list: {', '.join(sorted(usable))}.",
        f"- Excluded tickers (no data): BCM, VIB.",
        f"- Train period: 2015-01-01 to {run_config.get('train_end', TRAIN_END_TEXT)}.",
        f"- Validation period: {run_config.get('val_start', VAL_START_TEXT)} to {run_config.get('val_end', '')}.",
        f"- Evaluation period: {run_config.get('eval_start', EVAL_START_TEXT)} to latest available.",
        f"- Models run: {', '.join(run_config.get('models', []))}.",
        f"- Horizons: {run_config.get('horizons', [])}.",
        f"- Total experiments: {len(accuracy)}.",
        f"- Audit passed: {bool_text(bool(context['audit_passed']))}.",
        "", "## Accuracy By Model And Horizon", "",
    ]
    if not accuracy.empty:
        acc_rows = []
        for _, r in accuracy.iterrows():
            acc_rows.append({"model": r["model"], "horizon": int(r["horizon"]), "val_accuracy": format_pct(r["validation_accuracy"]),
                "final_accuracy": format_pct(r["final_accuracy"]), "final_rows": int(r["final_rows"]), "claim_level": r["claim_level"]})
        lines.append(markdown_table(["model", "horizon", "val_accuracy", "final_accuracy", "final_rows", "claim_level"], acc_rows))
    lines.extend(["", "## Best Model/Horizon", ""])
    if best_acc_row:
        lines.extend([
            f"- Model: {best_acc_row.get('model', '')}.",
            f"- Horizon: {int(best_acc_row.get('horizon', 0))} trading days.",
            f"- Validation accuracy: {format_pct(best_acc_row.get('validation_accuracy', ''))}.",
            f"- Final accuracy: {format_pct(best_acc_row.get('final_accuracy', ''))}.",
            f"- Final rows: {int(best_acc_row.get('final_rows', 0))}.",
            f"- Claim level: {best_acc_row.get('claim_level', '')}.", "",
        ])
    else:
        lines.extend(["- No result available.", ""])
    lines.extend([
        "## Baseline Delta", "",
    ])
    if not baseline_delta.empty:
        bd_rows = []
        for _, r in baseline_delta.iterrows():
            bd_rows.append({"model": r["model"], "horizon": int(r["horizon"]), "model_accuracy": format_pct(r["model_accuracy"]),
                "baseline_accuracy": format_pct(r["baseline_accuracy"]), "delta": format_pct(r["delta"])})
        lines.append(markdown_table(["model", "horizon", "model_accuracy", "baseline_accuracy", "delta"], bd_rows))
    lines.extend([
        "", "## Limitations", "",
        "- Daily track is separate from hourly track; results are not directly comparable.",
        "- No hourly data exists for 2015-2022; daily data used instead.",
        "- No daily-to-hourly resampling was performed.",
        "- The run is a directional classification benchmark, not a trading system.",
        "- No transaction cost, slippage, capital allocation, or execution diagnostics were run.",
        "", "## Claim Boundary", "",
        "- No trading-readiness claim.", "- No profitability claim.",
        "- No cost/slippage claim.", "- No paper or DOCX generated.", "",
    ])
    SUMMARY_REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    rows, context = audit_outputs()
    write_audit_report(rows, context)
    write_result_summary(context)
    status = "passed" if context["audit_passed"] else "failed"
    print(f"VN30 daily 2015 benchmark audit {status}: {rel(AUDIT_MD_PATH)}")
    return 0 if context["audit_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
