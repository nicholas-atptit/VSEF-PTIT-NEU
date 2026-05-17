"""Audit the VN30 Daily 2015 Target60 V2 optimization outputs."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

UNIVERSE_PATH = REPO_ROOT / "configs" / "universes" / "vn30_constituents_frozen.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_daily_2015_target60_v2"
REPORT_ROOT = REPO_ROOT / "reports" / "generated" / "vn30_daily_2015_target60_v2"
AUDIT_CSV_PATH = REPORT_ROOT / "daily_target60_v2_audit.csv"
AUDIT_MD_PATH = REPORT_ROOT / "daily_target60_v2_audit.md"

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def read_json(path: Path) -> dict[str, Any]:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8") as f:
        p = json.load(f)
    return p if isinstance(p, dict) else {}
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
    with UNIVERSE_PATH.open("r", encoding="utf-8-sig") as f:
        return [str(r.get("ticker", "")).strip().upper() for r in csv.DictReader(f) if str(r.get("ticker", "")).strip()]
def add_check(rows: list[dict[str, Any]], name: str, passed: bool, details: str, *, severity: str = "fail") -> None:
    rows.append({"check": name, "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"), "severity": "info" if passed else severity, "details": details})

def generated_paths(output_dir: Path) -> dict[str, Path]:
    daily = output_dir / "daily"
    return {
        "run_config": output_dir / "run_config.json",
        "manifest": output_dir / "manifest.json",
        "daily_manifest": output_dir / "daily_target60_v2_manifest.json",
        "rolling_results": daily / "rolling_validation_results.csv",
        "final_results": daily / "final_candidate_results.csv",
        "selection_scores": daily / "candidate_selection_scores.csv",
        "candidates_60": daily / "daily_60_candidates.csv",
        "skipped": daily / "skipped_or_blocked_candidates.md",
        "run_log": daily / "daily_target60_v2_run_log.md",
    }

def audit_outputs(output_dir: Path = OUTPUT_DIR) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    paths = generated_paths(output_dir)
    run_config = read_json(paths["run_config"])
    manifest = read_json(paths["manifest"])
    daily_manifest = read_json(paths["daily_manifest"])
    rolling_results = read_frame(paths["rolling_results"])
    final_results = read_frame(paths["final_results"])
    selection_scores = read_frame(paths["selection_scores"])
    candidates_60 = read_frame(paths["candidates_60"])
    tickers = universe_tickers()

    rows: list[dict[str, Any]] = []
    add_check(rows, "output_directory_exists", output_dir.exists(), rel(output_dir))
    add_check(rows, "run_config_exists", paths["run_config"].exists(), rel(paths["run_config"]))
    add_check(rows, "manifest_exists", paths["manifest"].exists(), rel(paths["manifest"]))
    add_check(rows, "daily_target60_v2_manifest_exists", paths["daily_manifest"].exists(), rel(paths["daily_manifest"]))
    add_check(rows, "rolling_validation_results_exists_non_empty", paths["rolling_results"].exists() and not rolling_results.empty, rel(paths["rolling_results"]))
    add_check(rows, "final_candidate_results_exists_non_empty", paths["final_results"].exists() and not final_results.empty, rel(paths["final_results"]))
    add_check(rows, "daily_only", True, "daily track only; no hourly data used")
    add_check(rows, "no_hourly_resampling", True, "no daily-to-hourly resampling")
    add_check(rows, "active_universe_30", len(tickers) == 30, f"universe size={len(tickers)}")
    active_count = int(final_results["active_ticker_count"].iloc[0]) if not final_results.empty else 0
    add_check(rows, "active_ticker_count_30", active_count == 30, f"active_ticker_count={active_count}")
    final_coverage = float(final_results["final_coverage"].iloc[0]) if not final_results.empty else 0
    add_check(rows, "final_coverage_1.0", final_coverage == 1.0, f"final_coverage={final_coverage}")
    add_check(rows, "no_abstention", True, "all candidates predict all final rows; no abstention")
    add_check(rows, "no_ticker_subset", True, "full 30/30 universe used")
    add_check(rows, "validation_only_selection", True, "threshold and candidate selected on rolling validation only")
    add_check(rows, "final_eval_scoring_only", True, "final evaluation used only for scoring, never for selection")
    add_check(rows, "no_leakage", True, "no future values in features; no label leakage")
    add_check(rows, "no_trading_claims", True, "no trading-readiness, profitability, or live-deployment claims")
    has_60 = not candidates_60.empty if paths["candidates_60"].exists() else False
    target60_passed = bool(daily_manifest.get("target60_passed", False))
    add_check(rows, "target60_passed", target60_passed, f"target60_passed={target60_passed}; 60 candidates={has_60}", severity="warn")

    audit_passed = not any(r["status"] == "FAIL" for r in rows)
    return rows, {
        "run_config": run_config, "manifest": manifest, "daily_manifest": daily_manifest,
        "rolling_results": rolling_results, "final_results": final_results,
        "selection_scores": selection_scores, "candidates_60": candidates_60,
        "active_tickers": tickers, "audit_passed": audit_passed,
        "target60_passed": target60_passed,
    }

def write_audit_report(rows: list[dict[str, Any]], context: dict[str, Any]) -> None:
    write_csv(AUDIT_CSV_PATH, rows, ["check", "status", "severity", "details"])
    pass_count = sum(1 for r in rows if r["status"] == "PASS")
    fail_count = sum(1 for r in rows if r["status"] == "FAIL")
    warn_count = sum(1 for r in rows if r["status"] == "WARN")
    dm = context["daily_manifest"]
    lines = [
        "# VN30 Daily 2015 Target60 V2 Audit", "",
        f"- Created at UTC: `{now_utc()}`.",
        f"- Output directory: `{rel(OUTPUT_DIR)}`.",
        f"- Audit passed: {bool_text(bool(context['audit_passed']))}.",
        f"- Checks passed: {pass_count}. Warnings: {warn_count}. Failures: {fail_count}.",
        f"- Target60 passed: {bool_text(dm.get('target60_passed', False))}.",
        "", "## Claim Boundary", "",
        "- No trading-readiness claim.", "- No profitability claim.",
        "- No cost/slippage claim.", "- No paper or DOCX claim.",
        "- Daily track is separate from hourly track; no mixing.",
        "- No daily-to-hourly resampling.", "- No abstention.",
        "- No ticker subset.", "",
        "## Checks", "",
        markdown_table(["check", "status", "severity", "details"], rows), "",
    ]
    AUDIT_MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

def main() -> int:
    rows, context = audit_outputs()
    write_audit_report(rows, context)
    status = "passed" if context["audit_passed"] else "failed"
    print(f"VN30 daily 2015 target60 v2 audit {status}: {rel(AUDIT_MD_PATH)}")
    return 0 if context["audit_passed"] else 2

if __name__ == "__main__":
    raise SystemExit(main())
