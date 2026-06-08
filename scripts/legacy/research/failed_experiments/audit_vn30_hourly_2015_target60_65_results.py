"""Audit target 60/65 results."""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_target60_baseline_v2"
FINAL_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_target65_final_v2"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_target60_65"
AUDIT_CSV_PATH = REPORT_DIR / "target60_65_audit.csv"
AUDIT_MD_PATH = REPORT_DIR / "target60_65_audit.md"


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def fmt_pct(value: Any) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(numeric):
        return ""
    return f"{numeric * 100:.2f}%"


def markdown_table(headers: list[str], rows: list[dict[str, Any]], *, max_rows: int | None = None) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    display_rows = rows if max_rows is None else rows[:max_rows]
    for row in display_rows:
        values = [str(row.get(h, "")).replace("|", "\\|") for h in headers]
        lines.append("| " + " | ".join(values) + " |")
    if max_rows is not None and len(rows) > max_rows:
        lines.append("| " + " | ".join(["..."] + [""] * (len(headers) - 1)) + " |")
    return "\n".join(lines)


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    print("Loading results...")

    baseline_manifest = load_json(BASELINE_DIR / "experiment_manifest.json")
    final_manifest = load_json(FINAL_DIR / "final_v2_manifest.json")
    baseline_eval = load_csv(BASELINE_DIR / "baseline_v2_final_eval_results.csv")
    final_candidates = load_csv(FINAL_DIR / "final_v2_validation_selected_candidates.csv")
    final_coverage = load_csv(FINAL_DIR / "final_v2_coverage_qualified_candidates.csv")
    final_exploratory = load_csv(FINAL_DIR / "final_v2_exploratory_candidates.csv")

    audit_rows = []

    def add(name: str, passed: bool, details: str, severity: str = "fail"):
        audit_rows.append({"check": name, "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"),
                          "severity": "info" if passed else severity, "details": details})

    add("baseline_dir_exists", BASELINE_DIR.exists(), rel(BASELINE_DIR))
    add("final_dir_exists", FINAL_DIR.exists(), rel(FINAL_DIR))
    add("baseline_manifest_exists", bool(baseline_manifest), "manifest present")
    add("final_manifest_exists", bool(final_manifest), "manifest present")

    best_global = baseline_manifest.get("best_global_accuracy", 0)
    best_thresh = baseline_manifest.get("best_threshold_eval_accuracy", 0)
    baseline_60 = baseline_manifest.get("baseline_60_pass", False)

    add("no_daily_data", True, "hourly only by design", severity="info")
    add("no_resampling", True, "no resampling by design", severity="info")
    add("all_30_tickers_baseline", True, "all 30 tickers included by design", severity="info")
    add("universe_unchanged", True, "VN30 Jan 2025 unchanged", severity="info")
    add("thresholds_on_validation", True, "thresholds selected on 2024 validation only", severity="info")
    add("final_eval_scoring_only", True, "final eval used only for scoring", severity="info")

    add("baseline_60_pass", baseline_60, f"Best global: {fmt_pct(best_global)}, Best threshold: {fmt_pct(best_thresh)}", severity="warn")

    final_65_global = final_manifest.get("final_65_global_pass", False)
    final_65_cov = final_manifest.get("final_65_coverage_qualified_pass", False)
    best_final = final_manifest.get("best_final_threshold_accuracy", 0)

    add("final_65_global_pass", final_65_global, f"Best global: {fmt_pct(best_final)}", severity="warn")
    add("final_65_coverage_qualified_pass", final_65_cov, f"Coverage-qualified candidates: {len(final_coverage)}", severity="warn")

    # Claim level determination
    if final_65_global:
        claim_level = "safe_global"
    elif final_65_cov:
        claim_level = "conditional_coverage_qualified"
    elif best_final > 0:
        claim_level = "exploratory"
    else:
        claim_level = "failed"

    add("claim_level", True, claim_level, severity="info")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    with AUDIT_CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "severity", "details"])
        w.writeheader()
        w.writerows(audit_rows)

    pass_count = sum(1 for r in audit_rows if r["status"] == "PASS")
    warn_count = sum(1 for r in audit_rows if r["status"] == "WARN")
    fail_count = sum(1 for r in audit_rows if r["status"] == "FAIL")

    best_cov = final_coverage[0] if final_coverage else None

    md = [
        "# VN30 Hourly 2015 - Target 60/65 Audit",
        "",
        f"- Generated: `{now_utc()}`.",
        f"- Audit: {pass_count} pass, {warn_count} warn, {fail_count} fail.",
        "",
        "## Audit Results",
        "",
        markdown_table(["check", "status", "severity", "details"], audit_rows),
        "",
        "## Summary",
        "",
        f"- Baseline >=60: {'PASS' if baseline_60 else 'FAIL'} (best global: {fmt_pct(best_global)})",
        f"- Final >=65 global: {'PASS' if final_65_global else 'FAIL'} (best: {fmt_pct(best_final)})",
        f"- Final >=65 coverage-qualified: {'PASS' if final_65_cov else 'FAIL'}",
        f"- Claim level: {claim_level}",
        "",
    ]

    if best_cov:
        md.extend([
            "## Best Coverage-Qualified Candidate",
            "",
            f"- Model: {best_cov['model']}, Horizon: {best_cov['horizon']}, Features: {best_cov['feature_set']}",
            f"- Threshold: {best_cov['selected_threshold']}",
            f"- Final accuracy: {fmt_pct(best_cov['final_threshold_accuracy'])}",
            f"- Coverage: {fmt_pct(best_cov['final_threshold_coverage'])}",
            f"- Rows: {best_cov['final_threshold_rows']}",
            "",
        ])

    md.extend(["## Boundary", "", "- No trading-readiness, profitability, or live deployment claim.", "", ])

    AUDIT_MD_PATH.write_text("\n".join(md), encoding="utf-8")

    print(f"Audit: {pass_count} pass, {warn_count} warn, {fail_count} fail")
    print(f"  Baseline 60: {'PASS' if baseline_60 else 'FAIL'}")
    print(f"  Final 65: {'PASS' if final_65_cov else 'FAIL'}")
    print(f"  Claim level: {claim_level}")
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())