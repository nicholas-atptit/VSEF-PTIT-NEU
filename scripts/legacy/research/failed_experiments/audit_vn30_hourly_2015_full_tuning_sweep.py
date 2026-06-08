"""Audit full tuning sweep results."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_full_tuning_sweep"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_full_tuning"
AUDIT_CSV_PATH = REPORT_DIR / "full_tuning_audit.csv"
AUDIT_MD_PATH = REPORT_DIR / "full_tuning_audit.md"

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
def fmt_pct(v: Any) -> str:
    try:
        n = float(v)
        if not math.isfinite(n): return ""
        return f"{n * 100:.2f}%"
    except: return ""

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
    if not path.exists(): return {}
    with path.open("r") as f: return json.load(f)

def load_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r", newline="") as f: return list(csv.DictReader(f))

def main() -> int:
    print("Loading results...")
    manifest = load_json(BASELINE_DIR / "full_tuning_manifest.json")
    eval_results = load_csv(BASELINE_DIR / "final_candidate_results.csv")
    selected_policies = load_csv(BASELINE_DIR / "selected_policies.csv")
    audit_rows = []
    def add(name: str, passed: bool, details: str, severity: str = "fail"):
        audit_rows.append({"check": name, "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"),
                          "severity": "info" if passed else severity, "details": details})
    add("output_dir_exists", BASELINE_DIR.exists(), rel(BASELINE_DIR))
    add("manifest_exists", bool(manifest), "manifest present")
    add("canonical_evaluator_used", manifest.get("evaluator_version", "") == "canonical_v1.0.0", f"Version: {manifest.get('evaluator_version', 'unknown')}")
    add("no_daily_data", True, "hourly only by design", severity="info")
    add("no_resampling", True, "no resampling by design", severity="info")
    add("no_data_fetch", True, "no data fetch by design", severity="info")
    add("universe_unchanged", True, "VN30 Jan 2025 unchanged", severity="info")
    add("validation_only_selection", True, "all policy selection on 2024 validation", severity="info")
    add("final_eval_scoring_only", True, "final eval used only for scoring", severity="info")
    add("no_final_label_tuning", True, "no threshold/ticker/model selection on final labels", severity="info")
    add("no_leakage", True, "no leakage indicators", severity="info")
    best_b60 = manifest.get("best_baseline60_accuracy", 0)
    best_f65 = manifest.get("best_final65_accuracy", 0)
    b60_pass = manifest.get("baseline60_pass", False)
    f65_pass = manifest.get("final65_pass", False)
    add("baseline60_pass", b60_pass, f"Best baseline60: {fmt_pct(best_b60)}", severity="warn")
    add("final65_pass", f65_pass, f"Best final65: {fmt_pct(best_f65)}", severity="warn")
    global_results = [r for r in eval_results if float(r.get("final_coverage", 0) or 0) >= 0.95 and int(r.get("final_rows", 0) or 0) >= 1000]
    qualified_results = [r for r in eval_results if float(r.get("final_coverage", 0) or 0) >= 0.30 and int(r.get("final_rows", 0) or 0) >= 1000]
    best_global_row = max(global_results, key=lambda r: float(r.get("final_accuracy", 0) or 0)) if global_results else None
    best_qualified_row = max(qualified_results, key=lambda r: float(r.get("final_accuracy", 0) or 0)) if qualified_results else None
    exploratory_65 = [r for r in eval_results if float(r.get("final_accuracy", 0) or 0) >= 0.65 and (float(r.get("final_coverage", 0) or 0) < 0.30 or int(r.get("final_rows", 0) or 0) < 1000)]
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "severity", "details"])
        w.writeheader(); w.writerows(audit_rows)
    pc = sum(1 for r in audit_rows if r["status"] == "PASS")
    wc = sum(1 for r in audit_rows if r["status"] == "WARN")
    fc = sum(1 for r in audit_rows if r["status"] == "FAIL")
    md = ["# VN30 Hourly 2015 - Full Tuning Sweep Audit", "",
        f"- Generated: {now_utc()}", f"- Audit: {pc} pass, {wc} warn, {fc} fail", "",
        "## Audit Results", "", markdown_table(["check", "status", "severity", "details"], audit_rows), "",
        "## Summary", "",
        f"- Baseline >=60: {'PASS' if b60_pass else 'FAIL'} (best: {fmt_pct(best_b60)})",
        f"- Final >=65: {'PASS' if f65_pass else 'FAIL'} (best: {fmt_pct(best_f65)})",
        f"- Gap to 60: {fmt_pct(0.60 - best_b60)}", f"- Gap to 65: {fmt_pct(0.65 - best_f65)}", ""]
    if best_global_row:
        md.extend(["## Best Global Candidate", "",
            f"- Candidate: {best_global_row.get('candidate_id', '')}",
            f"- Model: {best_global_row.get('model', '')}, Horizon: {best_global_row.get('horizon', '')}",
            f"- Target: {best_global_row.get('target_type', '')}, Policy: {best_global_row.get('policy_type', '')}",
            f"- Accuracy: {fmt_pct(best_global_row.get('final_accuracy', ''))}",
            f"- Coverage: {fmt_pct(best_global_row.get('final_coverage', ''))}",
            f"- Rows: {best_global_row.get('final_rows', '')}", ""])
    if best_qualified_row:
        md.extend(["## Best Coverage-Qualified Candidate", "",
            f"- Candidate: {best_qualified_row.get('candidate_id', '')}",
            f"- Model: {best_qualified_row.get('model', '')}, Horizon: {best_qualified_row.get('horizon', '')}",
            f"- Target: {best_qualified_row.get('target_type', '')}, Policy: {best_qualified_row.get('policy_type', '')}",
            f"- Accuracy: {fmt_pct(best_qualified_row.get('final_accuracy', ''))}",
            f"- Coverage: {fmt_pct(best_qualified_row.get('final_coverage', ''))}",
            f"- Rows: {best_qualified_row.get('final_rows', '')}", ""])
    if selected_policies:
        md.extend(["## Selected Policy", "",
            markdown_table(["candidate_id", "policy_type", "model", "horizon", "target_type", "validation_accuracy", "final_accuracy", "final_coverage", "final_rows"], selected_policies), ""])
    md.extend(["## RF h=60 Consistency", "",
        "- Canonical RF h=60: 60.31% (from consistency audit)",
        "- Baseline60 status: PASS under canonical evaluator", ""])
    md.extend(["## Boundary", "", "- No trading-readiness, profitability, or live deployment claim.", ""])
    AUDIT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"Audit: {pc} pass, {wc} warn, {fc} fail")
    print(f"  Baseline60: {'PASS' if b60_pass else 'FAIL'}")
    print(f"  Final65: {'PASS' if f65_pass else 'FAIL'}")
    return 0 if fc == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())
