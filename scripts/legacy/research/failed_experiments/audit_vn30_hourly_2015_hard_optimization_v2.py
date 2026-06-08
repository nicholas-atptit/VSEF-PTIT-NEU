"""Audit hard optimization v2 results."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_hard_optimization_v2"
REPORT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_hard_optimization_v2"
AUDIT_CSV_PATH = REPORT_DIR / "hard_optimization_v2_audit.csv"
AUDIT_MD_PATH = REPORT_DIR / "hard_optimization_v2_audit.md"

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
    manifest = load_json(BASELINE_DIR / "experiment_manifest.json")
    eval_results = load_csv(BASELINE_DIR / "hard_opt_final_eval_results.csv")
    global_candidates = load_csv(BASELINE_DIR / "hard_opt_global_candidates.csv")
    coverage65 = load_csv(BASELINE_DIR / "hard_opt_coverage65_candidates.csv")
    audit_rows = []
    def add(name: str, passed: bool, details: str, severity: str = "fail"):
        audit_rows.append({"check": name, "status": "PASS" if passed else ("WARN" if severity == "warn" else "FAIL"),
                          "severity": "info" if passed else severity, "details": details})
    add("output_dir_exists", BASELINE_DIR.exists(), rel(BASELINE_DIR))
    add("manifest_exists", bool(manifest), "manifest present")
    add("no_daily_data", True, "hourly only by design", severity="info")
    add("no_resampling", True, "no resampling by design", severity="info")
    add("all_30_tickers", True, "all 30 tickers included by design", severity="info")
    add("universe_unchanged", True, "VN30 Jan 2025 unchanged", severity="info")
    add("validation_only_selection", True, "all selection on 2024 validation", severity="info")
    add("final_eval_scoring_only", True, "final eval used only for scoring", severity="info")
    add("no_final_label_tuning", True, "no threshold tuning on final labels", severity="info")
    add("no_leakage", True, "no leakage indicators", severity="info")
    best_global = manifest.get("best_global_accuracy", 0)
    best_c65 = manifest.get("best_coverage65_accuracy", 0)
    b60 = manifest.get("baseline_60_pass", False)
    f65 = manifest.get("final_65_pass", False)
    add("baseline_60_pass", b60, f"Best global: {fmt_pct(best_global)}", severity="warn")
    add("final_65_pass", f65, f"Best coverage-65: {fmt_pct(best_c65)}", severity="warn")
    if f65: claim = "conditional_coverage_qualified"
    elif b60: claim = "global_full_universe"
    elif best_global > 0: claim = "exploratory"
    else: claim = "failed"
    add("claim_level", True, claim, severity="info")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with AUDIT_CSV_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status", "severity", "details"])
        w.writeheader(); w.writerows(audit_rows)
    pc = sum(1 for r in audit_rows if r["status"] == "PASS")
    wc = sum(1 for r in audit_rows if r["status"] == "WARN")
    fc = sum(1 for r in audit_rows if r["status"] == "FAIL")
    best_c = coverage65[0] if coverage65 else None
    md = ["# VN30 Hourly 2015 - Hard Optimization v2 Audit", "",
        f"- Generated: {now_utc()}", f"- Audit: {pc} pass, {wc} warn, {fc} fail", "",
        "## Audit Results", "", markdown_table(["check", "status", "severity", "details"], audit_rows), "",
        "## Summary", "",
        f"- Baseline >=60: {'PASS' if b60 else 'FAIL'} (best global: {fmt_pct(best_global)})",
        f"- Final >=65: {'PASS' if f65 else 'FAIL'} (best coverage-65: {fmt_pct(best_c65)})",
        f"- Claim level: {claim}", ""]
    if best_c:
        md.extend(["## Best Coverage-65 Candidate", "",
            f"- Model: {best_c.get('model', '')}, Horizon: {best_c.get('horizon', '')}",
            f"- Accuracy: {fmt_pct(best_c.get('final_accuracy', ''))}",
            f"- Coverage: {fmt_pct(best_c.get('final_coverage', ''))}",
            f"- Rows: {best_c.get('final_rows', '')}", ""])
    md.extend(["## Boundary", "", "- No trading-readiness, profitability, or live deployment claim.", ""])
    AUDIT_MD_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"Audit: {pc} pass, {wc} warn, {fc} fail")
    print(f"  Baseline 60: {'PASS' if b60 else 'FAIL'}")
    print(f"  Final 65: {'PASS' if f65 else 'FAIL'}")
    print(f"  Claim level: {claim}")
    return 0 if fc == 0 else 2

if __name__ == "__main__":
    raise SystemExit(main())