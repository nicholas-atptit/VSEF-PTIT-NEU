"""Audit script for VN30 hourly 2015 overall directional final65 results."""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_overall_directional_final65"
AUDIT_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_overall_directional_final65"

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def load_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_json(path: Path) -> dict:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Overall Directional Final65 Audit")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_json(OUTPUT_DIR / "overall_directional_manifest.json")
    val_rows = load_csv(OUTPUT_DIR / "validation_candidate_results.csv")
    eval_rows = load_csv(OUTPUT_DIR / "final_candidate_results.csv")
    issues = []
    # Check canonical evaluator used
    for r in eval_rows:
        if r.get("evaluator_version") != "canonical_v1.0.0":
            issues.append(f"Candidate {r.get('candidate_id')}: wrong evaluator version")
    # Check metric is overall directional accuracy
    for r in eval_rows:
        if "validation_accuracy" not in r or "final_accuracy" not in r:
            issues.append(f"Candidate {r.get('candidate_id')}: missing accuracy metric")
    # Check no ranking/top-k metric used
    for r in eval_rows:
        if any(k in r for k in ["precision_at_k", "hit_rate_at_k", "ranking_policy"]):
            issues.append(f"Candidate {r.get('candidate_id')}: ranking metric found")
    # Check no confidence abstention
    for r in eval_rows:
        if r.get("validation_coverage") != "1.0" or r.get("final_coverage") != "1.0":
            issues.append(f"Candidate {r.get('candidate_id')}: coverage not 100%")
    # Check no ticker subset
    for r in eval_rows:
        if r.get("full_universe") != "yes":
            issues.append(f"Candidate {r.get('candidate_id')}: not full universe")
    # Check all 30 tickers included
    for r in eval_rows:
        if int(r.get("active_ticker_count", 0)) != 30:
            issues.append(f"Candidate {r.get('candidate_id')}: not 30 tickers")
    # Check full final coverage
    for r in eval_rows:
        if r.get("full_coverage") != "yes":
            issues.append(f"Candidate {r.get('candidate_id')}: not full coverage")
    # Check selection done on validation only
    for r in eval_rows:
        if r.get("selected_on_validation") != "yes":
            issues.append(f"Candidate {r.get('candidate_id')}: not selected on validation")
    # Check leakage
    if not manifest.get("leakage_safe", False):
        issues.append("Manifest: leakage_safe not True")
    if manifest.get("daily_data_used", False):
        issues.append("Manifest: daily_data_used is True")
    if manifest.get("resampling_used", False):
        issues.append("Manifest: resampling_used is True")
    # Final65 pass check
    final65_passed = manifest.get("final65_passed", False)
    best_acc = manifest.get("best_final_accuracy", 0.0)
    # Write audit CSV
    audit_rows = [{"check": "canonical_evaluator_used", "status": "PASS" if not any("evaluator" in i for i in issues) else "FAIL"},
        {"check": "metric_overall_directional", "status": "PASS" if not any("accuracy" in i for i in issues) else "FAIL"},
        {"check": "no_ranking_topk_metric", "status": "PASS" if not any("ranking" in i for i in issues) else "FAIL"},
        {"check": "no_confidence_abstention", "status": "PASS" if not any("coverage" in i for i in issues) else "FAIL"},
        {"check": "no_ticker_subset", "status": "PASS" if not any("universe" in i for i in issues) else "FAIL"},
        {"check": "all_30_tickers", "status": "PASS" if not any("ticker" in i for i in issues) else "FAIL"},
        {"check": "full_final_coverage", "status": "PASS" if not any("coverage" in i for i in issues) else "FAIL"},
        {"check": "no_daily_data", "status": "PASS" if not manifest.get("daily_data_used", False) else "FAIL"},
        {"check": "no_resampling", "status": "PASS" if not manifest.get("resampling_used", False) else "FAIL"},
        {"check": "validation_only_selection", "status": "PASS" if not any("validation" in i for i in issues) else "FAIL"},
        {"check": "leakage_detected", "status": "PASS" if manifest.get("leakage_safe", False) else "FAIL"},
        {"check": "final65_pass", "status": "PASS" if final65_passed else "FAIL"},
        {"check": "best_final_accuracy", "status": str(round(best_acc, 6))},
        {"check": "total_candidates", "status": str(manifest.get("total_candidates_processed", 0))},
        {"check": "total_skipped", "status": str(manifest.get("total_skipped", 0))}]
    with (AUDIT_OUTPUT_DIR / "overall_directional_final65_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status"])
        w.writeheader(); w.writerows(audit_rows)
    # Write audit report
    report = ["# Overall Directional Final65 Audit Report", "", f"- Audit timestamp: {now_utc()}",
        f"- Total candidates: {manifest.get('total_candidates_processed', 0)}",
        f"- Skipped: {manifest.get('total_skipped', 0)}",
        f"- Best final accuracy: {best_acc * 100:.2f}%",
        f"- Final65 passed: {'YES' if final65_passed else 'NO'}", "",
        "## Audit Checks", ""]
    for row in audit_rows:
        report.append(f"- {row['check']}: {row['status']}")
    report.extend(["", "## Issues", ""])
    if issues:
        for i in issues: report.append(f"- {i}")
    else:
        report.append("- No issues found")
    report.append("")
    with (AUDIT_OUTPUT_DIR / "overall_directional_final65_audit.md").open("w") as f: f.write("\n".join(report))
    print(f"\nAudit complete. Issues: {len(issues)}")
    for i in issues: print(f"  - {i}")
    print(f"Final65 passed: {'YES' if final65_passed else 'NO'}")
    print(f"Best accuracy: {best_acc * 100:.2f}%")
    print(f"Audit outputs: {rel(AUDIT_OUTPUT_DIR)}")
    return 0 if len(issues) == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
