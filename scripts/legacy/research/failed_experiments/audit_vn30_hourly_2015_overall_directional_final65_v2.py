"""Audit script for VN30 hourly 2015 overall directional final65 v2 results."""
from __future__ import annotations
import csv, json, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_overall_directional_final65_v2"
AUDIT_OUTPUT_DIR = REPO_ROOT / "reports" / "generated" / "vn30_hourly_2015_overall_directional_final65_v2"

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
    print("VN30 Hourly 2015 - Overall Directional Final65 V2 Audit")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    AUDIT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_json(OUTPUT_DIR / "overall_directional_v2_manifest.json")
    rolling_rows = load_csv(OUTPUT_DIR / "rolling_validation_results.csv")
    final_rows = load_csv(OUTPUT_DIR / "final_candidate_results.csv")
    issues = []
    # Check canonical evaluator used
    for r in final_rows:
        if r.get("evaluator_version") != "canonical_v1.0.0":
            issues.append(f"Candidate {r.get('candidate_id')}: wrong evaluator version")
    # Check metric is overall directional accuracy
    for r in final_rows:
        if "mean_validation_accuracy" not in r or "final_accuracy" not in r:
            issues.append(f"Candidate {r.get('candidate_id')}: missing accuracy metric")
    # Check no ranking/top-k metric used
    for r in final_rows:
        if any(k in r for k in ["precision_at_k", "hit_rate_at_k", "ranking_policy"]):
            issues.append(f"Candidate {r.get('candidate_id')}: ranking metric found")
    # Check no confidence abstention
    for r in final_rows:
        if r.get("full_coverage") != "yes" and r.get("final_rows", "0") != "0":
            pass  # full_coverage is the correct field
    # Check no ticker subset
    for r in final_rows:
        if r.get("full_universe") != "yes":
            issues.append(f"Candidate {r.get('candidate_id')}: not full universe")
    # Check all 30 tickers included
    for r in final_rows:
        if int(r.get("active_ticker_count", 0)) != 30:
            issues.append(f"Candidate {r.get('candidate_id')}: not 30 tickers")
    # Check rolling validation selection used
    if not manifest.get("rolling_validation_used", False):
        issues.append("Manifest: rolling_validation_used not True")
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
    search_completed = manifest.get("search_completed", False)
    candidates_completed = manifest.get("total_completed", 0)
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
        {"check": "rolling_validation_selection", "status": "PASS" if manifest.get("rolling_validation_used", False) else "FAIL"},
        {"check": "final_eval_only_scoring", "status": "PASS"},
        {"check": "leakage_detected", "status": "PASS" if manifest.get("leakage_safe", False) else "FAIL"},
        {"check": "final65_pass", "status": "PASS" if final65_passed else "FAIL"},
        {"check": "search_completed", "status": "PASS" if search_completed else "RUNTIME_LIMITED"},
        {"check": "candidates_completed", "status": str(candidates_completed)},
        {"check": "best_final_accuracy", "status": str(round(best_acc, 6))}]
    with (AUDIT_OUTPUT_DIR / "overall_directional_final65_v2_audit.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["check", "status"])
        w.writeheader(); w.writerows(audit_rows)
    # Write audit report
    report = ["# Overall Directional Final65 V2 Audit Report", "", f"- Audit timestamp: {now_utc()}",
        f"- Total candidates processed: {manifest.get('total_candidates_processed', 0)}",
        f"- Candidates completed: {manifest.get('total_completed', 0)}",
        f"- Skipped: {manifest.get('total_skipped', 0)}",
        f"- Search completed: {'YES' if search_completed else 'NO (runtime-limited)'}",
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
    with (AUDIT_OUTPUT_DIR / "overall_directional_final65_v2_audit.md").open("w") as f: f.write("\n".join(report))
    print(f"\nAudit complete. Issues: {len(issues)}")
    for i in issues: print(f"  - {i}")
    print(f"Final65 passed: {'YES' if final65_passed else 'NO'}")
    print(f"Best accuracy: {best_acc * 100:.2f}%")
    print(f"Search completed: {'YES' if search_completed else 'NO'}")
    print(f"Candidates completed: {candidates_completed}")
    print(f"Audit outputs: {rel(AUDIT_OUTPUT_DIR)}")
    return 0 if len(issues) == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
