"""Audit script for VN30 hourly 2015 top-k ranking results."""
from __future__ import annotations
import csv, json, math, sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_topk_ranking_experiments"
REPORTS_DIR = REPO_ROOT / "reports"
AUDIT_DIR = REPO_ROOT / "outputs" / "audits"

def rel(path: Path) -> str:
    try: return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError: return path.as_posix()

def now_utc() -> str: return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def fmt_pct(v) -> str:
    try:
        n = float(v)
        if not math.isfinite(n): return ""
        return f"{n * 100:.2f}%"
    except: return ""

def load_csv(path: Path) -> list[dict]:
    if not path.exists(): return []
    with path.open("r", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def load_json(path: Path) -> dict:
    if not path.exists(): return {}
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def audit_manifest(manifest: dict) -> list[str]:
    issues = []
    if not manifest.get("leakage_safe"): issues.append("leakage_safe not True")
    if manifest.get("daily_data_used"): issues.append("daily_data_used is True")
    if manifest.get("resampling_used"): issues.append("resampling_used is True")
    if manifest.get("total_experiments", 0) == 0: issues.append("no experiments recorded")
    return issues

def audit_validation_results(val_rows: list[dict]) -> list[str]:
    issues = []
    for r in val_rows:
        if r.get("selected_on_validation") != "yes": issues.append(f"row not selected on validation: {r}")
        try:
            p = float(r.get("validation_precision_at_k", 0))
            h = float(r.get("validation_hit_rate_at_k", 0))
            if p > 1.0 or p < 0.0: issues.append(f"invalid precision: {p}")
            if h > 1.0 or h < 0.0: issues.append(f"invalid hit_rate: {h}")
        except: pass
    return issues

def audit_final_results(eval_rows: list[dict], val_rows: list[dict]) -> list[str]:
    issues = []
    val_keys = {(r.get("model"), r.get("horizon"), r.get("k")) for r in val_rows}
    for r in eval_rows:
        key = (r.get("model"), r.get("horizon"), r.get("k"))
        if key not in val_keys: issues.append(f"eval row not in validation: {key}")
        try:
            p = float(r.get("final_precision_at_k", 0))
            h = float(r.get("final_hit_rate_at_k", 0))
            if p > 1.0 or p < 0.0: issues.append(f"invalid final precision: {p}")
            if h > 1.0 or h < 0.0: issues.append(f"invalid final hit_rate: {h}")
        except: pass
    return issues

def audit_cross_consistency(val_rows: list[dict], eval_rows: list[dict]) -> list[str]:
    issues = []
    val_map = {(r.get("model"), r.get("horizon"), r.get("k")): r for r in val_rows}
    for r in eval_rows:
        key = (r.get("model"), r.get("horizon"), r.get("k"))
        vr = val_map.get(key)
        if not vr: continue
        try:
            vp = float(r.get("validation_precision_at_k", 0))
            vp_orig = float(vr.get("validation_precision_at_k", 0))
            if abs(vp - vp_orig) > 1e-6: issues.append(f"validation precision mismatch: {key}")
        except: pass
    return issues

def audit_selected_policy(policy_rows: list[dict], val_rows: list[dict]) -> list[str]:
    issues = []
    if not policy_rows: return ["no selected policy"]
    if len(policy_rows) > 1: issues.append(f"multiple selected policies: {len(policy_rows)}")
    return issues

def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Top-K Ranking Audit")
    print("=" * 60)
    print(f"Started: {now_utc()}")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_json(OUTPUT_DIR / "topk_manifest.json")
    val_rows = load_csv(OUTPUT_DIR / "validation_topk_results.csv")
    eval_rows = load_csv(OUTPUT_DIR / "final_topk_results.csv")
    policy_rows = load_csv(OUTPUT_DIR / "selected_topk_policy.csv")
    p65_rows = load_csv(OUTPUT_DIR / "topk_65_candidates.csv")
    exploratory_rows = load_csv(OUTPUT_DIR / "exploratory_topk_candidates.csv")

    all_issues = []
    all_issues.extend(audit_manifest(manifest))
    all_issues.extend(audit_validation_results(val_rows))
    all_issues.extend(audit_final_results(eval_rows, val_rows))
    all_issues.extend(audit_cross_consistency(val_rows, eval_rows))
    all_issues.extend(audit_selected_policy(policy_rows, val_rows))

    # Summary
    print(f"\nManifest: {manifest.get('total_experiments', 0)} experiments")
    print(f"Validation rows: {len(val_rows)}")
    print(f"Final eval rows: {len(eval_rows)}")
    print(f"Selected policies: {len(policy_rows)}")
    print(f"Ranking65 candidates: {len(p65_rows)}")
    print(f"Exploratory candidates: {len(exploratory_rows)}")
    print(f"\nAudit issues: {len(all_issues)}")
    for i, issue in enumerate(all_issues):
        print(f"  {i+1}. {issue}")

    # Write audit report
    audit_report = {
        "audit_timestamp": now_utc(),
        "manifest_issues": audit_manifest(manifest),
        "validation_issues": audit_validation_results(val_rows),
        "final_issues": audit_final_results(eval_rows, val_rows),
        "consistency_issues": audit_cross_consistency(val_rows, eval_rows),
        "policy_issues": audit_selected_policy(policy_rows, val_rows),
        "total_issues": len(all_issues),
        "passed": len(all_issues) == 0,
        "summary": {
            "total_experiments": manifest.get("total_experiments", 0),
            "validation_rows": len(val_rows),
            "final_rows": len(eval_rows),
            "selected_policies": len(policy_rows),
            "ranking65_candidates": len(p65_rows),
            "best_precision_at_k": manifest.get("best_precision_at_k", 0),
            "best_hit_rate_at_k": manifest.get("best_hit_rate_at_k", 0),
            "ranking65_pass": manifest.get("ranking65_pass", False),
        }
    }

    audit_path = AUDIT_DIR / "vn30_hourly_2015_topk_ranking_audit.json"
    with audit_path.open("w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print(f"\nAudit report: {rel(audit_path)}")
    print(f"Audit passed: {'YES' if len(all_issues) == 0 else 'NO'}")
    return 0 if len(all_issues) == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
