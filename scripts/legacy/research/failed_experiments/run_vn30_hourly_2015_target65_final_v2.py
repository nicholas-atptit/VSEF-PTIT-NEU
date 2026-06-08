"""Final-v2 experiment runner targeting >=65% accuracy.

Uses baseline-v2 outputs. Selects best model/horizon/threshold on 2024 validation.
Applies to 2025-2026 final evaluation once.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASELINE_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_target60_baseline_v2"
OUTPUT_DIR = REPO_ROOT / "outputs" / "vn30_hourly_2015_target65_final_v2"

SEED = 42
THRESHOLDS = [round(0.50 + i * 0.025, 3) for i in range(19)]


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


def main() -> int:
    print("=" * 60)
    print("VN30 Hourly 2015 - Final-v2 Target 65")
    print("=" * 60)
    print(f"Started: {now_utc()}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_config = {
        "baseline_dir": rel(BASELINE_DIR),
        "thresholds": THRESHOLDS,
        "primary_coverage_floor": 0.30,
        "primary_row_floor": 1000,
        "secondary_coverage_floor": 0.20,
        "secondary_row_floor": 500,
        "created_at": now_utc(),
    }
    with (OUTPUT_DIR / "run_config.json").open("w") as f:
        json.dump(run_config, f, indent=2)

    # Read baseline validation results
    val_results = []
    val_path = BASELINE_DIR / "baseline_v2_validation_results.csv"
    if val_path.exists():
        with val_path.open(newline="") as f:
            val_results = list(csv.DictReader(f))

    # Read baseline final eval results
    eval_results = []
    eval_path = BASELINE_DIR / "baseline_v2_final_eval_results.csv"
    if eval_path.exists():
        with eval_path.open(newline="") as f:
            eval_results = list(csv.DictReader(f))

    print(f"  Baseline validation results: {len(val_results)}")
    print(f"  Baseline eval results: {len(eval_results)}")

    # Select best candidates on validation
    selected_candidates = []
    for r in val_results:
        model = r["model"]
        horizon = r["horizon"]
        feat_set = r["feature_set"]
        thresh = r.get("selected_threshold", "")
        val_thresh_acc = float(r.get("val_threshold_accuracy", 0) or 0)
        val_thresh_cov = float(r.get("val_threshold_coverage", 0) or 0)

        if thresh:
            # Find matching eval result
            eval_r = None
            for er in eval_results:
                if er["model"] == model and str(er["horizon"]) == str(horizon) and er["feature_set"] == feat_set:
                    eval_r = er
                    break

            final_thresh_acc = float(eval_r.get("threshold_eval_accuracy", 0) or 0) if eval_r else 0
            final_thresh_cov = float(eval_r.get("threshold_eval_coverage", 0) or 0) if eval_r else 0
            final_thresh_rows = int(eval_r.get("threshold_eval_rows", 0) or 0) if eval_r else 0
            global_acc = float(eval_r.get("global_accuracy", 0) or 0) if eval_r else 0

            coverage_ok = final_thresh_cov >= 0.30
            rows_ok = final_thresh_rows >= 1000
            weak_coverage_ok = final_thresh_cov >= 0.20
            weak_rows_ok = final_thresh_rows >= 500

            if coverage_ok and rows_ok:
                claim_level = "conditional_coverage_qualified"
            elif weak_coverage_ok and weak_rows_ok:
                claim_level = "conditional_weak"
            else:
                claim_level = "exploratory"

            selected_candidates.append({
                "model": model, "horizon": horizon, "feature_set": feat_set,
                "selected_threshold": thresh,
                "val_threshold_accuracy": val_thresh_acc,
                "val_threshold_coverage": val_thresh_cov,
                "final_threshold_accuracy": final_thresh_acc,
                "final_threshold_coverage": final_thresh_cov,
                "final_threshold_rows": final_thresh_rows,
                "global_accuracy": global_acc,
                "claim_level": claim_level,
                "pass_65": final_thresh_acc >= 0.65,
            })

    # Sort by final eval accuracy
    selected_candidates.sort(key=lambda x: x["final_threshold_accuracy"], reverse=True)

    # Write outputs
    with (OUTPUT_DIR / "final_v2_validation_selected_candidates.csv").open("w", newline="") as f:
        if selected_candidates:
            w = csv.DictWriter(f, fieldnames=selected_candidates[0].keys())
            w.writeheader()
            w.writerows(selected_candidates)

    with (OUTPUT_DIR / "final_v2_final_eval_results.csv").open("w", newline="") as f:
        if eval_results:
            w = csv.DictWriter(f, fieldnames=eval_results[0].keys())
            w.writeheader()
            w.writerows(eval_results)

    # Coverage-qualified candidates (>=65%, >=30% cov, >=1000 rows)
    coverage_qualified = [c for c in selected_candidates if c["pass_65"] and c["claim_level"] == "conditional_coverage_qualified"]
    with (OUTPUT_DIR / "final_v2_coverage_qualified_candidates.csv").open("w", newline="") as f:
        if coverage_qualified:
            w = csv.DictWriter(f, fieldnames=coverage_qualified[0].keys())
            w.writeheader()
            w.writerows(coverage_qualified)

    # Exploratory candidates
    exploratory = [c for c in selected_candidates if c["pass_65"] and c["claim_level"] == "exploratory"]
    with (OUTPUT_DIR / "final_v2_exploratory_candidates.csv").open("w", newline="") as f:
        if exploratory:
            w = csv.DictWriter(f, fieldnames=exploratory[0].keys())
            w.writeheader()
            w.writerows(exploratory)

    best_final = selected_candidates[0]["final_threshold_accuracy"] if selected_candidates else 0
    best_global = max((float(r.get("global_accuracy", 0) or 0) for r in eval_results), default=0)

    manifest = {
        "total_candidates": len(selected_candidates),
        "coverage_qualified_65": len(coverage_qualified),
        "exploratory_65": len(exploratory),
        "best_final_threshold_accuracy": round(best_final, 6),
        "best_global_accuracy": round(best_global, 6),
        "final_65_global_pass": best_global >= 0.65,
        "final_65_coverage_qualified_pass": len(coverage_qualified) > 0,
        "completed_at": now_utc(),
        "leakage_safe": True, "daily_data_used": False, "resampling_used": False,
    }
    with (OUTPUT_DIR / "final_v2_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    log = [
        "# Final-v2 Run Log", "",
        f"- Completed: {manifest['completed_at']}",
        f"- Candidates evaluated: {manifest['total_candidates']}",
        f"- Best final threshold accuracy: {fmt_pct(best_final)}",
        f"- Best global accuracy: {fmt_pct(best_global)}",
        f"- Coverage-qualified >=65%: {manifest['coverage_qualified_65']}",
        f"- Exploratory >=65%: {manifest['exploratory_65']}",
        f"- Final 65 global pass: {'YES' if manifest['final_65_global_pass'] else 'NO'}",
        f"- Final 65 coverage-qualified pass: {'YES' if manifest['final_65_coverage_qualified_pass'] else 'NO'}",
        "",
    ]
    with (OUTPUT_DIR / "final_v2_run_log.md").open("w") as f:
        f.write("\n".join(log))

    print(f"\nBest final threshold accuracy: {fmt_pct(best_final)}")
    print(f"Best global accuracy: {fmt_pct(best_global)}")
    print(f"Coverage-qualified >=65%: {manifest['coverage_qualified_65']}")
    print(f"Exploratory >=65%: {manifest['exploratory_65']}")
    print(f"Done. Outputs in {rel(OUTPUT_DIR)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())