"""Lightweight verification gate for the manifest-driven ML pipeline.

Run this script before merging into main to catch regressions in:
  1. Import smoke
  2. CLI --help
  3. ML unit test subset
  4. One live smoke inference (CART on SSI from local CSV)

Exit code:
  0  -- all checks passed
  1  -- one or more checks failed (see output)

Usage:
  python scripts/verify_ml_baseline.py
  python scripts/verify_ml_baseline.py --skip-inference   # skip live inference if no models exist
  python scripts/verify_ml_baseline.py --full-suite       # run the entire tests/ml tree
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

PASS = "PASS"
FAIL = "FAIL"
FOCUSED_ML_TESTS = [
    "tests/ml/test_phase1_hardening.py",
    "tests/ml/test_phase2_hardening.py",
    "tests/ml/test_phase3_hardening.py",
    "tests/ml/test_strategy_backtest.py",
    "tests/ml/test_real_data_backtest.py",
    "tests/ml/test_model_comparison.py",
    "tests/ml/test_integration.py",
    "tests/ml/test_risk.py",
]


def run(cmd: list[str], *, label: str) -> bool:
    """Run a subprocess command and return True on success."""
    print(f"\n{'='*60}")
    print(f"CHECK: {label}")
    print(f"CMD:   {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    ok = result.returncode == 0
    print(f"RESULT: {PASS if ok else FAIL} (exit={result.returncode})")
    return ok


def check_imports() -> bool:
    code = (
        "from src.ml.inference.engine import InferenceEngine; "
        "from src.ml.pipelines.inference_pipeline import InferencePipeline; "
        "from src.ml.trainer import DualModelTrainer; "
        "from src.ml.models.factory import create_model, supported_algorithms; "
        "print('Import smoke OK. Supported algorithms:', supported_algorithms())"
    )
    return run([PYTHON, "-c", code], label="Import smoke")


def check_cli_help() -> bool:
    return run(
        [PYTHON, "scripts/train_ml_tickers.py", "--help"],
        label="CLI --help",
    )


def check_ml_tests(*, full_suite: bool) -> bool:
    test_args = ["tests/ml/"] if full_suite else FOCUSED_ML_TESTS
    return run(
        [PYTHON, "-m", "pytest", *test_args, "-v", "--tb=short", "-q"],
        label="ML unit tests (focused reliability subset)" if not full_suite else "ML unit tests (tests/ml/)",
    )


def check_smoke_inference(skip: bool) -> bool:
    """Train CART on SSI then run inference."""
    smoke_model_dir = ROOT / "models" / "ci_smoke"
    smoke_report = ROOT / "reports" / "ci_smoke_cart.csv"
    daily_csv = ROOT / "data" / "daily_market_split_data" / "SSI.csv"

    if not daily_csv.exists():
        print(f"  [SKIP] {daily_csv} not found -- skipping live inference check")
        return True

    if skip:
        print("  [SKIP] --skip-inference flag set")
        return True

    # Train
    train_ok = run(
        [
            PYTHON, "scripts/train_ml_tickers.py",
            "--tickers", "SSI",
            "--algorithms", "cart",
            "--primary-algorithm", "cart",
            "--output", str(smoke_model_dir),
            "--report", str(smoke_report),
        ],
        label="Smoke retrain CART SSI",
    )
    if not train_ok:
        return False

    # Inference
    infer_code = (
        "import sys, json; from pathlib import Path; import pandas as pd; "
        "sys.path.insert(0, '.'); "
        "from src.ml.inference.engine import InferenceEngine; "
        f"engine = InferenceEngine(model_root=r'{smoke_model_dir}'); "
        "df = pd.read_csv(r'data/daily_market_split_data/SSI.csv'); "
        "result = engine.predict_ticker('SSI', df, horizon='short'); "
        "assert 'predicted_return' in result, 'missing predicted_return'; "
        "assert 'trend_probabilities' in result, 'missing trend_probabilities'; "
        "print('Inference OK:', json.dumps({k: result[k] for k in ['ticker','algorithm','horizon','predicted_return']}))"
    )
    return run([PYTHON, "-c", infer_code], label="Smoke inference SSI via InferenceEngine")


def main() -> None:
    parser = argparse.ArgumentParser(description="ML baseline verification gate")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip the live smoke inference check (use when no data CSVs available)",
    )
    parser.add_argument(
        "--full-suite",
        action="store_true",
        help="Run the entire tests/ml tree instead of the focused reliability subset",
    )
    args = parser.parse_args()

    results: dict[str, bool] = {}
    results["import_smoke"] = check_imports()
    results["cli_help"] = check_cli_help()
    results["ml_unit_tests"] = check_ml_tests(full_suite=args.full_suite)
    results["smoke_inference"] = check_smoke_inference(skip=args.skip_inference)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    all_passed = True
    for name, ok in results.items():
        status = PASS if ok else FAIL
        print(f"  {status}  {name}")
        if not ok:
            all_passed = False

    print()
    if all_passed:
        print("All checks PASSED. Baseline is healthy.")
        sys.exit(0)
    else:
        print("One or more checks FAILED. Do not merge until resolved.")
        sys.exit(1)


if __name__ == "__main__":
    main()
