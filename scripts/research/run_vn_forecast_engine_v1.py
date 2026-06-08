"""Run the local-data-only VN Forecast Engine v1."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.forecasting.engine import EngineConfig, OfflineVNForecastEngine  # noqa: E402


def parse_csv_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("horizons must be positive comma-separated integers")
    return values


def parse_csv_strings(value: str) -> tuple[str, ...]:
    return tuple(item.strip().upper() for item in value.split(",") if item.strip())


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run VN Forecast Engine v1 from local historical/cache data only.")
    result.add_argument("--offline-historical-only", action="store_true", required=True)
    mode = result.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full-run", action="store_true")
    mode.add_argument("--forecast-latest", action="store_true")
    mode.add_argument("--forecast-asof")
    mode.add_argument("--build-evaluate", action="store_true")
    result.add_argument("--frequency", choices=("hourly",), default="hourly")
    result.add_argument("--horizons", type=parse_csv_ints, default=(5, 10, 20, 40, 60))
    result.add_argument("--index-codes", type=parse_csv_strings, default=("VNINDEX", "VN30", "HNXINDEX", "HNX30", "UPCOMINDEX", "VNXALL"))
    result.add_argument("--timeout-seconds", type=int, default=14400)
    result.add_argument("--enable-qml-features", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    asof = pd.to_datetime(args.forecast_asof, errors="raise") if args.forecast_asof else None
    config = EngineConfig(args.frequency, args.horizons, args.index_codes, args.timeout_seconds, args.enable_qml_features, asof)
    engine = OfflineVNForecastEngine(config)
    latest = engine.run(forecast=not args.build_evaluate, asof=asof)
    print(json.dumps({"status": "ok", "mode": "build_evaluate" if args.build_evaluate else "forecast", "forecast_rows": len(latest), "output_dir": "reports/generated/vn_forecast_engine_v1", "claim_label": "offline_diagnostic_forecast_only"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
