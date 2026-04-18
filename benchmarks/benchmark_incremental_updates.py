"""Local incremental-update benchmarks for Phase 4 artifact handling."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT
from src.ml.data_loader import (
    build_market_breadth_from_csv,
    build_sector_proxies_from_csv,
    clear_artifact_frame_cache,
    load_foreign_flow,
    load_macro_context,
    load_ticker_sectors,
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


class IncrementalUpdateBenchmark:
    """Benchmark derived-artifact rebuilds and cache hit behavior."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or PROJECT_ROOT / "artifacts" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.daily_csv_dir = PROJECT_ROOT / "data" / "daily_market_split_data"
        self.results: dict[str, Any] = {}

    @staticmethod
    def _time_call(callback: Any) -> tuple[Any, float]:
        start_ns = time.perf_counter_ns()
        result = callback()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        return result, elapsed_ms

    def _benchmark_loader(self, name: str, loader: Any, **kwargs: Any) -> dict[str, Any]:
        clear_artifact_frame_cache()
        miss_df, miss_ms = self._time_call(lambda: loader(**kwargs))
        hit_df, hit_ms = self._time_call(lambda: loader(**kwargs))
        result = {
            "name": name,
            "miss_ms": round(miss_ms, 3),
            "hit_ms": round(hit_ms, 3),
            "speedup": round(miss_ms / hit_ms, 3) if hit_ms > 0 else None,
            "rows": int(len(hit_df)),
            "cache_status": hit_df.attrs.get("artifact_cache_status"),
            "source_name": hit_df.attrs.get("source_name"),
            "source_provenance": hit_df.attrs.get("source_provenance"),
        }
        if hit_df.attrs.get("source_provenance") == "stub_todo":
            result["note"] = str(hit_df.attrs.get("source_notes") or "explicit_stub")
        return result

    def _benchmark_local_builder(self, name: str, builder: Any, *, output_path: Path, **kwargs: Any) -> dict[str, Any]:
        if not self.daily_csv_dir.exists():
            return {
                "name": name,
                "status": "missing_daily_csv_dir",
                "csv_dir": str(self.daily_csv_dir),
            }

        if output_path.exists():
            output_path.unlink()

        clear_artifact_frame_cache()
        full_result, full_ms = self._time_call(
            lambda: builder(
                csv_dir=self.daily_csv_dir,
                output_path=output_path,
                incremental_update=False,
                **kwargs,
            )
        )

        clear_artifact_frame_cache()
        incremental_result, incremental_ms = self._time_call(
            lambda: builder(
                csv_dir=self.daily_csv_dir,
                output_path=output_path,
                incremental_update=True,
                **kwargs,
            )
        )

        return {
            "name": name,
            "full_rebuild_ms": round(full_ms, 3),
            "incremental_rebuild_ms": round(incremental_ms, 3),
            "speedup": round(full_ms / incremental_ms, 3) if incremental_ms > 0 else None,
            "full_rows": int(len(full_result)),
            "incremental_rows": int(len(incremental_result)),
            "artifact_path": str(output_path),
        }

    def benchmark_market_breadth(self) -> dict[str, Any]:
        return self._benchmark_local_builder(
            "market_breadth",
            build_market_breadth_from_csv,
            output_path=self.output_dir / "tmp_market_breadth_incremental.csv",
        )

    def benchmark_sector_proxies(self) -> dict[str, Any]:
        ticker_sectors = load_ticker_sectors()
        return self._benchmark_local_builder(
            "sector_proxies",
            build_sector_proxies_from_csv,
            output_path=self.output_dir / "tmp_sector_proxies_incremental.csv",
            ticker_sectors=ticker_sectors,
        )

    def run_all(self) -> None:
        print("\n" + "=" * 80)
        print("PHASE 4 INCREMENTAL UPDATE BENCHMARKS")
        print("=" * 80 + "\n")

        print("1. Derived local artifact rebuilds:")
        breadth = self.benchmark_market_breadth()
        sector = self.benchmark_sector_proxies()
        self.results["derived_rebuilds"] = {
            "market_breadth": breadth,
            "sector_proxies": sector,
        }
        for stats in (breadth, sector):
            if "status" in stats:
                print(f"  {stats['name']:14s}: {stats['status']}")
            else:
                print(
                    f"  {stats['name']:14s}: full={stats['full_rebuild_ms']:.2f} ms, "
                    f"incremental={stats['incremental_rebuild_ms']:.2f} ms, "
                    f"speedup={stats['speedup']:.2f}x"
                )

        print("\n2. Loader miss/hit behavior:")
        loaders = {
            "macro_context": self._benchmark_loader("macro_context", load_macro_context),
            "foreign_flow": self._benchmark_loader("foreign_flow", load_foreign_flow),
        }
        self.results["loader_cache"] = loaders
        for name, stats in loaders.items():
            speedup = stats.get("speedup")
            speedup_label = f"{speedup:.2f}x" if isinstance(speedup, (int, float)) else "n/a"
            note = f" | note={stats['note']}" if "note" in stats else ""
            print(
                f"  {name:14s}: miss={stats['miss_ms']:.2f} ms, hit={stats['hit_ms']:.2f} ms, "
                f"speedup={speedup_label}, cache={stats['cache_status']}{note}"
            )

        self._save_results()
        print("\n" + "=" * 80)
        print(f"Results saved to: {self.output_dir / 'incremental_benchmark_results.json'}")
        print("=" * 80 + "\n")

    def _save_results(self) -> None:
        json_path = self.output_dir / "incremental_benchmark_results.json"
        json_path.write_text(json.dumps(_json_ready(self.results), indent=2), encoding="utf-8")

        report_path = self.output_dir / "incremental_benchmark_report.txt"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("PHASE 4 INCREMENTAL UPDATE BENCHMARK REPORT\n")
            handle.write("=" * 80 + "\n\n")

            handle.write("DERIVED LOCAL ARTIFACT REBUILDS\n")
            for name, stats in self.results.get("derived_rebuilds", {}).items():
                if "status" in stats:
                    handle.write(f"- {name}: {stats['status']}\n")
                else:
                    handle.write(
                        f"- {name}: full={stats['full_rebuild_ms']:.2f} ms, "
                        f"incremental={stats['incremental_rebuild_ms']:.2f} ms, "
                        f"speedup={stats['speedup']:.2f}x\n"
                    )
            handle.write("\nLOADER MISS/HIT BEHAVIOR\n")
            for name, stats in self.results.get("loader_cache", {}).items():
                handle.write(
                    f"- {name}: miss={stats['miss_ms']:.2f} ms, "
                    f"hit={stats['hit_ms']:.2f} ms, "
                    f"speedup={stats.get('speedup')}, "
                    f"cache={stats.get('cache_status')}\n"
                )
                if "note" in stats:
                    handle.write(f"  note: {stats['note']}\n")


if __name__ == "__main__":
    IncrementalUpdateBenchmark().run_all()
