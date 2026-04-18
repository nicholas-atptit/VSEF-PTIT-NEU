"""Phase 4 performance benchmarks for the feature and context pipeline."""

from __future__ import annotations

import datetime as dt
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.settings import PROJECT_ROOT
from src.ml.data_loader import (
    clear_artifact_frame_cache,
    load_foreign_flow,
    load_macro_context,
    load_market_breadth,
    load_sector_proxies,
)
from src.ml.feature_engineering import FEATURE_BUILD_MODE_CONFIGS, FeatureEngineer
from src.ml.trainer import DualModelTrainer


def generate_synthetic_ohlcv(ticker: str, rows: int = 500) -> pd.DataFrame:
    """Generate deterministic synthetic OHLCV for repeatable benchmarks."""
    seed = abs(hash((ticker.upper(), rows))) % (2**32)
    rng = np.random.default_rng(seed)
    end = pd.Timestamp(dt.date.today()).normalize()
    if end.dayofweek >= 5:
        end = end - pd.tseries.offsets.BDay(1)
    dates = pd.bdate_range(end=end, periods=rows)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, rows)))
    open_price = np.roll(close, 1)
    open_price[0] = close[0]
    high = np.maximum(open_price, close) * (1.0 + np.abs(rng.normal(0.0, 0.01, rows)))
    low = np.minimum(open_price, close) * (1.0 - np.abs(rng.normal(0.0, 0.01, rows)))
    volume = rng.lognormal(mean=10.0, sigma=1.0, size=rows).astype(int)
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": np.repeat(ticker.upper(), rows),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def enrich_with_context(df: pd.DataFrame) -> pd.DataFrame:
    """Attach deterministic context columns so the benchmark hits the main path."""
    frame = df.copy()
    pct_return = pd.to_numeric(frame["close"], errors="coerce").pct_change().fillna(0.0)
    turnover = pd.to_numeric(frame["close"], errors="coerce") * pd.to_numeric(frame["volume"], errors="coerce")
    angles = np.linspace(0.0, 12.0, len(frame))
    breadth_cycle = np.sin(angles)

    frame["m_ret"] = pct_return * 0.75
    frame["s_ret"] = pct_return * 0.55
    frame["sector_dispersion"] = 0.015 + (0.01 * np.abs(np.cos(angles / 2.0)))
    frame["sector_member_count"] = 18
    frame["market_breadth"] = breadth_cycle * 0.35
    frame["advancing_share"] = 0.5 + (breadth_cycle * 0.12)
    frame["declining_share"] = 1.0 - frame["advancing_share"]
    frame["advance_decline_ratio"] = 1.0 + (breadth_cycle * 0.4)
    frame["new_highs_252"] = np.clip(np.round(8.0 + (6.0 * np.sin(angles / 3.0))), 0, None)
    frame["new_lows_252"] = np.clip(np.round(5.0 + (6.0 * np.cos(angles / 3.0))), 0, None)
    frame["new_high_low_spread"] = (
        pd.to_numeric(frame["new_highs_252"], errors="coerce")
        - pd.to_numeric(frame["new_lows_252"], errors="coerce")
    ) / 25.0
    frame["up_volume"] = np.where(pct_return > 0.0, frame["volume"], frame["volume"] * 0.2)
    frame["down_volume"] = np.where(pct_return < 0.0, frame["volume"], frame["volume"] * 0.2)
    frame["foreign_net_value"] = turnover * (0.03 * np.sin(angles / 4.0))
    frame["foreign_net_volume"] = pd.to_numeric(frame["volume"], errors="coerce") * (0.05 * np.cos(angles / 5.0))
    frame["fx_usdvnd"] = 24_000.0 + np.linspace(0.0, 400.0, len(frame))
    frame["interest_rate"] = 4.0 + np.linspace(0.0, 0.35, len(frame))
    frame["gold_price"] = 2_000.0 + np.linspace(0.0, 125.0, len(frame))
    frame["oil_price"] = 80.0 + np.linspace(0.0, 18.0, len(frame))
    return frame


def synthetic_context_sources(df: pd.DataFrame, ticker: str) -> dict[str, pd.DataFrame]:
    """Build context sources that mimic the trainer's main-path joins."""
    enriched = enrich_with_context(df)
    return {
        "market_df": enriched[["date", "m_ret"]].copy(),
        "sector_df": enriched[
            ["date", "s_ret", "sector_dispersion", "sector_member_count"]
        ].rename(columns={"s_ret": "ret"}).assign(industry="SyntheticSector"),
        "ticker_sectors": pd.DataFrame({"ticker": [ticker.upper()], "industry": ["SyntheticSector"]}),
        "breadth_df": enriched[
            [
                "date",
                "market_breadth",
                "advancing_share",
                "declining_share",
                "advance_decline_ratio",
                "new_highs_252",
                "new_lows_252",
                "new_high_low_spread",
                "up_volume",
                "down_volume",
            ]
        ].copy(),
        "foreign_flow_df": enriched[
            ["date", "ticker", "foreign_net_value", "foreign_net_volume"]
        ].copy(),
        "macro_df": enriched[["date", "fx_usdvnd", "interest_rate", "gold_price", "oil_price"]].copy(),
        "sentiment_df": pd.DataFrame(),
    }


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


class FeaturePipelineBenchmark:
    """Benchmark suite for the Phase 4 feature/data pipeline."""

    def __init__(self, output_dir: Path | None = None):
        self.output_dir = output_dir or PROJECT_ROOT / "artifacts" / "benchmarks"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: dict[str, Any] = {}

    @staticmethod
    def _run_with_memory(callback: Any) -> tuple[Any, float, float]:
        tracemalloc.start()
        start_ns = time.perf_counter_ns()
        result = callback()
        elapsed_ms = (time.perf_counter_ns() - start_ns) / 1e6
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return result, elapsed_ms, peak_bytes / (1024 * 1024)

    @staticmethod
    def _loader_result(name: str, loader: Any, **kwargs: Any) -> dict[str, Any]:
        clear_artifact_frame_cache()
        miss_df, miss_ms, _ = FeaturePipelineBenchmark._run_with_memory(lambda: loader(**kwargs))
        hit_df, hit_ms, _ = FeaturePipelineBenchmark._run_with_memory(lambda: loader(**kwargs))
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

    def benchmark_build_mode(
        self,
        build_mode: str,
        ticker: str = "TEST",
        rows: int = 500,
        n_runs: int = 3,
    ) -> dict[str, Any]:
        clear_artifact_frame_cache()
        df = enrich_with_context(generate_synthetic_ohlcv(ticker, rows))
        feature_engineer = FeatureEngineer()

        try:
            feature_engineer.build_feature_frame(df.copy(), build_mode=build_mode)
        except Exception as exc:
            return {"build_mode": build_mode, "error": str(exc)}

        times_ms: list[float] = []
        peak_memory_mb: list[float] = []
        result_df = pd.DataFrame()
        for run_idx in range(n_runs):
            try:
                result_df, elapsed_ms, peak_mb = self._run_with_memory(
                    lambda: feature_engineer.build_feature_frame(df.copy(), build_mode=build_mode)
                )
            except Exception as exc:
                return {"build_mode": build_mode, "run": run_idx, "error": str(exc)}
            times_ms.append(elapsed_ms)
            peak_memory_mb.append(peak_mb)

        return {
            "build_mode": build_mode,
            "ticker": ticker,
            "rows": rows,
            "n_runs": n_runs,
            "time_ms_mean": round(float(np.mean(times_ms)), 3),
            "time_ms_std": round(float(np.std(times_ms)), 3),
            "time_ms_min": round(float(np.min(times_ms)), 3),
            "time_ms_max": round(float(np.max(times_ms)), 3),
            "time_per_row_us": round(float(np.mean(times_ms) * 1000.0 / rows), 3),
            "rows_per_sec": round(float(rows / (np.mean(times_ms) / 1000.0)), 3),
            "memory_peak_mb_mean": round(float(np.mean(peak_memory_mb)), 3),
            "memory_peak_mb_max": round(float(np.max(peak_memory_mb)), 3),
            "stage_timings": dict(result_df.attrs.get("feature_build_stages", {})),
            "output_cols": int(len(result_df.columns)),
            "output_rows": int(len(result_df)),
        }

    def benchmark_all_modes(self, rows: int = 500) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for mode_name in FEATURE_BUILD_MODE_CONFIGS:
            result = self.benchmark_build_mode(mode_name, rows=rows)
            results[mode_name] = result
            if "error" in result:
                print(f"  {mode_name:20s}: ERROR - {result['error']}")
            else:
                print(
                    f"  {mode_name:20s}: {result['time_ms_mean']:7.2f} ms/ticker, "
                    f"{result['memory_peak_mb_mean']:.2f} MiB peak"
                )
        return results

    def benchmark_scaling(self, tickers: list[str], rows: int = 500) -> dict[str, Any]:
        feature_engineer = FeatureEngineer()
        clear_artifact_frame_cache()
        elapsed_times: list[float] = []
        total_rows = 0
        for ticker in tickers:
            df = enrich_with_context(generate_synthetic_ohlcv(ticker, rows))
            _, elapsed_ms, _ = self._run_with_memory(
                lambda current_df=df: feature_engineer.build_feature_frame(
                    current_df.copy(),
                    build_mode="regime_risk_mode",
                )
            )
            elapsed_times.append(elapsed_ms)
            total_rows += len(df)
        total_ms = float(sum(elapsed_times))
        return {
            "n_tickers": len(tickers),
            "rows_per_ticker": rows,
            "total_rows": total_rows,
            "total_time_ms": round(total_ms, 3),
            "time_per_ticker_ms": round(total_ms / len(tickers), 3),
            "time_per_row_us": round((total_ms * 1000.0) / total_rows, 3),
            "rows_per_sec": round(total_rows / (total_ms / 1000.0), 3),
        }

    def benchmark_prepare_ticker(self, rows: int = 1250, n_runs: int = 3) -> dict[str, Any]:
        ticker = "PREP"
        trainer = DualModelTrainer(model_dir=self.output_dir / "tmp_models")
        df = generate_synthetic_ohlcv(ticker, rows)
        context_sources = synthetic_context_sources(df, ticker)

        try:
            trainer.prepare_ticker_data(
                ticker=ticker,
                df=df,
                max_sequence_length=20,
                feature_build_mode="full_research_mode",
                context_sources=context_sources,
            )
        except Exception as exc:
            return {"ticker": ticker, "error": str(exc)}

        elapsed_times: list[float] = []
        peak_memory_mb: list[float] = []
        prepared = None
        for _ in range(n_runs):
            prepared, elapsed_ms, peak_mb = self._run_with_memory(
                lambda: trainer.prepare_ticker_data(
                    ticker=ticker,
                    df=df,
                    max_sequence_length=20,
                    feature_build_mode="full_research_mode",
                    context_sources=context_sources,
                )
            )
            elapsed_times.append(elapsed_ms)
            peak_memory_mb.append(peak_mb)

        assert prepared is not None
        return {
            "ticker": ticker,
            "rows": rows,
            "n_runs": n_runs,
            "time_ms_mean": round(float(np.mean(elapsed_times)), 3),
            "time_ms_std": round(float(np.std(elapsed_times)), 3),
            "memory_peak_mb_mean": round(float(np.mean(peak_memory_mb)), 3),
            "feature_rows": int(len(prepared.feature_frame)),
            "feature_columns": int(len(prepared.feature_columns)),
            "indicator_warmup_rows": int(prepared.raw_stats.get("indicator_warmup_rows", 0)),
            "feature_build_mode": prepared.feature_build_mode,
        }

    def benchmark_cache_impact(self) -> dict[str, dict[str, Any]]:
        return {
            "sector_proxies": self._loader_result("sector_proxies", load_sector_proxies),
            "market_breadth": self._loader_result("market_breadth", load_market_breadth),
            "macro_context": self._loader_result("macro_context", load_macro_context),
            "foreign_flow": self._loader_result("foreign_flow", load_foreign_flow),
        }

    def run_all(self) -> None:
        print("\n" + "=" * 80)
        print("PHASE 4 PERFORMANCE BENCHMARKING")
        print("=" * 80 + "\n")

        print("1. Feature generation by explicit build mode (500 business days):")
        self.results["build_modes"] = self.benchmark_all_modes(rows=500)

        print("\n2. End-to-end trainer prepare_ticker_data (1,250 business days):")
        prepared = self.benchmark_prepare_ticker(rows=1250)
        self.results["prepare_ticker"] = prepared
        if "error" in prepared:
            print(f"  ERROR - {prepared['error']}")
        else:
            print(
                f"  {prepared['time_ms_mean']:.2f} ms/ticker, "
                f"{prepared['memory_peak_mb_mean']:.2f} MiB peak, "
                f"{prepared['indicator_warmup_rows']} warmup rows"
            )

        print("\n3. Sequential per-ticker scaling (10 tickers x 500 rows, regime_risk_mode):")
        scaling = self.benchmark_scaling([f"TICK{i:03d}" for i in range(10)], rows=500)
        self.results["scaling"] = scaling
        print(
            f"  total {scaling['total_time_ms']:.2f} ms, "
            f"{scaling['time_per_ticker_ms']:.2f} ms/ticker, "
            f"{scaling['rows_per_sec']:.0f} rows/sec"
        )

        print("\n4. Artifact cache impact:")
        cache_impact = self.benchmark_cache_impact()
        self.results["cache_impact"] = cache_impact
        for name, stats in cache_impact.items():
            speedup = stats.get("speedup")
            note = f" | note={stats['note']}" if "note" in stats else ""
            speedup_label = f"{speedup:.1f}x" if isinstance(speedup, (int, float)) else "n/a"
            print(
                f"  {name:14s}: miss={stats['miss_ms']:.2f} ms, hit={stats['hit_ms']:.2f} ms, "
                f"speedup={speedup_label}, rows={stats['rows']}, cache={stats['cache_status']}{note}"
            )

        self._save_results()
        print("\n" + "=" * 80)
        print(f"Results saved to: {self.output_dir / 'benchmark_results.json'}")
        print("=" * 80 + "\n")

    def _save_results(self) -> None:
        json_path = self.output_dir / "benchmark_results.json"
        json_path.write_text(json.dumps(_json_ready(self.results), indent=2), encoding="utf-8")

        report_path = self.output_dir / "benchmark_report.txt"
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("PHASE 4 PERFORMANCE BENCHMARK REPORT\n")
            handle.write("=" * 80 + "\n\n")

            handle.write("BUILD MODES\n")
            for mode, stats in self.results.get("build_modes", {}).items():
                if "error" in stats:
                    handle.write(f"- {mode}: ERROR - {stats['error']}\n")
                    continue
                handle.write(
                    f"- {mode}: {stats['time_ms_mean']:.2f} ms/ticker, "
                    f"{stats['memory_peak_mb_mean']:.2f} MiB peak, "
                    f"{stats['output_cols']} columns\n"
                )
                for stage_name, stage_time in stats.get("stage_timings", {}).items():
                    handle.write(f"  stage {stage_name}: {stage_time}\n")
            handle.write("\n")

            prepared = self.results.get("prepare_ticker", {})
            if prepared:
                if "error" in prepared:
                    handle.write(f"PREPARE_TICKER\n- ERROR - {prepared['error']}\n\n")
                else:
                    handle.write("PREPARE_TICKER\n")
                    handle.write(
                        f"- {prepared['time_ms_mean']:.2f} ms/ticker, "
                        f"{prepared['memory_peak_mb_mean']:.2f} MiB peak, "
                        f"{prepared['indicator_warmup_rows']} warmup rows\n\n"
                    )

            scaling = self.results.get("scaling", {})
            if scaling:
                handle.write("SCALING\n")
                handle.write(
                    f"- {scaling['n_tickers']} tickers x {scaling['rows_per_ticker']} rows: "
                    f"{scaling['total_time_ms']:.2f} ms total, "
                    f"{scaling['time_per_ticker_ms']:.2f} ms/ticker\n\n"
                )

            handle.write("CACHE IMPACT\n")
            for name, stats in self.results.get("cache_impact", {}).items():
                handle.write(
                    f"- {name}: miss={stats['miss_ms']:.2f} ms, "
                    f"hit={stats['hit_ms']:.2f} ms, "
                    f"speedup={stats.get('speedup')}, "
                    f"cache={stats.get('cache_status')}, "
                    f"rows={stats.get('rows')}\n"
                )
                if "note" in stats:
                    handle.write(f"  note: {stats['note']}\n")


if __name__ == "__main__":
    FeaturePipelineBenchmark().run_all()
