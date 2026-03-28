import argparse
import asyncio
import yaml
from pathlib import Path

from src.benchmark.simulator import HistoricalSimulator
from src.benchmark.evaluator import MetricsEvaluator
from src.benchmark.report_generator import ReportGenerator

def load_config(path: str = "config/benchmark/default.yaml"):
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

async def main_async():
    parser = argparse.ArgumentParser(description="Multi-Agent Benchmark Harness (CLI).")
    parser.add_argument("--profile", type=str, default="local_ollama", help="Config Profile (vd: local_ollama).")
    parser.add_argument("--regime", type=str, default="uptrend", help="Market regime (uptrend/downtrend/all).")
    parser.add_argument("--horizon", type=str, default="swing", help="Trade horizon.")
    args = parser.parse_args()
    
    # 1. Khởi tạo cấu hình và System
    cfg = load_config()
    print(f"=== KHỞI ĐỘNG BENCHMARK ===")
    print(f"Profile: {args.profile} | Regime: {args.regime} | Horizon: {args.horizon}")
    
    # 2. Lọc cấu hình Time Regime
    regime_cfg = cfg["market_regimes"].get(args.regime)
    if not regime_cfg:
        print(f"[ERROR] Không tìm thấy Regime: {args.regime}")
        return
        
    start_date = regime_cfg["start_date"]
    end_date = regime_cfg["end_date"]
    
    tickers_to_test = cfg["universes"]["VN30"]["tickers"][:3] # Test 3 mã 
    print(f"Lịch chạy: {start_date} tới {end_date} (Test mã: {tickers_to_test})")
    
    # Ở đây chúng ta khóa delay 2 giây để giữ sức cho Ollama
    sim = HistoricalSimulator(use_llm_provider='ollama', batch_delay_sec=2.0)
    
    # 3. Simulate The Agents Time-Travel
    # Cần phải pass kwargs start_date và end_date
    decision_cards = await sim.run_simulation(
        tickers=tickers_to_test, 
        start_date=start_date, 
        end_date=end_date, 
        horizon=args.horizon
    )
    
    # 4. Tính toán Metrics (Math + LLM Performance)
    evaluator = MetricsEvaluator()
    fin_metrics = evaluator.evaluate_financial_metrics(decision_cards)
    model_metrics = evaluator.evaluate_model_quality(decision_cards)
    
    # 5. Sinh Báo Cáo
    reporter = ReportGenerator()
    config_state = {
        "profile": args.profile,
        "model": "qwen2.5:7b",
        "regime": args.regime,
        "horizon": args.horizon
    }
    
    md_file_path = reporter.generate(config_state, fin_metrics, model_metrics)
    print(f"\n[DONE] Báo cáo đã sinh ra tại: {md_file_path}")


if __name__ == "__main__":
    # Điểm entrypoint chuẩn
    asyncio.run(main_async())
