from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

from config.settings import get_settings
from src.agents.orchestrator import AgentOrchestrator
from src.signals.builder import build_market_signal
from src.ml.data_loader import VN100DataLoader
from src.ml.feature_engineering import FeatureEngineer
from src.ml.trainer import DualModelTrainer
from src.ml.backtest.event_driven import simulate_execution_cost
from src.ml.backtest.metrics import calculate_risk_adjusted_returns
from src.utils.logging import get_logger

logger = get_logger(__name__)


class PortfolioBacktester:
    """Simulates a multi-agent portfolio strategy over historical data.
    
    This engine pushes historical OHLCV through the AgentOrchestrator day-by-day,
    simulating realistic execution costs and portfolio scaling.
    """

    def __init__(
        self,
        tickers: list[str],
        initial_capital: float = 1_000_000_000.0, # 1 Billion VND
        start_date: dt.date | None = None,
        end_date: dt.date | None = None,
    ) -> None:
        self.settings = get_settings()
        self.tickers = [t.upper() for t in tickers]
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.start_date = start_date or dt.date(2023, 1, 1)
        self.end_date = end_date or dt.date.today()
        
        self.data_loader = VN100DataLoader(prefer_source="csv")
        self.fe = FeatureEngineer()
        self.trainer = DualModelTrainer()
        self.orchestrator = AgentOrchestrator()
        
        self.portfolio_history = []
        self.positions = {} # { ticker: {volume, entry_price} }
        self.daily_returns = []

    async def run(self) -> dict:
        """Execute the backtest loop."""
        logger.info("starting_portfolio_backtest", tickers=self.tickers, start=self.start_date)
        
        # 1. Load data for all tickers
        dataset = self.data_loader.build_dataset(
            tickers=self.tickers,
            start_date=self.start_date - dt.timedelta(days=400), # Need buffer for features
            end_date=self.end_date,
            join_market=True,
            join_sectors=True
        )
        
        if dataset.empty:
            raise ValueError("No data found for backtest tickers.")

        # 2. Pre-compute features (Vectorized where possible)
        all_features = []
        for ticker in self.tickers:
            ticker_df = dataset[dataset["ticker"] == ticker]
            if ticker_df.empty: continue
            feat_df = self.fe.transform(ticker_df, drop_na=False)
            all_features.append(feat_df)
        
        full_df = pd.concat(all_features).sort_values("date")
        
        # 3. Filter to backtest range
        backtest_df = full_df[full_df["date"] >= pd.Timestamp(self.start_date)]
        unique_dates = sorted(backtest_df["date"].unique())
        
        prev_equity = self.initial_capital

        # 4. Day-by-Day Simulation
        for current_date in tqdm(unique_dates, desc="Backtesting"):
            day_data = backtest_df[backtest_df["date"] == current_date]
            market_signals = []
            
            # Step A: Aggregate signals for the day
            for _, row in day_data.iterrows():
                ticker = row["ticker"]
                price = row["close"]
                
                try:
                    # Generate ML prediction (Trend/Range)
                    model_output = self.trainer.predict(ticker, row)
                    if not model_output: continue
                    
                    # Wrap in MarketSignal contract
                    signal = build_market_signal(
                        ticker=ticker,
                        current_price=price,
                        model_output=model_output,
                        feature_snapshot=row.to_dict(),
                        sentiment_payload={"sentiment_score": 0.05} # Static for now
                    )
                    market_signals.append(signal)
                except Exception as e:
                    continue

            # Step B: Run Multi-Agent Orchestrator
            if market_signals:
                decision_output = await self.orchestrator.run(market_signals)
                portfolio_proposal = decision_output["portfolio"]
                
                # Step C: Execute target weights
                # For this simulation, we rebalance to target weights each day
                self._rebalance(current_date, day_data, portfolio_proposal["positions"])
            
            # Step D: Track Daily Equity
            current_equity = self._calculate_equity(day_data)
            daily_ret = (current_equity / prev_equity) - 1
            self.daily_returns.append(daily_ret)
            
            self.portfolio_history.append({
                "date": current_date.isoformat(),
                "equity": current_equity,
                "cash": self.cash,
                "drawdown": 0.0 # Will calculate later
            })
            prev_equity = current_equity

        # 5. Final Report
        metrics = calculate_risk_adjusted_returns(self.daily_returns)
        return {
            "metrics": metrics,
            "history": self.portfolio_history
        }

    def _rebalance(self, date: pd.Timestamp, day_data: pd.DataFrame, target_positions: list[dict]):
        """Simulate rebalancing to target weights."""
        # 1. Sell current positions not in targets
        target_tickers = {p["ticker"] for p in target_positions}
        for ticker in list(self.positions.keys()):
            if ticker not in target_tickers:
                self._execute_trade(ticker, 0, day_data, "SELL")

        # 2. Adjust to target weights
        total_equity = self._calculate_equity(day_data)
        for target in target_positions:
            ticker = target["ticker"]
            weight = target["weight"]
            
            ticker_row = day_data[day_data["ticker"] == ticker]
            if ticker_row.empty: continue
            price = ticker_row["close"].values[0]
            
            target_value = total_equity * weight
            target_volume = int(target_value / price)
            
            current_volume = self.positions.get(ticker, {}).get("volume", 0)
            diff = target_volume - current_volume
            
            if diff > 0:
                self._execute_trade(ticker, diff, day_data, "BUY")
            elif diff < 0:
                self._execute_trade(ticker, abs(diff), day_data, "SELL")

    def _execute_trade(self, ticker: str, volume: int, day_data: pd.DataFrame, action: str):
        ticker_row = day_data[day_data["ticker"] == ticker]
        if ticker_row.empty: return
        price = ticker_row["close"].values[0]
        
        # Simulate slippage/fees
        adj_price, fees, _ = simulate_execution_cost(price, volume, action)
        
        if action == "BUY":
            cost = adj_price * volume + fees
            if cost <= self.cash:
                self.cash -= cost
                curr = self.positions.get(ticker, {"volume": 0, "avg_price": 0.0})
                new_vol = curr["volume"] + volume
                # Weighted average price
                new_avg = ((curr["avg_price"] * curr["volume"]) + (adj_price * volume)) / new_vol if new_vol > 0 else 0
                self.positions[ticker] = {"volume": new_vol, "avg_price": new_avg}
        else: # SELL
            proceeds = adj_price * volume - fees
            self.cash += proceeds
            curr = self.positions.get(ticker, {"volume": 0, "avg_price": 0.0})
            new_vol = max(0, curr["volume"] - volume)
            if new_vol == 0:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker]["volume"] = new_vol

    def _calculate_equity(self, day_data: pd.DataFrame) -> float:
        market_value = 0.0
        for ticker, pos in self.positions.items():
            ticker_row = day_data[day_data["ticker"] == ticker]
            price = ticker_row["close"].values[0] if not ticker_row.empty else pos["avg_price"]
            market_value += price * pos["volume"]
        return self.cash + market_value


async def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Portfolio Backtester")
    parser.add_argument("--tickers", type=str, default="FPT,HPG,VNM,SSI,TCB", help="Comma-separated tickers")
    parser.add_argument("--start", type=str, default="2023-01-01", help="YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=1_000_000_000, help="Initial capital in VND")
    args = parser.parse_args()
    
    tickers = args.tickers.split(",")
    backtester = PortfolioBacktester(
        tickers=tickers,
        initial_capital=args.capital,
        start_date=dt.datetime.strptime(args.start, "%Y-%m-%d").date()
    )
    
    results = await backtester.run()
    
    print("\n" + "="*40)
    print("      PORTFOLIO BACKTEST RESULTS")
    print("="*40)
    print(f"Tickers:      {args.tickers}")
    print(f"Start Date:   {args.start}")
    print(f"End Date:     {dt.date.today()}")
    print("-"*40)
    for k, v in results["metrics"].items():
        if k != "error":
            print(f"{k:<25}: {v}")
    print("="*40)
    
    # Save history to JSON
    output_path = Path("reports/backtest_portfolio_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Full history saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
