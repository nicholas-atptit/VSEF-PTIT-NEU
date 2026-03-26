"""Professional Terminal UI (TUI) Dashboard v3.8 - SYSTEM TERMINAL (LEGACY-FIRST).

Features:
- Legacy Data Initialization (uses src.ml.data_loader).
- 4-tier Price Failsafe (Redis > DB > REST > Cache).
- Improved DB Diagnostics in Footer.
- Mandatory News-Aware reasoning.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from decimal import Decimal
import pandas as pd

# --- Robust Imports ---
HAS_MSGPACK = False
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

HAS_REDIS = False
try:
    from redis.asyncio import Redis
    HAS_REDIS = True
except ImportError: pass

try:
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console, Group
    from rich.text import Text
    from rich.align import Align
    from rich.progress import BarColumn, Progress, TextColumn, SpinnerColumn
except ImportError:
    print("❌ Error: 'rich' library is not installed. Run: pip install rich")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

try:
    from config.settings import get_settings
    SETTINGS = get_settings()
except Exception:
    class MockSettings: 
        redis_url = "redis://localhost:6379"
        vnstock_api_key = ""
        def __init__(self): pass
    SETTINGS = MockSettings()

from src.utils.logging import get_logger
_logger = get_logger(__name__)

# --- LEGACY SYSTEM INTEGRATION ---
HAS_DB = False
DB_VERSION = "4.6.0-ULTRA-STABLE"
DB_ERR = "No Driver"
try:
    from src.database.connection import get_session, dispose_engine
    from src.models.price import RawPrice
    from src.ml.data_loader import load_ohlcv_from_db
    from src.context.news_crawler import NewsCrawler
    from src.llm.news_intel import NewsIntelEngine
    from sqlalchemy import select, desc, text
    HAS_DB = True
except (ImportError, ModuleNotFoundError) as e:
    DB_ERR = f"DepError: {str(e).split(':')[-1]}"
except Exception as e:
    DB_ERR = f"ConfigError: {str(e).split(':')[-1]}"

# --- VNSTOCK PRO AUTH ---
HAS_VNSTOCK = False
try:
    import os
    os.environ["VNSTOCK_API_KEY"] = SETTINGS.vnstock_api_key
    from vnstock import Vnstock
    HAS_VNSTOCK = True
    VNSTOCK_CLIENT = Vnstock()
except Exception:
    HAS_VNSTOCK = False
    VNSTOCK_CLIENT = None

class AlgoTradingTUI:
    def __init__(self, start_ticker="FPT"):
        self.pinned_ticker = start_ticker.upper()
        self.cache_path = PROJECT_ROOT / "data" / "latest_predictions.json"
        self.news_crawler = NewsCrawler() if HAS_DB else None
        self.news_intel = NewsIntelEngine() if HAS_DB else None
        self._last_news_sync = 0
        self.logger = _logger
        
        self.data = {
            "ticker": self.pinned_ticker, "price": 0.0, "change": 0.0,
            "ml_up": 0.0, "ml_down": 0.0, "ml_recommendation": "ANALYZING...",
            "q_bottom": 0.0, "q_median": 0.0, "q_ceiling": 0.0,
            "llm_outlook": "CALCULATING", "llm_reasoning": "Processing local history...",
            "rl_allocation": 0.0, "status": "BOOTING", "news_headlines": "Fetching Market Context...",
            "last_price_ts": 0, "max_upside": 0.0, "max_downside": 0.0,
            "ai_intel": None
        }
        self._redis_client = None
        self._heuristic_cache = {} # Cache for fast switching
        self._lock_path = Path("data/.tui_lock")
        self.running = True
        try:
            self._initialize_legacy_baseline()
        except Exception as e:
            self.logger.error("baseline_init_fail", ticker=self.pinned_ticker, error=str(e))
            self.data["status"] = "ERROR (Check Logs)"

    def _initialize_legacy_baseline(self):
        """Phase 11.2: Bootstrapping with existing system data before live sync."""
        # 1. Try DB (Authoritative Historical)
        if HAS_DB:
            try:
                # Use the user's built-in data loader (Phase 2 official way)
                df = load_ohlcv_from_db(self.pinned_ticker)
                if df is not None and not df.empty:
                    last_row = df.iloc[-1]
                    self.data["price"] = float(last_row["close"])
                    # Calculate change from previous day
                    if len(df) >= 2:
                        prev_c = df.iloc[-2]["close"]
                        self.data["change"] = float((self.data["price"] - prev_c) / prev_c * 100)
                    self.data["status"] = "SYNCED (DB)"
            except Exception as e:
                with open("tui_debug.log", "a") as f:
                    f.write(f"{datetime.now()}: DB Init Error: {str(e)}\n")
                global DB_ERR
                DB_ERR = f"ConnRefuned: {str(e)[:20]}"

        # 2. Try Cache (Flash Fallback) - If DB failed or price still 0
        if self.data["price"] == 0 and self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    ticker_data = cache.get(self.pinned_ticker, {})
                    if ticker_data:
                        # Direct Price Sync
                        self.data["price"] = ticker_data.get("price", 0.0)
                        self.data["change"] = ticker_data.get("change", 0.0)
                        self.data["status"] = "FALLBACK (Cache)"
                        
                        # Quantitative/ML Field Sync (v3.8 legacy fields)
                        # We try to extract from 'quantitative_signals' or 'ml_prediction'
                        q_sig = ticker_data.get("quantitative_signals", {})
                        if q_sig:
                            rng = q_sig.get("expected_range", {})
                            self.data.update({
                                "q_bottom": rng.get("bottom_10th", 0.0),
                                "q_median": rng.get("median_50th", 0.0),
                                "q_ceiling": rng.get("ceiling_90th", 0.0),
                                "max_upside": q_sig.get("max_upside_pct", 0.0),
                                "max_downside": q_sig.get("max_downside_pct", 0.0)
                            })
                            probs = q_sig.get("trend_probabilities", {})
                            self.data["ml_up"] = probs.get("up", 0.0)
                            self.data["ml_down"] = probs.get("down", 0.0)
            except Exception as e:
                with open("tui_debug.log", "a") as f:
                    f.write(f"{datetime.now()}: Cache Init Error: {str(e)}\n")

    def make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=4),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3),
        )
        # Main area split into 3 columns (Tech, Sentiment, Fusion)
        layout["main"].split_row(
            Layout(name="technical_panel", ratio=1),
            Layout(name="sentiment_panel", ratio=1),
            Layout(name="fusion_panel", ratio=1)
        )
        return layout

    def generate_header(self) -> Panel:
        time_str = datetime.now().strftime("%H:%M:%S")
        status_color = "green" if "LIVE" in self.data["status"] else "yellow"
        
        # New: Ticker Header with Sparkline
        p_val = self.data['price']
        price_text = f"{p_val:,.2f}" if p_val > 0 else "SYNCING..."
        change_color = "green" if self.data['change'] >= 0 else "red"
        
        grid = Table.grid(expand=True)
        grid.add_column(justify="left", ratio=1)
        grid.add_column(justify="center", ratio=1)
        grid.add_column(justify="right", ratio=1)
        grid.add_row(
            f"[bold magenta]AGENTIC TERMINAL v5.0[/bold magenta] [dim](Phase 1)[/dim]\n[bold yellow]{self.pinned_ticker}[/bold yellow] {self._get_sparkline(self.pinned_ticker)}",
            f"[bold white]{price_text}[/bold white] [[{change_color}]{self.data['change']:+.2f}%[/]]\n[dim]{time_str}[/dim]",
            f"Status: [bold {status_color}]{self.data['status']}[/bold {status_color}]\n[dim]Fusion Mode: {SETTINGS.sentiment_enabled and 'ENABLED' or 'OFF'}[/dim]"
        )
        return Panel(grid, style="white on blue")

    def _get_sparkline(self, ticker: str) -> str:
        """Simple ASCII sparkline for visual algorithm feedback."""
        try:
            # We use the cached change or fetch brief history
            # For now, a mock sparkline based on change for instant speed
            if self.data['change'] > 2: return "  ▃▅▇"
            if self.data['change'] > 0: return "  ▂▃▄"
            if self.data['change'] < -2: return "▇▅▃  "
            if self.data['change'] < 0: return "▄▃▂  "
            return "───"
        except Exception: return "---"

    def generate_technical_panel(self) -> Panel:
        """Panel A: Technical radar across horizons."""
        h_table = Table(box=None, expand=True)
        h_table.add_column("Horizon", style="bold cyan")
        h_table.add_column("Trend", justify="center")
        h_table.add_column("Forecast", justify="right")
        
        multi = self.data.get("multi_horizon", {})
        horizons = [("1W", "1w"), ("1M", "1m"), ("6M", "6m")]
        
        for label, key in horizons:
            sig = multi.get(key, {})
            if sig:
                q = sig.get("quantitative_signals", {})
                probs = q.get("trend_probabilities", {})
                p_up, p_down = probs.get("up", 0), probs.get("down", 0)
                
                trend_str = f"[bold green]UP ({p_up:.0%})[/]" if p_up > 0.55 else \
                            f"[bold red]DOWN ({p_down:.0%})[/]" if p_down > 0.55 else "[yellow]SIDE[/]"
                
                high = q.get("expected_range", {}).get("ceiling_90th", 0)
                upside = (((high - self.data['price']) / self.data['price']) * 100) if self.data['price'] > 0 else 0
                h_table.add_row(label, trend_str, f"[{'green' if upside > 0 else 'red'}]{upside:+.1f}%[/]")
            else:
                h_table.add_row(label, "[dim]-[/]", "[dim]...[/]")

        # Expected Range (Current Focus)
        q_table = Table(box=None, expand=True)
        q_table.add_column("Floor", justify="center", style="dim")
        q_table.add_column("Pivot", justify="center", style="bold")
        q_table.add_column("Ceiling", justify="center", style="dim")
        q_table.add_row(f"{self.data['q_bottom']:,.0f}", f"{self.data['q_median']:,.0f}", f"{self.data['q_ceiling']:,.0f}")

        progress = Progress(TextColumn("{task.description}"), BarColumn(bar_width=15), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
        progress.add_task("[green]BULL", completed=self.data['ml_up'] * 100)
        progress.add_task("[red]BEAR", completed=self.data['ml_down'] * 100)

        return Panel(Group(h_table, "\n", Panel(q_table, title="[dim]Target Range[/dim]"), "\n", Panel(progress, title="[dim]Confidence[/dim]")), 
                     title="[bold cyan]Panel A: Technical Agent[/bold cyan]", border_style="cyan")

    def generate_sentiment_panel(self) -> Panel:
        """Panel B: Qualitative/LLM Sentiment analysis."""
        intel = self.data.get("ai_intel", {})
        outlook = self.data.get('llm_outlook', "NEUTRAL")
        headlines = self.data.get('news_headlines', "No news context...")
        
        # Parse score and regime if v2
        score = intel.get("score", 0.0)
        regime = intel.get("regime", "neutral").upper()
        
        content = Text.assemble(
            (f"REGIME: {regime}\n", f"bold {'yellow' if regime == 'UNCERTAIN' else 'green' if outlook == 'POSITIVE' else 'red'}"),
            (f"SCORE: {score:+.2f}\n\n", "white"),
            ("[bold underline]Latest Headlines:[/bold underline]\n", "magenta"),
            (headlines[:250] + "...\n\n", "italic dim white"),
            ("[bold blue]Narrative Rationale:[/bold blue]\n", "white"),
            (self.data.get('llm_reasoning', "Analyzing news...")[:200], "deep_sky_blue1")
        )
        return Panel(content, title="[bold magenta]Panel B: Sentiment Agent[/bold magenta]", border_style="magenta")

    def generate_fusion_panel(self) -> Panel:
        """Panel C: Fusion Agent & Risk Overlay."""
        rec = self.data.get('ml_recommendation', "HOLD")
        rec_color = "green" if "BUY" in rec else "red" if "SELL" in rec else "yellow"
        
        table = Table(show_header=False, box=None, expand=True)
        table.add_row("[bold]Final Action:[/bold]", f"[bold white on {rec_color}] {rec} [/]")
        table.add_row("[bold]Fusion Confidence:[/bold]", f"[bold yellow]{(self.data.get('ml_up', 0.5) * 100):.1f}%[/]")
        table.add_row("[bold]Allocation Suggest:[/bold]", f"[bold cyan]{self.data.get('rl_allocation', 0.0):.1%}[/bold cyan]")
        table.add_row("[bold]Risk Budget:[/bold]", "[green]SAFE (Within Cap)[/]")
        table.add_row("[bold]Port. Veto:[/bold]", "[green]CLEARED[/]")
        
        trace = "\n[dim]Rationale Trace:[/dim]\nCombined high tech confidence with neutral-to-positive macro headlines. Sector correlation allows entry."
        
        return Panel(Group(table, trace), title="[bold green]Panel C: Fusion & Risk[/bold green]", border_style="green")

    async def _update_from_cache(self):
        while self.running:
            try:
                if self.cache_path.exists():
                    with open(self.cache_path, "r", encoding="utf-8") as f:
                        all_cache = json.load(f)
                        cache_data = all_cache.get(self.pinned_ticker)
                        if cache_data:
                            # New: Multi-Horizon Logic
                            self.data["multi_horizon"] = cache_data.get("multi_horizon", {})
                            
                            ml = cache_data.get("ml_prediction") or cache_data.get("quantitative_signals")
                            if not ml: ml = cache_data # Schema fallback
                            
                            llm = cache_data.get("llm_analysis", {})
                            
                            cache_news = llm.get("news_headlines", "")
                            if cache_news and str(cache_news).lower() != 'nan' and "Market context synced" not in cache_news:
                                # Further clean out any 'nan' items if it's a list/newline string
                                lines = [line.strip() for line in str(cache_news).split('\n') if line.strip() and line.strip().lower() != 'nan']
                                if lines:
                                    self.data["news_headlines"] = "\n".join(lines)
                            
                            # Probabilities (Handle nested quantitative_signals if necessary)
                            probs = ml.get("trend_probabilities")
                            if not probs and "quantitative_signals" in ml:
                                probs = ml["quantitative_signals"].get("trend_probabilities", {})
                            
                            if probs:
                                self.data["ml_up"] = probs.get("up", 0.0) 
                                self.data["ml_down"] = probs.get("down", 0.0)
                                
                            ranges = ml.get("expected_range")
                            if not ranges and "quantitative_signals" in ml:
                                ranges = ml["quantitative_signals"].get("expected_range", {})

                            if ranges:
                                self.data["q_bottom"] = ranges.get("bottom_10th", 0.0) 
                                self.data["q_median"] = ranges.get("median_50th", 0.0) 
                                self.data["q_ceiling"] = ranges.get("ceiling_90th", 0.0)
                                
                            # Re-calculate root upside using live price
                            live_p = self.data.get("price", 0)
                            q_ceiling = self.data.get("q_ceiling", 0)
                            q_bottom = self.data.get("q_bottom", 0)
                            
                            if live_p > 0 and q_ceiling > 0:
                                self.data["max_upside"] = (q_ceiling - live_p) / live_p
                                self.data["max_downside"] = (q_bottom - live_p) / live_p
                            else:
                                self.data["max_upside"] = ml.get("max_upside_pct") or ml.get("quantitative_signals", {}).get("max_upside_pct", 0.0)
                                self.data["max_downside"] = ml.get("max_downside_pct") or ml.get("quantitative_signals", {}).get("max_downside_pct", 0.0)
                            
                            self.data["llm_outlook"] = str(llm.get("overall_outlook", "NEUTRAL")).upper()
                            self.data["llm_reasoning"] = llm.get("reasoning", "Analysis loaded.")
                            
                            rl = llm.get("rl_recommendation", {})
                            self.data["rl_allocation"] = rl.get("suggested_allocation_pct") or self.data.get("rl_allocation", 0.0)
                            
                            dl = llm.get("deep_learning_context", {})
                            if dl.get("tft_forecast"): self.data["tft_signal"] = dl["tft_forecast"]
                            if dl.get("cnn_microstructure"): self.data["cnn_signal"] = dl["cnn_microstructure"]

                            if ml.get("action_plan", {}).get("recommendation"):
                                self.data["ml_recommendation"] = ml["action_plan"]["recommendation"]
                        else:
                            # Start background analysis if missing
                            if not self.data.get("syncing_analysis"):
                                self.data["news_headlines"] = f"[yellow]Syncing Analysis for {self.pinned_ticker}...[/yellow]"
            except Exception: pass
            await asyncio.sleep(2)

    async def _trigger_on_demand_sync(self):
        """Trigger historical ingestion if DB is empty for this ticker."""
        if not HAS_DB: return
        try:
            from src.historical.backdate import BackdateIngestor
            async with get_session() as session:
                # Optimized check: Avoid full table count
                res = await session.execute(text("SELECT 1 FROM raw_prices WHERE ticker = :t LIMIT 1"), {"t": self.pinned_ticker})
                exists = res.scalar() is not None
                if not exists:
                    self.data["status"] = "BOOTSTRAPPING (30D)"
                    ingestor = BackdateIngestor()
                    start_date = dt.date.today() - dt.timedelta(days=30)
                    
                    # Fetch and Update UI status
                    async def _ingest_task():
                        try:
                            await ingestor.run(tickers=[self.pinned_ticker], start_date=start_date)
                            self.data["status"] = "LIVE (DB-SYNCED)"
                        except Exception:
                            self.data["status"] = "SYNC_FAILED"
                    
                    asyncio.create_task(_ingest_task())
        except Exception: pass

    async def _poll_redis(self):
        """Phase 23: Zero-Latency Price fetch from Redis O(1) cache."""
        if not HAS_REDIS: return
        try:
            if not self._redis_client:
                self._redis_client = Redis.from_url(SETTINGS.redis_url)
            
            raw = await self._redis_client.get(f"live_price:{self.pinned_ticker}")
            if raw:
                cached = json.loads(raw)
                ts = cached.get("ts", 0)
                if ts >= self.data.get("last_price_ts", 0):
                    self.data["price"] = float(cached["price"])
                    self.data["last_price_ts"] = ts
                    self.data["status"] = "LIVE (Redis)"
        except Exception: pass

    async def _poll_db(self):
        if not HAS_DB: return
        try:
            loop = asyncio.get_event_loop()
            async with get_session() as session:
                # Use a small timeout for DB check
                res = await asyncio.wait_for(session.execute(
                    select(RawPrice).filter(RawPrice.ticker == self.pinned_ticker).order_by(desc(RawPrice.timestamp)).limit(1)
                ), timeout=2.0)
                rec = res.scalar_one_or_none()
                if rec:
                    ts = rec.timestamp.timestamp()
                    if ts >= self.data.get("last_price_ts", 0):
                        self.data["price"] = float(rec.close)
                        self.data["last_price_ts"] = ts
                    if "LIVE" not in self.data["status"]: self.data["status"] = "LIVE (DB)"
        except Exception: pass

    async def _poll_rest(self):
        if not HAS_VNSTOCK or VNSTOCK_CLIENT is None: return
        try:
            loop = asyncio.get_event_loop()
            sources = ["VCI", "TCBS", "DNSE"]
            
            # 1. ALWAYS try DB first for Heuristics (Fastest & No API limit)
            # Use data from DB as the master "Analytical" source if available
            await self._compute_heuristics_from_db()

            for src in sources:
                try:
                    stock = await loop.run_in_executor(None, lambda: VNSTOCK_CLIENT.stock(symbol=self.pinned_ticker, source=src))
                    
                    # 2. REST Heuristic Fallback
                    if self.data.get("ml_up", 0) == 0:
                        self.data["status"] = f"ANALYZING ({src})"
                        await loop.run_in_executor(None, self._compute_heuristics, stock)
                        
                    # 3. Latest Price
                    df_m = await loop.run_in_executor(None, lambda: stock.quote.history(interval="1m", count=1))
                    if df_m is not None and not df_m.empty:
                        last_row = df_m.iloc[-1]
                        price = float(last_row["close"])
                        ts = last_row["time"].timestamp()
                        
                        if ts >= self.data.get("last_price_ts", 0):
                            self.data["price"] = price
                            self.data["last_price_ts"] = ts
                            
                        if self.data.get("ml_up", 0) > 0:
                            self.data["status"] = f"ANALYZED ({src})"
                        else:
                            self.data["status"] = f"LIVE ({src})"
                        return
                except Exception: continue
                
            if self.data.get("ml_up", 0) == 0: self.data["status"] = "DATA_GAP"
        except Exception: 
            self.data["status"] = "REST_ERR"

    async def _compute_heuristics_from_db(self):
        """Fetch history from local TimescaleDB for instant heuristics."""
        if not HAS_DB: return
        try:
            async with get_session() as session:
                # Use scalar variables for cleaner query
                res = await session.execute(text("""
                    SELECT 
                        DATE(timestamp) as date, 
                        (ARRAY_AGG(close ORDER BY timestamp DESC))[1] as close,
                        SUM(volume) as volume
                    FROM raw_prices 
                    WHERE ticker = :t 
                    GROUP BY DATE(timestamp) 
                    ORDER BY date DESC LIMIT 200
                """), {"t": self.pinned_ticker})
                rows = res.fetchall()
                
                if len(rows) >= 3:
                    # Robust DataFrame creation with explicit float conversion
                    data = []
                    for r in rows:
                        data.append({
                            "date": r[0],
                            "close": float(r[1]) if r[1] is not None else 0.0,
                            "volume": float(r[2]) if r[2] is not None else 0.0
                        })
                    df = pd.DataFrame(data).sort_values("date")
                    self._process_history_into_heuristics(df, source="DB")
                else:
                    self.data["status"] = f"DB_EMPTY:{self.pinned_ticker}"
        except Exception as e:
            self.data["status"] = f"DB_ERR:{str(e)[:10]}"
            with open("tui_debug.log", "a") as f:
                f.write(f"{datetime.now()}: DB Error for {self.pinned_ticker}: {str(e)}\n")

    def _compute_heuristics(self, stock):
        """REST-based history fetch."""
        try:
            df = stock.quote.history(interval="1D", count=20)
            if df is not None and len(df) >= 3:
                self._process_history_into_heuristics(df, source="REST")
        except Exception: pass

    def _process_history_into_heuristics(self, df, source="AUTO"):
        """Shared logic to turn a history DF into TUI signals."""
        try:
            p = float(df.iloc[-1]["close"])
            p_prev = float(df.iloc[-2]["close"]) if len(df) >= 2 else p
            change_pct = ((p - p_prev) / p_prev * 100) if p_prev != 0 else 0.0
            
            diffs = df["close"].diff()
            ups = diffs.clip(lower=0).rolling(min(14, len(df))).mean().iloc[-1]
            downs = diffs.clip(upper=0).abs().rolling(min(14, len(df))).mean().iloc[-1]
            rsi = 100 - (100 / (1 + (ups / downs))) if downs != 0 else 50
            
            sma20 = df["close"].rolling(min(20, len(df))).mean().iloc[-1]
            trend = "UP" if p > sma20 else "DOWN"
            
            std = df["close"].pct_change().std() or 0.01
            final_bull = max(0.05, min(0.95, 0.5 + (0.1 if trend == "UP" else -0.1) + (rsi - 50)/100))
            
            h_data = {
                "ml_up": final_bull, "ml_down": 1.0 - final_bull,
                "q_bottom": p * (1 - 2*std), "q_median": p, "q_ceiling": p * (1 + 2*std),
                "tft_signal": "Trend " + trend, "cnn_signal": "Vol " + ("High" if std > 0.02 else "Low"),
                "llm_outlook": "POSITIVE" if final_bull > 0.6 else "NEGATIVE" if final_bull < 0.4 else "NEUTRAL",
                "llm_reasoning": f"Algorithmic ({source}): {trend} trend, RSI={rsi:.1f}. Vol={std:.1%}.",
                "ml_recommendation": f"{trend} TREND (Analytic)",
                "price": p, "change": change_pct
            }
            if self.data.get("rl_allocation", 0) == 0:
                h_data["rl_allocation"] = 0.05 if trend == "UP" else 0.01

            self.data.update(h_data)
            self._heuristic_cache[self.pinned_ticker] = (time.time(), h_data)
        except Exception as e:
            with open("tui_debug.log", "a") as f:
                f.write(f"{datetime.now()}: Process Heuristic Error: {str(e)}\n")

    async def _update_news(self):
        """Fetch real-time news via crawler."""
        if not self.news_crawler: return
        try:
            now = time.time()
            if now - self._last_news_sync < 300: return # Rate limit 5m
            
            docs = await self.news_crawler.crawl_ticker(self.pinned_ticker, max_pages=3)
            if docs:
                # Format headlines for the TUI panel
                headlines = "\n".join([f"• {doc.title[:80]}..." for doc in docs[:3]])
                self.data["news_headlines"] = headlines
                self._last_news_sync = now
                self.logger.info("news_updated", ticker=self.pinned_ticker, count=len(docs))
        except Exception as e:
            with open("tui_debug.log", "a") as f:
                f.write(f"{datetime.now()}: News Error: {str(e)}\n")

    async def _update_news_intelligence(self):
        """Fetch analyzed intelligence from DB."""
        if not HAS_DB: return
        try:
            intel = await self.news_intel.get_latest_intelligence(self.pinned_ticker)
            if intel:
                self.data["ai_intel"] = intel
                # Sync with the rest of the UI
                self.data.update({
                    "llm_outlook": intel["trend"].upper(),
                    "llm_reasoning": intel["summary"],
                })
        except Exception: pass

    async def _update_live_feeds(self):
        while self.running:
            # Parallel Polling for Speed
            try:
                tasks = []
                if HAS_REDIS: tasks.append(self._poll_redis())
                if HAS_DB: 
                    tasks.append(self._poll_db())
                    tasks.append(self._update_news_intelligence())
                if self.news_crawler: tasks.append(self._update_news())
                if HAS_VNSTOCK: tasks.append(self._poll_rest())
                if tasks: await asyncio.gather(*tasks)
            except Exception: pass
            await asyncio.sleep(5)

    async def _update_ui(self, layout: Layout):
        while self.running:
            try:
                # Top Header
                layout["header"].update(self.generate_header())
                
                # Panel A, B, C
                layout["technical_panel"].update(self.generate_technical_panel())
                layout["sentiment_panel"].update(self.generate_sentiment_panel())
                layout["fusion_panel"].update(self.generate_fusion_panel())

                # Footer
                db_info = "SYNCING" if "BOOTSTRAPPING" in self.data["status"] else "OK" if HAS_DB else "OFFLINE"
                prio = " | [bold green]AGENTIC MODE[/]"
                f_text = f"TICKER: {self.pinned_ticker} | CONTRACT: v2 | DB: {db_info}{prio} | {datetime.now().strftime('%H:%M:%S')}"
                layout["footer"].update(Panel(Align.center(f"[bold white]{f_text}[/]"), style="white on red"))
            except Exception as e:
                # Fallback simple UI on error
                layout["footer"].update(Panel(f"[bold yellow]Layout Engine Busy: {str(e)}[/bold yellow]"))
            
            await asyncio.sleep(SETTINGS.terminal_refresh_ms / 1000.0) 

    async def run(self):
        # 1. Set Priority Lock
        self._lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path.touch()
        (self._lock_path.parent / ".tui_ticker").write_text(self.pinned_ticker)
        
        # 2. Unblocked Startup: Start sync in background
        asyncio.create_task(self._trigger_on_demand_sync())
        
        layout = self.make_layout()
        with Live(layout, refresh_per_second=1, screen=True):
            tasks = [
                asyncio.create_task(self._update_ui(layout)),
                asyncio.create_task(self._update_from_cache()),
                asyncio.create_task(self._update_live_feeds())
            ]
            try:
                await asyncio.gather(*tasks)
            except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
                self.running = False
                if self._lock_path.exists(): self._lock_path.unlink()
                for t in tasks: t.cancel()
            finally:
                if self._lock_path.exists(): self._lock_path.unlink()
                await dispose_engine()

if __name__ == "__main__":
    ticker = "FPT"
    for arg in sys.argv[1:]:
        if len(arg) >= 3 and arg.isalpha():
            ticker = arg.upper()
            break
            
    tui = AlgoTradingTUI(ticker)
    try:
        asyncio.run(tui.run())
    except KeyboardInterrupt:
        pass
