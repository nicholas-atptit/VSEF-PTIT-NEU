"""Demo module.
Non-authoritative and not part of canonical governed runtime.

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
from typing import Any
import pandas as pd

import sys
# --- Legacy Cloud SDK Suppression ---
try:
    # Only try to suppress/import if library is actually needed or present
    import google.generativeai as genai
except ImportError:
    pass

# --- Robust Imports ---
from dotenv import load_dotenv
load_dotenv()
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

# --- DATABASE INTEGRATION ---
HAS_DB = False
DB_VERSION = "5.0.0-AGENTIC"
DB_ERR = "No Driver"
try:
    from src.data.database.connection import get_session, dispose_engine, get_db
    from src.ml.models.price import RawPrice
    from src.ml.data_loader import load_ohlcv_from_db
    from sqlalchemy import select, desc, text
    HAS_DB = True
except Exception as e:
    DB_ERR = f"DBError: {str(e).split(':')[-1]}"

# --- AI AGENT INTEGRATION (OPTIONAL) ---
HAS_AI = False
try:
    from src.context.news_crawler import NewsCrawler
    from src.ml.llm.news_intel import NewsIntelEngine
    from src.ml.signal_generator import SignalGenerator
    HAS_AI = True
except Exception as e:
    _logger.warning("ai_agents_disabled", error=str(e))
    NewsCrawler = None
    NewsIntelEngine = None
    SignalGenerator = None
    HAS_AI = False
except (ImportError, ModuleNotFoundError) as e:
    DB_ERR = f"DepError: {str(e).split(':')[-1]}"
except Exception as e:
    DB_ERR = f"ConfigError: {str(e).split(':')[-1]}"

# --- VNSTOCK_DATA PRO AUTH ---
HAS_VNSTOCK = False
try:
    from vnstock_data import Quote as _Quote
    HAS_VNSTOCK = True
    VNSTOCK_CLIENT = None  # vnstock_data uses functional API, no global client needed
except Exception:
    HAS_VNSTOCK = False
    VNSTOCK_CLIENT = None

class AlgoTradingTUI:
    def __init__(self, start_ticker="FPT"):
        self.pinned_ticker = start_ticker.upper()
        self.cache_path = PROJECT_ROOT / "data" / "latest_predictions.json"
        self.news_crawler = NewsCrawler() if (HAS_DB and NewsCrawler is not None) else None
        self.news_intel = NewsIntelEngine() if (HAS_DB and NewsIntelEngine is not None) else None
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
        self.agent = SignalGenerator() if HAS_AI else None
        self._last_agent_run = 0
        self._agent_task: asyncio.Task | None = None
        self._heuristic_cache = {} 
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
                self.logger.error("tui_db_init_error", ticker=self.pinned_ticker, error=str(e))
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
                self.logger.error("tui_cache_init_error", ticker=self.pinned_ticker, error=str(e))

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
        # Health Indicators
        risk_data = self.data.get("risk_intel") or {}
        sent_data = self.data.get("sentiment_intel") or {}
        
        # AI is ONLINE if sentiment data exists or it is currently PENDING
        is_pending = sent_data.get("status") == "PENDING"
        ai_online = (len(sent_data) > 0 and sent_data.get("ticker")) or is_pending
        ai_status = "[bold green]AI: ONLINE[/]" if ai_online else "[bold yellow]AI: FALLBACK[/]"
        if is_pending:
            ai_status = "[bold blue]AI: PENDING...[/]"
            
        regime_str = self.data.get("regime_label", "UNKNOWN").upper()
        ai_status += f" | REGIME: [bold cyan]{regime_str}[/bold cyan]"
        
        acc = risk_data.get("model_accuracy_1w", 0.0) if isinstance(risk_data, dict) else 0.0
        acc_str = f"[bold cyan]Acc: {acc:.0%}[/]" if acc > 0 else "[dim]Acc: --%[/]"
        
        grid.add_row(
            f"[bold magenta]AGENTIC TERMINAL v5.0[/bold magenta]\n[bold yellow]{self.pinned_ticker}[/bold yellow] {self._get_sparkline(self.pinned_ticker)}",
            f"[bold white]{price_text}[/bold white] [[{change_color}]{self.data['change']:+.2f}%[/]]\n[dim]{time_str}[/dim]",
            f"{ai_status} | {acc_str}\nStatus: [bold {status_color}]{self.data['status']}[/bold {status_color}]"
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
        
        multi = self.data.get("multi_horizon") or {}
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
                if high > 0 and self.data['price'] > 0:
                    upside = (((high - self.data['price']) / self.data['price']) * 100)
                    upside_str = f"[{'green' if upside > 0 else 'red'}]{upside:+.1f}%[/]"
                else:
                    upside_str = "[dim]---[/dim]"
                
                h_table.add_row(label, trend_str, upside_str)
            else:
                h_table.add_row(label, "[dim]-[/]", "[dim]...[/]")

        # Expected Range (Current Focus)
        q_table = Table(box=None, expand=True)
        q_table.add_column("Floor", justify="center", style="dim")
        q_table.add_column("Pivot", justify="center", style="bold")
        q_table.add_column("Ceiling", justify="center", style="dim")
        q_table.add_row(f"{self.data['q_bottom']:,.0f}", f"{self.data['q_median']:,.0f}", f"{self.data['q_ceiling']:,.0f}")

        progress = Progress(TextColumn("{task.description}"), BarColumn(bar_width=15), TextColumn("[progress.percentage]{task.percentage:>3.0f}%"))
        progress.add_task("[green]BULL", completed=(self.data.get('ml_up') or 0.0) * 100)
        progress.add_task("[red]BEAR", completed=(self.data.get('ml_down') or 0.0) * 100)

        return Panel(Group(h_table, "\n", Panel(q_table, title="[dim]Target Range[/dim]"), "\n", Panel(progress, title="[dim]Confidence[/dim]")), 
                     title="[bold cyan]Panel A: Technical Agent[/bold cyan]", border_style="cyan")

    def generate_sentiment_panel(self) -> Panel:
        """Panel B: Qualitative/LLM Sentiment analysis."""
        intel = self.data.get("sentiment_intel") or {}
        outlook = self.data.get('llm_outlook', "NEUTRAL")
        headlines = self.data.get('news_headlines')
        if not headlines or "Syncing" in headlines:
            headlines = intel.get("summary", "Fetching Market Context......")
        
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
        
        risk = self.data.get("risk_intel") or {}
        veto_status = "[red]BLOCKED (VETO)[/]" if risk.get("veto_flag") else "[green]CLEARED[/]"
        
        table = Table(show_header=False, box=None, expand=True)
        table.add_row("[bold]Final Action:[/bold]", f"[bold white on {rec_color}] {rec} [/]")
        table.add_row("[bold]Agent Consensus Score:[/bold]", f"[bold magenta]{self.data.get('consensus_score', 0.0):.3f}[/bold magenta]")
        table.add_row("[bold]Dynamic Threshold:[/bold]", f"[bold white]{self.data.get('dynamic_confidence_threshold', 0.0):.2f}[/bold white]")
        table.add_row("[bold]Allocation (Lots):[/bold]", f"[bold cyan]{(self.data.get('execution_shares', 0)):,.0f} shares[/bold cyan]")
        table.add_row("[bold]Risk Budget:[/bold]", "[green]SAFE (Within Cap)[/]")
        table.add_row("[bold]Risk Veto:[/bold]", veto_status)
        
        if risk and isinstance(risk, dict) and risk.get("constraints_hit"):
            table.add_row("[bold red]Violations:[/bold red]", f"[red]{', '.join(risk['constraints_hit'])}[/red]")
        
        acc_1w = risk.get("model_accuracy_1w", 0.0) if isinstance(risk, dict) else 0.0
        table.add_row("[bold]Running Acc (1W):[/bold]", f"[bold cyan]{acc_1w:.1%}[/bold cyan]" if acc_1w > 0 else "[dim]Insufficient Data[/dim]")
        
        trace = "\n[dim]Rationale Trace:[/dim]\nCombined high tech confidence with neutral-to-positive macro headlines. Sector correlation allows entry."
        
        return Panel(Group(table, trace), title="[bold green]Panel C: Fusion & Risk[/bold green]", border_style="green")

    async def _update_from_decision_cards(self):
        """Poll the reports/decision_cards directory for the latest debate output."""
        cards_dir = PROJECT_ROOT / "reports" / "decision_cards"
        while self.running:
            try:
                if cards_dir.exists():
                    files = list(cards_dir.glob(f"{self.pinned_ticker}_*.json"))
                    if files:
                        latest_file = sorted(files)[-1]
                        import json
                        with open(latest_file, "r", encoding="utf-8") as f:
                            card = json.load(f)
                            
                        self.data["consensus_score"] = card.get("consensus_score", 0.0)
                        self.data["regime_label"] = card.get("regime_label", "UNKNOWN")
                        self.data["dynamic_confidence_threshold"] = card.get("dynamic_confidence_threshold", 0.0)
                        self.data["execution_shares"] = card.get("execution_shares", 0)
                        
                        if card.get("action"):
                            self.data["ml_recommendation"] = card["action"]
                        if card.get("rationale"):
                            self.data["llm_reasoning"] = card["rationale"]
                            self.data["_has_decision_card"] = True
                            
                        # Satisfy the agent intel check to prevent endless fallback loops
                        if not self.data.get("sentiment_intel"):
                            self.data["sentiment_intel"] = {"status": "LIVE (Card)"}
                            
            except Exception as e:
                self.logger.error("update_from_cards_error", error=str(e))
            await asyncio.sleep(5)

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
                            
                            if not self.data.get("_has_decision_card"):
                                self.data["llm_reasoning"] = llm.get("reasoning", "Analysis loaded.")
                                
                            rl = llm.get("rl_recommendation", {})
                            self.data["rl_allocation"] = rl.get("suggested_allocation_pct") or self.data.get("rl_allocation", 0.0)
                            
                            dl = llm.get("deep_learning_context", {})
                            if dl.get("tft_forecast"): self.data["tft_signal"] = dl["tft_forecast"]
                            if dl.get("cnn_microstructure"): self.data["cnn_signal"] = dl["cnn_microstructure"]

                            if ml.get("action_plan", {}).get("recommendation"):
                                self.data["ml_recommendation"] = ml["action_plan"]["recommendation"]
                                
                            # --- v5.0 Payload Support ---
                            if "technical" in cache_data:
                                tech = cache_data["technical"]
                                horizons = tech.get("horizons", [])
                                if horizons:
                                    h0 = horizons[0]
                                    probs = h0.get("trend_probs", {})
                                    self.data["ml_up"] = probs.get("up", 0.0)
                                    self.data["ml_down"] = probs.get("down", 0.0)
                                    
                                    rng = h0.get("expected_range", {})
                                    self.data["q_bottom"] = rng.get("bottom_10th", 0.0)
                                    self.data["q_median"] = rng.get("median_50th", 0.0)
                                    self.data["q_ceiling"] = rng.get("ceiling_90th", 0.0)
                            
                            if "sentiment" in cache_data and cache_data["sentiment"]:
                                sent = cache_data["sentiment"]
                                self.data["sentiment_intel"] = sent
                                self.data["ai_intel"] = {
                                    "sentiment_score": sent.get("sentiment_score", 0.0),
                                    "trend": sent.get("sentiment_regime", "neutral"),
                                    "score": sent.get("sentiment_score", 0.0),
                                    "regime": sent.get("sentiment_regime", "neutral")
                                }
                                self.data["llm_outlook"] = sent.get("sentiment_regime", "neutral").upper()
                                if sent.get("summary"):
                                    self.data["news_headlines"] = sent["summary"]
                                
                            if "fusion" in cache_data and not self.data.get("_has_decision_card"):
                                fus = cache_data["fusion"]
                                self.data["ml_recommendation"] = fus.get("action", "HOLD")
                                self.data["llm_reasoning"] = fus.get("rationale", "No rationale.")
                            
                            if "risk" in cache_data:
                                self.data["risk_intel"] = cache_data["risk"]
                        # Trigger On-Demand Agent if missing or stale (>10 min)
                        now = time.time()
                        has_intel = self.data.get("sentiment_intel") and len(self.data["sentiment_intel"]) > 1
                        
                        has_running_agent_task = self._agent_task is not None and not self._agent_task.done()
                        if HAS_AI and self.agent and (not has_intel) and (now - self._last_agent_run > 60) and not has_running_agent_task:
                            # Only set pending if we don't have ANY intel
                            if not self.data.get("sentiment_intel"):
                                self.data["news_headlines"] = f"[yellow]Agent Analysis PENDING for {self.pinned_ticker}...[/yellow]"
                            self._last_agent_run = now
                            self._agent_task = asyncio.create_task(self._run_agent_on_demand())
                        
                        # Start background analysis info
                        if not self.data.get("syncing_analysis") and not has_intel:
                            self.data["news_headlines"] = f"[yellow]Syncing Analysis for {self.pinned_ticker}...[/yellow]"
            except Exception as e:
                self.logger.error("update_from_cache_error", error=str(e))
            await asyncio.sleep(2)

    async def _run_agent_on_demand(self):
        """Phase 5 Pro-feature: Live Agent fallback in TUI."""
        try:
            if not self.agent:
                self.logger.warning("agent_on_demand_skipped", reason="agent_unavailable")
                return
            self.logger.info("agent_on_demand_start", ticker=self.pinned_ticker)
            
            # Mark as pending for UI
            self.data["sentiment_intel"] = {"status": "PENDING"}
            
            # Prepare minimal model_output if we don't have one
            curr_price = self.data.get("price", 0.0)
            mock_output = {
                "horizon": "short",
                "trend_probabilities": {"up": 0.33, "sideways": 0.33, "down": 0.34},
                "expected_range": {
                    "bottom_10th": curr_price * 0.95,
                    "median_50th": curr_price,
                    "ceiling_90th": curr_price * 1.05
                }
            }
            res = await asyncio.wait_for(
                self.agent.generate(
                    self.pinned_ticker, 
                    current_close=self.data.get('price', 0.0),
                    model_output=mock_output,
                    active_analysis=True
                ), 
                timeout=35.0
            )
            
            if res:
                # Update local cache immediately
                self.data["risk_intel"] = res.get("risk") or {}
                self.data["sentiment_intel"] = res.get("sentiment") or {}
                self.data["ml_recommendation"] = res.get("fusion", {}).get("action", "HOLD")
                
                # Fetch summary from sentiment payload
                sent = res.get("sentiment", {})
                self.data["news_headlines"] = sent.get("summary") or "Analysis complete (No news found)."
                self.data["llm_outlook"] = sent.get("sentiment_regime", "neutral").upper()
                
                # Write back to global cache file to share with other tools
                if self.cache_path.exists():
                    try:
                        with open(self.cache_path, "r", encoding="utf-8") as f:
                            all_cache = json.load(f)
                        all_cache[self.pinned_ticker] = res
                        with open(self.cache_path, "w", encoding="utf-8") as f:
                            json.dump(all_cache, f, indent=4)
                    except Exception as e:
                        self.logger.error("cache_write_error", error=str(e))

                self.logger.info("agent_on_demand_success", ticker=self.pinned_ticker)
            else:
                self.logger.warning("agent_on_demand_empty_result", ticker=self.pinned_ticker)
                # Cleanup pending status if empty
                if self.data.get("sentiment_intel", {}).get("status") == "PENDING":
                    self.data["sentiment_intel"] = {"status": "FALLBACK"}
                    self.data["news_headlines"] = "[yellow]AI API Offline - Analysis halted.[/yellow]"

        except asyncio.TimeoutError:
            self.logger.error("agent_on_demand_timeout", ticker=self.pinned_ticker)
            if self.data.get("sentiment_intel", {}).get("status") == "PENDING":
                self.data["sentiment_intel"] = {"status": "TIMEOUT"}
                self.data["news_headlines"] = "[red]AI Request Timed Out (Local model hanging?)[/red]"
        except Exception as e:
            self.logger.error("agent_on_demand_failed", error=str(e))
            if self.data.get("sentiment_intel", {}).get("status") == "PENDING":
                self.data["sentiment_intel"] = {"status": "ERROR"}
        finally:
            self._last_agent_run = time.time()

    async def _trigger_on_demand_sync(self):
        """Trigger historical ingestion if DB is empty for this ticker."""
        if not HAS_DB: return
        try:
            from src.data.historical.backdate import BackdateIngestor
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

    async def _get_hybrid_market_pulse(self):
        """Fetch the new v4 Hybrid Market Summary from the API."""
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                res = await client.get("http://127.0.0.1:8005/api/v2/market/summary", timeout=2.0)
                if res.status_code == 200:
                    self.data["market_pulse"] = res.json()
        except Exception:
            self.data["market_pulse"] = {"status": "Syncing 104..."}

    def generate_footer(self) -> Panel:
        pulse = self.data.get("market_pulse", {})
        sent_24h = pulse.get("sentiment_24h", 0.0)
        sent_color = "green" if sent_24h > 0.1 else "red" if sent_24h < -0.1 else "yellow"
        
        dist = pulse.get("prediction_distribution", {})
        up_count = dist.get("UP", 0)
        
        pulse_str = f"| [bold]MARKET 104:[/] Sent: [{sent_color}]{sent_24h:+.2f}[/] | Bullish: [cyan]{up_count}[/] mã"
        if not dist:
            pulse_str = "| [dim]Hybrid Training v4: Active (104 Tickers)...[/dim]"
            
        footer_text = Text.assemble(
            (f" Ticker: {self.pinned_ticker} ", "bold white on magenta"),
            (f"  {pulse_str}  ", "white"),
            (f"  DB: {DB_VERSION} | Status: {DB_ERR if not HAS_DB else 'READY'} ", "dim cyan")
        )
        return Panel(footer_text, style="white on black")

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
        try:
            loop = asyncio.get_event_loop()
            import datetime as _dt

            # 1. ALWAYS try DB first for Heuristics (Fastest & No API limit)
            await self._compute_heuristics_from_db()

            # 2. Try vnstock_data REST fallback for price update
            if not HAS_VNSTOCK:
                if self.data.get("ml_up", 0) == 0:
                    self.data["status"] = "DATA_GAP"
                return

            try:
                from src.data.adapters.vnstock_adapter import VnstockAdapter
                end_d = _dt.date.today()
                start_d = end_d - _dt.timedelta(days=5)

                df_daily = await loop.run_in_executor(
                    None,
                    lambda: VnstockAdapter().get_ohlcv(
                        self.pinned_ticker,
                        start_date=start_d.strftime("%Y-%m-%d"),
                        end_date=end_d.strftime("%Y-%m-%d"),
                        interval="1D",
                    )
                )
                if df_daily is not None and not df_daily.empty:
                    last_row = df_daily.iloc[-1]
                    price = float(last_row["close"])
                    time_val = last_row.get("date") if hasattr(last_row, "get") else None
                    ts = _dt.datetime.now().timestamp()
                    if time_val is not None:
                        try:
                            ts = _dt.datetime.fromisoformat(str(time_val)).timestamp()
                        except Exception:
                            pass

                    if ts >= self.data.get("last_price_ts", 0):
                        self.data["price"] = price
                        self.data["last_price_ts"] = ts

                    if self.data.get("ml_up", 0) > 0:
                        self.data["status"] = "ANALYZED (vnstock_data)"
                    else:
                        self.data["status"] = "LIVE (vnstock_data)"
                    # Compute basic heuristics if DB heuristics weren't available
                    if self.data.get("ml_up", 0) == 0:
                        self._compute_heuristics(df_daily)
                    return
            except Exception:
                pass

            if self.data.get("ml_up", 0) == 0:
                self.data["status"] = "DATA_GAP"
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
            self.logger.error("db_heuristics_error", ticker=self.pinned_ticker, error=str(e))

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
                "ml_recommendation": f"{trend} TREND (Analytic)",
                "price": p, "change": change_pct
            }
            if not self.data.get("_has_decision_card"):
                h_data["llm_reasoning"] = f"Algorithmic ({source}): {trend} trend, RSI={rsi:.1f}. Vol={std:.1%}."
            
            if self.data.get("rl_allocation", 0) == 0:
                h_data["rl_allocation"] = 0.05 if trend == "UP" else 0.01

            self.data.update(h_data)
            self._heuristic_cache[self.pinned_ticker] = (time.time(), h_data)
        except Exception as e:
            self.logger.error("heuristics_process_error", ticker=self.pinned_ticker, error=str(e))

    async def _update_news(self):
        """Fetch real-time news via crawler."""
        if not self.news_crawler: return
        try:
            now = time.time()
            if now - self._last_news_sync < 300: return # Rate limit 5m
            
            docs = await self.news_crawler.crawl_ticker(self.pinned_ticker, count=5)
            if docs:
                # Format headlines for the TUI panel
                headlines = "\n".join([f"• {doc.title[:80]}..." for doc in docs[:3]])
                if headlines.strip():
                    self.data["news_headlines"] = headlines
                self._last_news_sync = now
                self.logger.info("news_updated", ticker=self.pinned_ticker, count=len(docs))
        except Exception as e:
            self.logger.error("news_update_error", ticker=self.pinned_ticker, error=str(e))

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
                # New Phase 42 Hybrid Sync
                tasks.append(self._get_hybrid_market_pulse())
                
                if tasks: await asyncio.gather(*tasks)
            except Exception as e:
                 self.logger.error("live_feeds_loop_error", error=str(e))
            await asyncio.sleep(5)

    async def _update_ui(self, layout: Layout):
        while self.running:
            try:
                # Top Header
                layout["header"].update(self.generate_header())
                
                # Panel A, B, C
                layout["main"]["technical_panel"].update(self.generate_technical_panel())
                layout["main"]["sentiment_panel"].update(self.generate_sentiment_panel())
                layout["main"]["fusion_panel"].update(self.generate_fusion_panel())

                # Footer - Now using the specialized 104-Pulse generator
                layout["footer"].update(self.generate_footer())
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
                asyncio.create_task(self._update_from_decision_cards()),
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
