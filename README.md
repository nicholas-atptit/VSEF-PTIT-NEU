# 📈 Algo Trading AI — Professional Stack (v4.6.0)

> **Enterprise-grade Vietnamese stock market analysis system with 1,600+ LightGBM models, Ollama LLM, Kafka message broker, and a High-Performance Terminal UI (TUI) Dashboard.**

---

## 🏗️ Architecture

```
Market Data (vnstock_pro) ──► Redis (Live Cache) ──► TimescaleDB (5-Year History)
                                                       │
[Background Sync] ◄────────────────────────────────────┘
       │
[ML Feed] ──► Parallel Inferencing (1600+ Tickers) ──► JSON Cache ──► TUI Dashboard (v4.6)
                                                                       │
LLM Context (Ollama) ◄─────────────────────────────────────────────────┘
```

## 🚀 Key Features (v4.6.0 Upgrade)

- ✅ **Professional TUI Dashboard**: Real-time monitoring with `Rich` rendering, Market Reality (Heuristics) vs. Psychology (ML) panels.
- ✅ **5-Year Historical Data**: Instant local bootstrapping for all tickers (2019-2024).
- ✅ **Live Heartbeat Sync**: Ultra-fast (10s cycle) database updates using `vnstock_data` Pro (300 req/min).
- ✅ **All-Ticker ML Engine**: Parallel processing of 1,600+ models with intelligent TUI prioritization and incremental caching.
- ✅ **Max Forecast Metrics**: Probabilistic % Upside/Downside calculations based on 90th percentile ML quantiles.
- ✅ **Zero-Gap Reliability**: Automatic connection disposal, REST fallbacks, and DB existence validation.

---

## 🔧 Service Entry Points

| Script | Command | Description |
|--------|---------|-------------|
| **Dashboard** | `python src/ui/dashboard.py <TICKER>` | Start the Professional TUI (v4.6) |
| **Live Sync** | `python scripts/live_heartbeat_sync.py` | 10s Live DB Heartbeat (Pro) |
| **ML Engine** | `python scripts/per_session_predict.py --loop 15` | All-ticker ML update (15m cycle) |
| **Backdate** | `python scripts/run_backdate.py` | Historical data bulk fetcher |
| **Importer** | `python scripts/local_importer.py` | High-speed CSV -> DB importer |

---

## 📂 Project Structure

```
├── data/                          # Persistent Data & Cache
│   ├── prediction_cache/          # Incremental ML predictions (v4.6)
│   ├── listing/                   # VIP & Market symbols list
│   └── .tui_ticker                # Priority lock for background services
├── models/                        # 1,625 trained LightGBM models (Quantile + Trend)
├── scripts/                       # High-Performance Entry Points
│   ├── live_heartbeat_sync.py     # ⭐ Live 10s Heartbeat Service
│   ├── per_session_predict.py     # ⭐ All-Ticker ML Prediction Engine
│   └── local_importer.py          # 5-Year Data Bootstrapping
├── src/
│   ├── ui/                        # Professional TUI Dashboard (Rich)
│   ├── ml/                        # DualModelTrainer (v3) + SignalGenerator
│   ├── llm/                       # Ollama/Groq/Gemini Multi-Provider
│   ├── streaming/                 # Kafka/Redis Infrastructure
│   └── historical/                # TimescaleDB Ingestion Logic
└── docker-compose.yml             # Full Infrastructure Stack
```

---

## 🔧 Development (Local)

```bash
# Setup Python environment
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt

# Start Infrastructure
docker compose up -d timescaledb redis chromadb kafka ollama

# 1. Bootstrapping (One-time)
python scripts/local_importer.py  # Load 5-year history

# 2. Start Background Services
start python scripts/live_heartbeat_sync.py
start python scripts/per_session_predict.py --loop 15

# 3. Launch TUI
python src/ui/dashboard.py FPT
```

---

## 📊 TUI Controls
- `python src/ui/dashboard.py <TICKER>`: Launch with specific symbol.
- **Auto-Sync**: Background services automatically detect your active ticker and prioritize it.
- **ML Insights**: Click or switch tickers to see updated quantiles and % max forecast.

---

## License
Private — Proprietary Vietnamese Stock Analysis System.
