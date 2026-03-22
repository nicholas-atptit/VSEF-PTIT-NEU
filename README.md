# 📈 Algo Trading AI — Full Stack (Data + ML + LLM + Kafka)

> **Enterprise-grade Vietnamese stock market analysis system with 1,500 LightGBM models, Ollama LLM, Kafka message broker, and automated AM/PM data ingestion.**

---

## 🏗️ Architecture

```
VNStock API ──11:35 & 15:15──► [market.data.raw] ──► DB Writer (TimescaleDB)
                                                  ──► ML Consumer (1500 LightGBM)
                                                        ──► [ml.predictions] ──► LLM Consumer (Ollama)
                                                                                   ──► [llm.analysis] ──► Cache Writer
                                                                                                           ──► /predict API (sub-ms)
News Crawler ──► [market.news.raw] ──► News Embedder (ChromaDB)
```

## 🚀 One-Command Deployment (Server)

### Prerequisites
- Docker + Docker Compose installed
- NVIDIA GPU drivers (for Ollama LLM acceleration, optional)

### Start Everything
```bash
# Clone the repo on your server
git clone <repo-url> && cd AI-ML-LLM-Stock

# Start ALL services (DB, Kafka, LLM, API, Consumers)
docker compose up -d --build

# Pull the LLM model into Ollama
docker exec algo_ollama ollama pull qwen3:8b
```

That's it! The system will:
1. ✅ **Auto-start** TimescaleDB, ChromaDB, Kafka, Zookeeper, Kafka-UI, Ollama
2. ✅ **Auto-start** FastAPI server on port `8888`
3. ✅ **Auto-start** all 5 Kafka consumer daemons (with crash auto-restart)
4. ✅ **Auto-schedule** AM (11:35) and PM (15:15) data ingestion + news crawling
5. ✅ **Auto-restart** everything on server reboot (`unless-stopped` policy)

### Service Ports
| Service | Port | URL |
|---------|------|-----|
| FastAPI | 8888 | http://localhost:8888/api/v1/health |
| Kafka UI | 8080 | http://localhost:8080 |
| ChromaDB | 8000 | http://localhost:8000 |
| TimescaleDB | 5432 | postgresql://localhost:5432 |
| Ollama LLM | 11434 | http://localhost:11434 |

---

## 📂 Project Structure

```
├── data/                          # Persistent Data & Cache
│   ├── prediction_cache/          # Kafka-populated prediction cache
│   ├── listing/                   # Market symbols
│   └── hourly/                    # 1H price datasets
├── models/                        # 1,584 trained LightGBM models
├── scripts/                       # Entry Points
│   ├── run_consumers.py           # ⭐ Kafka Consumer Supervisor (5 daemons)
│   ├── run_stream.py              # WebSocket stream manager
│   ├── run_backdate.py            # Historical backfill
│   └── train_ml_tickers.py        # Model training pipeline
├── src/
│   ├── api/                       # FastAPI endpoints (/predict, /chat, /analyze)
│   ├── ml/                        # LightGBM models & feature engineering
│   ├── llm/                       # Ollama LLM pipeline
│   ├── streaming/                 # Kafka clients, scheduler, producers, consumers
│   │   ├── kafka_client.py        # KafkaPublisher + KafkaSubscriber
│   │   ├── scheduler.py           # AM/PM cron scheduler
│   │   ├── producers/             # Market data + news producers
│   │   └── consumers/             # 5 consumer daemons
│   ├── context/                   # RAG, news crawler, ChromaDB
│   └── historical/                # TimescaleDB services
├── docker-compose.yml             # ⭐ Full stack (8 services)
├── Dockerfile                     # Python app container
└── requirements.txt               # Pinned dependencies
```

---

## 🔧 Development (Local)

```bash
# Setup Python environment
python -m venv .venv312
.venv312\Scripts\activate  # Windows
pip install -r requirements.txt

# Start infra only
docker compose up -d timescaledb chromadb kafka zookeeper kafka-ui ollama

# Run API locally
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8888

# Run consumer daemons locally
python scripts/run_consumers.py

# Train ML models (one-time)
python scripts/train_ml_tickers.py
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/predict?ticker=SSI` | ML prediction (cache → live fallback) |
| GET | `/api/v1/analyze?ticker=SSI` | ML + LLM qualitative analysis |
| POST | `/api/v1/chat` | Conversational AI with real-time data |
| GET | `/api/v1/health` | Service health check |
| GET | `/api/v1/news/{ticker}` | Latest news for a ticker |

---

## License
Private — Internal use only.
