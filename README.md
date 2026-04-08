# 📈 Algo Trading AI — Hybrid Agentic Intelligence (v5.3.1)

> **Hệ thống giao dịch lai (Hybrid) thế hệ mới, tích hợp Tin tức (Sentiment) và Kỹ thuật (Quant) cho 104 mã (VN100 + Viettel Group). Sử dụng kiến trúc Multi-Agent Debate và Parkinson + Yang-Zhang Volatility Alpha.**

---

## 🏗️ Kiến Trúc Hệ Thống (V5 Agentic Hybrid)

Hệ thống hoạt động theo mô hình **Agentic Graph**, nơi dữ liệu được xử lý qua 4 miền (Domains) độc lập trước khi đưa ra quyết định cuối cùng:

1.  **Domain A: Technical Radar (Quantitative)**:
    - Sử dụng **CART (Classification And Regression Tree)** cho tốc độ và diễn giải, **LSTM/BiLSTM** cho mô hình hóa chuỗi thời gian.
    - **Cửa sổ dữ liệu**: Rolling 5 năm mới nhất, được tính toán lại tại mỗi lần huấn luyện.
    - Alpha Factors: **Parkinson Volatility** (biên độ High-Low), **Yang-Zhang Volatility** (tối ưu gap mở cửa), **Sentiment Momentum**, và 80+ chỉ số kỹ thuật khác.
2.  **Domain B: Sentiment Intelligence (Qualitative)**:
    - **News Crawler**: Quét tin tức thời gian thực từ Google News RSS & Vnstock cho 104 mã.
    - **Active Analysis Path**: Tự động kích hoạt crawl & analyze on-demand nếu dữ liệu cache thiếu/stale.
    - **LLM Engine (Ollama/Gemini)**: Phân tích 104 mã dựa trên bộ từ khóa tối ưu tại `news_keywords_baseline.csv`.
3.  **Domain C: Fused Decision Matrix (The Brain)**:
    - **Hybrid Fusion**: Kết hợp dự báo ML với điểm cảm xúc LLM theo trọng số động.
    - **Multi-Agent Debate (MAD)**: Luồng tranh biện Bull vs Bear để lọc nhiễu tín hiệu.
    - **Risk Veto Logic**: Quyền phủ quyết (Kill Switch) từ Risk Management Agent khi có rủi ro vĩ mô.
4.  **Domain D: Operational Interface**:
    - **API v2**: Cung cấp dữ liệu theo Domain-driven Design (Technical, Sentiment, Fusion).
    - **TUI Dashboard (v5.1)**: Giao diện Terminal chuyên dụng với thanh **Hybrid Pulse Monitor**.

---

## 💾 Hệ Thống Dữ Liệu (Data Engine)

Hệ thống lưu trữ dữ liệu tại 3 lớp chính:

### 1. TimescaleDB (PostgreSQL) — "Linh hồn" hệ thống
- `raw_prices`: Lưu trữ OHLCV lịch sử 5 năm cho 1,600+ mã.
- `news_intelligence`: Lưu trữ kết quả phân tích tin tức (Summary, Sentiment Score, Market Trend).
- `agent_predictions`: Lưu trữ các dự báo "Hybrid" mới nhất (Label, Confidence, Target Price).

### 2. Dữ liệu Feature & Report (CSV/Local)
- `data/sentiment_features.csv`: Cầu nối từ tin tức sang các đặc trưng ML định lượng.
- `reports/performance_universal_v4.csv`: Báo cáo độ chính xác của 104 mã vừa được huấn luyện.

### 3. Cấu trúc Tri thức (Knowledge Base)
- `reports/news_keywords_baseline.csv`: Danh mục 104 mã (VN100 + Viettel) với các từ khóa tìm kiếm tiếng Việt/tiếng Anh chuẩn hóa.

---

## 📚 Thư Viện Sử Dụng (Tech Stack)

| Nhóm | Thư viện chính |
| :--- | :--- |
| **ML Engine** | `scikit-learn` (CART), `torch` (LSTM/BiLSTM), `joblib` (artifact persistence) |
| **Data Logic** | `pandas`, `pandas_ta` (Technical Indicators), `sqlalchemy` (asyncio) |
| **AI/LLM** | `ollama` (Local LLM), `langchain` (Orchestrator), `httpx` |
| **Data Source** | `vnstock_pro` (Official Data), `beautifulsoup4` (Crawling) |
| **UI/UX** | `rich` (Terminal Layout), `pydantic` (Schema Validation) |
| **Infrastructure** | `timescaledb` (Docker), `redis` (High-speed price cache) |

---

## 🚀 Hướng Dẫn Sử Dụng (Usage Guide)

### Bước 1: Thu thập Trí tuệ Tin tức (News Ingestion)
Chạy script quét tin tức cho toàn bộ 104 mã trọng tâm:
```powershell
python scripts/run_news_crawler.py
```

### Bước 2: Huấn luyện Mô hình Kỹ thuật (ML Training)
Huấn luyện các mô hình CART, LSTM, BiLSTM trên cửa sổ dữ liệu rolling 5 năm.

**Train CART cho tất cả mã VN100:**
```powershell
python scripts/train_ml_tickers.py --vn100 --algorithms cart --primary-algorithm cart
```

**Train LSTM với tuning cơ bản:**
```powershell
python scripts/train_ml_tickers.py --tickers "SSI,HPG,VGI" --algorithms lstm --primary-algorithm lstm --sequence-length 20 --epochs 50 --batch-size 32
```

**Train cả ba thuật toán trên một mã:**
```powershell
python scripts/train_ml_tickers.py --tickers "SSI" --algorithms cart,lstm,bilstm --primary-algorithm lstm --sequence-length 20 --epochs 50
```

**Chỉ BUILD features (không train model):**
```powershell
python scripts/train_ml_tickers.py --tickers "SSI" --prepare-only
```

### Bước 3: Vận hành Dashboard
Khởi động Terminal Dashboard để xem tín hiệu trực tiếp:
```powershell
python src/ui/dashboard.py <TICKER>
```

### Bước 4: Tương tác với AI Agent (Q&A)
Gửi câu hỏi trực tiếp cho hệ thống qua API (hoặc tích hợp vào Chat Terminal):
```bash
# Sử dụng Curl để hỏi về mã VGI
curl -X POST http://127.0.0.1:8005/api/v2/chat `
     -H "Content-Type: application/json" `
     -d '{"message": "Hãy phân tích tin tức và kỹ thuật mã VGI", "history": []}'
```

---

## 💬 Hệ thống Phân tích Tương tác (AI Agent Q&A)

Điểm nâng cấp của Phase 5 là hệ thống **RAG-based Chat**. AI Agent không chỉ trả lời dựa trên kiến thức chung mà còn truy vấn trực tiếp cơ sở dữ liệu để đưa ra thông tin thực tế:

- **Tự động nhận diện Ticker**: Khi bạn hỏi về một mã (VD: "SSI có tốt không?"), Agent tự động nhận diện `SSI`.
- **Truy vấn Đa tầng**:
    - **Tầng 1 (News)**: Lấy tóm tắt tin tức và điểm Sentiment mới nhất từ bảng `news_intelligence`.
    - **Tầng 2 (Quant)**: Lấy dự báo kỹ thuật và giá mục tiêu từ bảng `agent_predictions`.
- **Context Fusion**: Agent tổng hợp hai luồng thông tin trên để đưa ra lời khuyên "Hybrid" chuẩn xác nhất.

---

## 📂 Cấu Trúc Thư Mục (Project Structure)

```
├── data/                          # Data Cache & Sentiment Features
├── models/                        # Joblib models cho 104 mã (v4 Hybrid)
├── scripts/                       # Scripts vận hành chính
│   ├── run_news_crawler.py        # Quét tin tức (Domain B)
│   ├── train_ml_tickers.py        # Huấn luyện Hybrid (Domain A)
│   └── sync_predictions_to_db.py  # Đồng bộ API/DB
├── src/
│   ├── api/                       # Domain v2 (REST API FastAPI)
│   ├── ml/                        # Parkinson & Yang-Zhang Volatility
│   ├── engine/                    # Multi-Agent Debate Logic
│   └── ui/                        # TUI Dashboard (v5.1)
├── reports/                       
│   ├── news_keywords_baseline.csv # Danh mục 104 mã & Keywords
│   └── performance_universal_v4.csv # Accuracy report (Hybrid)
└── requirements.txt               
```

---

## 🧬 Inspiration & Credits
Hệ thống được thiết kế dựa trên triết lý từ các nghiên cứu:
- **TradingAgents**: Multi-agent orchestration.
- **FinRL-X**: Alpha Research & Deep Reinforcement Learning frameworks.
- **Machine Learning for Trading (Stefan Jansen)**.

---

## License
Private — Proprietary Vietnamese Stock Analysis System - Lương Minh Quân.
v5.1.3 — Universal Hybrid Edition.
---

## Technical ML Architecture (Domain A — Quantitative)

The technical forecasting stack has been rebuilt around a **single manifest-driven registry pipeline** supporting three production algorithms.

### Supported Algorithms

| Algorithm | Type | Framework | Sequences? | Artifact format |
|-----------|------|-----------|------------|-----------------|
| **CART** | Classification & Regression Tree | scikit-learn | No | `.joblib` |
| **LSTM** | Unidirectional RNN | PyTorch | Yes | `.pt` file + metadata |
| **BiLSTM** | Bidirectional RNN | PyTorch | Yes | `.pt` file + metadata |

### Data & Feature Engineering

**Rolling 5-year window:**
- The trainer auto-detects the latest trading date in your source CSV
- Keeps only the 5-year window ending on that date (recalculated every run)
- Adds a ~180-day warmup before the window for indicator computation only
- Recomputes all features from raw OHLCV within the scope
- Prevents data leakage via chronological train/validation/test split

**Features:**
- 80+ technical indicators: RSI, MACD, Bollinger, Parkinson Volatility, Yang-Zhang Volatility, Sentiment Momentum
- Context features: market return, sector return, relative performance
- Auto-computed via `src/ml/feature_engineering.py` + `src/ml/features/`

### Artifact & Manifest System

**Per-ticker directory:** `models/<TICKER>/`

**Files present after training:**
```
models/SSI/
├── manifest.json                              # Central metadata contract
├── trend_classifier_cart_short.joblib         # CART classifier
├── return_regressor_cart_short.joblib         # CART regressor
├── trend_classifier_lstm_short.pt             # LSTM weights (PyTorch)
├── trend_classifier_lstm_short.meta.joblib    # LSTM config
├── trend_classifier_lstm_short.scaler.joblib  # Feature scaler (fit on train only)
├── [same for return_regressor_lstm_short.*]
└── [same pattern for bilstm]
```

**What `manifest.json` contains:**
- `schema_version` – artifact format compatibility  
- `primary_algorithm` – default inference algorithm  
- `feature_columns` – exact feature names (prevents misalignment)  
- `data_window.start` / `data_window.end` – reproducible data scope  
- `raw_stats` – row counts per ticker  
- `horizons.<horizon>.algorithms.<algorithm>` – per-algo metrics, calibration, artifact files

### Inference Expected Behavior

**For CART models:**
- Input: a row or small dataframe with required feature columns
- Output: class probabilities (trend: up/down/sideways) + expected range
- Latency: ~1-5ms per sample on CPU
- Handles: single-row prediction or batch

**For LSTM/BiLSTM models:**
- Input: historical feature sequences (configurable window, default 20 trading days)
- Output: same as CART (probabilities + range)
- Latency: ~10-50ms per sample on CPU
- **Error if insufficient history:** raises `ValueError("Insufficient history...")`

**Via the InferenceEngine facade:**
- Manifest auto-discovery via `models/<TICKER>/manifest.json`
- Automatic model loading based on primary_algorithm
- Batch-friendly interface for multiple tickers
- Clear error messages on missing artifacts or insufficient history

### Supported Training Commands

**Install dependencies:**
```powershell
pip install -r requirements.txt
```
> Python 3.12+ recommended. If stuck on 3.13, use a 3.12 venv for LSTM/BiLSTM.

**Basic usage:**
```powershell
python scripts/train_ml_tickers.py \
  --tickers "SSI" \
  --algorithms cart,lstm,bilstm \
  --primary-algorithm lstm \
  --sequence-length 20 \
  --epochs 50 \
  --batch-size 32
```

**CLI arguments:**
- `--tickers` – comma-separated symbols (default: required unless --all or --vn100)
- `--all` – train every .csv in `data/daily_market_split_data/`
- `--vn100` – train current dynamic VN100 universe
- `--algorithms` – comma-separated: `cart`, `lstm`, `bilstm` (default: `cart`)
- `--primary-algorithm` – which to use by default in inference
- `--sequence-length` – rolling window size for LSTM/BiLSTM (default: 20)
- `--hidden-size` – RNN hidden dimension (default: 64)
- `--num-layers` – RNN layer count (default: 2)
- `--dropout` – RNN dropout fraction (default: 0.2)
- `--learning-rate` – optimizer lr for RNN (default: 1e-3)
- `--batch-size` – training batch size (default: 32)
- `--epochs` – max RNN training epochs (default: 30)
- `--patience` – early stopping patience (default: 5)
- `--max-depth` – CART max tree depth (default: None = unlimited)
- `--min-samples-split` – CART split threshold (default: 2)
- `--min-samples-leaf` – CART leaf threshold (default: 1)
- `--prepare-only` – build features without training models
- `--report` – output benchmark CSV path (default: `reports/ml_benchmark.csv`)

**Example commands:**

Rebuild feature dataset only:
```powershell
python scripts/train_ml_tickers.py --tickers SSI --prepare-only
```

Train CART on entire VN100:
```powershell
python scripts/train_ml_tickers.py --vn100 --algorithms cart
```

Train LSTM with custom params:
```powershell
python scripts/train_ml_tickers.py --tickers "SSI,HPG,VGI" \
  --algorithms lstm \
  --primary-algorithm lstm \
  --sequence-length 20 \
  --hidden-size 128 \
  --epochs 100 \
  --batch-size 16
```

Train all algorithms and compare:
```powershell
python scripts/train_ml_tickers.py --tickers "SSI" \
  --algorithms cart,lstm,bilstm \
  --primary-algorithm bilstm \
  --report reports/ssi_full_benchmark.csv
```

### Testing the ML System

Run all ML tests:
```powershell
pytest tests/ml/ -v
```

Run specific test suites:
```powershell
pytest tests/ml/test_training_cli.py -v                        # CLI parsing
pytest tests/ml/test_trainer_pipeline.py -v                    # Training + inference
pytest tests/ml/test_sequence_dataset.py -v                    # Sequence building
pytest tests/ml/test_model_artifacts.py -v                     # Artifact save/load
```

Quick smoke test of inference:
```powershell
python -c "
from src.ml.trainer import DualModelTrainer
from src.ml.data_loader import generate_mock_data
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    trainer = DualModelTrainer(model_dir=tmpdir)
    df = generate_mock_data(ticker='TEST', num_days=900)
    result = trainer.train(ticker='TEST', df=df, algorithms=['cart'], horizons=['short'])
    print(f'✅ Training: {result[\"ticker\"]} OK')
    
    features = trainer.compute_features_for_ticker('TEST', df)
    pred = trainer.predict('TEST', features, horizon='short')
    print(f'✅ Inference: algorithm={pred[\"algorithm\"]}, trend={list(pred.get(\"trend_probabilities\", {}).keys())}')
"
```
