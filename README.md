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

### Bước 2: Huấn luyện Mô hình Lai (Hybrid Training)
Huấn luyện **toàn bộ** mã trong thư mục data bằng 1 lệnh duy nhất:
```powershell
python scripts/train_ml_tickers.py --all --optuna
```
> Hoặc chỉ train một vài mã cụ thể: `python scripts/train_ml_tickers.py --tickers "SSI,HPG,VGI"`

### Bước 3: Đồng bộ kết quả (Sync to DB)
Đẩy kết quả huấn luyện từ file models vào Dashboard API:
```powershell
python scripts/sync_predictions_to_db.py
```

### Bước 4: Vận hành Dashboard
Khởi động Terminal Dashboard để xem tín hiệu trực tiếp:
```powershell
python src/ui/dashboard.py <TICKER>

# 6. Tương tác với AI Agent (Q&A)
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

## Technical ML Rebuild

The technical forecasting stack has been rebuilt around a single registry-driven pipeline.

- Supported algorithms: `cart`, `lstm`, `bilstm`
- Training window: the latest rolling 5 years available per ticker at execution time
- Split policy: chronological train/validation/test only, with horizon purge gaps
- Feature scaling: sequence-model scalers are fit on training sequences only
- Artifact root: `models/<TICKER>/`
- Manifest contract: `models/<TICKER>/manifest.json`

### Model Registry

- `src/ml/models/base.py`: shared ML model contract plus ORM base objects
- `src/ml/models/factory.py`: central registry and lazy model loading
- `src/ml/models/cart.py`: `DecisionTreeClassifier` and `DecisionTreeRegressor`
- `src/ml/models/lstm.py`: PyTorch LSTM implementation
- `src/ml/models/bilstm.py`: true bidirectional PyTorch LSTM
- `src/ml/sequence_dataset.py`: rolling-window builder for sequence models
- `src/ml/trainer.py`: unified training/inference facade used by CLI, API, batch inference, and backtests

### Data Window

For every ticker, the trainer:

1. finds the latest available trading date in the source CSV
2. keeps only the rolling 5-year window ending on that date
3. adds a warmup buffer before the 5-year start for indicator computation only
4. recomputes features from raw OHLCV inside that scope
5. logs the effective start date, effective end date, raw rows, indicator warmup rows, target rows lost, sequence rows lost, and final usable rows

### Artifact Contract

- CART trend classifier: `trend_classifier_cart_<horizon>.joblib`
- CART return regressor: `return_regressor_cart_<horizon>.joblib`
- LSTM trend classifier: `trend_classifier_lstm_<horizon>.pt`
- LSTM return regressor: `return_regressor_lstm_<horizon>.pt`
- BiLSTM trend classifier: `trend_classifier_bilstm_<horizon>.pt`
- BiLSTM return regressor: `return_regressor_bilstm_<horizon>.pt`
- Companion metadata: `*.meta.joblib`
- Torch scaler bundles: `*.scaler.joblib`
- Per-ticker manifest: `manifest.json`

The manifest stores the primary algorithm, feature columns, horizon metadata, calibration values for range reconstruction, and the exact data window used for that ticker.

### Commands

Install dependencies:

```powershell
pip install -r requirements.txt
```

If your default `python` is `3.13`, run the deep-model commands under a Python `3.12` interpreter because PyTorch support in this environment is on `3.12`.

Rebuild the 5-year feature dataset without training:

```powershell
python scripts/train_ml_tickers.py --tickers SSI --prepare-only --report reports/ssi_prepare_report.csv
```

Train CART:

```powershell
python scripts/train_ml_tickers.py --tickers SSI --algorithms cart --primary-algorithm cart
```

Train LSTM:

```powershell
python scripts/train_ml_tickers.py --tickers SSI --algorithms lstm --primary-algorithm lstm --sequence-length 20 --epochs 50 --batch-size 32
```

Train BiLSTM:

```powershell
python scripts/train_ml_tickers.py --tickers SSI --algorithms bilstm --primary-algorithm bilstm --sequence-length 20 --epochs 50 --batch-size 32
```

Train all selected algorithms:

```powershell
python scripts/train_ml_tickers.py --tickers SSI --algorithms cart,lstm,bilstm --primary-algorithm lstm --sequence-length 20 --epochs 50 --batch-size 32 --report reports/ssi_benchmark.csv
```

Run benchmark/report generation:

```powershell
python -m src.ml.benchmark.run --tickers SSI,HPG --algorithms cart,lstm,bilstm --sequence-length 20 --epochs 50 --batch-size 32 --report reports/ml_benchmark.csv
```

Run an inference smoke test:

```powershell
python -c "import pandas as pd; from src.ml.trainer import DualModelTrainer; df=pd.read_csv('data/daily_market_split_data/SSI.csv'); t=DualModelTrainer(); feat=t.compute_features_for_ticker('SSI', df); print(t.predict('SSI', feat, horizon='short'))"
```

Run the rebuilt ML tests:

```powershell
pytest tests/ml/test_sequence_dataset.py tests/ml/test_model_artifacts.py tests/ml/test_training_cli.py tests/ml/test_trainer_pipeline.py
```
