# 📈 Algo Trading AI — Hybrid Agentic Intelligence (v5.0.0)

> **Hệ thống giao dịch lai (Hybrid) thế hệ mới, tích hợp Tin tức (Sentiment) và Kỹ thuật (Quant) cho 104 mã (VN100 + Viettel Group). Sử dụng kiến trúc Multi-Agent Debate và Parkinson + Yang-Zhang Volatility Alpha.**

---

## 🏗️ Kiến Trúc Hệ Thống (V5 Agentic Hybrid)

Hệ thống hoạt động theo mô hình **Agentic Graph**, nơi dữ liệu được xử lý qua 4 miền (Domains) độc lập trước khi đưa ra quyết định cuối cùng:

1.  **Domain A: Technical Radar (Quantitative)**:
    - Sử dụng **LightGBM & XGBoost** tối ưu qua **Optuna**.
    - Alpha Factors mới: **Parkinson Volatility** (biên độ High-Low), **Yang-Zhang Volatility** (mới - tối ưu gap mở cửa), **Sentiment Momentum**, và 80+ chỉ số kỹ thuật khác.
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
| **ML Engine** | `lightgbm`, `xgboost`, `optuna` (Tuning), `scikit-learn` |
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
Huấn luyện 104 mã với bộ tham số tối ưu Alpha:
```powershell
python scripts/train_ml_tickers.py --tickers "AAA,ACB,VGI,..." --optuna
```

### Bước 3: Đồng bộ kết quả (Sync to DB)
Đẩy kết quả huấn luyện từ file models vào Dashboard API:
```powershell
python scripts/sync_predictions_to_db.py
```

### Bước 4: Vận hành Dashboard
Khởi động Terminal Dashboard để xem tín hiệu trực tiếp:
```powershell
python src/ui/dashboard.py VGI
```

---

## 📁 Cấu Trúc Thư Mục (Project Structure)

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
