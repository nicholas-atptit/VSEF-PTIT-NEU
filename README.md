# 📈 Algo Trading AI — Hệ Thống Phân Tích Chứng Khoán Chuyên Nghiệp (v4.6.0)

> **Hệ thống phân tích thị trường chứng khoán Việt Nam cấp doanh nghiệp, tích hợp 1,600+ mô hình LightGBM, LLM Ollama, Kafka Message Broker và Dashboard TUI (Terminal UI) hiệu năng cao.**

---

## 🏗️ Cấu Trúc Hệ Thống (System Architecture)

Hệ thống được thiết kế theo mô hình 5 lớp (5-Layer Architecture) để đảm bảo tính thời gian thực và khả năng mở rộng:

1.  **Lớp Thu Thập (Ingestion Layer)**: 
    - Sử dụng `vnstock_pro` để lấy dữ liệu OHLCV và News với tốc độ 300 request/phút.
    - Kết nối WebSocket để nhận dữ liệu live-tick.
2.  **Lớp Vận Chuyển (Transport Layer)**:
    - **Kafka**: Điều phối dữ liệu giữa các thành phần (Market Data -> ML Inference -> LLM Analysis).
    - **Redis**: Bộ nhớ đệm (Cache) tốc độ cực cao O(1) để TUI truy xuất giá tức thời.
3.  **Lớp Lưu Trữ (Storage Layer)**:
    - **TimescaleDB**: Cơ sở dữ liệu chuỗi thời gian (Time-series) tối ưu cho hàng tỷ bản ghi giá. 
    - Sử dụng **Hypertables** cho `raw_prices` và `adjusted_prices` với lịch sử 5 năm.
4.  **Lớp Trí Tuệ (Intelligence Layer)**:
    - **ML Core**: 1,625 mô hình LightGBM chạy song song, dự báo xu hướng (Trend) và Biên độ (Quantile Regression).
    - **LLM Core**: Sử dụng Ollama (Qwen-3) để phân tích tâm lý thị trường và tin tức theo SOP (Standard Operating Procedure).
5.  **Lớp Hiển Thị (Interface Layer)**:
    - **TUI Dashboard (v4.6)**: Giao diện dòng lệnh chuyên nghiệp, cập nhật 0.5s/lần, hỗ trợ On-demand Sync.

---

## 📊 Chỉ Số Chứng Khoán & Feature Engineering

Hệ thống sử dụng bộ chỉ số kỹ thuật nâng cao được tính toán tự động qua `FeatureEngineer` và `HeuristicEngine`:

### 1. Nhóm Chỉ Số Biến Động (Volatility)
- **ATR (14)**: Đo lường độ biến động trung bình của giá.
- **Historical Volatility (HV-20)**: Biến động lịch sử được chuẩn hóa theo năm.
- **Bollinger Bandwidth (5, 20, 60)**: Độ rộng dải Bollinger giúp xác định các điểm "nén" giá để bùng nổ.

### 2. Nhóm Chỉ Số Động Lượng (Momentum)
- **RSI (14)**: Chỉ số sức mạnh tương đối để xác định vùng quá mua/quá bán.
- **ROC (5, 20, 60)**: Tốc độ thay đổi giá qua các khung thời gian Ngắn/Trung/Dài hạn.
- **Price Gap**: Phân tích các khoảng trống giá (Gaps) khi mở phiên.

### 3. Nhóm Chỉ Số Cấu Trúc (Structure)
- **Pivot Points (Classic)**: Xác định các mức xoay chiều của thị trường.
- **Support & Resistance (S1/R1)**: Các ngưỡng hỗ trợ và kháng cự cứng dựa trên khung D1.
- **Dist to Pivot**: Khoảng cách từ giá hiện tại đến các ngưỡng kỹ thuật để tính điểm Entry.

### 4. Nhóm Chỉ Số Khối Lượng (Volume)
- **Money Flow Ratio**: Dòng tiền luân chuyển giữa các mã.
- **Volume Surge**: Cảnh báo bùng nổ khối lượng khi Vol hiện tại vượt mức trung bình 20 phiên.
- **OBV (On-Balance Volume)**: Sự tương quan giữa giá và khối lượng để xác nhận xu hướng.

### 5. Dự Báo Nâng Cao (ML Forecast)
- **Max Upside (90th Quantile)**: Mục tiêu giá tối đa dự kiến trong phiên.
- **Max Downside (10th Quantile)**: Ngưỡng rủi ro tối đa dựa trên xác suất 90%.

---

## 🔧 Hướng Dẫn Vận Hành (Quick Start)

```bash
# 1. Cài đặt môi trường
python -m venv .venv312
.\.venv312\Scripts\activate
pip install -r requirements.txt

# 2. Khởi động hạ tầng (Docker)
docker compose up -d

# 3. Nạp dữ liệu lịch sử 5 năm (Chỉ chạy 1 lần đầu)
python scripts/local_importer.py

# 4. Chạy các dịch vụ ngầm
# Chạy Live Sync (10s/vòng)
start python scripts/live_heartbeat_sync.py
# Chạy ML Engine (Cập nhật dự báo toàn thị trường)
start python scripts/per_session_predict.py --loop 15

# 5. Mở Dashboard
python src/ui/dashboard.py <TICKER>
```

---

## 📂 Cấu Trúc Thư Mục (Project Structure)

```
├── data/                          # Dữ liệu & Cache dự báo
├── models/                        # 1,625 file model LightGBM (.joblib)
├── scripts/                       # Các entry points vận hành hệ thống
│   ├── live_heartbeat_sync.py     # Đồng bộ giá thời gian thực
│   ├── per_session_predict.py     # Engine dự báo ML song song
│   └── local_importer.py          # Script nạp dữ liệu 5 năm
├── src/
│   ├── ui/                        # TUI Dashboard (Rich)
│   ├── ml/                        # Feature Engineering & Trainer
│   ├── llm/                       # Pipeline AI phân tích tin tức
│   └── historical/                # Logic xử lý TimescaleDB
└── requirements.txt               # Danh sách thư viện cần thiết
```

---

## 🚀 Quick Update Commands

If you need to manually refresh the system data, use these specialized commands:

### 🔄 1. Cập nhật Toàn bộ (Universal Update)
Lệnh này là **"Tất cả trong một"**. Nó sẽ thực hiện 3 bước: Cập nhật Giá mới nhất -> Chạy dự báo ML -> Thu thập Tin tức mới:
```powershell
.\.venv312\Scripts\python.exe scripts/update_market.py
```

### 📰 2. Cập nhật Tin tức (Manual News)
Lệnh này sẽ chủ động quét tin tức mới cho danh mục theo dõi:
```powershell
.\.venv312\Scripts\python.exe scripts/update_news.py
# Hoặc cập nhật riêng cho các mã cụ thể:
.\.venv312\Scripts\python.exe scripts/update_news.py FPT VGI VHM
```

### ⌚ 3. Chế độ Chờ Cuối Phiên (Session Standby)
Để hệ thống tự động dậy và quét dự báo vào lúc 11:35 và 15:15:
```powershell
.\.venv312\Scripts\python.exe scripts/per_session_predict.py --session
```

### 💓 4. Đồng bộ giá thời thực (Live Heartbeat)
Giữ giá nhảy liên tục trên Dashboard (chu kỳ 10s):
```powershell
.\.venv312\Scripts\python.exe scripts/live_heartbeat_sync.py
```

### ⚡ 5. Manual Update (Full System Refresh)
Dùng lệnh này để ép hệ thống cập nhập Giá + ML + News bất cứ lúc nào:
- **Mặc định (Top-10)**:
  ```powershell
  .\.venv312\Scripts\python.exe scripts/update_market.py
  ```
- **Cập nhật TOÀN BỘ thị trường (1,500+ mã)**:
  ```powershell
  .\.venv312\Scripts\python.exe scripts/update_market.py --all
  ```
- **Lấp hố dữ liệu cũ (VD: Quét lại 30 ngày qua)**:
  ```powershell
  .\.venv312\Scripts\python.exe scripts/update_market.py --all --days 30
  ```

---

## License
Private — Proprietary Vietnamese Stock Analysis System - In custody of Lương Minh Quân.
