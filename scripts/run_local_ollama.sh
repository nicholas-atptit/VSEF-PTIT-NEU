#!/bin/bash
# ==============================================================================
# Script: run_local_ollama.sh
# MÔ TẢ: Khởi động hệ thống theo cấu hình Local-first sử dụng Ollama cho LLM
# ==============================================================================

echo "=== 1/4 Khởi động hạ tầng Docker (Timescale, Chroma, Kafka, Redis, Ollama) ==="
docker compose up -d timescaledb chromadb zookeeper kafka redis ollama

echo "=== 2/4 Đợi dịch vụ Ollama ổn định và kéo Model Qwen2.5:7b ==="
# Đợi 5 giây để ollama container sẵn sàng
sleep 5
echo "Đang pull qwen2.5:7b (Có thể tốn vài phút nếu chạy máy mới)..."
docker exec -tt algo_ollama ollama pull qwen2.5:7b

echo "=== 3/4 Chuẩn bị môi trường Python ==="
if [ ! -d ".venv312" ]; then
    echo "Khởi tạo Virtual Environment .venv312..."
    python -m venv .venv312
fi

echo "Kích hoạt Virtual Environment..."
# Lưu ý: Tuỳ HĐH, đối với Windows (Git Bash/WSL) sử dụng `source .venv312/Scripts/activate`
# Đối với Linux/Mac: `source .venv312/bin/activate`
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    source .venv312/Scripts/activate
else
    source .venv312/bin/activate
fi

echo "Cài đặt các gói phụ thuộc..."
pip install -r requirements.txt

echo "=== 4/4 Thiết lập biến môi trường và chạy API ==="
# Load configuration local ollama
set -a
source .env.local_ollama
set +a

echo "Khởi động FastAPI Server trên cổng 8888..."
uvicorn src.api.main:app --host 0.0.0.0 --port 8888
