<#
.SYNOPSIS
Khởi động hệ thống theo cấu hình Local-first sử dụng Ollama cho LLM trên Windows (PowerShell).
#>

Write-Host "=== 1/4 Khởi động hạ tầng Docker (Timescale, Chroma, Kafka, Redis, Ollama) ===" -ForegroundColor Cyan
docker compose up -d timescaledb chromadb zookeeper kafka redis ollama

Write-Host "=== 2/4 Đợi dịch vụ Ollama ổn định và kéo Model Qwen2.5:7b ===" -ForegroundColor Cyan
Start-Sleep -Seconds 5
Write-Host "Đang pull qwen2.5:7b (Có thể tốn vài phút nếu chạy máy mới)..." -ForegroundColor Yellow
docker exec -tt algo_ollama ollama pull qwen2.5:7b

Write-Host "=== 3/4 Chuẩn bị môi trường Python ===" -ForegroundColor Cyan
if (-not (Test-Path -Path ".venv312")) {
    Write-Host "Khởi tạo Virtual Environment .venv312..."
    python -m venv .venv312
}

Write-Host "Kích hoạt Virtual Environment..."
if (Test-Path -Path ".venv312\Scripts\Activate.ps1") {
    . "\.venv312\Scripts\Activate.ps1"
}

Write-Host "Cài đặt các gói phụ thuộc..."
pip install -r requirements.txt

Write-Host "=== 4/4 Thiết lập biến môi trường và chạy API ===" -ForegroundColor Cyan
# Đọc file .env.local_ollama và thiết lập biến môi trường
if (Test-Path -Path ".env.local_ollama") {
    Get-Content ".env.local_ollama" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $parts = $line -split '=', 2
            if ($parts.Count -eq 2) {
                $name = $parts[0].Trim()
                $value = $parts[1].Trim()
                [Environment]::SetEnvironmentVariable($name, $value, "Process")
            }
        }
    }
}

Write-Host "Khởi động FastAPI Server trên cổng 8888..." -ForegroundColor Green
uvicorn src.api.main:app --host 0.0.0.0 --port 8888
