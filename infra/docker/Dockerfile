FROM python:3.12-slim

WORKDIR /app

# System dependencies for building native extensions (lightgbm, xgboost, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for Docker layer caching
COPY pyproject.toml ./
RUN pip install --no-cache-dir pip --upgrade && \
    pip install --no-cache-dir "."

COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Copy the full application
COPY . .

# Set PYTHONPATH so all modules resolve
ENV PYTHONPATH=/app

# Default command (overridden by docker-compose per-service)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8888"]
