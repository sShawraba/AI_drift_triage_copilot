# backend/ml_system/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy application code
COPY backend/ ./backend/
COPY data/ ./data/
COPY mlruns/ ./mlruns/
COPY shared/ ./shared/

# Create directory for predictions db
RUN mkdir -p /data

# Expose port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8001/api/v1/health || exit 1

# Run the application
CMD ["uv", "run", "uvicorn", "backend.ml_system.main:app", "--host", "0.0.0.0", "--port", "8001"]