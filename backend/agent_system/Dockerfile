FROM python:3.12-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Copy dependency files first for caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only
RUN uv sync --frozen --no-dev

# Copy the rest of the source
COPY . .

# Run the FastAPI server
CMD ["uv", "run", "uvicorn", "backend.agent_system.main:app", "--host", "0.0.0.0", "--port", "8000"]