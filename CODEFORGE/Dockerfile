FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir ruff mypy pytest

# Application code
COPY codeforge/ codeforge/
COPY inference.py .
COPY pyproject.toml .

# Expose ports: FastAPI on 7860 (HF default), MCP SSE on 7861
EXPOSE 7860 7861

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python3 -c "import httpx; r = httpx.get('http://localhost:7860/'); assert r.status_code == 200"

# Start server
CMD ["uvicorn", "codeforge.app:app", "--host", "0.0.0.0", "--port", "7860"]
