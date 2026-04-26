# syntax=docker/dockerfile:1.6
# Unified DriftCall Space — same base + deps as env Space, plus the
# pre-built frontend dist/ mounted at root.

FROM python:3.11-slim AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

# Pre-pull TTS / ASR weights so the runtime container can run offline.
RUN pip install --prefix=/install huggingface_hub
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('hexgrad/Kokoro-82M', cache_dir='/weights'); \
               snapshot_download('Systran/faster-whisper-small', cache_dir='/weights')"

# -------- runtime --------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_PROJECT=driftcall \
    WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /weights /root/.cache/huggingface

WORKDIR /app

# Application code (cells/ + app.py + openenv.yaml + data/) and the
# pre-built frontend dist/ (mounted at / by unified_app.py).
COPY cells/        ./cells/
COPY data/         ./data/
COPY app.py openenv.yaml unified_app.py ./
COPY site/         ./site/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import urllib.request; \
                   urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4).read()" \
        || exit 1

# unified_app:app exposes both the OpenEnv routes (at root) and the
# static frontend (mounted at /).
CMD ["uvicorn", "unified_app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
