# syntax=docker/dockerfile:1.6
# DriftCall Env Space — multi-stage CPU-only image.
# Target final image: < 2 GB (DESIGN.md Risk 10, deploy_env_space.md §4.2).

# -------- Stage 1: builder --------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Build-time system deps (dropped from the runtime stage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt

# Pre-pull the Kokoro TTS and faster-whisper ASR weights into /weights so the
# runtime container can run HF_HUB_OFFLINE=1 without any network access.
RUN pip install --prefix=/install huggingface_hub
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('hexgrad/Kokoro-82M', cache_dir='/weights'); \
               snapshot_download('Systran/faster-whisper-small', cache_dir='/weights')"

# -------- Stage 2: runtime --------
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_OFFLINE=1 \
    WANDB_PROJECT=driftcall \
    WANDB_MODE=disabled

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        ffmpeg \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /weights /root/.cache/huggingface

WORKDIR /app

# Application code — copy the notebook cells (importable modules), the
# FastAPI entrypoint, the OpenEnv manifest, and any authored fixtures.
COPY cells/ ./cells/
COPY app.py openenv.yaml ./
COPY data/ ./data/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s \
    CMD python -c "import urllib.request; \
                   urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4).read()" \
        || exit 1

CMD ["uvicorn", "app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "2", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
