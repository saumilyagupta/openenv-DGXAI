# syntax=docker/dockerfile:1.6
# Unified DriftCall Space — single image serving:
#   /                  static React site
#   /reset … /healthz  OpenEnv API
#   /demo              Gradio voice demo (mounted, not iframed)
#   /env /lora /source server-rendered HTML pages
#
# Heavier than the env-only Space because we bake the demo's model deps
# (transformers + peft + unsloth + torch + audio) so /demo runs locally.
# HF_HUB_OFFLINE is OFF so /demo can pull Gemma-3n + the LoRA on demand.

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

# Pre-pull TTS / ASR weights so the audio path is offline-fast.
# The Gemma base + LoRA stay online — too big to bake; they'll cache on
# first /demo session.
RUN PYTHONPATH=/install/lib/python3.11/site-packages \
    python -c "from huggingface_hub import snapshot_download; \
               snapshot_download('hexgrad/Kokoro-82M', cache_dir='/weights'); \
               snapshot_download('Systran/faster-whisper-small', cache_dir='/weights')"

# -------- runtime --------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    WANDB_PROJECT=driftcall \
    WANDB_MODE=disabled \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /weights /root/.cache/huggingface

WORKDIR /app

COPY cells/        ./cells/
COPY data/         ./data/
COPY scripts/      ./scripts/
COPY app.py demo_app.py openenv.yaml unified_app.py online_trainer.py ./
COPY site/         ./site/

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; \
                   urllib.request.urlopen('http://127.0.0.1:7860/healthz', timeout=4).read()" \
        || exit 1

# Single worker — both Gradio (stateful UI) and our SessionPool prefer
# single-process; horizontal scaling is for multi-instance deployments.
CMD ["uvicorn", "unified_app:app", \
     "--host", "0.0.0.0", \
     "--port", "7860", \
     "--workers", "1", \
     "--timeout-keep-alive", "30", \
     "--log-level", "info"]
