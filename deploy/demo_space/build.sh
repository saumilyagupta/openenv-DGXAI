#!/usr/bin/env bash
# Build a self-contained HF Space build dir for the DriftCall Gradio demo.
# Idempotent: rsyncs canonical sources (demo/app_gradio.py, cells/, data/)
# into deploy/demo_space/build/ and adds a Gradio-SDK Space card.
#
# Usage:
#   bash deploy/demo_space/build.sh                 # build only
#   bash deploy/demo_space/build.sh --push          # build + hf upload
#
# Env vars:
#   HF_SPACE_REPO  default: DGXAI/driftcall-demo
#   HF_TOKEN       must be set when --push is used
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
SPACE_DIR="$REPO_ROOT/deploy/demo_space/build"
HF_SPACE_REPO="${HF_SPACE_REPO:-DGXAI/driftcall-demo}"

mkdir -p "$SPACE_DIR"

echo "[build] copying canonical sources -> $SPACE_DIR"

# Standalone Gradio entrypoint at the repo top-level (the demo SDK convention).
cp "$REPO_ROOT/demo/app_gradio.py"   "$SPACE_DIR/app.py"

# Importable modules + authored fixtures (env, models, drift, audio, …).
rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/cells/"  "$SPACE_DIR/cells/"
rsync -a --delete \
    "$REPO_ROOT/data/"   "$SPACE_DIR/data/"

# Space-card README + .gitignore + requirements.txt live only in deploy/.
cp "$REPO_ROOT/deploy/demo_space/README.md"          "$SPACE_DIR/README.md"
cp "$REPO_ROOT/deploy/demo_space/.gitignore"         "$SPACE_DIR/.gitignore"
cp "$REPO_ROOT/deploy/demo_space/requirements.txt"   "$SPACE_DIR/requirements.txt"

echo "[build] done. files in $SPACE_DIR:"
ls -la "$SPACE_DIR" | head -20

if [[ "${1:-}" == "--push" ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be set to push}"
    echo "[push] hf upload -> $HF_SPACE_REPO"
    hf upload "$HF_SPACE_REPO" "$SPACE_DIR" \
        --repo-type=space \
        --token "$HF_TOKEN"
    echo "[push] done. live: https://huggingface.co/spaces/$HF_SPACE_REPO"
fi
