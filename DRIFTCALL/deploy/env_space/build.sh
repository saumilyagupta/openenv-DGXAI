#!/usr/bin/env bash
# Build a self-contained HF Space build dir for DriftCall env Space.
# Idempotent: rsyncs the canonical sources at REPO root into deploy/env_space/build/.
#
# Usage:
#   bash deploy/env_space/build.sh                 # build only
#   bash deploy/env_space/build.sh --push          # build + hf upload
#
# Env vars:
#   HF_SPACE_REPO  default: DGXAI/driftcall-env
#   HF_TOKEN       must be set when --push is used
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
SPACE_DIR="$REPO_ROOT/deploy/env_space/build"
HF_SPACE_REPO="${HF_SPACE_REPO:-DGXAI/driftcall-env}"

mkdir -p "$SPACE_DIR"

echo "[build] copying canonical sources -> $SPACE_DIR"

# App + manifest + Dockerfile (from canonical root)
cp "$REPO_ROOT/app.py"              "$SPACE_DIR/app.py"
cp "$REPO_ROOT/openenv.yaml"        "$SPACE_DIR/openenv.yaml"
cp "$REPO_ROOT/Dockerfile"          "$SPACE_DIR/Dockerfile"
cp "$REPO_ROOT/requirements.txt"    "$SPACE_DIR/requirements.txt"

# Importable modules (cells) + authored fixtures (data)
rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/cells/"  "$SPACE_DIR/cells/"
rsync -a --delete \
    "$REPO_ROOT/data/"   "$SPACE_DIR/data/"

# HF Space card (specific to env_space, lives only in deploy/)
cp "$REPO_ROOT/deploy/env_space/README.md"   "$SPACE_DIR/README.md"
cp "$REPO_ROOT/deploy/env_space/.gitignore"  "$SPACE_DIR/.gitignore"

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
