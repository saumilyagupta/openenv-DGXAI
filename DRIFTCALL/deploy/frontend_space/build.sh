#!/usr/bin/env bash
# Build the DriftCall frontend (Vite + React + TS) to static files and
# bundle them as an HF Static SDK Space.
#
# Idempotent: runs `npm ci && npm run build` in ../../frontend/, then copies
# the produced dist/ + the Space card README.md into deploy/frontend_space/build/.
#
# Usage:
#   bash deploy/frontend_space/build.sh                 # build only
#   bash deploy/frontend_space/build.sh --push          # build + hf upload
#
# Env vars:
#   HF_SPACE_REPO  default: DGXAI/driftcall-site
#   HF_TOKEN       must be set when --push is used
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
FRONTEND_DIR="$REPO_ROOT/frontend"
SPACE_DIR="$REPO_ROOT/deploy/frontend_space/build"
HF_SPACE_REPO="${HF_SPACE_REPO:-DGXAI/driftcall-site}"

echo "[build] installing frontend deps (npm ci, may take a moment)"
cd "$FRONTEND_DIR"
if [[ ! -d node_modules ]]; then
    npm ci --no-audit --no-fund --silent
fi

echo "[build] vite build -> $FRONTEND_DIR/dist"
npm run build --silent

echo "[build] staging Space dir at $SPACE_DIR"
rm -rf "$SPACE_DIR"
mkdir -p "$SPACE_DIR"
cp -r "$FRONTEND_DIR/dist/." "$SPACE_DIR/"

# Space card + .gitignore (only the README ships, .gitignore is for our repo).
cp "$REPO_ROOT/deploy/frontend_space/README.md"   "$SPACE_DIR/README.md"

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
