#!/usr/bin/env bash
# Build a SELF-CONTAINED unified DriftCall HF Space.
#
# Everything needed to run is copied into deploy/unified_space/build/.
# After this script runs, the build dir has zero references to anything
# outside it — you could `cd build && docker build .` and it would work,
# or zip it and host anywhere.
#
# Layout produced under build/:
#   app.py                  ← canonical OpenEnv FastAPI server (copy)
#   unified_app.py          ← extends app.py + static mount + /demo redirect
#   openenv.yaml            ← OpenEnv v1.0 manifest
#   requirements.txt        ← runtime deps (no training stack)
#   Dockerfile              ← multi-stage CPU image, Kokoro + whisper baked
#   cells/                  ← DriftCallEnv + 5 reward components + drift + …
#   data/                   ← briefs, drift patterns, API schemas
#   site/                   ← Vite-built frontend dist/
#   README.md               ← HF Space card (Docker SDK)
#   .gitignore
#
# Usage:
#   bash deploy/unified_space/build.sh                 # build only
#   bash deploy/unified_space/build.sh --push          # build + hf upload
#
# Env vars:
#   HF_SPACE_REPO  default: DGXAI/driftcall
#   HF_TOKEN       must be set when --push is used
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/../.. && pwd)"
SPACE_DIR="$REPO_ROOT/deploy/unified_space/build"
HF_SPACE_REPO="${HF_SPACE_REPO:-DGXAI/driftcall}"

# 1) Build the frontend so we have dist/ to vendor in.
FRONTEND_DIR="$REPO_ROOT/frontend"
echo "[build] vite build (frontend)"
cd "$FRONTEND_DIR"
if [[ ! -d node_modules ]]; then
    npm ci --no-audit --no-fund --silent
fi
npm run build --silent

# 2) Stage a fresh build dir.
echo "[build] staging $SPACE_DIR"
rm -rf "$SPACE_DIR"
mkdir -p "$SPACE_DIR"

# 3) Copy canonical sources from the repo root — everything below this
#    point is self-contained inside the build dir.
cp "$REPO_ROOT/app.py"            "$SPACE_DIR/app.py"
cp "$REPO_ROOT/openenv.yaml"      "$SPACE_DIR/openenv.yaml"

# Demo Gradio app — renamed to demo_app.py so unified_app.py can import it
# as a top-level module. The original lives at demo/app_gradio.py.
cp "$REPO_ROOT/demo/app_gradio.py"  "$SPACE_DIR/demo_app.py"

# Importable runtime modules (used by app.py at request time).
rsync -a --delete \
    --exclude '__pycache__' --exclude '*.pyc' \
    "$REPO_ROOT/cells/"  "$SPACE_DIR/cells/"
rsync -a --delete \
    "$REPO_ROOT/data/"   "$SPACE_DIR/data/"

# Project content browsable via the HF Space "Files" tab — not imported
# by the running container, but bundled so the Space is a complete
# project mirror (per user request).
EXTRA_CONTENT=(
    "scripts"      # training scripts: train_driftcall_grpo.py, train_full_gemma3n.sh, smoke_gemma3n_boot.py
    "notebooks"    # Colab-runnable notebook + builder
    "docs"         # 14 module specs + 14 test plans + DESIGN drafts
    "tests"        # pytest suite (35+ files)
)
for d in "${EXTRA_CONTENT[@]}"; do
    if [[ -d "$REPO_ROOT/$d" ]]; then
        rsync -a --delete \
            --exclude '__pycache__' --exclude '*.pyc' \
            --exclude '.pytest_cache' --exclude '.coverage' \
            --exclude '*.ipynb_checkpoints' \
            "$REPO_ROOT/$d/" "$SPACE_DIR/$d/"
    fi
done

# Top-level docs / metadata files. README.md is renamed to PROJECT_README.md
# so it doesn't shadow the Space-card README.md that HF uses for the page.
for f in DESIGN.md CLAUDE.md BLOG.md pyproject.toml; do
    if [[ -f "$REPO_ROOT/$f" ]]; then
        cp "$REPO_ROOT/$f" "$SPACE_DIR/$f"
    fi
done
if [[ -f "$REPO_ROOT/README.md" ]]; then
    cp "$REPO_ROOT/README.md" "$SPACE_DIR/PROJECT_README.md"
fi

# 4) Copy unified-specific files (requirements.txt OVERRIDES the root copy
#    because we add Gradio + model deps for the bundled /demo).
cp "$REPO_ROOT/deploy/unified_space/requirements.txt" "$SPACE_DIR/requirements.txt"
cp "$REPO_ROOT/deploy/unified_space/unified_app.py"   "$SPACE_DIR/unified_app.py"
cp "$REPO_ROOT/deploy/unified_space/online_trainer.py" "$SPACE_DIR/online_trainer.py"
cp "$REPO_ROOT/deploy/unified_space/Dockerfile"       "$SPACE_DIR/Dockerfile"
cp "$REPO_ROOT/deploy/unified_space/README.md"        "$SPACE_DIR/README.md"
cp "$REPO_ROOT/deploy/unified_space/.gitignore"       "$SPACE_DIR/.gitignore"

# 5) Vendor the pre-built frontend as ./site/ (mounted at root by unified_app).
mkdir -p "$SPACE_DIR/site"
cp -r "$FRONTEND_DIR/dist/." "$SPACE_DIR/site/"

echo "[build] done. tree:"
ls -la "$SPACE_DIR" | head -25
echo "[build] site/:"
ls "$SPACE_DIR/site/" | head -10

if [[ "${1:-}" == "--push" ]]; then
    : "${HF_TOKEN:?HF_TOKEN must be set to push}"
    echo "[push] hf upload -> $HF_SPACE_REPO"
    hf upload "$HF_SPACE_REPO" "$SPACE_DIR" \
        --repo-type=space \
        --token "$HF_TOKEN"
    echo "[push] done. live: https://huggingface.co/spaces/$HF_SPACE_REPO"
fi
