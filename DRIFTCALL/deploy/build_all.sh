#!/usr/bin/env bash
# Build (and optionally push) every DriftCall deployment target.
#
# Targets:
#   - deploy/env_space/    OpenEnv REST env Space          (Docker SDK)
#   - deploy/demo_space/   Voice-first Gradio demo Space   (Gradio SDK)
#   - deploy/inference/    OpenEnv gym client (no build)   (CLI module)
#
# Usage:
#   bash deploy/build_all.sh            # build env_space + demo_space
#   bash deploy/build_all.sh --push     # build + push both Spaces
#
# Env vars:
#   HF_SPACE_ENV_REPO   default: DGXAI/driftcall-env
#   HF_SPACE_DEMO_REPO  default: DGXAI/driftcall-demo
#   HF_TOKEN            required for --push
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
PUSH_FLAG=""
[[ "${1:-}" == "--push" ]] && PUSH_FLAG="--push"

echo "================================================================"
echo "[build_all] env_space"
echo "================================================================"
HF_SPACE_REPO="${HF_SPACE_ENV_REPO:-DGXAI/driftcall-env}" \
    bash "$REPO_ROOT/deploy/env_space/build.sh" $PUSH_FLAG

echo
echo "================================================================"
echo "[build_all] demo_space"
echo "================================================================"
HF_SPACE_REPO="${HF_SPACE_DEMO_REPO:-DGXAI/driftcall-demo}" \
    bash "$REPO_ROOT/deploy/demo_space/build.sh" $PUSH_FLAG

echo
echo "================================================================"
echo "[build_all] frontend_space"
echo "================================================================"
HF_SPACE_REPO="${HF_SPACE_SITE_REPO:-DGXAI/driftcall-site}" \
    bash "$REPO_ROOT/deploy/frontend_space/build.sh" $PUSH_FLAG

echo
echo "================================================================"
echo "[build_all] unified_space"
echo "================================================================"
HF_SPACE_REPO="${HF_SPACE_UNIFIED_REPO:-DGXAI/driftcall}" \
    bash "$REPO_ROOT/deploy/unified_space/build.sh" $PUSH_FLAG

echo
echo "================================================================"
echo "[build_all] DONE"
echo "================================================================"
echo "Unified:    https://huggingface.co/spaces/${HF_SPACE_UNIFIED_REPO:-DGXAI/driftcall}"
echo "Env Space:  https://huggingface.co/spaces/${HF_SPACE_ENV_REPO:-DGXAI/driftcall-env}"
echo "Demo Space: https://huggingface.co/spaces/${HF_SPACE_DEMO_REPO:-DGXAI/driftcall-demo}"
echo "Site:       https://huggingface.co/spaces/${HF_SPACE_SITE_REPO:-DGXAI/driftcall-site}"
echo "LoRA:       https://huggingface.co/DGXAI/gemma-3n-e2b-driftcall-lora"
echo
echo "Inference smoke (run from repo root):"
echo "  python -m deploy.inference.run --num-episodes 1 --seed 42 --curriculum-stage 2"
