#!/usr/bin/env bash
# DriftCall — build + push the training image to Docker Hub.
#
# Run this from /home/krrish/Desktop/Project/openenv-DGXAI/DRIFTCALL after
# you've pushed your local commits to GitHub. The script:
#
#   1. Verifies git tree is clean and synced with origin/main
#   2. Logs into Docker Hub (interactive — uses your saved creds if any)
#   3. Builds krrishchoudhary109/driftcall-train:v1 (and :latest)
#   4. Pushes both tags
#   5. Prints the SDL line you should verify
#
# Usage:
#   ./scripts/build_and_push.sh
#   ./scripts/build_and_push.sh --skip-push          # build only
#   ./scripts/build_and_push.sh --tag v2             # custom tag
#   ./scripts/build_and_push.sh --user yourdockeruser
#   GITHUB_TOKEN=ghp_... ./scripts/build_and_push.sh # for private repo clone
#   HF_TOKEN_BUILD=hf_... ./scripts/build_and_push.sh # pre-pull Gemma weights

set -euo pipefail

DOCKER_USER="${DOCKER_USER:-krrishchoudhary109}"
TAG="${TAG:-v1}"
PUSH=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-push) PUSH=0; shift ;;
        --tag)       TAG="$2"; shift 2 ;;
        --user)      DOCKER_USER="$2"; shift 2 ;;
        -h|--help)
            grep -E "^# " "$0" | head -25 | sed 's/^# //'
            exit 0
            ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

IMAGE="${DOCKER_USER}/driftcall-train"

echo "==> DriftCall train-image build"
echo "    image:   ${IMAGE}:${TAG} (+ :latest)"
echo "    push:    ${PUSH}"
echo "    DOCKER_HOST: ${DOCKER_HOST:-default}"
echo

# 1. Confirm we're in DRIFTCALL/
if [[ ! -f Dockerfile.train ]]; then
    echo "ERROR: run from DRIFTCALL/ directory (Dockerfile.train missing)"
    exit 1
fi

# 2. Git pre-flight
echo "==> Git status"
git fetch origin --quiet || true
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "unknown")
if [[ "$LOCAL" != "$REMOTE" ]]; then
    echo "WARNING: local HEAD ($LOCAL) != origin/main ($REMOTE)"
    AHEAD=$(git rev-list --count origin/main..HEAD 2>/dev/null || echo "?")
    echo "  $AHEAD local commits not on origin/main"
    echo "  The Dockerfile clones from origin/main inside the build —"
    echo "  these unpushed commits will NOT be in the resulting image."
    read -p "  Push to origin/main now? [y/N] " yn
    if [[ "$yn" =~ ^[Yy]$ ]]; then
        git push origin HEAD:main
    else
        echo "  Continuing without push (image will be built from origin/main as-is)."
    fi
fi

DIRTY=$(git status --porcelain | wc -l | tr -d ' ')
if [[ "$DIRTY" -gt 0 ]]; then
    echo "WARNING: $DIRTY dirty / untracked files. They won't be in the image (clone-based build)."
    git status --short | head -10
    echo
fi

# 3. Docker daemon
echo "==> Docker daemon"
docker info > /dev/null 2>&1 || {
    echo "ERROR: docker daemon not reachable. Try DOCKER_HOST=unix:///var/run/docker.sock $0"
    exit 1
}
docker info | grep -E "Server Version|Storage Driver" | head -2

# 4. Docker login
if [[ "$PUSH" == 1 ]]; then
    echo "==> Docker Hub login"
    if ! docker info 2>/dev/null | grep -q "Username:"; then
        docker login || { echo "ERROR: docker login failed"; exit 1; }
    else
        echo "  already logged in"
    fi
fi

# 5. Build
echo "==> Building ${IMAGE}:${TAG}"
BUILD_ARGS=()
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
    BUILD_ARGS+=(--build-arg "GITHUB_TOKEN=${GITHUB_TOKEN}")
fi
if [[ -n "${HF_TOKEN_BUILD:-}" ]]; then
    BUILD_ARGS+=(--build-arg "HF_TOKEN_BUILD=${HF_TOKEN_BUILD}")
fi
if [[ -n "${GIT_REF:-}" ]]; then
    BUILD_ARGS+=(--build-arg "GIT_REF=${GIT_REF}")
fi

docker build \
    -f Dockerfile.train \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE}:${TAG}" \
    -t "${IMAGE}:latest" \
    .

# 6. Push
if [[ "$PUSH" == 1 ]]; then
    echo "==> Pushing ${IMAGE}:${TAG}"
    docker push "${IMAGE}:${TAG}"
    docker push "${IMAGE}:latest"
fi

# 7. Final summary
echo
echo "==> Done."
echo "    Image: ${IMAGE}:${TAG}"
echo "    Size:  $(docker image inspect "${IMAGE}:${TAG}" --format '{{.Size}}' 2>/dev/null | awk '{printf "%.1f GB\n", $1/1024/1024/1024}')"
echo
echo "    SDL reference (verify line 22 of akasha_config/akasha_deployment.sdl):"
echo "      image: ${IMAGE}:${TAG}"
