#!/usr/bin/env bash
# DriftCall container entrypoint — clones the repo at container start (not at
# docker build), installs the package editable, then starts sshd and runs the
# full training pipeline in tmux.
#
# Default root password is "driftcall" (set in Dockerfile.train). Override at
# runtime via `-e ROOT_PASSWORD=<new>` if exposed publicly.

set -euo pipefail

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
LOG_DIR="${DRIFTCALL_LOG_DIR:-${WORKSPACE_DIR}/logs}"
VENV_PY="${WORKSPACE_DIR}/.venv/driftcall/bin/python"
VENV_PIP="${WORKSPACE_DIR}/.venv/driftcall/bin/pip"

GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/saumilyagupta/openenv-DGXAI.git}"
GIT_REF="${GIT_REF:-main}"
SUBPATH="${SUBPATH:-DRIFTCALL}"

mkdir -p "$LOG_DIR"

driftcall_clone_and_install() {
    if [[ "${DRIFTCALL_SKIP_GIT_CLONE:-0}" == "1" ]] && [[ -f "${WORKSPACE_DIR}/repo/pyproject.toml" ]]; then
        echo "[entrypoint] DRIFTCALL_SKIP_GIT_CLONE=1 and repo present — skipping git clone"
    else
        echo "[entrypoint] Cloning ${GIT_REPO_URL} (ref=${GIT_REF}, subpath=${SUBPATH})"
        rm -rf /tmp/driftcall-src
        export GIT_TERMINAL_PROMPT=0
        local url
        if [[ -n "${GITHUB_TOKEN:-}" ]]; then
            url=$(echo "$GIT_REPO_URL" | sed "s|https://|https://${GITHUB_TOKEN}@|")
        else
            url="$GIT_REPO_URL"
        fi
        git clone --depth 1 --branch "$GIT_REF" "$url" /tmp/driftcall-src
        rm -rf "${WORKSPACE_DIR}/repo"
        mkdir -p "${WORKSPACE_DIR}/repo"
        cp -a "/tmp/driftcall-src/${SUBPATH}/." "${WORKSPACE_DIR}/repo/"
        rm -rf /tmp/driftcall-src
    fi

    echo "[entrypoint] pip install -e ${WORKSPACE_DIR}/repo (no-deps; deps from image)"
    "$VENV_PIP" install --no-deps -e "${WORKSPACE_DIR}/repo"
}

driftcall_optional_pre_pull() {
    if [[ "${DRIFTCALL_PRE_PULL_AT_START:-0}" != "1" ]]; then
        return 0
    fi
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[entrypoint] DRIFTCALL_PRE_PULL_AT_START=1 but HF_TOKEN empty — skipping pre-pull"
        return 0
    fi
    echo "[entrypoint] Pre-pulling base model weights (DRIFTCALL_PRE_PULL_AT_START=1)"
    HF_TOKEN="$HF_TOKEN" "$VENV_PY" -c \
        "from huggingface_hub import snapshot_download; snapshot_download('unsloth/gemma-4-E2B-it-bnb-4bit')" \
        || true
}

driftcall_clone_and_install
driftcall_optional_pre_pull

# Optional override of root password before sshd starts.
if [[ -n "${ROOT_PASSWORD:-}" ]]; then
    echo "root:${ROOT_PASSWORD}" | chpasswd
fi

# Start sshd in background — port 22 stays accessible for tmux attach.
/usr/sbin/sshd

cd "${WORKSPACE_DIR}/repo"

# shellcheck disable=SC1090
source "${WORKSPACE_DIR}/.venv/driftcall/bin/activate"

tmux new-session -d -s train \
    "/workspace/repo/scripts/train_full.sh 2>&1 | tee -a ${LOG_DIR}/train.log"

echo "[entrypoint] DriftCall training started in tmux session 'train'"
echo "[entrypoint] Attach with: ssh root@<host> -p 22 ; tmux attach -t train"

while tmux has-session -t train 2>/dev/null; do
    sleep 30
done

echo "[entrypoint] tmux session ended; container will exit"
tail -50 "${LOG_DIR}/train.log" || true
