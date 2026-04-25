#!/usr/bin/env bash
# DriftCall container entrypoint — starts sshd in background, runs the full
# training pipeline inside tmux so users can `ssh root@<pod>` and `tmux
# attach -t train` to watch progress live.
#
# Default root password is "driftcall" (set in Dockerfile.train). Override
# at runtime via `-e ROOT_PASSWORD=<new>` if exposed publicly.

set -e

WORKSPACE_DIR="${WORKSPACE_DIR:-/workspace}"
LOG_DIR="${DRIFTCALL_LOG_DIR:-${WORKSPACE_DIR}/logs}"

mkdir -p "$LOG_DIR"

# Optional override of root password before sshd starts.
if [ -n "${ROOT_PASSWORD:-}" ]; then
    echo "root:${ROOT_PASSWORD}" | chpasswd
fi

# Start sshd in background — port 22 stays accessible for tmux attach.
/usr/sbin/sshd

# Move into the cloned repo dir so cells/ + data/ resolve correctly.
cd "${WORKSPACE_DIR}/repo"

# Activate the UV venv so subsequent python invocations use the right interpreter.
# shellcheck disable=SC1090
source "${WORKSPACE_DIR}/.venv/driftcall/bin/activate"

# Run the training pipeline inside a named tmux session.
tmux new-session -d -s train \
    "/workspace/repo/scripts/train_full.sh 2>&1 | tee -a ${LOG_DIR}/train.log"

echo "[entrypoint] DriftCall training started in tmux session 'train'"
echo "[entrypoint] Attach with: ssh root@<host> -p 22 ; tmux attach -t train"

# Hold the container open while tmux runs the training in detached mode.
# Container exits when train_full.sh finishes (success or failure).
while tmux has-session -t train 2>/dev/null; do
    sleep 30
done

echo "[entrypoint] tmux session ended; container will exit"
tail -50 "${LOG_DIR}/train.log" || true
