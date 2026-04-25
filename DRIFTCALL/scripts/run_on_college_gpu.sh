#!/usr/bin/env bash
# DriftCall — one-shot launcher for the college GPU server.
#
# Runs from your laptop. SSH-es into the college GPU box, clones the repo,
# sets up a Python venv, installs deps, exports training env vars, and
# launches the full pipeline inside a tmux session named 'train'. Survives
# Cloudflare tunnel idle drops because tmux runs server-side.
#
# Usage:
#   ./scripts/run_on_college_gpu.sh                                # uses defaults
#   ./scripts/run_on_college_gpu.sh --host <other-ssh-alias>       # different host
#   ./scripts/run_on_college_gpu.sh --hardware v100                # if not h100
#   ./scripts/run_on_college_gpu.sh --steps-1 5                    # smoke run
#   ./scripts/run_on_college_gpu.sh --attach                       # ssh + tmux attach only
#   ./scripts/run_on_college_gpu.sh --logs                         # tail server log
#
# Required env (or set via cells/_secrets.py / .env):
#   HF_TOKEN           — Hugging Face write token (push trained LoRA)
#   WANDB_API_KEY      — wandb auth (cells/_secrets.py is fallback)
#   DRIFTCALL_HF_REPO  — e.g. DGXAI/gemma-4-e2b-driftcall-lora
#   SSH_PASS           — optional; college server password if key auth fails

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults — match the user's ~/.ssh/config + akasha SDL
# ---------------------------------------------------------------------------
SSH_HOST="${SSH_HOST:-QWERTY-GPU-proxy}"
REMOTE_WORKSPACE="${REMOTE_WORKSPACE:-/workspace}"
GIT_REPO_URL="${GIT_REPO_URL:-https://github.com/saumilyagupta/openenv-DGXAI.git}"
GIT_REF="${GIT_REF:-main}"
SUBPATH="${SUBPATH:-DRIFTCALL}"
DRIFTCALL_HARDWARE="${DRIFTCALL_HARDWARE:-h100}"
DRIFTCALL_HF_REPO="${DRIFTCALL_HF_REPO:-DGXAI/gemma-4-e2b-driftcall-lora}"
DRIFTCALL_NUM_STEPS_STAGE1="${DRIFTCALL_NUM_STEPS_STAGE1:-150}"
DRIFTCALL_NUM_STEPS_STAGE2="${DRIFTCALL_NUM_STEPS_STAGE2:-200}"
DRIFTCALL_NUM_STEPS_STAGE3="${DRIFTCALL_NUM_STEPS_STAGE3:-150}"
DRIFTCALL_EVAL_EPISODES="${DRIFTCALL_EVAL_EPISODES:-50}"
DRIFTCALL_PROBE_EPISODES="${DRIFTCALL_PROBE_EPISODES:-200}"

# Hardcoded SSH password — private-repo policy. Rotate by editing here.
# Override at runtime: `SSH_PASS=<other> ./scripts/run_on_college_gpu.sh`
HARDCODED_SSH_PASS="qazplmqa"
SSH_PASS="${SSH_PASS:-$HARDCODED_SSH_PASS}"

# Tokens — pull from env, fall back to the .env file at repo root if present.
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi
HF_TOKEN="${HF_TOKEN:-}"
WANDB_API_KEY="${WANDB_API_KEY:-}"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
MODE="run"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host)        SSH_HOST="$2"; shift 2 ;;
        --hardware)    DRIFTCALL_HARDWARE="$2"; shift 2 ;;
        --steps-1)     DRIFTCALL_NUM_STEPS_STAGE1="$2"; shift 2 ;;
        --steps-2)     DRIFTCALL_NUM_STEPS_STAGE2="$2"; shift 2 ;;
        --steps-3)     DRIFTCALL_NUM_STEPS_STAGE3="$2"; shift 2 ;;
        --git-ref)     GIT_REF="$2"; shift 2 ;;
        --workspace)   REMOTE_WORKSPACE="$2"; shift 2 ;;
        --attach)      MODE="attach"; shift ;;
        --logs)        MODE="logs"; shift ;;
        --kill)        MODE="kill"; shift ;;
        --status)      MODE="status"; shift ;;
        -h|--help)
            grep -E "^# " "$0" | head -25 | sed 's/^# //'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

# ---------------------------------------------------------------------------
# SSH wrapper — uses sshpass if available + needed, else key auth from config
# ---------------------------------------------------------------------------
ssh_cmd() {
    local cmd_args=()
    if [[ -n "$SSH_PASS" ]] && command -v sshpass >/dev/null 2>&1; then
        cmd_args=(sshpass -p "$SSH_PASS" ssh -o StrictHostKeyChecking=no \
                  -o UserKnownHostsFile=/dev/null \
                  -o ServerAliveInterval=60 -o ServerAliveCountMax=3)
    else
        cmd_args=(ssh -o StrictHostKeyChecking=no \
                  -o UserKnownHostsFile=/dev/null \
                  -o ServerAliveInterval=60 -o ServerAliveCountMax=3)
    fi
    "${cmd_args[@]}" "$@"
}

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
case "$MODE" in
    attach)
        echo "==> Attaching to remote tmux session 'train' on $SSH_HOST"
        ssh_cmd -t "$SSH_HOST" "tmux attach -t train"
        exit 0
        ;;
    logs)
        echo "==> Tailing remote training log on $SSH_HOST"
        ssh_cmd "$SSH_HOST" "tail -f -n 200 $REMOTE_WORKSPACE/logs/train.log 2>/dev/null || \
                              echo 'no log yet at $REMOTE_WORKSPACE/logs/train.log'"
        exit 0
        ;;
    kill)
        echo "==> Killing remote tmux session 'train' on $SSH_HOST"
        ssh_cmd "$SSH_HOST" "tmux kill-session -t train 2>/dev/null && echo killed || echo 'no session'"
        exit 0
        ;;
    status)
        echo "==> Status on $SSH_HOST"
        ssh_cmd "$SSH_HOST" "
            echo '--- nvidia-smi ---'
            nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv 2>/dev/null || echo 'no nvidia-smi'
            echo
            echo '--- tmux sessions ---'
            tmux ls 2>/dev/null || echo 'no tmux'
            echo
            echo '--- last 20 log lines ---'
            tail -n 20 $REMOTE_WORKSPACE/logs/train.log 2>/dev/null || echo 'no log'
        "
        exit 0
        ;;
esac

# ---------------------------------------------------------------------------
# Pre-flight checks (run mode)
# ---------------------------------------------------------------------------
echo "==> DriftCall college-GPU launcher"
echo "    host:           $SSH_HOST"
echo "    workspace:      $REMOTE_WORKSPACE"
echo "    hardware:       $DRIFTCALL_HARDWARE"
echo "    steps:          $DRIFTCALL_NUM_STEPS_STAGE1 / $DRIFTCALL_NUM_STEPS_STAGE2 / $DRIFTCALL_NUM_STEPS_STAGE3"
echo "    repo:           $GIT_REPO_URL @ $GIT_REF"
echo "    hf_repo:        $DRIFTCALL_HF_REPO"
echo "    sshpass:        $([[ -n "$SSH_PASS" ]] && echo yes || echo "no (using key auth)")"
echo

if [[ -z "$HF_TOKEN" || "$HF_TOKEN" == hf_REPLACE_ME* ]]; then
    echo "ERROR: HF_TOKEN is not set. Export it or put it in .env." >&2
    exit 1
fi
if [[ -z "$WANDB_API_KEY" ]]; then
    echo "WARNING: WANDB_API_KEY not set in env — relying on cells/_secrets.py fallback"
fi
if [[ -n "$SSH_PASS" ]] && ! command -v sshpass >/dev/null 2>&1; then
    echo "WARNING: SSH_PASS provided but 'sshpass' not installed locally. Falling back to key auth."
    echo "         Install: sudo apt install sshpass   (or)   brew install sshpass"
fi

# ---------------------------------------------------------------------------
# Generate the remote bootstrap+launch script and pipe it via SSH
# ---------------------------------------------------------------------------
REMOTE_SCRIPT=$(cat <<REMOTE_EOF
#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="$REMOTE_WORKSPACE"
GIT_REPO_URL="$GIT_REPO_URL"
GIT_REF="$GIT_REF"
SUBPATH="$SUBPATH"
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

mkdir -p "\$WORKSPACE"
cd "\$WORKSPACE"

# 1. Clone (or update) the repo
if [[ -d openenv-DGXAI/.git ]]; then
    echo "[remote] repo exists, fetching latest"
    cd openenv-DGXAI
    git fetch origin
    git checkout "\$GIT_REF"
    git reset --hard "origin/\$GIT_REF"
    cd ..
else
    echo "[remote] cloning \$GIT_REPO_URL"
    if [[ -n "\$GITHUB_TOKEN" ]]; then
        AUTHED_URL=\$(echo "\$GIT_REPO_URL" | sed "s|https://|https://\${GITHUB_TOKEN}@|")
    else
        AUTHED_URL="\$GIT_REPO_URL"
    fi
    git clone --branch "\$GIT_REF" "\$AUTHED_URL" openenv-DGXAI
fi

cd "openenv-DGXAI/\$SUBPATH"
APP_DIR=\$(pwd)
echo "[remote] using \$APP_DIR"

# 2. Python venv
if [[ ! -d .venv ]]; then
    echo "[remote] creating Python venv"
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install --upgrade pip wheel >/dev/null

# 3. Install deps (skip if already done)
if ! python -c "import unsloth, trl, peft, transformers" 2>/dev/null; then
    echo "[remote] installing Python deps (~5-15 min first time)"
    pip install -e . >/dev/null
    pip install "unsloth>=2026.4.5" "trl>=0.23" hf_transfer >/dev/null
fi

# 4. Output / cache dirs
mkdir -p "\$WORKSPACE"/{checkpoints,eval_reports,logs,hf_cache}

# 5. Write env file inside the venv dir for the tmux session to source
cat > .driftcall.env <<ENV_EOF
export HF_TOKEN="$HF_TOKEN"
export WANDB_API_KEY="$WANDB_API_KEY"
export DRIFTCALL_HF_REPO="$DRIFTCALL_HF_REPO"
export DRIFTCALL_HARDWARE="$DRIFTCALL_HARDWARE"
export DRIFTCALL_NUM_STEPS_STAGE1="$DRIFTCALL_NUM_STEPS_STAGE1"
export DRIFTCALL_NUM_STEPS_STAGE2="$DRIFTCALL_NUM_STEPS_STAGE2"
export DRIFTCALL_NUM_STEPS_STAGE3="$DRIFTCALL_NUM_STEPS_STAGE3"
export DRIFTCALL_EVAL_EPISODES="$DRIFTCALL_EVAL_EPISODES"
export DRIFTCALL_PROBE_EPISODES="$DRIFTCALL_PROBE_EPISODES"
export DRIFTCALL_OUTPUT_DIR="\$WORKSPACE/checkpoints"
export DRIFTCALL_EVAL_DIR="\$WORKSPACE/eval_reports"
export DRIFTCALL_LOG_DIR="\$WORKSPACE/logs"
export DRIFTCALL_BRIEFS_PATH="\$APP_DIR/data/val/briefs.jsonl"
export HF_HOME="\$WORKSPACE/hf_cache"
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_PROJECT=driftcall
export WANDB_MODE=online
export DRIFTCALL_PUSH_TO_HUB=true
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=0
ENV_EOF

# 6. Kill any stale tmux session, then start fresh
if tmux has-session -t train 2>/dev/null; then
    echo "[remote] killing existing tmux 'train' session"
    tmux kill-session -t train
fi

echo "[remote] launching training in tmux 'train'"
tmux new-session -d -s train "
    cd '\$APP_DIR' && \
    source .venv/bin/activate && \
    source .driftcall.env && \
    bash scripts/train_full.sh 2>&1 | tee -a '\$WORKSPACE/logs/train.log'
"

echo "[remote] training started. Tail log: tail -f \$WORKSPACE/logs/train.log"
echo "[remote] attach: tmux attach -t train"
sleep 3
tmux ls
REMOTE_EOF
)

echo "==> Streaming bootstrap script to $SSH_HOST"
echo "$REMOTE_SCRIPT" | ssh_cmd "$SSH_HOST" "bash -s"

echo
echo "==> Training launched."
echo "    Tail log:  $0 --logs"
echo "    Attach:    $0 --attach"
echo "    Status:    $0 --status"
echo "    Kill:      $0 --kill"
