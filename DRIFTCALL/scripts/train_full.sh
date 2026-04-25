#!/usr/bin/env bash
# DriftCall full training pipeline — Stage 1 -> 2 -> 3 -> eval -> probe -> push.
#
# Thin wrapper over scripts/run_pipeline.py — the orchestrator wires
# task_gen + env_factory + rollout_group_fn + training_eval from the
# real cells (training.md §2.2, evaluation.md §6.1) and dispatches every
# stage with consistent contracts. The cell modules themselves do NOT
# expose CLIs.
#
# Designed to run inside the Dockerfile.train image on Akash Network.
# Reads env vars set in the Dockerfile (override via `docker run -e`):
#
#   WANDB_API_KEY                  wandb auth (optional; offline-safe)
#   WANDB_RUN_ID                   wandb run id for plot history fetch
#   HF_TOKEN                       huggingface_hub write token (required if PUSH_TO_HUB=true)
#   DRIFTCALL_HF_REPO              e.g. "krrishchoudhary109/gemma-4-e2b-driftcall-lora"
#   DRIFTCALL_HARDWARE             "v100" | "h100" (default h100)
#   DRIFTCALL_NUM_STEPS_STAGE{1,2,3}  per-stage GRPO step counts
#   DRIFTCALL_EVAL_EPISODES        baseline + final eval episode count
#   DRIFTCALL_PROBE_EPISODES       reward-hacking probe episode count
#   DRIFTCALL_OUTPUT_DIR           where to write LoRA checkpoints
#   DRIFTCALL_EVAL_DIR             where to write eval reports + plots
#   DRIFTCALL_PUSH_TO_HUB          "true" | "false" (default true)

set -euo pipefail

OUT="${DRIFTCALL_OUTPUT_DIR:-/app/checkpoints}"
LOG_DIR="${DRIFTCALL_LOG_DIR:-/app/logs}"
EVAL_DIR="${DRIFTCALL_EVAL_DIR:-/app/eval_reports}"
HARDWARE="${DRIFTCALL_HARDWARE:-h100}"
APP_DIR="${DRIFTCALL_APP_DIR:-/app}"

NUM_STEPS_STAGE1="${DRIFTCALL_NUM_STEPS_STAGE1:-150}"
NUM_STEPS_STAGE2="${DRIFTCALL_NUM_STEPS_STAGE2:-200}"
NUM_STEPS_STAGE3="${DRIFTCALL_NUM_STEPS_STAGE3:-150}"
EVAL_EPISODES="${DRIFTCALL_EVAL_EPISODES:-50}"
PROBE_EPISODES="${DRIFTCALL_PROBE_EPISODES:-200}"
PUSH_TO_HUB="${DRIFTCALL_PUSH_TO_HUB:-true}"

mkdir -p "$OUT" "$LOG_DIR" "$EVAL_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/train.log"; }

trap 'log "ERROR on line $LINENO; exiting with code $?"' ERR

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
log "DriftCall full training run starting on hardware=$HARDWARE"
log "Output: $OUT  Eval: $EVAL_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU info:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "$LOG_DIR/train.log"
else
    log "nvidia-smi not available — skipping GPU probe (CPU-only run)"
fi

log "Python env:"
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')" \
    | tee -a "$LOG_DIR/train.log" || log "torch import failed (CPU-only smoke?)"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    log "WANDB_API_KEY not set — wandb runs in offline mode"
fi

if [[ "$PUSH_TO_HUB" == "true" && -z "${HF_TOKEN:-}" ]]; then
    log "WARNING: PUSH_TO_HUB=true but HF_TOKEN unset; final push will fail. Set HF_TOKEN or set PUSH_TO_HUB=false."
fi

cd "$APP_DIR"

PIPELINE="python3 scripts/run_pipeline.py"

# ---------------------------------------------------------------------------
# 1. Stage 1 — warmup, no drift
# ---------------------------------------------------------------------------
log "=== Stage 1: $NUM_STEPS_STAGE1 GRPO steps (no drift) ==="
$PIPELINE stage1 \
    --num-steps "$NUM_STEPS_STAGE1" \
    --hardware "$HARDWARE" \
    --output-dir "$OUT/stage1" \
    2>&1 | tee -a "$LOG_DIR/stage1.log"
log "Stage 1 complete."

# ---------------------------------------------------------------------------
# 2. Stage 2 — single drift; resumes from stage1 final
# ---------------------------------------------------------------------------
log "=== Stage 2: $NUM_STEPS_STAGE2 GRPO steps (single drift) ==="
$PIPELINE stage2 \
    --num-steps "$NUM_STEPS_STAGE2" \
    --hardware "$HARDWARE" \
    --resume-from "$OUT/stage1/final" \
    --output-dir "$OUT/stage2" \
    2>&1 | tee -a "$LOG_DIR/stage2.log"
log "Stage 2 complete."

# ---------------------------------------------------------------------------
# 3. Stage 3 — compound drift; resumes from stage2 final
# ---------------------------------------------------------------------------
log "=== Stage 3: $NUM_STEPS_STAGE3 GRPO steps (compound drift) ==="
$PIPELINE stage3 \
    --num-steps "$NUM_STEPS_STAGE3" \
    --hardware "$HARDWARE" \
    --resume-from "$OUT/stage2/final" \
    --output-dir "$OUT/stage3" \
    2>&1 | tee -a "$LOG_DIR/stage3.log"
log "Stage 3 complete. Final LoRA at $OUT/stage3/final"

# ---------------------------------------------------------------------------
# 4. Baseline eval
# ---------------------------------------------------------------------------
log "=== Baseline eval: $EVAL_EPISODES episodes ==="
$PIPELINE eval-baseline \
    --episodes "$EVAL_EPISODES" \
    --output "$EVAL_DIR/baseline.json" \
    2>&1 | tee -a "$LOG_DIR/eval_baseline.log"
log "Baseline eval complete."

# ---------------------------------------------------------------------------
# 5. Final eval — paired against baseline
# ---------------------------------------------------------------------------
log "=== Final eval: $EVAL_EPISODES episodes ==="
$PIPELINE eval-final \
    --episodes "$EVAL_EPISODES" \
    --checkpoint "$OUT/stage3/final" \
    --output "$EVAL_DIR/final.json" \
    2>&1 | tee -a "$LOG_DIR/eval_final.log"
log "Final eval complete."

# ---------------------------------------------------------------------------
# 6. Reward-hacking probe
# ---------------------------------------------------------------------------
log "=== Reward-hacking probe: $PROBE_EPISODES episodes ==="
$PIPELINE probe \
    --episodes "$PROBE_EPISODES" \
    --checkpoint "$OUT/stage3/final" \
    --output "$EVAL_DIR/probe.json" \
    2>&1 | tee -a "$LOG_DIR/probe.log"
log "Probe complete."

# ---------------------------------------------------------------------------
# 7. Plots + summary
# ---------------------------------------------------------------------------
log "=== Generating eval plots + summary ==="
$PIPELINE plots \
    --baseline "$EVAL_DIR/baseline.json" \
    --final "$EVAL_DIR/final.json" \
    --output-dir "$EVAL_DIR/plots" \
    2>&1 | tee -a "$LOG_DIR/plots.log"

$PIPELINE summary \
    --baseline "$EVAL_DIR/baseline.json" \
    --final "$EVAL_DIR/final.json" \
    --probe "$EVAL_DIR/probe.json" \
    --output "$EVAL_DIR/summary.md" \
    2>&1 | tee -a "$LOG_DIR/summary.log"
log "Reports written to $EVAL_DIR"

# ---------------------------------------------------------------------------
# 8. Push trained LoRA to HF Hub (optional)
# ---------------------------------------------------------------------------
if [[ "$PUSH_TO_HUB" == "true" && -n "${HF_TOKEN:-}" && -n "${DRIFTCALL_HF_REPO:-}" ]]; then
    log "=== Pushing trained LoRA to HF Hub: $DRIFTCALL_HF_REPO ==="
    $PIPELINE deploy \
        --checkpoint "$OUT/stage3/final" \
        --eval-reports "$EVAL_DIR" \
        --repo "$DRIFTCALL_HF_REPO" \
        2>&1 | tee -a "$LOG_DIR/deploy.log"
    log "Push complete."
else
    log "Skipping HF Hub push (PUSH_TO_HUB=$PUSH_TO_HUB, HF_TOKEN set=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no), repo=${DRIFTCALL_HF_REPO:-unset})"
fi

# ---------------------------------------------------------------------------
# 9. Final summary
# ---------------------------------------------------------------------------
log "=== TRAINING RUN COMPLETE ==="
log "Checkpoints:    $OUT/{stage1,stage2,stage3}/final"
log "Eval reports:   $EVAL_DIR/{baseline,final,probe}.json"
log "Plots:          $EVAL_DIR/plots/"
log "Summary:        $EVAL_DIR/summary.md"
log "Logs:           $LOG_DIR/"

# Surface key metrics so they show up in `docker logs`.
if [[ -f "$EVAL_DIR/summary.md" ]]; then
    log ""
    log "--- summary.md ---"
    cat "$EVAL_DIR/summary.md" | tee -a "$LOG_DIR/train.log"
fi
