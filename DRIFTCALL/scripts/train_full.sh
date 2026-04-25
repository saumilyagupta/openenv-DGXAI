#!/usr/bin/env bash
# DriftCall full training pipeline — Stage 1 → 2 → 3 → eval → probe → push.
#
# Designed to run inside the Dockerfile.train image on Akash Network. Reads
# env vars set in the Dockerfile (override via `docker run -e`):
#
#   WANDB_API_KEY            wandb auth (else falls back to cells/_secrets.py)
#   HF_TOKEN                 huggingface_hub write token (required if PUSH_TO_HUB=true)
#   DRIFTCALL_HF_REPO        e.g. "krrishchoudhary109/gemma-3n-e2b-driftcall-lora"
#   DRIFTCALL_HARDWARE       "v100" | "h100" (default h100)
#   DRIFTCALL_NUM_STEPS_*    per-stage step counts
#   DRIFTCALL_EVAL_EPISODES  baseline + final eval episode count
#   DRIFTCALL_PROBE_EPISODES reward-hacking probe episode count
#   DRIFTCALL_OUTPUT_DIR     where to write LoRA checkpoints
#   DRIFTCALL_PUSH_TO_HUB    "true" | "false"

set -euo pipefail

OUT="${DRIFTCALL_OUTPUT_DIR:-/app/checkpoints}"
LOG_DIR="/app/logs"
EVAL_DIR="/app/eval_reports"
HARDWARE="${DRIFTCALL_HARDWARE:-h100}"

mkdir -p "$OUT" "$LOG_DIR" "$EVAL_DIR"

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "[$(ts)] $*" | tee -a "$LOG_DIR/train.log"; }

trap 'log "ERROR on line $LINENO; exiting with code $?"' ERR

# ---------------------------------------------------------------------------
# 0. Pre-flight checks
# ---------------------------------------------------------------------------
log "DriftCall full training run starting on hardware=$HARDWARE"
log "Output: $OUT"
log "GPU info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee -a "$LOG_DIR/train.log"

log "Python env:"
python3 -c "import torch; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} device_count={torch.cuda.device_count()}')" \
    | tee -a "$LOG_DIR/train.log"

if [[ -z "${WANDB_API_KEY:-}" ]]; then
    log "WANDB_API_KEY not set — init_wandb will fall back to cells/_secrets.py"
fi

if [[ "${DRIFTCALL_PUSH_TO_HUB:-true}" == "true" && -z "${HF_TOKEN:-}" ]]; then
    log "WARNING: PUSH_TO_HUB=true but HF_TOKEN unset; final push will fail. Set HF_TOKEN or set PUSH_TO_HUB=false."
fi

cd /app

# ---------------------------------------------------------------------------
# 1. Stage 1 — 150 steps, no drift
# ---------------------------------------------------------------------------
log "=== Stage 1: $DRIFTCALL_NUM_STEPS_STAGE1 GRPO steps (no drift) ==="
python3 -m cells.step_15_train_stage1 \
    --num-steps "$DRIFTCALL_NUM_STEPS_STAGE1" \
    --hardware "$HARDWARE" \
    --output-dir "$OUT/stage1" \
    2>&1 | tee -a "$LOG_DIR/stage1.log"
log "Stage 1 complete."

# ---------------------------------------------------------------------------
# 2. Stage 2 — 200 steps, single drift; resumes from stage1 final
# ---------------------------------------------------------------------------
log "=== Stage 2: $DRIFTCALL_NUM_STEPS_STAGE2 GRPO steps (single drift) ==="
python3 -m cells.step_16_train_stage2 \
    --num-steps "$DRIFTCALL_NUM_STEPS_STAGE2" \
    --hardware "$HARDWARE" \
    --resume-from "$OUT/stage1/final" \
    --output-dir "$OUT/stage2" \
    2>&1 | tee -a "$LOG_DIR/stage2.log"
log "Stage 2 complete."

# ---------------------------------------------------------------------------
# 3. Stage 3 — 150 steps, compound drift; resumes from stage2 final
# ---------------------------------------------------------------------------
log "=== Stage 3: $DRIFTCALL_NUM_STEPS_STAGE3 GRPO steps (compound drift) ==="
python3 -m cells.step_17_train_stage3 \
    --num-steps "$DRIFTCALL_NUM_STEPS_STAGE3" \
    --hardware "$HARDWARE" \
    --resume-from "$OUT/stage2/final" \
    --output-dir "$OUT/stage3" \
    2>&1 | tee -a "$LOG_DIR/stage3.log"
log "Stage 3 complete. Final LoRA at $OUT/stage3/final"

# ---------------------------------------------------------------------------
# 4. Baseline eval (untrained Gemma 3n E2B, same 50 seeds)
# ---------------------------------------------------------------------------
log "=== Baseline eval: $DRIFTCALL_EVAL_EPISODES episodes ==="
python3 -m cells.step_18_eval_baseline \
    --episodes "$DRIFTCALL_EVAL_EPISODES" \
    --output "$EVAL_DIR/baseline.json" \
    2>&1 | tee -a "$LOG_DIR/eval_baseline.log"
log "Baseline eval complete."

# ---------------------------------------------------------------------------
# 5. Final eval (trained LoRA, same 50 seeds)
# ---------------------------------------------------------------------------
log "=== Final eval: $DRIFTCALL_EVAL_EPISODES episodes ==="
python3 -m cells.step_19_eval_final \
    --episodes "$DRIFTCALL_EVAL_EPISODES" \
    --checkpoint "$OUT/stage3/final" \
    --output "$EVAL_DIR/final.json" \
    2>&1 | tee -a "$LOG_DIR/eval_final.log"
log "Final eval complete."

# ---------------------------------------------------------------------------
# 6. Reward-hacking probe
# ---------------------------------------------------------------------------
log "=== Reward-hacking probe: $DRIFTCALL_PROBE_EPISODES episodes ==="
python3 -m cells.step_20_probe \
    --episodes "$DRIFTCALL_PROBE_EPISODES" \
    --checkpoint "$OUT/stage3/final" \
    --output "$EVAL_DIR/probe.json" \
    2>&1 | tee -a "$LOG_DIR/probe.log"
log "Probe complete."

# ---------------------------------------------------------------------------
# 7. Plots + summary
# ---------------------------------------------------------------------------
log "=== Generating eval plots + summary ==="
python3 -m cells.step_21_plots \
    --baseline "$EVAL_DIR/baseline.json" \
    --final "$EVAL_DIR/final.json" \
    --output-dir "$EVAL_DIR/plots" \
    2>&1 | tee -a "$LOG_DIR/plots.log"
python3 -m cells.step_22_summary \
    --baseline "$EVAL_DIR/baseline.json" \
    --final "$EVAL_DIR/final.json" \
    --probe "$EVAL_DIR/probe.json" \
    --output "$EVAL_DIR/summary.md" \
    2>&1 | tee -a "$LOG_DIR/summary.log"
log "Reports written to $EVAL_DIR"

# ---------------------------------------------------------------------------
# 8. Push trained LoRA + eval reports to HF Hub (optional)
# ---------------------------------------------------------------------------
if [[ "${DRIFTCALL_PUSH_TO_HUB:-true}" == "true" && -n "${HF_TOKEN:-}" && -n "${DRIFTCALL_HF_REPO:-}" ]]; then
    log "=== Pushing trained LoRA + eval reports to HF Hub: $DRIFTCALL_HF_REPO ==="
    python3 -m cells.step_24_deploy_hf \
        --checkpoint "$OUT/stage3/final" \
        --eval-reports "$EVAL_DIR" \
        --repo "$DRIFTCALL_HF_REPO" \
        2>&1 | tee -a "$LOG_DIR/deploy.log"
    log "Push complete."
else
    log "Skipping HF Hub push (PUSH_TO_HUB=$DRIFTCALL_PUSH_TO_HUB, HF_TOKEN set=$([ -n "${HF_TOKEN:-}" ] && echo yes || echo no), repo=$DRIFTCALL_HF_REPO)"
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
