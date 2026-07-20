#!/usr/bin/env bash
# DriftCall Gemma 3n — three-stage GRPO curriculum (native loop).
#
# Drives the new self-contained training script
# (scripts/train_driftcall_grpo.py) for stages 1 -> 2 -> 3 with the
# correct resume plumbing.
#
# Env vars:
#   DRIFTCALL_HARDWARE             "v100" | "h100"     (default v100)
#   DRIFTCALL_NUM_STEPS_STAGE1     int                 (default 150)
#   DRIFTCALL_NUM_STEPS_STAGE2     int                 (default 200)
#   DRIFTCALL_NUM_STEPS_STAGE3     int                 (default 150)
#   DRIFTCALL_NUM_GENERATIONS      int (4 | 8)         (default 8)
#   DRIFTCALL_OUTPUT_DIR           dir                 (default /workspace/openenv-DGXAI/DRIFTCALL/checkpoints)
#   DRIFTCALL_LOG_DIR              dir                 (default /workspace/openenv-DGXAI/DRIFTCALL/logs)
#   CUDA_VISIBLE_DEVICES           int                 (default 3)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$REPO_ROOT"

OUT="${DRIFTCALL_OUTPUT_DIR:-$REPO_ROOT/checkpoints}"
LOGS="${DRIFTCALL_LOG_DIR:-$REPO_ROOT/logs}"
HW="${DRIFTCALL_HARDWARE:-v100}"
S1="${DRIFTCALL_NUM_STEPS_STAGE1:-150}"
S2="${DRIFTCALL_NUM_STEPS_STAGE2:-200}"
S3="${DRIFTCALL_NUM_STEPS_STAGE3:-150}"
G="${DRIFTCALL_NUM_GENERATIONS:-8}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

mkdir -p "$OUT/stage1" "$OUT/stage2" "$OUT/stage3" "$LOGS"

ts() { date -u +"%Y%m%d_%H%M%S"; }
TS_RUN="$(ts)"

echo "[full] hardware=$HW  steps=($S1, $S2, $S3)  G=$G  CUDA=$CUDA_VISIBLE_DEVICES"
echo "[full] output_dir=$OUT  logs=$LOGS  run_ts=$TS_RUN"

# ---------------------------------------------------------------------------
# Stage 1 — no drift
# ---------------------------------------------------------------------------
echo "[full] === Stage 1: $S1 GRPO steps (no drift) ==="
python3 scripts/train_driftcall_grpo.py \
    --stage 1 --num-steps "$S1" \
    --hardware "$HW" \
    --num-generations "$G" \
    --output-dir "$OUT/stage1/final" \
    2>&1 | tee "$LOGS/stage1_${TS_RUN}.log"

# ---------------------------------------------------------------------------
# Stage 2 — single drift, resumes from stage1
# ---------------------------------------------------------------------------
echo "[full] === Stage 2: $S2 GRPO steps (single drift) ==="
python3 scripts/train_driftcall_grpo.py \
    --stage 2 --num-steps "$S2" \
    --hardware "$HW" \
    --num-generations "$G" \
    --resume-from "$OUT/stage1/final" \
    --output-dir "$OUT/stage2/final" \
    2>&1 | tee "$LOGS/stage2_${TS_RUN}.log"

# ---------------------------------------------------------------------------
# Stage 3 — compound drift, resumes from stage2
# ---------------------------------------------------------------------------
echo "[full] === Stage 3: $S3 GRPO steps (compound drift) ==="
python3 scripts/train_driftcall_grpo.py \
    --stage 3 --num-steps "$S3" \
    --hardware "$HW" \
    --num-generations "$G" \
    --resume-from "$OUT/stage2/final" \
    --output-dir "$OUT/stage3/final" \
    2>&1 | tee "$LOGS/stage3_${TS_RUN}.log"

echo "[full] === ALL STAGES COMPLETE ==="
echo "[full] Final LoRA at: $OUT/stage3/final"
echo "[full] Logs at:       $LOGS/stage{1,2,3}_${TS_RUN}.log"
