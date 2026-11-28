#!/usr/bin/env bash
# Step 47 training launcher with the recovery protocol defaults.
# Run in tmux/nohup; add RESUME=1 only when last_checkpoint.pt exists.
#
# Usage:
#   MANIFEST=/.../controlled_aug_manifest.csv RUN_ID=recovery_E2_01 \
#     bash run_training.sh
set -euo pipefail

: "${MANIFEST:?Set MANIFEST to the controlled augmented manifest}"
: "${RUN_ID:?Set RUN_ID to a new run folder name}"

OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/acne_project/output}"
TRAIN_DIR="${TRAIN_DIR:-$OUTPUT_ROOT/training_runs}"
OUTPUT_DIR="${OUTPUT_DIR:-$TRAIN_DIR/$RUN_ID}"
WORKDIR="${WORKDIR:-/workspace/acne_segmentation_github}"

BATCH_SIZE="${BATCH_SIZE:-4}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-20}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-30}"
CLASS_WEIGHTING="${CLASS_WEIGHTING:-none}"
CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-val_macro_f1}"
PATIENCE="${PATIENCE:-5}"
MAX_OPTIMIZER_STEPS="${MAX_OPTIMIZER_STEPS:-0}"   # isi dari Step 46 reference_updates; 0 = nonaktif
NUM_WORKERS="${NUM_WORKERS:-4}"
RESUME="${RESUME:-0}"

if [ -f "$OUTPUT_DIR/metrics.json" ]; then
  echo "REFUSE: $OUTPUT_DIR already has metrics.json. Use a new RUN_ID." >&2
  exit 2
fi

EXTRA=()
if [ "$RESUME" = "1" ]; then
  if [ ! -f "$OUTPUT_DIR/last_checkpoint.pt" ]; then
    echo "REFUSE: RESUME=1 but $OUTPUT_DIR/last_checkpoint.pt is missing." >&2
    exit 2
  fi
  EXTRA+=(--resume)
fi

mkdir -p "$OUTPUT_DIR"
echo "Commit : $(git -C "$WORKDIR" rev-parse HEAD)"
echo "Manifest: $MANIFEST"
echo "Output  : $OUTPUT_DIR"

cd "$WORKDIR"
exec python scripts/train_proposal_hybrid_resnet50_malds.py \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --image-size 512 \
  --batch-size "$BATCH_SIZE" \
  --phase1-epochs "$PHASE1_EPOCHS" \
  --phase2-epochs "$PHASE2_EPOCHS" \
  --pretrained \
  --num-workers "$NUM_WORKERS" \
  --device cuda \
  --class-weighting "$CLASS_WEIGHTING" \
  --checkpoint-metric "$CHECKPOINT_METRIC" \
  --early-stopping-patience "$PATIENCE" \
  --max-optimizer-steps "$MAX_OPTIMIZER_STEPS" \
  --skip-test-eval \
  "${EXTRA[@]}"
