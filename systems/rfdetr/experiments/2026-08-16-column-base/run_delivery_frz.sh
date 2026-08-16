#!/usr/bin/env bash
# Train the frozen-backbone recipe on the delivery split and score it there.
#
# The cross-validation arm measured +0.188 precision and -0.41 boxes per sound
# image against its control, but those folds mix the frozen test split into
# training and the absolute numbers are inflated for every arm alike. A paired
# difference under that protocol says the recipe is better; it does not say what
# the delivery would report. Only the 45-image protocol can say that, and it
# needs a model trained on the 179 training images alone.
#
# Two seeds, because this is a training-dependent claim and the project's
# single-run reproducibility is 0.06-0.10. Both are scored as single models and
# as WBF members alongside the two shipped checkpoints.
# Usage: run_delivery_frz.sh <tag> <seed> <cuda-index> [extra trainer args...]
set -uo pipefail
TAG="$1"; SEED="$2"; GPU="$3"; shift 3
REPO=/workspace/Shimizu-2026
PY=/workspace/.venv-rfdetr/bin/python
RUN=/workspace/exp_cb/$TAG
BASE=/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"; cd "$REPO"
echo "[$(date -u +%FT%TZ)] $TAG 开始 (seed $SEED, $*)"
timeout --signal=KILL 2400 "$PY" systems/rfdetr/scripts/train_rfdetr_router.py \
  --config systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml \
  --experiment medium --device "cuda:${GPU}" \
  --dataset-dir data/rfdetr_column_base_bcd_20260725_test_as_valid \
  --output-dir "$RUN" --checkpoint "$BASE" \
  --epochs 12 --batch-size 10 --grad-accum-steps 1 --lr 0.00005 \
  --num-workers 4 --seed "$SEED" --checkpoint-interval 999 "$@" \
  > "$RUN/train.log" 2>&1 || echo "  训练退出码非零,用已存 epoch 继续"
rm -f "$RUN"/*.ckpt "$RUN"/checkpoint_best_*.pth
echo "[$(date -u +%FT%TZ)] $TAG 训练完成,$(ls $RUN/epoch_pth/*.pth 2>/dev/null|wc -l) 个 epoch"
