#!/usr/bin/env bash
# A short, low-rate fine-tune: the recipe the epoch curves point at.
#
# Across six arms the best epoch was the first in five cases and the third in the
# sixth, and the last epoch was always markedly worse. Part of that is
# contamination decay -- the starting checkpoint saw every image including each
# fold's evaluation half -- so the curve alone cannot prove that training hurts.
# But it does mean nothing measured so far has tested the obvious alternative:
# stop early and move less.
#
# Twelve epochs at lr 5e-5 becomes three at 1e-5, holding everything else fixed.
# If a short schedule loses less than the standard one against the same control,
# the fine-tune recipe rather than the data is what has been costing precision.
# Usage: run_cv_arm_short.sh <arm-tag> <seed> <cuda-index> [epochs] [lr]
set -uo pipefail
ARM="$1"; SEED="$2"; GPU="$3"; EPOCHS="${4:-3}"; LR="${5:-0.00001}"
REPO=/workspace/Shimizu-2026
PY=/workspace/.venv-rfdetr/bin/python
BASE=/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
for f in 0 1 2 3 4; do
  RUN=/workspace/exp_cb/cv_${ARM}_f${f}
  mkdir -p "$RUN"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 开始 (epochs $EPOCHS, lr $LR)"
  timeout --signal=KILL 2400 "$PY" systems/rfdetr/scripts/train_rfdetr_router.py \
    --config systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml \
    --experiment medium --device "cuda:${GPU}" \
    --dataset-dir "data/rfdetr_column_base_cv5/fold${f}" --output-dir "$RUN" \
    --checkpoint "$BASE" --epochs "$EPOCHS" --batch-size 10 --grad-accum-steps 1 \
    --lr "$LR" --num-workers 4 --seed "$SEED" --checkpoint-interval 999 \
    > "$RUN/train.log" 2>&1 || echo "  训练退出码非零,用已存 epoch 继续"
  rm -f "$RUN"/*.ckpt "$RUN"/checkpoint_best_*.pth
  "$PY" /workspace/scripts_exp/cv_dump.py "$RUN" \
    "$REPO/data/rfdetr_column_base_cv5/fold${f}" "cuda:${GPU}" > "$RUN/dump.log" 2>&1 \
    || echo "  dump 失败"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 完成"
done
echo "[$(date -u +%FT%TZ)] $ARM 全部完成"
