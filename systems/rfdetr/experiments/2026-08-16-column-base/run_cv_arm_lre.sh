#!/usr/bin/env bash
# A cross-validation arm at a reduced backbone learning rate.
#
# Freezing the encoder outright is one end of a range that has never been
# explored on this project: the delivered recipe moves the backbone at 1.5e-4
# against a head rate of 1e-4, and freezing sets it to zero. If the frozen arm
# wins, the useful question is immediately whether zero is the optimum or merely
# better than 1.5x, since a backbone that adapts a little may beat one that
# cannot adapt at all on a corpus this small.
#
# Everything else matches the control exactly, so the encoder rate is the only
# variable.
# Usage: run_cv_arm_lre.sh <arm-tag> <seed> <cuda-index> <lr-encoder>
set -uo pipefail
ARM="$1"; SEED="$2"; GPU="$3"; LRE="$4"
REPO=/workspace/Shimizu-2026
PY=/workspace/.venv-rfdetr/bin/python
BASE=/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
for f in 0 1 2 3 4; do
  RUN=/workspace/exp_cb/cv_${ARM}_f${f}
  mkdir -p "$RUN"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 开始 (lr_encoder $LRE)"
  timeout --signal=KILL 2400 "$PY" systems/rfdetr/scripts/train_rfdetr_router.py \
    --config systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml \
    --experiment medium --device "cuda:${GPU}" --lr-encoder "$LRE" \
    --dataset-dir "data/rfdetr_column_base_cv5/fold${f}" --output-dir "$RUN" \
    --checkpoint "$BASE" --epochs 12 --batch-size 10 --grad-accum-steps 1 \
    --lr 0.00005 --num-workers 4 --seed "$SEED" --checkpoint-interval 999 \
    > "$RUN/train.log" 2>&1 || echo "  训练退出码非零,用已存 epoch 继续"
  rm -f "$RUN"/*.ckpt "$RUN"/checkpoint_best_*.pth
  "$PY" /workspace/scripts_exp/cv_dump.py "$RUN" \
    "$REPO/data/rfdetr_column_base_cv5/fold${f}" "cuda:${GPU}" > "$RUN/dump.log" 2>&1 \
    || echo "  dump 失败"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 完成"
done
echo "[$(date -u +%FT%TZ)] $ARM 全部完成"
