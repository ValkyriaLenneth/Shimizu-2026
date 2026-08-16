#!/usr/bin/env bash
# A cross-validation arm with the DINOv2 backbone frozen.
#
# The delivered checkpoints were fine-tuned with lr_encoder 1.5e-4 against a head
# lr of 1e-4 -- the backbone moved faster than the detector on 179 images. That is
# the classic setup for overfitting a large pretrained encoder to a small corpus,
# and it has never been varied on this project. Freezing is the strongest single
# test of whether backbone adaptation is helping or hurting here.
# Usage: run_cv_arm_frz.sh <arm-tag> <seed> <cuda-index>
set -uo pipefail
ARM="$1"; SEED="$2"; GPU="$3"
REPO=/workspace/Shimizu-2026
PY=/workspace/.venv-rfdetr/bin/python
BASE=/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
for f in 0 1 2 3 4; do
  RUN=/workspace/exp_cb/cv_${ARM}_f${f}
  mkdir -p "$RUN"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 开始 (freeze-encoder)"
  timeout --signal=KILL 2400 "$PY" systems/rfdetr/scripts/train_rfdetr_router.py \
    --config systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml \
    --experiment medium --device "cuda:${GPU}" --freeze-encoder \
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
