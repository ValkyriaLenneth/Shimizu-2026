#!/usr/bin/env bash
# A cross-validation arm at a non-default input resolution.
#
# Effective batch is held at 10 by pairing a smaller per-step batch with gradient
# accumulation, so resolution is the only thing that differs from the control.
# Raising resolution alone would have forced the batch down and made the arm a
# two-variable change, which is how several of this week's claims became
# uninterpretable.
#
# Usage: run_cv_arm_res.sh <arm-tag> <seed> <cuda-index> <resolution> <batch> <accum>
set -uo pipefail
ARM="$1"; SEED="$2"; GPU="$3"; RES="$4"; BS="${5:-5}"; ACC="${6:-2}"
REPO=/workspace/Shimizu-2026
PY=/workspace/.venv-rfdetr/bin/python
BASE=/workspace/handoff_20260726/checkpoints/column_base_negatives_v1_epoch_016.pth
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
for f in 0 1 2 3 4; do
  RUN=/workspace/exp_cb/cv_${ARM}_f${f}
  mkdir -p "$RUN"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 开始 (res $RES, batch $BS x accum $ACC)"
  timeout --signal=KILL 2400 "$PY" systems/rfdetr/scripts/train_rfdetr_router.py \
    --config systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml \
    --experiment medium --device "cuda:${GPU}" --resolution "$RES" \
    --dataset-dir "data/rfdetr_column_base_cv5/fold${f}" --output-dir "$RUN" \
    --checkpoint "$BASE" --epochs 12 --batch-size "$BS" --grad-accum-steps "$ACC" \
    --lr 0.00005 --num-workers 4 --seed "$SEED" --checkpoint-interval 999 \
    > "$RUN/train.log" 2>&1 || echo "  训练退出码非零,用已存 epoch 继续"
  rm -f "$RUN"/*.ckpt "$RUN"/checkpoint_best_*.pth
  "$PY" /workspace/scripts_exp/cv_dump.py "$RUN" \
    "$REPO/data/rfdetr_column_base_cv5/fold${f}" "cuda:${GPU}" > "$RUN/dump.log" 2>&1 \
    || echo "  dump 失败"
  echo "[$(date -u +%FT%TZ)] $ARM fold$f 完成"
done
echo "[$(date -u +%FT%TZ)] $ARM 全部完成"
