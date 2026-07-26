#!/usr/bin/env bash
# 5-fold cross-validation for the new categories, with a protocol designed to give
# an unbiased performance estimate rather than another selected-on-test maximum.
#
# The problem it solves: the single 9:1 split holds 39 boxes for ブレース and 38 for
# 柱脚. Across six experiments we evaluated roughly 60000 (checkpoint, threshold)
# configurations on those boxes and reported the maximum, with valid mirroring test.
# That maximum is optimistically biased and cannot distinguish a real gain from
# noise.
#
# Protocol, chosen so nothing is selected on the fold's own test data:
#   * fixed recipe    - medium, crop2 view, lr 3e-5, 12 epochs, batch 28
#   * fixed epoch     - an a-priori epoch index, NOT the best epoch on this fold
#   * fixed thresholds- the per-category B/C/D triple already chosen on the 9:1
#                       split, not re-tuned per fold
#   * pooled scoring  - tp/fp/fn summed over all five folds, so the estimate rests
#                       on 477 / 320 boxes instead of 39 / 38
#
# Residual caveat, stated rather than hidden: the fixed thresholds were originally
# chosen on the 9:1 test split, whose images also appear in the CV folds. That is
# one configuration carried over instead of 60000 selected in place, so the bias is
# drastically reduced but not zero.
#
# CPU 1 on this host is unschedulable and wedges anything pinned to it, so every
# command runs under an affinity mask. --checkpoint-interval 999 suppresses the
# 511 MB per-epoch Lightning .ckpt files.

set -uo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
FOLDS=${FOLDS:-5}
EPOCHS=${EPOCHS:-12}
# EVAL_EPOCH is fixed in advance from the peak-step analysis of the earlier runs:
# peak mAP has landed at roughly 200 optimizer steps in every run, and a crop2 fold
# gives about 35 steps per epoch, so epoch 6 is where the peak is expected. It is
# chosen from prior runs, never from a fold's own test data.
#
# The first attempt at this used the FINAL epoch instead. That was a mistake: at 30
# epochs (~1050 steps) the final checkpoint was far past the peak and had collapsed
# on the D grade - fold 0 gave D recall 1/18 = 0.056 against 0.62 for the best
# checkpoint on the single split. It measured epoch decay, not selection bias.
EVAL_EPOCH=${EVAL_EPOCH:-6}
LR=${LR:-0.00003}
VIEW=${VIEW:-cv5crop2_20260725}
# CV_TAG separates run directories per train view, so a baseline CV and a crop
# CV can coexist and be compared rather than overwrite one another.
CV_TAG=${CV_TAG:-crop2}
# Optional augmentation preset, passed through to --aug-config. RF-DETR builds
# only a single horizontal-flip transform by default, so this is the knob that
# actually changes how much augmentation the run sees.
AUG_CONFIG=${AUG_CONFIG:-}
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs/cv"

# Fixed per-category thresholds, carried over from the 9:1 crop2+lr3e-5 runs.
declare -A GRID=( [brace]="0.3,0.35,0.4" [column_base]="0.25,0.45,0.5" )

cd "$REPO"
mkdir -p "$LOG_DIR"
taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }

train_fold() {
  local category="$1" fold="$2" device="$3"
  local dataset="data/rfdetr_${category}_${VIEW}_fold${fold}_test_as_valid"
  local run_dir="outputs/rfdetr_single_crack/cv/${CV_TAG}/${category}_fold${fold}"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"

  if [[ -f "$run_dir/cv_done" ]]; then
    echo "[$(date -u +%FT%TZ)] skip ${category} fold${fold} (already done)"
    return 0
  fi

  echo "[$(date -u +%FT%TZ)] train ${category} fold${fold} on ${device}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" --experiment medium --device "$device" \
    --dataset-dir "$dataset" --output-dir "$run_dir" \
    --epochs "$EPOCHS" --lr "$LR" --checkpoint-interval 999 \
    ${AUG_CONFIG:+--aug-config "$AUG_CONFIG"} \
    > "$LOG_DIR/${category}_fold${fold}_train.log" 2>&1
  rm -f "$run_dir"/checkpoint_*.ckpt

  # Primary, unbiased: the a-priori fixed epoch.
  local fixed
  fixed=$(printf '%s/epoch_pth/checkpoint_epoch_%03d.pth' "$run_dir" "$EVAL_EPOCH")
  if [[ ! -f "$fixed" ]]; then
    echo "[$(date -u +%FT%TZ)] ${category} fold${fold}: epoch ${EVAL_EPOCH} checkpoint missing" >&2
    return 1
  fi
  taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
    --checkpoint "$fixed" --dataset-dir "$dataset" --split test \
    --threshold-grid "${GRID[$category]}" --iou-threshold 0.229 --num-classes 3 \
    --output-csv "$run_dir/cv_grid.csv" --device "$device" \
    > "$LOG_DIR/${category}_fold${fold}_eval.log" 2>&1 || return 1

  # Secondary, deliberately biased: every epoch at the same fixed thresholds, so the
  # best-epoch-per-fold number can be reported as an upper bound alongside the
  # unbiased one. Reporting both brackets the truth instead of guessing where it is.
  for ckpt in "$run_dir"/epoch_pth/checkpoint_epoch_*.pth; do
    local ep
    ep=$(basename "$ckpt" .pth | sed 's/checkpoint_epoch_//')
    taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
      --checkpoint "$ckpt" --dataset-dir "$dataset" --split test \
      --threshold-grid "${GRID[$category]}" --iou-threshold 0.229 --num-classes 3 \
      --output-csv "$run_dir/cv_grid_epoch_${ep}.csv" --device "$device" \
      >> "$LOG_DIR/${category}_fold${fold}_eval.log" 2>&1
  done

  # Keep only the fixed-epoch checkpoint once every epoch has been scored.
  find "$run_dir/epoch_pth" -name "checkpoint_epoch_*.pth" ! -samefile "$fixed" -delete 2>/dev/null
  : > "$run_dir/cv_done"
  echo "[$(date -u +%FT%TZ)] done ${category} fold${fold}"
}

status=0
for fold in $(seq 0 $((FOLDS - 1))); do
  train_fold brace "$fold" cuda:0 &
  A=$!
  train_fold column_base "$fold" cuda:1 &
  B=$!
  wait "$A" || status=1
  wait "$B" || status=1
done

echo "[$(date -u +%FT%TZ)] cv training finished status=$status"
taskset -c "$CPU_LIST" "$PYTHON" systems/rfdetr/scripts/report_new_class_cv.py \
  --run-root "outputs/rfdetr_single_crack/cv/${CV_TAG}"
echo "[$(date -u +%FT%TZ)] cv pipeline finished"
