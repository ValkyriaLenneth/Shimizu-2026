#!/usr/bin/env bash
# Baseline training for the two new downstream categories, ブレース and 柱脚.
#
# The recipe is identical to the RC柱 / RC壁 / 内壁 / 天井 baselines: RFDETRMedium,
# 80 epochs, batch 28, grad accum 1, lr 1e-4, 16-mixed, default resolution.
# `small` is not trained - every documented downstream run in this project uses
# `--experiment medium`, and all four released checkpoints come from the
# `*_medium_*` runs.
#
# One category per GPU, both concurrently:
#
#   GPU 0: brace medium
#   GPU 1: column_base medium
#
# After training, each run goes through the two evaluation steps that the other
# four categories were selected with. They matter: the automatic
# checkpoint_best_total.pth is picked by mAP, not recall-first, so it is not the
# checkpoint to ship.
#
#   1. sweep_rfdetr_router_test.py    - reload every saved epoch checkpoint and
#                                       force-evaluate it on the official test
#                                       split, giving a recall-first ranking
#   2. evaluate_rfdetr_class_threshold_grid.py - per-class B/C/D thresholds at
#                                       match IoU 0.229, which is the protocol
#                                       the delivered client numbers use
#
# CPU 1 on this host is advertised as online but is not schedulable: any process
# pinned to it wedges permanently in state R, unkillable even by SIGKILL. torch's
# cpuinfo topology probe pins itself to each CPU of its affinity mask in turn, so
# a plain `import torch` hangs forever on CPU 1. Every command below is therefore
# launched under an affinity mask that excludes it. Remove CPU_LIST only after
# verifying `taskset -c 1 /bin/true` returns.
#
# Usage:
#   systems/rfdetr/scripts/run_new_classes_baseline_comparison.sh          # both GPUs
#   systems/rfdetr/scripts/run_new_classes_baseline_comparison.sh brace 0  # one category

set -euo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
EXPERIMENT=${EXPERIMENT:-medium}
SUFFIX=${SUFFIX:-bcd_20260725_split91_test_as_valid}
# LR and TAG make this a general experiment runner: TAG is appended to the run
# directory so variants never overwrite each other, and LR overrides the config.
LR=${LR:-}
TAG=${TAG:-}
MATCH_IOU=0.229
THRESHOLD_GRID="0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50"
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs"

cd "$REPO"
mkdir -p "$LOG_DIR"

if taskset -c "$CPU_LIST" /bin/true; then
  echo "affinity mask $CPU_LIST is schedulable"
else
  echo "affinity mask $CPU_LIST is not usable; aborting" >&2
  exit 1
fi

run_category() {
  local category="$1"
  local device="$2"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"
  local dataset="data/rfdetr_${category}_${SUFFIX}"
  local run_dir="outputs/rfdetr_single_crack/${category}_${EXPERIMENT}_${TAG:+${TAG}_}${SUFFIX}"

  echo "[$(date -u +%FT%TZ)] start ${category}/${EXPERIMENT} on ${device}"
  # --checkpoint-interval 999 suppresses the per-epoch Lightning .ckpt files.
  # Those are full training-resume state at ~511 MB each; at the config default of
  # 1 they accumulate to ~41 GB per 80-epoch run and filled the 199 GB disk, which
  # made the sweep fail with "checkpoint_epoch_043.pth cannot be opened". The
  # project's own overnight scripts use 999 for the same reason. Evaluation only
  # needs epoch_pth/*.pth, which save_epoch_pth still writes (~134 MB each).
  # --dataset-dir / --output-dir are passed explicitly and derived from SUFFIX, so
  # SUFFIX is the single source of truth. Without this the config's dataset.dir
  # would silently win for training while the sweep below used the SUFFIX path -
  # they would train and evaluate on different datasets.
  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" \
    --experiment "$EXPERIMENT" \
    --device "$device" \
    --dataset-dir "$dataset" \
    --output-dir "$run_dir" \
    ${EPOCHS:+--epochs "$EPOCHS"} \
    ${LR:+--lr "$LR"} \
    --checkpoint-interval 999 \
    2>&1 | tee "$LOG_DIR/${category}_${EXPERIMENT}${TAG:+_${TAG}}.log"

  # Belt and braces: drop any .ckpt that still landed, before the sweep runs.
  rm -f "$run_dir"/checkpoint_*.ckpt
  echo "[$(date -u +%FT%TZ)] done ${category}/${EXPERIMENT}"

  # Step 1: recall-first ranking across all saved epoch checkpoints.
  echo "[$(date -u +%FT%TZ)] sweep ${category}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/sweep_rfdetr_router_test.py \
    --run-dir "$run_dir" \
    --dataset-dir "$dataset" \
    --output-csv "$run_dir/test_results.csv" \
    --device "$device" \
    --batch-size 28 \
    --num-workers 8 \
    --precision 16-mixed \
    2>&1 | tee "$LOG_DIR/${category}${TAG:+_${TAG}}_sweep.log"
  echo "[$(date -u +%FT%TZ)] done sweep ${category}"

  # Step 2: per-class threshold grid at the report match IoU, for the top
  # recall-first checkpoints picked from the sweep.
  echo "[$(date -u +%FT%TZ)] threshold grid ${category}"
  local candidates
  candidates=$(taskset -c "$CPU_LIST" "$PYTHON" \
    systems/rfdetr/scripts/report_new_class_training_status.py \
    --list-top-checkpoints "$run_dir" --top 3 2>/dev/null || true)
  if [[ -z "$candidates" ]]; then
    echo "no sweep candidates resolved for ${category}; skipping threshold grid" >&2
  else
    while read -r ckpt; do
      [[ -n "$ckpt" ]] || continue
      local tag
      tag=$(basename "$ckpt" .pth)
      taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
        --checkpoint "$ckpt" \
        --dataset-dir "$dataset" \
        --split test \
        --threshold-grid "$THRESHOLD_GRID" \
        --iou-threshold "$MATCH_IOU" \
        --num-classes 3 \
        --output-csv "$run_dir/class_threshold_grid_${tag}.csv" \
        --device "$device" \
        2>&1 | tee -a "$LOG_DIR/${category}${TAG:+_${TAG}}_threshold_grid.log"
    done <<< "$candidates"
  fi
  echo "[$(date -u +%FT%TZ)] done threshold grid ${category}"
}

if [[ $# -ge 1 ]]; then
  run_category "$1" "cuda:${2:-0}"
  exit 0
fi

run_category brace cuda:0 &
BRACE_PID=$!
run_category column_base cuda:1 &
COLUMN_PID=$!

status=0
wait "$BRACE_PID" || status=1
wait "$COLUMN_PID" || status=1

if [[ $status -ne 0 ]]; then
  echo "at least one run failed; check $LOG_DIR"
  exit 1
fi

echo "all runs finished; logs in $LOG_DIR"
