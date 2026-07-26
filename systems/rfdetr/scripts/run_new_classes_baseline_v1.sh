#!/usr/bin/env bash
# Plain baseline on the frozen single train/test split - the reference point that
# every later improvement is measured against.
#
# "Plain" is meant literally. Nothing is added on top of the recipe the four
# delivered categories (天井 / 内壁 / RC壁 / RC柱) were trained with:
#
#   RFDETRMedium, 80 epochs, batch 28, grad accum 1, lr 1e-4, 16-mixed,
#   default resolution (576 for medium), no crop view, no --aug-config,
#   no external eval profiles, valid mirroring test.
#
# Deliberately NOT included, so this stays a baseline:
#   * crop / oversampled / class-boosted train views
#   * augmentation presets
#   * lr_encoder overrides, num_queries changes, warm-start checkpoints
#   * tiled inference
#
# Two operational deviations, neither of which changes the recipe:
#   * --checkpoint-interval 999 suppresses the 511 MB per-epoch Lightning .ckpt
#     files (~41 GB per run, which previously filled the disk and broke a sweep).
#     Only epoch_pth/*.pth is needed for evaluation and those are still written.
#   * every command runs under an affinity mask, because CPU 1 on this host is
#     advertised online but is not schedulable - anything pinned to it wedges in
#     state R and cannot be killed, including /bin/true.
#
# Data is the frozen 8:2 split. Verify it before trusting any comparison:
#   systems/rfdetr/scripts/freeze_new_class_datasets.py --check

set -uo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
EPOCHS=${EPOCHS:-80}
LR=${LR:-0.0001}
BATCH=${BATCH:-28}
TAG=${TAG:-baseline_v1}
RUN_ROOT="outputs/rfdetr_single_crack/${TAG}"
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs/${TAG}"

# Full per-class threshold grid at the project's match IoU. The delivered numbers
# for all four existing categories are per-class threshold-tuned at IoU 0.229, so
# only this protocol is comparable to them.
GRID=${GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}
MATCH_IOU=${MATCH_IOU:-0.229}

cd "$REPO"
mkdir -p "$LOG_DIR"
taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }

taskset -c "$CPU_LIST" "$PYTHON" systems/rfdetr/scripts/freeze_new_class_datasets.py --check \
  || { echo "frozen dataset check failed - refusing to train on drifted data" >&2; exit 1; }

run_one() {
  local category="$1" device="$2"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"
  local run_dir="${RUN_ROOT}/${category}"
  local dataset="data/rfdetr_${category}_bcd_20260725_test_as_valid"

  echo "[$(date -u +%FT%TZ)] train ${category} on ${device}: batch ${BATCH}, lr ${LR}, ${EPOCHS} epochs"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" --experiment medium --device "$device" \
    --output-dir "$run_dir" --epochs "$EPOCHS" --lr "$LR" \
    --batch-size "$BATCH" --checkpoint-interval 999 \
    > "$LOG_DIR/${category}_train.log" 2>&1 || { echo "${category}: training failed" >&2; return 1; }
  rm -f "$run_dir"/checkpoint_*.ckpt

  # Step 1 of the established selection protocol: reload and force-evaluate every
  # saved epoch, because checkpoint_best_total.pth is chosen by mAP rather than
  # recall-first. This is how RC柱 epoch 47 was found to beat best_total.
  echo "[$(date -u +%FT%TZ)] sweep ${category}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/sweep_rfdetr_router_test.py \
    --run-dir "$run_dir" --dataset-dir "$dataset" --device "$device" \
    --output-csv "$run_dir/test_results.csv" --skip-existing \
    > "$LOG_DIR/${category}_sweep.log" 2>&1 || echo "${category}: sweep failed" >&2

  echo "[$(date -u +%FT%TZ)] done ${category}"
}

run_one brace cuda:0 &
A=$!
run_one column_base cuda:1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1

echo "[$(date -u +%FT%TZ)] baseline training finished status=${status}"
echo "next: per-class threshold grid at IoU ${MATCH_IOU} on the top recall epochs, e.g."
echo "  $PYTHON scripts/evaluate_rfdetr_class_threshold_grid.py \\"
echo "    --checkpoint ${RUN_ROOT}/<cat>/epoch_pth/checkpoint_epoch_0NN.pth \\"
echo "    --dataset-dir data/rfdetr_<cat>_bcd_20260725_test_as_valid --split test \\"
echo "    --threshold-grid ${GRID} --iou-threshold ${MATCH_IOU} --num-classes 3 \\"
echo "    --output-csv ${RUN_ROOT}/<cat>/grid.csv"
exit "$status"
