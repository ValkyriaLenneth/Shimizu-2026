#!/usr/bin/env bash
# Per-class B/C/D threshold grid for the two new downstream categories.
#
# Split out of the training launcher so it can be re-run on its own: training and
# the epoch sweep are expensive and their artifacts are already on disk, while
# this step is cheap and was the one that failed. It failed because
# evaluate_rfdetr_class_threshold_grid.py called match_counts() with 3 arguments
# after that function gained a required num_classes parameter - a pre-existing
# signature mismatch in the repo, now fixed.
#
# match IoU 0.229 is the report protocol used for the four delivered categories,
# so it is the only setting comparable to their published numbers.
#
# CPU 1 on this host is advertised as online but is not schedulable; anything
# pinned to it wedges unkillably, and torch pins itself to every core during init.
# Hence the affinity mask on every command.

set -euo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
EXPERIMENT=${EXPERIMENT:-medium}
SUFFIX=${SUFFIX:-bcd_20260725_split91_test_as_valid}
MATCH_IOU=0.229
THRESHOLD_GRID=${THRESHOLD_GRID:-"0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50"}
TOP=${TOP:-3}
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs"

cd "$REPO"
mkdir -p "$LOG_DIR"

grid_category() {
  local category="$1"
  local device="$2"
  local run_dir="outputs/rfdetr_single_crack/${category}_${EXPERIMENT}_${SUFFIX}"
  local dataset="data/rfdetr_${category}_${SUFFIX}"

  local candidates
  candidates=$("$PYTHON" systems/rfdetr/scripts/report_new_class_training_status.py \
    --list-top-checkpoints "$run_dir" --top "$TOP" 2>/dev/null || true)
  if [[ -z "$candidates" ]]; then
    echo "[$(date -u +%FT%TZ)] no sweep candidates for ${category}; is test_results.csv present?" >&2
    return 1
  fi

  while read -r ckpt; do
    [[ -n "$ckpt" ]] || continue
    local tag
    tag=$(basename "$ckpt" .pth)
    echo "[$(date -u +%FT%TZ)] grid ${category} ${tag} on ${device}"
    taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
      --checkpoint "$ckpt" \
      --dataset-dir "$dataset" \
      --split test \
      --threshold-grid "$THRESHOLD_GRID" \
      --iou-threshold "$MATCH_IOU" \
      --num-classes 3 \
      --output-csv "$run_dir/class_threshold_grid_${tag}.csv" \
      --device "$device" \
      >> "$LOG_DIR/${category}_threshold_grid.log" 2>&1
    echo "[$(date -u +%FT%TZ)] done grid ${category} ${tag}"
  done <<< "$candidates"
}

grid_category brace cuda:0 &
BRACE_PID=$!
grid_category column_base cuda:1 &
COLUMN_PID=$!

status=0
wait "$BRACE_PID" || status=1
wait "$COLUMN_PID" || status=1

if [[ $status -ne 0 ]]; then
  echo "threshold grid: at least one category failed; check $LOG_DIR"
  exit 1
fi

echo "threshold grid finished for all categories"
