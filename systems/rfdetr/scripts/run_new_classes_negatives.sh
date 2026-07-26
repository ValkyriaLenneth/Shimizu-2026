#!/usr/bin/env bash
# Train with background negatives, against baseline_v1.
#
# Tests one hypothesis: the models detect the element rather than the damage,
# because every training image contains damage and therefore "brace present" and
# "brace damaged" are indistinguishable in the corpus. See
# docs/development_records/2026-07-26-new-classes-shortcut-learning-finding.md
#
# The only change from baseline_v1 is the training set, which now includes the 141
# previously excluded zero-box images as background samples. Recipe, test split and
# evaluation protocol are identical, so any difference is attributable to the
# negatives alone.
#
#   ブレース  235 positives + 59 negatives = 294 (20% negative)
#   柱脚      179 positives + 82 negatives = 261 (31% negative)
#
# Accepted risk (2026-07-26 decision): all negatives are used without annotator
# triage, and a minority carry real but unannotated damage. If the result is worse
# rather than better, that is the first thing to suspect - rerun with
# --max-audit-score 0.5 on the view builder to drop the most suspicious ones.
#
# Prediction registered before the run, so the reading is not fitted afterwards:
#   * if the shortcut hypothesis holds, precision at fixed recall rises materially
#     and the false-positive count drops; recall may dip slightly
#   * if precision is unchanged, the shortcut hypothesis is wrong and the element
#     detection seen in the audit was ordinary false-positive noise

set -uo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
EPOCHS=${EPOCHS:-80}
TOPK=${TOPK:-3}
TAG=${TAG:-negatives_v1}
GRID=${GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}
MATCH_IOU=${MATCH_IOU:-0.229}
RUN_ROOT="outputs/rfdetr_single_crack/${TAG}"
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs/${TAG}"

cd "$REPO"
mkdir -p "$LOG_DIR"
taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }

run_one() {
  local category="$1" device="$2"
  local run_dir="${RUN_ROOT}/${category}"
  local dataset="data/rfdetr_${category}_bcd_20260725_neg_test_as_valid"
  local eval_dataset="data/rfdetr_${category}_bcd_20260725_test_as_valid"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"

  echo "[$(date -u +%FT%TZ)] START ${category} on ${device}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" --experiment medium --device "$device" \
    --dataset-dir "$dataset" --output-dir "$run_dir" \
    --epochs "$EPOCHS" --batch-size 28 --checkpoint-interval 999 \
    > "$LOG_DIR/${category}_train.log" 2>&1 \
    || { echo "[$(date -u +%FT%TZ)] FAIL train ${category}" >&2; return 1; }
  rm -f "$run_dir"/checkpoint_*.ckpt

  local eps
  eps=$(taskset -c "$CPU_LIST" "$PYTHON" - "$run_dir" "$TOPK" <<'PY'
import csv, sys
run, k = sys.argv[1], int(sys.argv[2])
rows = []
for r in csv.DictReader(open(f"{run}/metrics.csv")):
    try:
        rows.append((float(r["val/mAP_50"]), int(float(r["epoch"]))))
    except (ValueError, KeyError):
        pass
print(",".join(f"{e:03d}" for _, e in sorted(rows, reverse=True)[:k]))
PY
)
  echo "[$(date -u +%FT%TZ)] ${category} candidate epochs: ${eps}"

  # Grid runs against the frozen split, whose test is byte-identical to the neg
  # view's, so the number is directly comparable with baseline_v1.
  local IFS=,
  for ep in $eps; do
    local ck="$run_dir/epoch_pth/checkpoint_epoch_${ep}.pth"
    [[ -f "$ck" ]] || { echo "  missing ${ck}"; continue; }
    taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
      --checkpoint "$ck" --dataset-dir "$eval_dataset" --split test \
      --threshold-grid "$GRID" --iou-threshold "$MATCH_IOU" --num-classes 3 \
      --device "$device" --output-csv "$run_dir/grid_ep${ep}.csv" \
      >> "$LOG_DIR/${category}_grid.log" 2>&1 || echo "  grid ep${ep} failed"
  done
  unset IFS

  taskset -c "$CPU_LIST" "$PYTHON" - "$run_dir" "$eps" <<'PY'
import sys
from pathlib import Path
run, eps = Path(sys.argv[1]), set(sys.argv[2].split(","))
for p in (run / "epoch_pth").glob("checkpoint_epoch_*.pth"):
    if p.stem.replace("checkpoint_epoch_", "") not in eps:
        p.unlink()
PY
  : > "$run_dir/variant_done"
  echo "[$(date -u +%FT%TZ)] DONE ${category}"
}

run_one brace cuda:0 &
A=$!
run_one column_base cuda:1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1
echo "[$(date -u +%FT%TZ)] negatives run finished status=${status}"
exit "$status"
