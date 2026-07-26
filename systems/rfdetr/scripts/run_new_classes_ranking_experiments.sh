#!/usr/bin/env bash
# Ranking / score-separation experiments against baseline_v1.
#
# Motivation, from docs/development_records/2026-07-25-new-classes-baseline-v1-results.md:
# both categories already reach recall 0.80, but only while emitting 20-90 boxes
# per image; the client target needs about 2. The true boxes are present but rank
# 20-90 instead of 2. So every variant here targets score separation, and nothing
# here tries to find more damage.
#
#   varifocal   --use-varifocal-loss    IoU-aware classification score - the most
#                                       directly targeted lever
#   freezeenc   --freeze-encoder        stop degrading DINOv2 features on 235 images
#   lrenc15     --lr-encoder 1.5e-5     the soft version of freezeenc (default 1.5e-4)
#   q100        --num-queries 100       fewer slots competing for rank (default 300)
#
# Everything else is baseline_v1 verbatim, so each variant differs in exactly one
# thing and is directly comparable: RFDETRMedium, 80 epochs, batch 28, lr 1e-4,
# 16-mixed, default resolution, frozen 8:2 split, no augmentation, no crop view.
#
# One GPU per category, variants sequential on each. Candidate epochs are read from
# metrics.csv (valid mirrors test, so per-epoch val IS test) instead of running the
# 80-checkpoint sweep, which would roughly double wall clock for a screening run.
# The winner still gets the full sweep afterwards before anything is shipped.

set -uo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
EPOCHS=${EPOCHS:-80}
TAG=${TAG:-ranking_v1}
TOPK=${TOPK:-3}
GRID=${GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}
MATCH_IOU=${MATCH_IOU:-0.229}
RUN_ROOT="outputs/rfdetr_single_crack/${TAG}"
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs/${TAG}"

declare -a VARIANTS=(
  "varifocal:--use-varifocal-loss"
  "freezeenc:--freeze-encoder"
  "lrenc15:--lr-encoder 0.000015"
  "q100:--num-queries 100"
)

cd "$REPO"
mkdir -p "$LOG_DIR"
taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }
taskset -c "$CPU_LIST" "$PYTHON" systems/rfdetr/scripts/freeze_new_class_datasets.py --check \
  || { echo "frozen dataset check failed - refusing to train on drifted data" >&2; exit 1; }

run_variant() {
  local category="$1" device="$2" name="$3" flags="$4"
  local run_dir="${RUN_ROOT}/${category}_${name}"
  local dataset="data/rfdetr_${category}_bcd_20260725_test_as_valid"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"

  if [[ -f "$run_dir/variant_done" ]]; then
    echo "[$(date -u +%FT%TZ)] skip ${category}/${name} (done)"; return 0
  fi
  echo "[$(date -u +%FT%TZ)] START ${category}/${name} on ${device}: ${flags}"

  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" --experiment medium --device "$device" \
    --output-dir "$run_dir" --epochs "$EPOCHS" --batch-size 28 \
    --checkpoint-interval 999 ${flags} \
    > "$LOG_DIR/${category}_${name}_train.log" 2>&1 \
    || { echo "[$(date -u +%FT%TZ)] FAIL train ${category}/${name}" >&2; return 1; }
  rm -f "$run_dir"/checkpoint_*.ckpt

  # Top-K epochs by val mAP50; valid mirrors test so this is the test metric.
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
  echo "[$(date -u +%FT%TZ)] ${category}/${name} candidate epochs: ${eps}"

  local IFS=,
  for ep in $eps; do
    local ck="$run_dir/epoch_pth/checkpoint_epoch_${ep}.pth"
    [[ -f "$ck" ]] || { echo "  missing ${ck}"; continue; }
    taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
      --checkpoint "$ck" --dataset-dir "$dataset" --split test \
      --threshold-grid "$GRID" --iou-threshold "$MATCH_IOU" --num-classes 3 \
      --device "$device" --output-csv "$run_dir/grid_ep${ep}.csv" \
      >> "$LOG_DIR/${category}_${name}_grid.log" 2>&1 || echo "  grid ep${ep} failed"
  done
  unset IFS

  # Keep only the graded checkpoints; the rest is ~10 GB of dead weight per variant.
  taskset -c "$CPU_LIST" "$PYTHON" - "$run_dir" "$eps" <<'PY'
import sys
from pathlib import Path
run, eps = Path(sys.argv[1]), set(sys.argv[2].split(","))
for p in (run / "epoch_pth").glob("checkpoint_epoch_*.pth"):
    if p.stem.replace("checkpoint_epoch_", "") not in eps:
        p.unlink()
PY
  : > "$run_dir/variant_done"
  echo "[$(date -u +%FT%TZ)] DONE ${category}/${name}"
}

sweep_category() {
  local category="$1" device="$2"
  for spec in "${VARIANTS[@]}"; do
    run_variant "$category" "$device" "${spec%%:*}" "${spec#*:}"
  done
  echo "[$(date -u +%FT%TZ)] all variants finished for ${category}"
}

sweep_category brace cuda:0 &
A=$!
sweep_category column_base cuda:1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1
echo "[$(date -u +%FT%TZ)] ranking experiments finished status=${status}"
exit "$status"
