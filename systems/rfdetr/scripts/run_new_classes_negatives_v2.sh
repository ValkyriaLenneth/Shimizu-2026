#!/usr/bin/env bash
# Round 2 on the negatives line, which is the only intervention so far that moved
# the binding constraint (recall at precision >= 0.60: ブレース 0.434 -> 0.470,
# 柱脚 0.417 -> 0.514, with false positives roughly halved).
#
# Two variants, both raising negative exposure without new photographs:
#
#   neg2x     each of the 141 zero-box images emitted twice
#             ブレース 33% negative, 柱脚 48%
#             Motivated by the dose-response in round 1: 柱脚 ran at 31% negatives
#             and gained +0.097 recall at the floor, ブレース at 20% gained +0.036.
#
#   negcrop   crops around the *baseline* model's detections on those same images,
#             at context 3.0 matching the positive crop view
#             ブレース 33% negative, 柱脚 40%
#             A whole-image negative is mostly floor, sky and wall the model never
#             confused for damage. The sharper negative is an intact element framed
#             the way positives are framed. We have no element boxes on undamaged
#             images, but baseline_v1 is effectively an element detector - that is
#             the shortcut it learned - so its own detections localise them.
#
# Everything else is baseline_v1 verbatim and the test split is byte-identical to
# the frozen one in all four datasets (verified), so results are directly
# comparable across rounds.
#
# Prediction registered before the run:
#   * if more negative exposure keeps helping, neg2x beats negatives_v1 and the
#     dose-response continues
#   * if negatives_v1 already saturated, neg2x is flat or slightly worse (too much
#     of the batch spent on empty images starves the positive signal)
#   * negcrop should beat neg2x at equal negative fraction if hardness matters more
#     than count; if it does not, whole-image negatives were already hard enough

set -uo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
EPOCHS=${EPOCHS:-80}
TOPK=${TOPK:-3}
TAG=${TAG:-negatives_v2}
GRID=${GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}
MATCH_IOU=${MATCH_IOU:-0.229}
RUN_ROOT="outputs/rfdetr_single_crack/${TAG}"
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs/${TAG}"

declare -a VARIANTS=("neg2x" "negcrop")

cd "$REPO"
mkdir -p "$LOG_DIR"
taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }

run_variant() {
  local category="$1" device="$2" name="$3"
  local run_dir="${RUN_ROOT}/${category}_${name}"
  local dataset="data/rfdetr_${category}_bcd_20260725_${name}_test_as_valid"
  local eval_dataset="data/rfdetr_${category}_bcd_20260725_test_as_valid"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"

  if [[ -f "$run_dir/variant_done" ]]; then echo "[$(date -u +%FT%TZ)] skip ${category}/${name}"; return 0; fi
  [[ -d "$dataset" ]] || { echo "[$(date -u +%FT%TZ)] FAIL ${category}/${name}: missing ${dataset}" >&2; return 1; }

  echo "[$(date -u +%FT%TZ)] START ${category}/${name} on ${device}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" --experiment medium --device "$device" \
    --dataset-dir "$dataset" --output-dir "$run_dir" \
    --epochs "$EPOCHS" --batch-size 28 --checkpoint-interval 999 \
    > "$LOG_DIR/${category}_${name}_train.log" 2>&1 \
    || { echo "[$(date -u +%FT%TZ)] FAIL train ${category}/${name}" >&2; return 1; }
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
  echo "[$(date -u +%FT%TZ)] ${category}/${name} candidate epochs: ${eps}"

  local IFS=,
  for ep in $eps; do
    local ck="$run_dir/epoch_pth/checkpoint_epoch_${ep}.pth"
    [[ -f "$ck" ]] || continue
    taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
      --checkpoint "$ck" --dataset-dir "$eval_dataset" --split test \
      --threshold-grid "$GRID" --iou-threshold "$MATCH_IOU" --num-classes 3 \
      --device "$device" --output-csv "$run_dir/grid_ep${ep}.csv" \
      >> "$LOG_DIR/${category}_${name}_grid.log" 2>&1 || echo "  grid ep${ep} failed"
  done
  unset IFS

  # Report the headline number straight into the driver log so the monitor surfaces
  # it without a follow-up query.
  taskset -c "$CPU_LIST" "$PYTHON" - "$run_dir" "$category" "$name" <<'PY'
import csv, glob, sys
run, cat, name = sys.argv[1], sys.argv[2], sys.argv[3]
rows = []
for f in glob.glob(f"{run}/grid_ep*.csv"):
    for r in csv.DictReader(open(f)):
        try:
            rows.append((float(r["f1"]), float(r["precision"]), float(r["recall"])))
        except (ValueError, KeyError):
            pass
if rows:
    bf = max(rows, key=lambda t: t[0])
    at60 = max([t for t in rows if t[1] >= 0.60], key=lambda t: t[2], default=None)
    tail = f", R@P>=.60 {at60[2]:.3f}" if at60 else ", R@P>=.60 none"
    print(f"RESULT {cat}/{name}: bestF1 {bf[0]:.3f} (R {bf[2]:.3f}/P {bf[1]:.3f}){tail}")
PY

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
  for name in "${VARIANTS[@]}"; do run_variant "$category" "$device" "$name"; done
  echo "[$(date -u +%FT%TZ)] all variants finished for ${category}"
}

sweep_category brace cuda:0 &
A=$!
sweep_category column_base cuda:1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1
echo "[$(date -u +%FT%TZ)] negatives_v2 finished status=${status}"
exit "$status"
