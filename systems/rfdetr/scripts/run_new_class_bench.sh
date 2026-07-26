#!/usr/bin/env bash
# Fast iteration bench for the two new categories.
#
# One fixed test set, one command per experiment, results appended to a shared
# table. Roughly six minutes per experiment against the ~50 minutes a 5-fold CV
# costs, which is the trade this bench exists to make.
#
# The fixed split is fold 3 of the 5-fold layout, chosen for two reasons: it landed
# closest to the pooled cross-validated truth of all five folds (combined deviation
# 0.076, the smallest), and its test set is three times the size of the old 9:1
# split - 117 boxes for ブレース and 60 for 柱脚 against 39 and 38 - so it is also the
# quieter measurement.
#
# What this bench cannot do: resolve differences below roughly 0.05 F1. Fold-to-fold
# spread was measured at 0.070, and a single split carries noise of that order. Use
# it to rank ideas and discard the ones that clearly do nothing; confirm anything
# that looks like a winner with run_new_classes_cv.sh before reporting it.
#
# Selection bias also accumulates: every experiment scored against the same test set
# makes the best-so-far number a little more optimistic. The results table records
# the count so the discount is visible at the end.
#
# Usage:
#   run_new_class_bench.sh NAME [KEY=VALUE ...]
#
#   NAME              label for the results table and run directory
#   VIEW=...          dataset view stem, default cv5crop2_20260725
#   EPOCHS=...        default 12
#   LR=...            default 0.00003
#   EVAL_EPOCH=...    default 6
#   AUG_CONFIG=...    path to an augmentation yaml, default none
#   RESOLUTION=...    override model resolution, default model default
#
# Example:
#   run_new_class_bench.sh strongaug AUG_CONFIG=.../aug_new_class_strong.yaml

set -uo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
FOLD=${FOLD:-3}
VIEW=${VIEW:-cv5crop2_20260725}
EPOCHS=${EPOCHS:-12}
LR=${LR:-0.00003}
EVAL_EPOCH=${EVAL_EPOCH:-6}
AUG_CONFIG=${AUG_CONFIG:-}
RESOLUTION=${RESOLUTION:-}
BENCH_DIR="$REPO/outputs/rfdetr_new_classes/bench"
RESULTS="$BENCH_DIR/results.csv"
LOG_DIR="$BENCH_DIR/logs"

declare -A GRID=( [brace]="0.3,0.35,0.4" [column_base]="0.25,0.45,0.5" )
declare -A THR=( [brace]="0.3,0.35,0.4" [column_base]="0.25,0.5,0.45" )

NAME=${1:-}
if [[ -z "$NAME" ]]; then
  echo "usage: $0 NAME [VAR=VALUE ...]" >&2
  exit 2
fi
shift || true
for kv in "$@"; do export "${kv?}"; done
# Re-read after the caller's overrides so KEY=VALUE args work positionally too.
VIEW=${VIEW}; EPOCHS=${EPOCHS}; LR=${LR}; EVAL_EPOCH=${EVAL_EPOCH}
AUG_CONFIG=${AUG_CONFIG}; RESOLUTION=${RESOLUTION}

cd "$REPO"
mkdir -p "$LOG_DIR"
taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }

if [[ ! -f "$RESULTS" ]]; then
  echo "name,category,view,epochs,lr,eval_epoch,aug,resolution,tp,fp,fn,precision,recall,f1,B_recall,C_recall,D_recall" > "$RESULTS"
fi

run_one() {
  local category="$1" device="$2"
  local dataset="data/rfdetr_${category}_${VIEW}_fold${FOLD}_test_as_valid"
  local run_dir="outputs/rfdetr_single_crack/bench/${NAME}_${category}"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"

  [[ -d "$dataset" ]] || { echo "missing dataset $dataset" >&2; return 1; }

  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" --experiment medium --device "$device" \
    --dataset-dir "$dataset" --output-dir "$run_dir" \
    --epochs "$EPOCHS" --lr "$LR" --checkpoint-interval 999 \
    ${AUG_CONFIG:+--aug-config "$AUG_CONFIG"} \
    ${RESOLUTION:+--resolution "$RESOLUTION"} \
    > "$LOG_DIR/${NAME}_${category}_train.log" 2>&1 || return 1
  rm -f "$run_dir"/checkpoint_*.ckpt

  local ckpt
  ckpt=$(printf '%s/epoch_pth/checkpoint_epoch_%03d.pth' "$run_dir" "$EVAL_EPOCH")
  [[ -f "$ckpt" ]] || ckpt=$(ls "$run_dir"/epoch_pth/checkpoint_epoch_*.pth 2>/dev/null | sort | tail -1)
  [[ -f "$ckpt" ]] || { echo "no checkpoint for ${category}" >&2; return 1; }

  taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
    --checkpoint "$ckpt" --dataset-dir "$dataset" --split test \
    --threshold-grid "${GRID[$category]}" --iou-threshold 0.229 --num-classes 3 \
    --output-csv "$run_dir/bench_grid.csv" --device "$device" \
    > "$LOG_DIR/${NAME}_${category}_eval.log" 2>&1 || return 1

  # Keep only the evaluated checkpoint; the bench runs many experiments.
  find "$run_dir/epoch_pth" -name "checkpoint_epoch_*.pth" ! -samefile "$ckpt" -delete 2>/dev/null

  taskset -c "$CPU_LIST" "$PYTHON" - "$run_dir/bench_grid.csv" "${THR[$category]}" \
    "$NAME" "$category" "$VIEW" "$EPOCHS" "$LR" "$EVAL_EPOCH" "${AUG_CONFIG:-none}" "${RESOLUTION:-default}" \
    >> "$RESULTS" <<'PY'
import csv, sys
path, thr, name, cat, view, epochs, lr, ev, aug, res = sys.argv[1:11]
want = tuple(round(float(v), 6) for v in thr.split(","))
for row in csv.DictReader(open(path, encoding="utf-8")):
    got = tuple(round(float(row[f"threshold_class_{i}"]), 6) for i in range(3))
    if got != want:
        continue
    tp = sum(int(float(row[f"class_{i}_tp"])) for i in range(3))
    fp = sum(int(float(row[f"class_{i}_fp"])) for i in range(3))
    fn = sum(int(float(row[f"class_{i}_fn"])) for i in range(3))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    br = [float(row[f"class_{i}_recall"]) for i in range(3)]
    print(f"{name},{cat},{view},{epochs},{lr},{ev},{aug.split('/')[-1]},{res},"
          f"{tp},{fp},{fn},{p:.4f},{r:.4f},{f1:.4f},{br[0]:.4f},{br[1]:.4f},{br[2]:.4f}")
    break
PY
}

echo "[$(date -u +%FT%TZ)] bench '$NAME' view=$VIEW epochs=$EPOCHS lr=$LR aug=${AUG_CONFIG:-none} res=${RESOLUTION:-default}"
run_one brace cuda:0 &
A=$!
run_one column_base cuda:1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1
echo "[$(date -u +%FT%TZ)] bench '$NAME' done status=$status"
taskset -c "$CPU_LIST" "$PYTHON" systems/rfdetr/scripts/report_new_class_bench.py
exit $status
