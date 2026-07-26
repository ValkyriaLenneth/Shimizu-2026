#!/usr/bin/env bash
# Joint-pretrain then per-category fine-tune for ブレース and 柱脚.
#
# Rationale: both categories share the B/C/D damage-grade semantics but each has
# only a third of the training data of the delivered categories. Pretraining on
# their union (1907 crop-augmented images) and then fine-tuning per category gives
# each model an initialization that has seen far more damage examples, without
# changing the deployment shape - still one recognition model per category.
#
# Stages:
#   1. pretrain  medium on the joint corpus                        (cuda:0)
#   2. select    the pretrain checkpoint with the best valid mAP50
#   3. finetune  brace (cuda:0) and column_base (cuda:1) from it, in parallel
#   4. sweep + per-class threshold grid for each fine-tuned model
#
# Selection is on mAP50 for the pretrain stage because it is an initialization,
# not an operating point.
#
# CPU 1 on this host is advertised as online but is not schedulable; anything
# pinned to it wedges unkillably, and torch pins itself to every core during init.
# --checkpoint-interval 999 suppresses the 511 MB per-epoch Lightning .ckpt files
# that previously filled the disk.

set -euo pipefail

REPO=/workspace/Shimizu-2026
PYTHON=${PYTHON:-$REPO/.venv/bin/python}
CPU_LIST=${CPU_LIST:-0,2-63}
JOINT_CONFIG=systems/rfdetr/recognition_models/joint_bcd/configs/rfdetr_joint_bcd_pretrain.yaml
JOINT_RUN=outputs/rfdetr_single_crack/joint_bcd_medium_20260725_split91_crop2
# Peak mAP arrives after roughly 200 optimizer steps on this data regardless of
# how the corpus is expanded, and the joint corpus gives 68 steps per epoch, so
# 20 epochs already overshoots the peak by a wide margin. The best-mAP checkpoint
# is selected afterwards, so a shorter run loses nothing.
PRETRAIN_EPOCHS=${PRETRAIN_EPOCHS:-20}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-15}
FINETUNE_LR=${FINETUNE_LR:-0.00005}
SUFFIX=${SUFFIX:-bcd_20260725_split91_crop2_test_as_valid}
TAG=${TAG:-jointft}
MATCH_IOU=0.229
THRESHOLD_GRID="0.05,0.07,0.10,0.12,0.15,0.18,0.20,0.22,0.25,0.28,0.30,0.35,0.40,0.45,0.50"
LOG_DIR="$REPO/outputs/rfdetr_new_classes/logs"

cd "$REPO"
mkdir -p "$LOG_DIR"

taskset -c "$CPU_LIST" /bin/true || { echo "affinity mask unusable" >&2; exit 1; }

# ---- stage 1: joint pretrain -------------------------------------------------
echo "[$(date -u +%FT%TZ)] stage1 pretrain joint on cuda:0"
taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
  --config "$JOINT_CONFIG" \
  --experiment medium \
  --device cuda:0 \
  --epochs "$PRETRAIN_EPOCHS" \
  --checkpoint-interval 999 \
  2>&1 | tee "$LOG_DIR/joint_pretrain.log"
rm -f "$JOINT_RUN"/checkpoint_*.ckpt
echo "[$(date -u +%FT%TZ)] stage1 done"

# ---- stage 2: pick the best pretrain checkpoint by valid mAP50 ---------------
INIT=$(taskset -c "$CPU_LIST" "$PYTHON" - "$JOINT_RUN" <<'PY'
import csv, sys
from pathlib import Path
run = Path(sys.argv[1])
best = (None, -1.0)
metrics = run / "metrics.csv"
if metrics.exists():
    for row in csv.DictReader(metrics.open(encoding="utf-8")):
        raw = (row.get("val/mAP_50") or "").strip()
        epoch = (row.get("epoch") or "").strip()
        if not raw or not epoch:
            continue
        value = float(raw)
        if value > best[1]:
            best = (int(float(epoch)), value)
if best[0] is not None:
    candidate = run / "epoch_pth" / f"checkpoint_epoch_{best[0]:03d}.pth"
    if candidate.exists():
        print(candidate)
PY
)
if [[ -z "$INIT" ]]; then
  echo "could not resolve a pretrain checkpoint; aborting" >&2
  exit 1
fi
echo "[$(date -u +%FT%TZ)] stage2 init checkpoint: $INIT"

# ---- stage 3+4: per-category fine-tune, then sweep and grid ------------------
finetune_category() {
  local category="$1"
  local device="$2"
  local config="systems/rfdetr/recognition_models/${category}/configs/rfdetr_${category}_baseline.yaml"
  local dataset="data/rfdetr_${category}_${SUFFIX}"
  local run_dir="outputs/rfdetr_single_crack/${category}_medium_${TAG}_${SUFFIX}"

  echo "[$(date -u +%FT%TZ)] stage3 finetune ${category} on ${device} lr=${FINETUNE_LR}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/train_rfdetr_router.py \
    --config "$config" \
    --experiment medium \
    --device "$device" \
    --dataset-dir "$dataset" \
    --output-dir "$run_dir" \
    --checkpoint "$INIT" \
    --epochs "$FINETUNE_EPOCHS" \
    --lr "$FINETUNE_LR" \
    --checkpoint-interval 999 \
    2>&1 | tee "$LOG_DIR/${category}_${TAG}.log"
  rm -f "$run_dir"/checkpoint_*.ckpt
  echo "[$(date -u +%FT%TZ)] stage3 done ${category}"

  echo "[$(date -u +%FT%TZ)] stage4 sweep ${category}"
  taskset -c "$CPU_LIST" "$PYTHON" scripts/sweep_rfdetr_router_test.py \
    --run-dir "$run_dir" --dataset-dir "$dataset" \
    --output-csv "$run_dir/test_results.csv" --device "$device" \
    --batch-size 28 --num-workers 8 --precision 16-mixed \
    > "$LOG_DIR/${category}_${TAG}_sweep.log" 2>&1
  echo "[$(date -u +%FT%TZ)] stage4 sweep done ${category}"

  local candidates
  candidates=$(taskset -c "$CPU_LIST" "$PYTHON" \
    systems/rfdetr/scripts/report_new_class_training_status.py \
    --list-top-checkpoints "$run_dir" --top 3 2>/dev/null || true)
  while read -r ckpt; do
    [[ -n "$ckpt" ]] || continue
    local tag
    tag=$(basename "$ckpt" .pth)
    taskset -c "$CPU_LIST" "$PYTHON" scripts/evaluate_rfdetr_class_threshold_grid.py \
      --checkpoint "$ckpt" --dataset-dir "$dataset" --split test \
      --threshold-grid "$THRESHOLD_GRID" --iou-threshold "$MATCH_IOU" --num-classes 3 \
      --output-csv "$run_dir/class_threshold_grid_${tag}.csv" --device "$device" \
      >> "$LOG_DIR/${category}_${TAG}_grid.log" 2>&1
  done <<< "$candidates"
  echo "[$(date -u +%FT%TZ)] stage4 grid done ${category}"
}

finetune_category brace cuda:0 &
A=$!
finetune_category column_base cuda:1 &
B=$!
status=0
wait "$A" || status=1
wait "$B" || status=1
[[ $status -eq 0 ]] || { echo "joint pretrain/finetune: a category failed"; exit 1; }
echo "[$(date -u +%FT%TZ)] joint pretrain + finetune pipeline finished"
