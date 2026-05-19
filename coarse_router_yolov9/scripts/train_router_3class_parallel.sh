#!/usr/bin/env bash
set -euo pipefail

# Run full and cleaned 3-class router training in parallel on two physical GPUs.
# GPU0 trains the full dataset, GPU1 trains the cleaned dataset.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_DIR="$ROOT_DIR/yolov9"
PYTHON_BIN="${PYTHON_BIN:-python3}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
CFG="${CFG:-$YOLO_DIR/models/detect/gelan-c.yaml}"
HYP="${HYP:-$YOLO_DIR/data/hyps/hyp.scratch-high.yaml}"
WEIGHTS="${WEIGHTS:-}"
PROJECT="${PROJECT:-$ROOT_DIR/runs/train}"
FULL_DATA="${FULL_DATA:-$ROOT_DIR/datasets/coarse_router_3class_full/data.yaml}"
CLEANED_DATA="${CLEANED_DATA:-$ROOT_DIR/datasets/coarse_router_3class_cleaned/data.yaml}"
FULL_NAME="${FULL_NAME:-gelan_c_router_3class_full_e${EPOCHS}}"
CLEANED_NAME="${CLEANED_NAME:-gelan_c_router_3class_cleaned_e${EPOCHS}}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/train_parallel_logs/router_3class_$(date +%Y%m%d_%H%M%S)}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "$LOG_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[WARN] nvidia-smi not found"
else
  nvidia-smi > "$LOG_DIR/nvidia_smi_before.txt"
fi

run_train() {
  local physical_gpu="$1"
  local data_yaml="$2"
  local run_name="$3"
  local log_file="$4"
  shift 4

  echo "[INFO] starting $run_name on physical GPU $physical_gpu" | tee -a "$LOG_DIR/launcher.log" >&2
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" yolov9/train.py \
      --workers "$WORKERS" \
      --device "$physical_gpu" \
      --batch "$BATCH_SIZE" \
      --data "$data_yaml" \
      --img "$IMGSZ" \
      --cfg "$CFG" \
      --weights "$WEIGHTS" \
      --name "$run_name" \
      --project "$PROJECT" \
      --hyp "$HYP" \
      --min-items 0 \
      --epochs "$EPOCHS" \
      --close-mosaic 10 \
      --exist-ok \
      $EXTRA_ARGS
  ) > "$log_file" 2>&1 &
  LAST_TRAIN_PID=$!
}

run_train 0 "$FULL_DATA" "$FULL_NAME" "$LOG_DIR/full.log"
full_pid="$LAST_TRAIN_PID"
run_train 1 "$CLEANED_DATA" "$CLEANED_NAME" "$LOG_DIR/cleaned.log"
cleaned_pid="$LAST_TRAIN_PID"

cat > "$LOG_DIR/pids.txt" <<EOF
full $full_pid
cleaned $cleaned_pid
EOF

set +e
wait "$full_pid"
full_status=$?
wait "$cleaned_pid"
cleaned_status=$?
set -e

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "$LOG_DIR/nvidia_smi_after.txt"
fi

cat > "$LOG_DIR/summary.txt" <<EOF
full_status=$full_status
cleaned_status=$cleaned_status
full_run=$PROJECT/$FULL_NAME
cleaned_run=$PROJECT/$CLEANED_NAME
full_log=$LOG_DIR/full.log
cleaned_log=$LOG_DIR/cleaned.log
EOF

cat "$LOG_DIR/summary.txt"

if [[ "$full_status" -ne 0 || "$cleaned_status" -ne 0 ]]; then
  exit 1
fi
