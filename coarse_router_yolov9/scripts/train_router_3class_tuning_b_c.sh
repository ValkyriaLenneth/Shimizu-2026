#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
YOLO_DIR="$ROOT_DIR/yolov9"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-32}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-8}"
CFG="${CFG:-$YOLO_DIR/models/detect/gelan-c.yaml}"
HYP="${HYP:-$ROOT_DIR/hyps/hyp.router_finetune.yaml}"
BASE_WEIGHTS="${BASE_WEIGHTS:-$ROOT_DIR/runs/train/gelan_c_router_3class_cleaned_e50/weights/best.pt}"
PROJECT="${PROJECT:-$ROOT_DIR/runs/train}"
CLEANED_DATA="${CLEANED_DATA:-$ROOT_DIR/datasets/coarse_router_3class_cleaned/data.yaml}"
OVERSAMPLE_DATA="${OVERSAMPLE_DATA:-$ROOT_DIR/datasets/coarse_router_3class_cleaned_rc_column_oversample/data.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/train_parallel_logs/router_tuning_bc_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
nvidia-smi > "$LOG_DIR/nvidia_smi_before.txt" || true

run_train() {
  local gpu="$1"
  local data_yaml="$2"
  local name="$3"
  local log_file="$4"
  shift 4
  echo "[INFO] starting $name on GPU $gpu" | tee -a "$LOG_DIR/launcher.log" >&2
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" yolov9/train.py \
      --workers "$WORKERS" \
      --device "$gpu" \
      --batch "$BATCH_SIZE" \
      --data "$data_yaml" \
      --img "$IMGSZ" \
      --cfg "$CFG" \
      --weights "$BASE_WEIGHTS" \
      --name "$name" \
      --project "$PROJECT" \
      --hyp "$HYP" \
      --min-items 0 \
      --epochs "$EPOCHS" \
      --close-mosaic 10 \
      --exist-ok \
      "$@"
  ) > "$log_file" 2>&1 &
  LAST_PID=$!
}

run_train 0 "$CLEANED_DATA" "gelan_c_router_3class_cleaned_ft_imgw_e${EPOCHS}" "$LOG_DIR/image_weights.log" --image-weights
pid_b="$LAST_PID"
run_train 1 "$OVERSAMPLE_DATA" "gelan_c_router_3class_cleaned_ft_rc_os_e${EPOCHS}" "$LOG_DIR/rc_oversample.log"
pid_c="$LAST_PID"

cat > "$LOG_DIR/pids.txt" <<EOF
image_weights $pid_b
rc_oversample $pid_c
EOF

set +e
wait "$pid_b"
status_b=$?
wait "$pid_c"
status_c=$?
set -e

nvidia-smi > "$LOG_DIR/nvidia_smi_after.txt" || true

cat > "$LOG_DIR/summary.txt" <<EOF
image_weights_status=$status_b
rc_oversample_status=$status_c
image_weights_run=$PROJECT/gelan_c_router_3class_cleaned_ft_imgw_e${EPOCHS}
rc_oversample_run=$PROJECT/gelan_c_router_3class_cleaned_ft_rc_os_e${EPOCHS}
image_weights_log=$LOG_DIR/image_weights.log
rc_oversample_log=$LOG_DIR/rc_oversample.log
EOF

cat "$LOG_DIR/summary.txt"

if [[ "$status_b" -ne 0 || "$status_c" -ne 0 ]]; then
  exit 1
fi
