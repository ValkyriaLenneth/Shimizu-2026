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
DATA_D800="${DATA_D800:-$ROOT_DIR/datasets/coarse_router_3class_cleaned_rc_column_os800/data.yaml}"
DATA_D900="${DATA_D900:-$ROOT_DIR/datasets/coarse_router_3class_cleaned_rc_column_os900/data.yaml}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/runs/train_parallel_logs/router_tuning_d_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$LOG_DIR"
nvidia-smi > "$LOG_DIR/nvidia_smi_before.txt" || true

run_train() {
  local gpu="$1"
  local data_yaml="$2"
  local name="$3"
  local log_file="$4"
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
      --image-weights \
      --exist-ok
  ) > "$log_file" 2>&1 &
  LAST_PID=$!
}

run_train 0 "$DATA_D800" "gelan_c_router_3class_cleaned_ft_imgw_rc_os800_e${EPOCHS}" "$LOG_DIR/d800.log"
pid_d800="$LAST_PID"
run_train 1 "$DATA_D900" "gelan_c_router_3class_cleaned_ft_imgw_rc_os900_e${EPOCHS}" "$LOG_DIR/d900.log"
pid_d900="$LAST_PID"

cat > "$LOG_DIR/pids.txt" <<EOF
d800 $pid_d800
d900 $pid_d900
EOF

set +e
wait "$pid_d800"
status_d800=$?
wait "$pid_d900"
status_d900=$?
set -e

nvidia-smi > "$LOG_DIR/nvidia_smi_after.txt" || true

cat > "$LOG_DIR/summary.txt" <<EOF
d800_status=$status_d800
d900_status=$status_d900
d800_run=$PROJECT/gelan_c_router_3class_cleaned_ft_imgw_rc_os800_e${EPOCHS}
d900_run=$PROJECT/gelan_c_router_3class_cleaned_ft_imgw_rc_os900_e${EPOCHS}
d800_log=$LOG_DIR/d800.log
d900_log=$LOG_DIR/d900.log
EOF

cat "$LOG_DIR/summary.txt"

if [[ "$status_d800" -ne 0 || "$status_d900" -ne 0 ]]; then
  exit 1
fi
