#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATA="${DATA:-/workspace/Shimizu-2026/handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2_gemini_nb2_mix/data.yaml}"
WEIGHTS="${WEIGHTS:-/workspace/Shimizu-2026/coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv3_recall_ft_b48_e15/weights/epoch14.pt}"
HYP="${HYP:-$ROOT_DIR/hyps/hyp.router_finetune_recall_v4_long.yaml}"
CFG="${CFG:-$ROOT_DIR/yolov9/models/detect/gelan-c.yaml}"
PROJECT="${PROJECT:-$ROOT_DIR/runs/train}"
NAME="${NAME:-gelan_c_router_3class_merged4219_augv2_gemini_nb2_mix_e50_b64_lowlr}"

EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-64}"
IMGSZ="${IMGSZ:-640}"
WORKERS="${WORKERS:-12}"
DEVICE="${DEVICE:-0}"
SEED="${SEED:-2026052604}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$ROOT_DIR"
"$PYTHON_BIN" yolov9/train.py \
  --workers "$WORKERS" \
  --device "$DEVICE" \
  --batch "$BATCH_SIZE" \
  --data "$DATA" \
  --img "$IMGSZ" \
  --cfg "$CFG" \
  --weights "$WEIGHTS" \
  --name "$NAME" \
  --project "$PROJECT" \
  --hyp "$HYP" \
  --min-items 0 \
  --epochs "$EPOCHS" \
  --patience 100 \
  --save-period 1 \
  --image-weights \
  --test-every-epoch \
  --exist-ok \
  --seed "$SEED" \
  $EXTRA_ARGS
