#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TORCH_INDEX_URL="${TORCH_INDEX_URL:-}"

echo "[INFO] root: $ROOT_DIR"
echo "[INFO] venv: $VENV_DIR"

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if [[ -n "$TORCH_INDEX_URL" ]]; then
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
else
  python -m pip install torch torchvision
fi

python -m pip install -r "$ROOT_DIR/yolov9/requirements.txt"

python - <<'PY'
import torch
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_device_count={torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"gpu{i}={torch.cuda.get_device_name(i)}")
PY

echo "[INFO] YOLOv9 environment is ready."
