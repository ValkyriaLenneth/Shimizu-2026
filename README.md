# Shimizu 2026 Damage Detection Repository

This repository contains the project history and current engineering assets for
building-element routing and B/C/D damage recognition.

Current production direction:

```text
RF-DETR router
  -> RF-DETR downstream B/C/D recognition models
  -> RF-DETR production pipeline
  -> final display merge / report artifacts
```

Older YOLO / YOLO9 work is retained as baseline, migration history, and
reference implementation.

Start from the organized system index:

```text
systems/
  yolo/
  rfdetr/
```

See:

```text
systems/README.md
docs/repo_organization_20260623.md
docs/handoffs/README.md
```

## Current Repo Layout

| area | purpose |
|---|---|
| `systems/` | navigation layer by model family and role |
| `systems/rfdetr/pipeline/rfdetr_prod_pipeline/` | current RF-DETR end-to-end production pipeline |
| `systems/rfdetr/scripts/` | RF-DETR training, evaluation, report, and analysis utilities |
| `systems/yolo/router/coarse_router_yolov9/` | legacy YOLO9 router training/source reference |
| `systems/yolo/pipeline/router_crack_pipeline/` | legacy router + crack pipeline |
| `systems/*/*/configs/` | configs grouped by model family, role, and category |
| `docs/` | handoff notes, reports, progress summaries, and visual analysis |
| `final_release_20260615/` | git-tracked release docs, metrics, manifests, and checksums |
| `.local_artifacts/` | ignored local datasets, handoff archives, extracted packages, and old results |

Compatibility links are kept at the old root paths:

```text
rfdetr_prod_pipeline -> systems/rfdetr/pipeline/rfdetr_prod_pipeline
router_crack_pipeline -> systems/yolo/pipeline/router_crack_pipeline
coarse_router_yolov9 -> systems/yolo/router/coarse_router_yolov9
scripts/*.py -> systems/<area>/scripts/*.py
configs/*.yaml -> systems/<area>/configs/*.yaml
```

Large datasets, checkpoints, training outputs, compressed final releases, and
weekly handoff payloads are kept outside git. In this local checkout, recovered
large material is grouped under:

```text
.local_artifacts/
```

The complete final release archive currently known locally is:

```text
/Users/len/Downloads/final_release_20260615.tar.zst
sha256: 1daed69947449dd852873aabbed1c8413581c6bfb8651944ddb0228d47828820
```

## Categories

| repo name | label |
|---|---|
| `tenjo` | 天井 |
| `inner_wall` | 内壁 |
| `rc_wall` | RC壁 |
| `rc_column` | RC柱 |

Each model family is organized by:

```text
router
recognition_models
pipeline
```

## Quick Test

```bash
python3 -m pytest rfdetr_prod_pipeline/tests -q
```

## Legacy Classification Notes

The original project also trained image classifiers for four building element
classes using pretrained `timm` models and a fixed reusable train/val/test
split. That work is retained for reference below.

## Hardware Observed

- GPU: NVIDIA GeForce RTX 4090
- VRAM: 24564 MiB
- Driver: 580.126.09
- CUDA reported by driver: 13.0
- System Python: `/usr/bin/python3`, Python 3.12.3
- `conda` / `mamba`: not installed

## Dataset

Raw data for legacy workflows is expected under:

```text
.local_artifacts/data/unzip
```

Class mapping:

```text
a.天井  -> 天井
b.内壁  -> 内壁
c.RC壁  -> RC壁
d.RC柱  -> RC柱
```

Prepared ImageFolder data is written to an ignored local data directory:

```text
.local_artifacts/data/processed/building_cls_v1
```

Fixed split manifests are written to:

```text
.local_artifacts/data/manifests
```

## Environment

Preferred local virtualenv setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If conda is available later:

```bash
conda env create -f environment.yml
conda activate shimizu-building-cls
```

## Workflow

```bash
python scripts/audit_dataset.py --config configs/dataset.yaml
python scripts/prepare_dataset.py --config configs/dataset.yaml
python scripts/train_timm.py --config configs/train_efficientnet.yaml
python scripts/train_timm.py --config configs/train_resnet.yaml
python scripts/evaluate.py --run-dir outputs/runs/<run_name> --split test
python scripts/ensemble_predict.py --config configs/ensemble.yaml --split test
python scripts/infer.py --image path/to/image.jpg --run-dir outputs/runs/<run_name>
```

## Main Metrics

- Top-1 accuracy
- Macro precision
- Macro recall
- Macro F1
- Weighted precision
- Weighted recall
- Weighted F1
- Per-class precision/recall/F1
- R1: Recall@1, reported as per-class recall and macro recall for this single-label classification task
- Confusion matrix
- Loss and metric curves
