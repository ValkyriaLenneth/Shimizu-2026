# Building Type Image Classification

This project trains image classifiers for four building element classes using pretrained `timm` models and a fixed reusable train/val/test split.

## Hardware Observed

- GPU: NVIDIA GeForce RTX 4090
- VRAM: 24564 MiB
- Driver: 580.126.09
- CUDA reported by driver: 13.0
- System Python: `/usr/bin/python3`, Python 3.12.3
- `conda` / `mamba`: not installed

## Dataset

Raw data is expected under:

```text
data/unzip
```

Class mapping:

```text
a.天井  -> 天井
b.内壁  -> 内壁
c.RC壁  -> RC壁
d.RC柱  -> RC柱
```

Prepared ImageFolder data is written to:

```text
data/processed/building_cls_v1
```

Fixed split manifests are written to:

```text
data/manifests
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
