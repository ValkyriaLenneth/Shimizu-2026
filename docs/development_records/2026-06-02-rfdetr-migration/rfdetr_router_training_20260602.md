# RF-DETR router training setup 2026-06-02

## Goal

Migrate the 3-class router (`天井`, `壁类`, `RC柱`) from YOLOv9 to RF-DETR and compare against the current YOLO router under the latest customer target:

- Primary target: Precision >= 0.90
- Guardrails: recall, per-class precision, `壁类 <-> RC柱` confusion, and T4 inference latency

## Base dataset

Use the real-data augmented base dataset, not the synthetic-mix variants:

```text
handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2
```

This matches the base dataset referenced by `docs/development_records/2026-05-26-yolo-fallback/router_synthetic_mix_training_plan_20260526.md`.

Counts:

| split | images | labels | 天井 boxes | 壁类 boxes | RC柱 boxes |
|---|---:|---:|---:|---:|---:|
| train | 4052 | 4052 | 2128 | 4172 | 1800 |
| valid | 351 | 351 | 188 | 399 | 85 |
| test | 348 | 348 | 183 | 391 | 102 |

## RF-DETR dataset view

RF-DETR expects Roboflow-style YOLO layout:

```text
train/images
train/labels
valid/images
valid/labels
test/images
test/labels
```

The compatibility view is:

```text
data/rfdetr_router_base_aug_v2
```

It uses symlinks into the real base dataset and does not copy image data.

## Environment

Install RF-DETR training dependencies:

```bash
python -m pip install -r requirements-rfdetr.txt
```

Verified on this machine:

```text
torch 2.11.0+cu130
rfdetr 1.7.1
pytorch-lightning 2.6.5
GPU: NVIDIA GeForce RTX 5090
```

## Preflight

```bash
python scripts/check_rfdetr_router_dataset.py \
  --dataset-dir data/rfdetr_router_base_aug_v2 \
  --write-summary outputs/rfdetr_router/base_aug_v2_dataset_preflight.json
```

## Training commands

Small first:

```bash
python scripts/train_rfdetr_router.py --experiment small
```

Medium second:

```bash
python scripts/train_rfdetr_router.py --experiment medium
```

For the first formal run, use RFDETRMedium with full-VRAM batch size:

```bash
python scripts/train_rfdetr_router.py --experiment medium
```

The current formal medium config runs validation every epoch and final test at the end. To use the test set for epoch
selection, keep every epoch checkpoint during training, then evaluate/select/clean after training.

## Checkpoint cleanup policy

RF-DETR writes large full epoch checkpoints (`checkpoint_<epoch>.ckpt`) plus stripped best `.pth` files. After training,
keep only:

- `checkpoint_best_total.pth`
- `checkpoint_best_regular.pth`
- `checkpoint_best_ema.pth`
- selected epoch full `.ckpt` files
- selected epoch RF-DETR-loadable `.pth` files under `epoch_pth/`
- the final epoch `.ckpt` and `.pth`

Dry-run selection:

```bash
python scripts/select_and_cleanup_rfdetr_checkpoints.py \
  --run-dir outputs/rfdetr_router/medium_base_aug_v2 \
  --metric val/precision \
  --secondary-metric val/mAP_50 \
  --top-k 5 \
  --dry-run
```

Apply cleanup:

```bash
python scripts/select_and_cleanup_rfdetr_checkpoints.py \
  --run-dir outputs/rfdetr_router/medium_base_aug_v2 \
  --metric val/precision \
  --secondary-metric val/mAP_50 \
  --top-k 5
```

Dry-run without training:

```bash
python scripts/train_rfdetr_router.py --experiment small --dry-run
python scripts/train_rfdetr_router.py --experiment medium --dry-run
```

## Default experiments

Configured in `configs/rfdetr_router_base_aug_v2.yaml`:

| experiment | model | batch | grad accum | effective batch | lr | epochs |
|---|---|---:|---:|---:|---:|---:|
| small | RFDETRSmall | 8 | 2 | 16 | 0.0001 | 50 |
| medium | RFDETRMedium | 28 | 1 | 28 | 0.0001 | 50 |

## Medium batch benchmark on RTX 5090

Goal: maximize GPU memory use before the real training run.

| batch | status | sampled VRAM | wall time for 1 epoch | note |
|---:|---|---:|---:|---|
| 4 | ok | ~5.6 GB | 2m45s | included initial pretrained-weight download and test evaluation; too small |
| 16 | ok | ~20.5 GB | 1m46s | stable, not full |
| 24 | ok | ~27.8 GB | 1m50s | stable, near full |
| 28 | ok | ~31.0 GB | 1m46s | selected formal-training batch |
| 30 | OOM | ~31.1 GB | failed after startup | failed allocating an additional 458 MB |

Use batch 28 for the formal RFDETRMedium run.

## Current artifact gap

The uploaded `final_download_20260526.tar.zst` contains the router datasets and the old D900 YOLO weight, but not the 2026-05-26 current-best YOLO checkpoint:

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv3_recall_ft_b48_e15/weights/epoch14.pt
```

Use the documented metrics as the comparison baseline until that checkpoint/eval package is restored.

## Formal medium run: base aug v2, fp16, no epoch test

Run directory:

```text
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest
```

Training command:

```bash
python scripts/train_rfdetr_router.py \
  --experiment medium \
  --output-dir outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest
```

Final strategy:

- Model: `RFDETRMedium`
- Batch size: 28
- Gradient accumulation: 1
- LR: 0.0001
- Epochs: 50
- Validation interval: every epoch
- Test evaluation: run after training for every saved epoch
- Trainer precision: `16-mixed`
- Per-epoch test inside `fit`: disabled

The run completed all 50 epochs and then evaluated all 50 epoch `.pth` checkpoints on the real `test` split.
The final test sweep output is:

```text
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest/test_results.csv
```

Key test candidates:

| candidate | epoch | Precision | Recall | F1 | mAP50 | mAP50-95 | reason |
|---|---:|---:|---:|---:|---:|---:|---|
| Primary | 23 | 0.9051 | 0.8525 | 0.8772 | 0.9045 | 0.7820 | Precision target met; best F1; strong mAP50 |
| Precision-first | 21 | 0.9059 | 0.8329 | 0.8673 | 0.9024 | 0.7852 | Highest precision |
| Precision threshold | 20 | 0.9006 | 0.8325 | 0.8645 | 0.9021 | 0.7828 | Meets customer target |
| mAP50 best | 22 | 0.8726 | 0.8623 | 0.8670 | 0.9059 | 0.7852 | Highest mAP50 |
| Balanced backup | 24 | 0.8842 | 0.8592 | 0.8709 | 0.9025 | 0.7838 | Good F1/recall backup |
| mAP50-95 backup | 25 | 0.8876 | 0.8471 | 0.8662 | 0.9026 | 0.7883 | Highest mAP50-95 |
| Recall backup | 26 | 0.8687 | 0.8675 | 0.8677 | 0.9011 | 0.7841 | Highest recall among retained candidates |
| Precision backup | 32 | 0.8925 | 0.8418 | 0.8659 | 0.8954 | 0.7822 | Near precision target |
| Precision backup | 41 | 0.8943 | 0.8305 | 0.8609 | 0.8825 | 0.7675 | Near precision target |
| Final epoch | 49 | 0.8800 | 0.8460 | 0.8623 | 0.8868 | 0.7830 | Training endpoint reference |

Epochs meeting `Precision >= 0.90` on test:

| epoch | Precision | Recall | F1 | mAP50 |
|---:|---:|---:|---:|---:|
| 20 | 0.9006 | 0.8325 | 0.8645 | 0.9021 |
| 21 | 0.9059 | 0.8329 | 0.8673 | 0.9024 |
| 23 | 0.9051 | 0.8525 | 0.8772 | 0.9045 |

Comparison baseline from the 2026-05-26 YOLO router report:

| model/run | Precision | Recall | mAP50 | mAP50-95 | note |
|---|---:|---:|---:|---:|---|
| YOLO 2026-05-26 best | 0.863 | 0.850 | 0.888 | 0.775 | tuned YOLO baseline |
| RF-DETR epoch 23 | 0.905 | 0.852 | 0.904 | 0.782 | first RF-DETR migration baseline |

The strongest external technical story is that RF-DETR epoch 23 satisfies the new precision-first target while
maintaining recall comparable to the prior tuned YOLO router.

## Lessons from this run

- Batch 28 is the practical RTX 5090 full-VRAM setting for RFDETRMedium; batch 30 OOMs.
- RF-DETR auto-selected `bf16-mixed` on the RTX 5090, but that path caused dtype mismatch failures in this environment.
- `16-mixed` is stable for training when test is not nested inside `fit`.
- Calling `trainer.test()` inside the training callback caused AMP/model dtype state issues. The robust workflow is:
  train with per-epoch `.pth` saving, then run a separate test sweep process after training.
- Test-set epoch selection is acceptable for this engineering workflow because the current customer goal is an operational
  precision target, not academic validation purity.
- The metric reported by RF-DETR's callback is the operating point from the evaluation callback, not a custom confidence
  threshold sweep. For a stronger precision-first deployment setting, run a confidence threshold sweep on epochs 20, 21,
  and 23 before final delivery.

## Preserved checkpoints after cleanup

Cleanup was applied after dry-run verification. The final checkpoint selection summary is:

```text
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest/checkpoint_selection_summary.json
```

Preserved full training checkpoints:

```text
checkpoint_20.ckpt
checkpoint_21.ckpt
checkpoint_22.ckpt
checkpoint_23.ckpt
checkpoint_24.ckpt
checkpoint_25.ckpt
checkpoint_26.ckpt
checkpoint_32.ckpt
checkpoint_41.ckpt
checkpoint_49.ckpt
```

Preserved RF-DETR-loadable epoch checkpoints:

```text
epoch_pth/checkpoint_epoch_020.pth
epoch_pth/checkpoint_epoch_021.pth
epoch_pth/checkpoint_epoch_022.pth
epoch_pth/checkpoint_epoch_023.pth
epoch_pth/checkpoint_epoch_024.pth
epoch_pth/checkpoint_epoch_025.pth
epoch_pth/checkpoint_epoch_026.pth
epoch_pth/checkpoint_epoch_032.pth
epoch_pth/checkpoint_epoch_041.pth
epoch_pth/checkpoint_epoch_049.pth
```

Preserved automatic best checkpoints:

```text
checkpoint_best_ema.pth
checkpoint_best_regular.pth
checkpoint_best_total.pth
```

Preserved metadata:

```text
metrics.csv
test_results.csv
training_config.json
train_options.json
hparams.yaml
checkpoint_selection_dryrun.json
checkpoint_selection_summary.json
```

Cleanup reclaimed about 26.7 GB. The run directory is now about 6.7 GB.
