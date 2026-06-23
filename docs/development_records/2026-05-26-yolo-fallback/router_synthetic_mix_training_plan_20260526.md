# Router synthetic mix training plan 2026-05-26

## Goal

Improve the coarse router for the 2B project with Gemini/Nano Banana 2 synthetic data while judging progress on real validation and real test splits.

Current model to beat:

`coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv3_recall_ft_b48_e15/weights/epoch14.pt`

Primary business priority:

`Recall > mAP50 > Precision > mAP50-95`

The main regression risk is overfitting to synthetic image style. Therefore synthetic images are used only in the training split. The original real `val` and `test` splits stay unchanged.

## Dataset strategy

Base dataset:

`handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2`

Synthetic source:

`outputs/synthetic_router_pipeline_nb2_promptgen_300x4_c10/annotation_results.jsonl`

Mixed dataset output:

`handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2_gemini_nb2_mix`

Class merge:

| Source class | Router class |
|---|---|
| 天井 | 天井 |
| 内壁 | 壁类 |
| RC壁 | 壁类 |
| RC柱 | RC柱 |

Use all valid synthetic samples first. This should add about 1200 synthetic training images to 4052 real training images, so the synthetic image ratio is about 23%. This is aggressive enough to matter but not enough to dominate the real distribution.

## Epoch sampling

Training uses YOLOv9 `--image-weights`.

This keeps one mixed training pool but resamples image indices at every epoch, so each epoch can see a different subset/order of synthetic and real samples. This is cheaper and more stable than restarting 50 one-epoch runs.

## Training settings

Start from the current best epoch14 checkpoint, not from scratch.

Use:

- `epochs=50`
- `batch=64` on RTX 5090, adjust upward only after checking memory
- `hyp.router_finetune_recall_v4_long.yaml`
- low LR: `lr0=0.00018`
- `--save-period 1`
- `--test-every-epoch`
- `--image-weights`
- no early stop pressure: `--patience 100`

Best checkpoint from YOLO remains based on the normal validation fitness. Separately, `test_results.csv` records real test metrics every epoch. Model selection for reporting should compare every epoch against v3 epoch14 using real test recall and confusion/hard-case review.

## Commands

Build the mixed dataset after generation finishes:

```bash
cd /workspace/Shimizu-2026
python3 scripts/build_router_mixed_synthetic_dataset.py --overwrite
```

Start training:

```bash
cd /workspace/Shimizu-2026
BATCH_SIZE=64 EPOCHS=50 DEVICE=0 coarse_router_yolov9/scripts/train_router_3class_gemini_mix.sh
```

Important outputs:

- training run: `coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv2_gemini_nb2_mix_e50_b64_lowlr`
- per-epoch test metrics: `.../test_results.csv`
- per-epoch weights: `.../weights/epoch*.pt`

