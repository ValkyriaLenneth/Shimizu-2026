#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Shimizu-2026

python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_tenjo_baseline.yaml \
  --experiment medium \
  --dataset-dir data/rfdetr_tenjo_b_crop2_hard_e002_fpfn3_test_v1 \
  --output-dir outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2hard_e002_fpfn3_res896_lr5e7_long \
  --device cuda:0 \
  --epochs 120 \
  --batch-size 14 \
  --lr 0.0000005 \
  --resolution 896 \
  --trainer-precision 16-mixed \
  --checkpoint outputs/rfdetr_single_crack/tenjo_medium_e006_b_os2_res896_ft_lr1e5/epoch_pth/checkpoint_epoch_004.pth \
  --checkpoint-interval 999

python scripts/sweep_rfdetr_router_test.py \
  --run-dir outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2hard_e002_fpfn3_res896_lr5e7_long \
  --dataset-dir data/rfdetr_tenjo_all_non_legacy_test_v1 \
  --epochs 0,1,2,4,8,16,32,64,96,119 \
  --output-csv outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2hard_e002_fpfn3_res896_lr5e7_long/test_results_selected.csv \
  --device cuda:0 \
  --batch-size 14 \
  --num-workers 8 \
  --precision 16-mixed
