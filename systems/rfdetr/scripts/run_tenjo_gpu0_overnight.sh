#!/usr/bin/env bash
set -euo pipefail

cd /workspace/Shimizu-2026

python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_tenjo_baseline.yaml \
  --experiment medium \
  --dataset-dir data/rfdetr_tenjo_b_hard_e6_fpfn3_test_v1 \
  --output-dir outputs/rfdetr_single_crack/tenjo_medium_e004_b_hard_e6_fpfn3_res896_lr3e6 \
  --device cuda:0 \
  --epochs 80 \
  --batch-size 14 \
  --lr 0.000003 \
  --resolution 896 \
  --trainer-precision 16-mixed \
  --checkpoint outputs/rfdetr_single_crack/tenjo_medium_e006_b_os2_res896_ft_lr1e5/epoch_pth/checkpoint_epoch_004.pth \
  --checkpoint-interval 999

python scripts/sweep_rfdetr_router_test.py \
  --run-dir outputs/rfdetr_single_crack/tenjo_medium_e004_b_hard_e6_fpfn3_res896_lr3e6 \
  --dataset-dir data/rfdetr_tenjo_all_non_legacy_test_v1 \
  --epochs 0,1,2,4,8,16,32,48,64,79 \
  --output-csv outputs/rfdetr_single_crack/tenjo_medium_e004_b_hard_e6_fpfn3_res896_lr3e6/test_results_selected.csv \
  --device cuda:0 \
  --batch-size 14 \
  --num-workers 8 \
  --precision 16-mixed

python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_tenjo_baseline.yaml \
  --experiment medium \
  --dataset-dir data/rfdetr_tenjo_b_hard_e6_fpfn3_test_v1 \
  --output-dir outputs/rfdetr_single_crack/tenjo_medium_e007_crop_b_hard_e6_fpfn3_res896_lr1e6 \
  --device cuda:0 \
  --epochs 80 \
  --batch-size 14 \
  --lr 0.000001 \
  --resolution 896 \
  --trainer-precision 16-mixed \
  --checkpoint outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2_res896_ft_lr5e6/epoch_pth/checkpoint_epoch_007.pth \
  --checkpoint-interval 999

python scripts/sweep_rfdetr_router_test.py \
  --run-dir outputs/rfdetr_single_crack/tenjo_medium_e007_crop_b_hard_e6_fpfn3_res896_lr1e6 \
  --dataset-dir data/rfdetr_tenjo_all_non_legacy_test_v1 \
  --epochs 0,1,2,4,8,16,32,48,64,79 \
  --output-csv outputs/rfdetr_single_crack/tenjo_medium_e007_crop_b_hard_e6_fpfn3_res896_lr1e6/test_results_selected.csv \
  --device cuda:0 \
  --batch-size 14 \
  --num-workers 8 \
  --precision 16-mixed
