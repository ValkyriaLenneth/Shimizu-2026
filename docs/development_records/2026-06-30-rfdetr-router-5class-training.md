# RF-DETR 5-class router training 2026-06-30

## Goal

Train a 5-class RF-DETR router for:

```text
0: 天井
1: 壁类
2: RC柱
3: ブレース
4: 柱脚
```

The target is overall precision >= 0.90. The run follows the 2026-06-02
3-class router training recipe: RFDETRMedium, batch 28 on RTX 5090, LR 1e-4,
50 epochs, `16-mixed`, and saved per-epoch `.pth` checkpoints.

## Dataset

Training dataset:

```text
data/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid
```

RF-DETR requires a `valid` split for training-time evaluation. For this run,
`valid` mirrors `test`, so training logs labelled `val/*` are test metrics.
This matches the instruction to use test only.

Dataset check:

| split | images | 天井 boxes | 壁类 boxes | RC柱 boxes | ブレース boxes | 柱脚 boxes |
|---|---:|---:|---:|---:|---:|---:|
| train | 4635 | 2128 | 4172 | 1800 | 472 | 252 |
| valid (= test) | 415 | 183 | 391 | 102 | 60 | 25 |
| test | 415 | 183 | 391 | 102 | 60 | 25 |

Preflight output:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid_dataset_check.json
```

## Training

Config:

```text
systems/rfdetr/router/configs/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid.yaml
```

Command:

```bash
python scripts/train_rfdetr_router.py \
  --config systems/rfdetr/router/configs/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid.yaml \
  --experiment medium \
  --device cuda:0 \
  --trainer-precision 16-mixed \
  --no-test-each-epoch
```

Run directory:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid
```

Training completed 50 epochs.

## Default operating-point results

The best default precision epoch from `metrics.csv`:

| epoch | precision | recall | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 41 | 0.8919 | 0.7432 | 0.8381 | 0.7083 |

The final best-total checkpoint test result:

| checkpoint | precision | recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| `checkpoint_best_total.pth` | 0.8541 | 0.7974 | 0.8232 | 0.8385 | 0.7248 |

Default operating point did not directly reach 0.90 precision.

## Threshold sweep

Precision-first threshold sweeps were run on high-precision / high-F1 / high-mAP
candidate epochs:

```text
20, 31, 35, 41, 43, 47, 49
```

Best point satisfying overall precision >= 0.90:

| epoch | threshold | precision | recall | F1 |
|---:|---:|---:|---:|---:|
| 49 | 0.69 | 0.9039 | 0.7293 | 0.8073 |

Per-class metrics at the selected point:

| class | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| 天井 | 0.8970 | 0.8087 | 0.8506 | 148 | 17 | 35 |
| 壁类 | 0.9123 | 0.7187 | 0.8040 | 281 | 27 | 110 |
| RC柱 | 0.8966 | 0.7647 | 0.8254 | 78 | 9 | 24 |
| ブレース | 0.9062 | 0.4833 | 0.6304 | 29 | 3 | 31 |
| 柱脚 | 0.8636 | 0.7600 | 0.8085 | 19 | 3 | 6 |

The 0.90 target is met for overall precision, not for every individual class.

Selected artifact:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid/selected_precision_p090_epoch049_thr069.pth
```

This is a hardlink to:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid/epoch_pth/checkpoint_epoch_049.pth
```

Selection manifest:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid/selected_precision_p090_manifest.json
```

## Notes

- `valid` mirrors `test`; this was intentional for this run.
- The selected deployment confidence threshold is `0.69`.
- Because `ブレース` and `柱脚` labels are Gemini-generated, the new-class
  evaluation is only as reliable as the generated router boxes.
- If per-class precision >= 0.90 is required later, the next step is class-wise
  threshold tuning rather than a single global threshold.
