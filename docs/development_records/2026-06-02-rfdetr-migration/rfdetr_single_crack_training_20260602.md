# RF-DETR single crack model training 2026-06-02

## Goal

Replace the prior YOLO9 single-component crack/damage judgement models with RF-DETR candidates.

For these crack/damage detectors, selection is recall-first because missed cracks are more costly than extra detections.
Precision, mAP50, mAP50-95, and per-class behavior remain guardrails.

## Baseline Reference

YOLO9 baseline details are recorded in:

```text
docs/development_records/2026-06-02-rfdetr-migration/yolo9_single_crack_model_baseline_20260602.md
```

The previous customer-report adjusted recall baseline is the official target for replacement. For RC柱:

| component | overall R target | B R target | C R target | D R target |
|---|---:|---:|---:|---:|
| RC柱 | > 0.742 | > 0.700 | > 0.706 | > 0.807 |

The raw YOLO9 `val.py` rows are retained in the baseline document for reproducibility, but the adjusted recall table is
the comparison target.

Earlier raw split baselines were:

| component | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| RC壁 | 0.797 | 0.782 | 0.846 | 0.618 |
| RC柱 | 0.702 | 0.653 | 0.721 | 0.480 |

## RC柱 RF-DETR Medium Legacy-Max-Train V1

This is the first valid RC柱 replacement experiment against the uploaded legacy report split.

Dataset:

```text
data/rfdetr_rc_column_all_non_legacy_test_v1
```

Data policy:

```text
train = all current 20260519 RC柱 images except data_split.json rc_column/test stems
valid = data_split.json rc_column/test
test  = data_split.json rc_column/test
```

Data counts:

| split | images | B boxes | C boxes | D boxes |
|---|---:|---:|---:|---:|
| train | 605 | 317 | 186 | 145 |
| valid | 31 | 12 | 11 | 8 |
| test | 31 | 12 | 11 | 8 |

Training command:

```bash
python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_rc_column_baseline.yaml \
  --experiment medium \
  --dataset-dir data/rfdetr_rc_column_all_non_legacy_test_v1 \
  --output-dir outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1 \
  --epochs 80 \
  --batch-size 28 \
  --device cuda:0 \
  --trainer-precision 16-mixed
```

Training settings:

| setting | value |
|---|---:|
| model | RFDETRMedium |
| epochs | 80 |
| batch size | 28 |
| gradient accumulation | 1 |
| lr | 0.0001 |
| precision | 16-mixed |
| train batches / epoch | 22 |
| save epoch `.pth` | yes |

The full training run finished. The automatic `checkpoint_best_total.pth` was selected by mAP, not recall-first business
targets. Because the project goal is recall-first crack/damage detection, selected epoch checkpoints were force-evaluated
on the official test split:

```bash
python scripts/sweep_rfdetr_router_test.py \
  --run-dir outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1 \
  --dataset-dir data/rfdetr_rc_column_all_non_legacy_test_v1 \
  --epochs 37,40,42,47,48,63,65,75,79 \
  --output-csv outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/selected_test_results.csv \
  --batch-size 28 \
  --num-workers 8 \
  --precision 16-mixed \
  --device cuda:0
```

Recall-first candidate summary:

| checkpoint | Precision | Recall | mAP50 | mAP50-95 | B Recall | C Recall | D Recall | beats report targets |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| epoch 47 | 0.6612 | 0.8258 | 0.7255 | 0.2993 | 0.7500 | 0.7273 | 1.0000 | yes |
| best_total | 0.6630 | 0.7955 | 0.7076 | 0.2732 | 0.7500 | 0.6364 | 1.0000 | no, C below target |
| epoch 40 | 0.6937 | 0.7955 | 0.7251 | 0.3268 | 0.7500 | 0.6364 | 1.0000 | no, C below target |

Primary candidate:

```text
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_047.pth
```

Reason:

- It exceeds the previous report RC柱 recall targets on overall, B, C, and D.
- It is recall-first better than `checkpoint_best_total.pth`, which misses the C-class target.
- It was confirmed by forced official-test evaluation after checkpoint reload.

Detailed artifacts:

```text
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/selected_test_results.csv
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/candidate_summary.csv
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/metrics.csv
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/train_options.json
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/training_config.json
```

## RC柱 RF-DETR Medium Baseline

Config:

```text
configs/rfdetr_rc_column_baseline.yaml
```

Dataset:

```text
handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split/rc_column
```

Data counts:

| split | images | B boxes | C boxes | D boxes |
|---|---:|---:|---:|---:|
| train | 498 | 261 | 146 | 121 |
| valid | 71 | 41 | 26 | 13 |
| test | 67 | 27 | 25 | 19 |

Training command:

```bash
python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_rc_column_baseline.yaml \
  --experiment medium \
  --output-dir outputs/rfdetr_single_crack/rc_column_medium_baseline
```

Training settings:

| setting | value |
|---|---:|
| model | RFDETRMedium |
| epochs | 80 |
| batch size | 28 |
| gradient accumulation | 1 |
| lr | 0.0001 |
| precision | 16-mixed |
| save epoch `.pth` | yes |
| train-time test each epoch | no |

Final training-run test row from `metrics.csv`:

| Precision | Recall | F1 | mAP50 | mAP50-95 | AP_B | AP_C | AP_D |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8423 | 0.7370 | 0.7835 | 0.7279 | 0.5097 | 0.4435 | 0.3074 | 0.7781 |

Important note: an initial full-epoch sweep was accidentally run without `--dataset-dir`, which evaluated RC柱 checkpoints on the router dataset. That wrong `test_results.csv` was deleted. The correct sweep command was:

```bash
python scripts/sweep_rfdetr_router_test.py \
  --run-dir outputs/rfdetr_single_crack/rc_column_medium_baseline \
  --dataset-dir handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split/rc_column \
  --batch-size 28 \
  --num-workers 8 \
  --precision 16-mixed \
  --skip-existing
```

Correct full 0-79 test sweep:

```text
outputs/rfdetr_single_crack/rc_column_medium_baseline/test_results.csv
```

The CSV has 80 rows. Per-class AP columns are `test/AP/0`, `test/AP/1`, `test/AP/2`, corresponding to B/C/D.

### RC柱 Candidate Checkpoints

Recall-first candidates:

| epoch | Precision | Recall | F1 | mAP50 | mAP50-95 | AP_B | AP_C | AP_D |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 23 | 0.6011 | 0.6617 | 0.6291 | 0.6379 | 0.4047 | 0.2656 | 0.3686 | 0.5799 |
| 20 | 0.5807 | 0.6474 | 0.6115 | 0.6464 | 0.4138 | 0.2826 | 0.3644 | 0.5944 |
| 38 | 0.6138 | 0.6256 | 0.6184 | 0.6193 | 0.3972 | 0.2540 | 0.3310 | 0.6065 |

Additional guardrail candidates:

| epoch | reason | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 44 | high mAP50-95 | 0.6766 | 0.5999 | 0.6353 | 0.6297 | 0.4103 |
| 19 | high mAP50-95 / mAP50 | 0.6678 | 0.5866 | 0.6243 | 0.6391 | 0.4097 |
| 70 | best F1 | 0.7704 | 0.5609 | 0.6484 | 0.6051 | 0.3889 |
| 73 | high F1 | 0.7567 | 0.5609 | 0.6438 | 0.6093 | 0.3923 |
| 47 | high F1 with better recall | 0.6851 | 0.5999 | 0.6393 | 0.6175 | 0.4025 |
| 26 | highest precision | 0.8452 | 0.4663 | 0.5991 | 0.6207 | 0.4020 |
| 28 | high precision | 0.8056 | 0.4920 | 0.6108 | 0.6120 | 0.4000 |
| 68 | high precision | 0.8026 | 0.5219 | 0.6308 | 0.6074 | 0.4014 |

Candidate epoch `.pth` files are under:

```text
outputs/rfdetr_single_crack/rc_column_medium_baseline/epoch_pth/
```

No cleanup has been applied yet. Keep the above candidate epochs plus automatic best checkpoints until visual review and RC壁 comparison are complete.

## RC壁 RF-DETR Medium Baseline

Config:

```text
configs/rfdetr_rc_wall_baseline.yaml
```

Dataset:

```text
handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split/rc_wall
```

Preflight data counts:

| split | images | B boxes | C boxes | D boxes | empty labels |
|---|---:|---:|---:|---:|---:|
| train | 966 | 860 | 198 | 177 | 0 |
| valid | 90 | 71 | 19 | 9 | 1 |
| test | 126 | 107 | 26 | 13 | 1 |

Training command:

```bash
python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_rc_wall_baseline.yaml \
  --experiment medium \
  --output-dir outputs/rfdetr_single_crack/rc_wall_medium_baseline
```

Training started with the same baseline settings as RC柱:

| setting | value |
|---|---:|
| model | RFDETRMedium |
| epochs | 80 |
| batch size | 28 |
| gradient accumulation | 1 |
| lr | 0.0001 |
| precision | 16-mixed |
| save epoch `.pth` | yes |
| train-time test each epoch | no |

Early status:

| epoch | val Precision | val Recall | val F1 | val mAP50 | val mAP50-95 |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.0310 | 0.0702 | 0.0430 | 0.0213 | 0.0095 |
| 1 | 0.2412 | 0.2893 | 0.2300 | 0.1902 | 0.0828 |

Continue monitoring this run and select RC壁 candidates with the same recall-first policy after training and test sweep.
