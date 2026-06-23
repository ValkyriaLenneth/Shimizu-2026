# YOLO9 single crack model baseline 2026-06-02

## Goal

Confirm the current four deployed YOLO9 crack/damage judgement models before replacing or comparing them with a new
detector family.

The four models are the previous-phase single-component models:

| component | dataset key | deployed weight |
|---|---|---|
| 天井 | `tenjo` | `downloads/previous_phase_gpl_model_unpacked/infer_models/TIANJING.pt` |
| 内壁 | `inner_wall` | `downloads/previous_phase_gpl_model_unpacked/infer_models/NEIBI.pt` |
| RC壁 | `rc_wall` | `downloads/previous_phase_gpl_model_unpacked/infer_models/RCBI.pt` |
| RC柱 | `rc_column` | `downloads/previous_phase_gpl_model_unpacked/infer_models/RCZHU.pt` |

All four models detect three damage grades: B, C, D.

## Data

Test split source:

```text
handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split
```

This path is currently a compatibility symlink to:

```text
final_download_20260526/handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split
```

Test counts:

| dataset | test images | test instances |
|---|---:|---:|
| `tenjo` | 96 | 105 |
| `inner_wall` | 104 | 122 |
| `rc_wall` | 126 | 146 |
| `rc_column` | 67 | 71 |

## Evaluation command

Each model was evaluated with YOLOv9 `val.py` on the `test` split:

```bash
python val.py \
  --data <split>/<dataset>/data.yaml \
  --weights <weight.pt> \
  --task test \
  --img 960 \
  --batch 16 \
  --device 0 \
  --project /workspace/Shimizu-2026/outputs/yolo9_single_model_baseline_20260602 \
  --name <dataset>_test \
  --exist-ok
```

YOLOv9 defaults were used for metric evaluation:

```text
conf_thres=0.001
iou_thres=0.7
max_det=300
half=False
```

Environment:

```text
Python 3.14.3
torch 2.11.0+cu130
GPU: NVIDIA GeForce RTX 5090
```

## Baseline Results

### Reported Adjusted Recall Baseline

The table below is the final adjusted recall result from the previous customer report. This is the official comparison
target for replacing the previous YOLO9 single-component crack/damage models.

| component | class scope | previous R | reported new R | delta |
|---|---|---:|---:|---:|
| 天井 | 全体 | 0.750 | 0.845 | +0.095 |
| 天井 | B | 0.667 | 0.750 | +0.083 |
| 天井 | C | 0.583 | 0.826 | +0.243 |
| 天井 | D | 1.000 | 1.000 | +0.000 |
| 内壁 | 全体 | 0.694 | 0.750 | +0.056 |
| 内壁 | B | 0.600 | 0.747 | +0.147 |
| 内壁 | C | 0.682 | 0.773 | +0.091 |
| 内壁 | D | 0.800 | 0.800 | +0.000 |
| RC壁 | 全体 | 0.535 | 0.720 | +0.185 |
| RC壁 | B | 0.620 | 0.739 | +0.119 |
| RC壁 | C | 0.451 | 0.680 | +0.229 |
| RC壁 | D | 0.536 | 0.667 | +0.131 |
| RC柱 | 全体 | 0.695 | 0.742 | +0.047 |
| RC柱 | B | 0.659 | 0.700 | +0.041 |
| RC柱 | C | 0.647 | 0.706 | +0.059 |
| RC柱 | D | 0.779 | 0.807 | +0.028 |

For RF-DETR replacement experiments, the primary target is to exceed the `reported new R` values on the same
`data_split.json` test protocol. For the first RC柱 experiment, the target is:

| component | overall R target | B R target | C R target | D R target |
|---|---:|---:|---:|---:|
| RC柱 | > 0.742 | > 0.700 | > 0.706 | > 0.807 |

### YOLOv9 Raw Evaluation Record

The `all` rows below are YOLOv9's aggregate metrics over the three damage-grade classes B/C/D for each component-specific
model on the current 20260519 split. These are kept as a reproducible raw `val.py` record, but they are not the official
replacement target once the previous report's adjusted recall baseline above is available.

| model | images | instances | Precision | Recall | mAP50 | mAP50-95 | inference / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| 天井 `TIANJING.pt` | 96 | 105 | 0.961 | 0.941 | 0.967 | 0.845 | 5.4 ms |
| 内壁 `NEIBI.pt` | 104 | 122 | 0.934 | 0.884 | 0.953 | 0.767 | 5.0 ms |
| RC壁 `RCBI.pt` | 126 | 146 | 0.797 | 0.782 | 0.846 | 0.618 | 5.4 ms |
| RC柱 `RCZHU.pt` | 67 | 71 | 0.702 | 0.653 | 0.721 | 0.480 | 6.0 ms |

Per-class results:

| model | class | instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| 天井 | B | 24 | 0.952 | 0.958 | 0.955 | 0.800 |
| 天井 | C | 56 | 0.980 | 0.865 | 0.964 | 0.819 |
| 天井 | D | 25 | 0.951 | 1.000 | 0.983 | 0.916 |
| 内壁 | B | 94 | 0.938 | 0.804 | 0.928 | 0.676 |
| 内壁 | C | 15 | 0.893 | 1.000 | 0.991 | 0.825 |
| 内壁 | D | 13 | 0.970 | 0.846 | 0.941 | 0.800 |
| RC壁 | B | 107 | 0.866 | 0.846 | 0.908 | 0.655 |
| RC壁 | C | 26 | 0.773 | 0.731 | 0.830 | 0.585 |
| RC壁 | D | 13 | 0.750 | 0.769 | 0.800 | 0.613 |
| RC柱 | B | 27 | 0.629 | 0.519 | 0.615 | 0.438 |
| RC柱 | C | 25 | 0.754 | 0.492 | 0.622 | 0.367 |
| RC柱 | D | 19 | 0.722 | 0.947 | 0.927 | 0.637 |

Detailed CSV outputs:

```text
outputs/yolo9_single_model_baseline_20260602/summary.csv
outputs/yolo9_single_model_baseline_20260602/per_class_summary.csv
```

## Artifacts

```text
outputs/yolo9_single_model_baseline_20260602/tenjo_test
outputs/yolo9_single_model_baseline_20260602/inner_wall_test
outputs/yolo9_single_model_baseline_20260602/rc_wall_test
outputs/yolo9_single_model_baseline_20260602/rc_column_test
outputs/yolo9_single_model_baseline_20260602/logs
outputs/yolo9_single_model_baseline_20260602/summary.csv
```

## Notes

- The downloaded model package was stored at `downloads/previous_phase_gpl_models_drive/previous_phase_gpl_models`.
- It was extracted to `downloads/previous_phase_gpl_model_unpacked/infer_models`, matching the existing deployment config.
- The weakest current baselines are `RC柱` and then `RC壁`; 天井 and 内壁 are already strong.
- For future detector replacement, compare against these four model-specific baselines rather than averaging all components
  too early, because the current model quality is uneven across components.

## Legacy `data_split.json` Reproduction

On 2026-06-02, the original YOLO training split file was uploaded as:

```text
data_split.json
```

This JSON has four parts:

| JSON part | dataset view | model |
|---|---|---|
| `ceiling` | `tenjo` | `TIANJING.pt` |
| `interior` | `inner_wall` | `NEIBI.pt` |
| `rc_wall` | `rc_wall` | `RCBI.pt` |
| `rc_column` | `rc_column` | `RCZHU.pt` |

The split in the JSON is 8:1:1 with stem-only filenames:

| dataset | legacy train images | legacy valid images | legacy test images | legacy test instances B/C/D |
|---|---:|---:|---:|---|
| `tenjo` | 242 | 28 | 31 | 11 / 12 / 9 |
| `inner_wall` | 242 | 28 | 31 | 16 / 8 / 9 |
| `rc_wall` | 242 | 28 | 31 | 14 / 10 / 8 |
| `rc_column` | 242 | 28 | 31 | 12 / 11 / 8 |

A separate copied YOLO dataset view was created here:

```text
data/yolo9_legacy_split_eval
```

All 124 legacy test images were found in the current 20260519 dataset package. However, the legacy test samples mostly
belong to the current package's `train` or `valid` split:

| dataset | where legacy test images are located in current split |
|---|---|
| `tenjo` | 29 train + 2 valid |
| `inner_wall` | 29 train + 2 valid |
| `rc_wall` | 26 train + 3 valid + 2 test |
| `rc_column` | 26 train + 2 valid + 3 test |

This confirms that the previous "current split" baseline above and the uploaded legacy split evaluate different image
sets. They should not be treated as the same test protocol.

Evaluation command:

```bash
python coarse_router_yolov9/yolov9/val.py \
  --data data/yolo9_legacy_split_eval/<dataset>/data.yaml \
  --weights downloads/previous_phase_gpl_model_unpacked/infer_models/<weight>.pt \
  --task test \
  --img 960 \
  --batch 16 \
  --device 0 \
  --project outputs/yolo9_legacy_split_eval \
  --name <dataset> \
  --exist-ok
```

YOLOv9 defaults were used:

```text
conf_thres=0.001
iou_thres=0.7
max_det=300
half=False
```

Legacy test reproduction results:

| model | images | instances | Precision | Recall | mAP50 | mAP50-95 | inference / image |
|---|---:|---:|---:|---:|---:|---:|---:|
| 天井 `TIANJING.pt` | 31 | 32 | 0.593 | 0.624 | 0.562 | 0.328 | 9.3 ms |
| 内壁 `NEIBI.pt` | 31 | 33 | 0.636 | 0.694 | 0.647 | 0.352 | 8.6 ms |
| RC壁 `RCBI.pt` | 31 | 32 | 0.585 | 0.625 | 0.542 | 0.198 | 7.1 ms |
| RC柱 `RCZHU.pt` | 31 | 31 | 0.571 | 0.416 | 0.445 | 0.151 | 7.0 ms |

Per-class results:

| model | class | instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| 天井 | B | 11 | 0.393 | 0.545 | 0.507 | 0.259 |
| 天井 | C | 12 | 0.690 | 0.558 | 0.461 | 0.282 |
| 天井 | D | 9 | 0.697 | 0.768 | 0.718 | 0.444 |
| 内壁 | B | 16 | 0.812 | 0.541 | 0.596 | 0.333 |
| 内壁 | C | 8 | 0.626 | 0.875 | 0.744 | 0.305 |
| 内壁 | D | 9 | 0.472 | 0.667 | 0.600 | 0.418 |
| RC壁 | B | 14 | 0.684 | 0.500 | 0.541 | 0.253 |
| RC壁 | C | 10 | 0.475 | 0.500 | 0.359 | 0.096 |
| RC壁 | D | 8 | 0.595 | 0.875 | 0.725 | 0.245 |
| RC柱 | B | 12 | 0.747 | 0.167 | 0.313 | 0.097 |
| RC柱 | C | 11 | 0.311 | 0.331 | 0.206 | 0.117 |
| RC柱 | D | 8 | 0.655 | 0.750 | 0.816 | 0.238 |

Detailed outputs:

```text
outputs/yolo9_legacy_split_eval/summary.csv
outputs/yolo9_legacy_split_eval/per_class_summary.csv
outputs/yolo9_legacy_split_eval/split_alignment.csv
outputs/yolo9_legacy_split_eval/<dataset>
```

Interpretation:

- The uploaded `data_split.json` is treated as the official previous YOLO split for replacement comparison.
- The previous report's adjusted `reported new R` table above is the customer-facing baseline to beat.
- The raw YOLOv9 `val.py` results in this section are useful for reproducibility and debugging, but they are not a
  substitute for the adjusted report metric unless the same post-processing/aggregation rule is applied.
