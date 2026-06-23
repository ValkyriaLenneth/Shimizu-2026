# YOLO / YOLO9 System Index

YOLO is the legacy line in this repo. It remains important for comparison,
handoff history, and some router/pipeline fallback references, but RF-DETR is
the current production direction.

## Router

Primary locations:

```text
systems/yolo/router/coarse_router_yolov9/
systems/yolo/router/coarse_router_yolov9/scripts/
systems/yolo/router/coarse_router_yolov9/hyps/
systems/yolo/router/coarse_router_yolov9/yolov9/
```

Key scripts:

```text
coarse_router_yolov9/scripts/build_coarse_yolo_dataset.py
coarse_router_yolov9/scripts/build_router_3class_input_aug_v1.py
coarse_router_yolov9/scripts/train_router_3class_parallel.sh
coarse_router_yolov9/scripts/train_router_3class_gemini_mix.sh
coarse_router_yolov9/scripts/check_router_3class_training_ready.py
```

Important docs:

```text
docs/router_3class_data_cleaning_2026-05-19.md
docs/router_3class_training_preparation_2026-05-19.md
docs/router_tuning_b_c_results_20260519.md
docs/client_report_yolo_coarse_router.md
docs/yolo9_single_crack_model_baseline_20260602.md
```

Final release package also contains the retained YOLO router weight:

```text
final_release_20260615/data/final_download_20260526/.../coarse_router_yolov9/runs/train/.../weights/best.pt
```

The full file is in `/Users/len/Downloads/final_release_20260615.tar.zst`, not
in git.

Detailed index:

```text
systems/yolo/router/
```

## Recognition Models

YOLO recognition models are historical B/C/D crack or damage detectors by
component. Their active engineering has been superseded by RF-DETR, but their
metrics and failure cases are still used as baselines.

| category | meaning | current status |
|---|---|---|
| `tenjo` | ceiling damage recognition | legacy baseline/reference |
| `inner_wall` | inner wall damage recognition | legacy baseline/reference |
| `rc_wall` | RC wall damage recognition | legacy baseline/reference |
| `rc_column` | RC column damage recognition | legacy baseline/reference |

Relevant historical data and docs:

```text
docs/final_crack_dataset_20260519.md
docs/report_story_router_e2e_20260519.md
docs/original_prod_vs_router_e2e_20260519.md
router_crack_pipeline/
```

Detailed indexes:

```text
systems/yolo/recognition_models/tenjo/
systems/yolo/recognition_models/inner_wall/
systems/yolo/recognition_models/rc_wall/
systems/yolo/recognition_models/rc_column/
```

## Pipeline

Primary legacy pipeline locations:

```text
systems/yolo/pipeline/router_crack_pipeline/
systems/yolo/pipeline/router_crack_pipeline/pipeline/
systems/yolo/pipeline/router_crack_pipeline/configs/
systems/yolo/pipeline/router_crack_pipeline/scripts/
```

This pipeline is preserved for comparison and migration history. New production
work should start from:

```text
rfdetr_prod_pipeline/
```

Detailed index:

```text
systems/yolo/pipeline/
```
