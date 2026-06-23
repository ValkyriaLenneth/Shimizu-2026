# YOLO Router

Role: legacy three-class structural router.

Current historical code:

```text
systems/yolo/router/coarse_router_yolov9/
systems/yolo/router/coarse_router_yolov9/scripts/
systems/yolo/router/coarse_router_yolov9/hyps/
systems/yolo/router/coarse_router_yolov9/yolov9/
```

Important scripts:

```text
systems/yolo/router/coarse_router_yolov9/scripts/build_coarse_yolo_dataset.py
systems/yolo/router/coarse_router_yolov9/scripts/build_router_3class_input_aug_v1.py
systems/yolo/router/coarse_router_yolov9/scripts/train_router_3class_parallel.sh
systems/yolo/router/coarse_router_yolov9/scripts/train_router_3class_gemini_mix.sh
```

Release weight path inside the full release archive:

```text
final_release_20260615/data/final_download_20260526/.../coarse_router_yolov9/runs/train/.../weights/best.pt
```

Docs:

```text
docs/development_records/2026-05-19-yolo-router/router_3class_data_cleaning_2026-05-19.md
docs/development_records/2026-05-19-yolo-router/router_3class_training_preparation_2026-05-19.md
docs/development_records/2026-05-19-yolo-router/router_tuning_b_c_results_20260519.md
```
