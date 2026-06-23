# RF-DETR System Index

RF-DETR is the current production direction in this repo.

## Router

Primary locations:

```text
scripts/train_rfdetr_router.py
scripts/rfdetr_router_callbacks.py
scripts/check_rfdetr_router_dataset.py
configs/rfdetr_router_base_aug_v2.yaml
rfdetr_prod_pipeline/pipeline/rfdetr_router_infer.py
```

Actual organized locations:

```text
systems/rfdetr/scripts/train_rfdetr_router.py
systems/rfdetr/scripts/rfdetr_router_callbacks.py
systems/rfdetr/scripts/check_rfdetr_router_dataset.py
systems/rfdetr/router/configs/rfdetr_router_base_aug_v2.yaml
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/rfdetr_router_infer.py
```

Current release checkpoint path inside the full release package:

```text
final_release_20260615/models/rfdetr/router/checkpoint_epoch_023.pth
final_release_20260615/models/rfdetr/router/checkpoint_23.ckpt
```

The full checkpoint files are in:

```text
/Users/len/Downloads/final_release_20260615.tar.zst
```

They are intentionally not tracked in git.

Detailed index:

```text
systems/rfdetr/router/
```

## Recognition Models

The downstream recognition models classify B/C/D damage within routed regions.

| category | config | active/release checkpoint in final package | status |
|---|---|---|---|
| `tenjo` | `configs/rfdetr_tenjo_baseline.yaml`, `configs/rfdetr_tenjo_report_finetune.yaml` | `models/rfdetr/downstream/tenjo/tenjo_standard_orig_checkpoint_epoch_009.pth` | report candidate exists; NMS follow-up noted |
| `inner_wall` | `configs/rfdetr_inner_wall_baseline.yaml` | `models/rfdetr/downstream/inner_wall/inner_wall_checkpoint_epoch_026.pth` | current release model |
| `rc_wall` | `configs/rfdetr_rc_wall_baseline.yaml`, `configs/rfdetr_rc_wall_report_finetune.yaml` | `models/rfdetr/downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth` | optimized 2026-06-16 release model |
| `rc_column` | `configs/rfdetr_rc_column_baseline.yaml` | `models/rfdetr/downstream/rc_column/checkpoint_epoch_047.pth` | current release model |

Primary training/evaluation scripts:

```text
scripts/build_rfdetr_single_crack_views.py
scripts/train_rfdetr_router.py
scripts/evaluate_rfdetr_class_threshold_grid.py
scripts/evaluate_rfdetr_threshold_sweep.py
scripts/analyze_rfdetr_hard_cases.py
scripts/select_and_cleanup_rfdetr_checkpoints.py
```

Important docs:

```text
docs/rfdetr_router_training_20260602.md
docs/rfdetr_single_crack_training_20260602.md
docs/rfdetr_work_summary_20260609.md
docs/rfdetr_downstream_progress_20260608.md
docs/today_rc_wall_optimization_handoff_20260616.md
docs/tenjo_rfdetr_report_optimization_20260616.md
```

Detailed indexes:

```text
systems/rfdetr/recognition_models/tenjo/
systems/rfdetr/recognition_models/inner_wall/
systems/rfdetr/recognition_models/rc_wall/
systems/rfdetr/recognition_models/rc_column/
```

## Pipeline

Primary locations:

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml
systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/
```

Current production flow:

```text
input image
  -> RF-DETR router: tenjo / wall / rc_column
  -> region view by ndarray slice
  -> RF-DETR downstream B/C/D recognition model
  -> merge internal detections
  -> wall display rule
  -> final display merge
  -> JSONL + visualization outputs
```

Key pipeline modules:

```text
rfdetr_prod_pipeline/pipeline/run_full_pipeline.py
rfdetr_prod_pipeline/pipeline/crack_detector_registry.py
rfdetr_prod_pipeline/pipeline/rfdetr_backend.py
rfdetr_prod_pipeline/pipeline/rfdetr_router_infer.py
rfdetr_prod_pipeline/pipeline/wall_candidate_display.py
rfdetr_prod_pipeline/pipeline/display_merge.py
```

Current tests:

```text
python3 -m pytest rfdetr_prod_pipeline/tests -q
```

Detailed index:

```text
systems/rfdetr/pipeline/
```
