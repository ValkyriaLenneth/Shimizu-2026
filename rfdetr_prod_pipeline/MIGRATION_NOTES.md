# RF-DETR pipeline migration notes

Date: 2026-06-09

This folder is an independent RF-DETR-oriented copy of the previous router + crack pipeline.

## Source

- Local baseline: `router_crack_pipeline/`
- Original GitHub reference requested by the user: `Generative-AI-Tokyo/Shimizu-VLM-Crack-Detection-Prod`
- The GitHub repository was not accessible without authentication in this environment, so the migration uses the local pipeline copy already restored in this workspace.

## Default entrypoint

```bash
python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline \
  --config rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml \
  --source data/rfdetr_rc_wall_all_non_legacy_test_v1/test/images/data_add100__3-B-00009.jpg \
  --output-dir outputs/rfdetr_prod_pipeline/smoke_real \
  --device cpu \
  --skip-visualization
```

## RF-DETR models

- Router: `rfdetr_model_candidates_20260602/router_epoch23/checkpoint_epoch_023.pth`
- Ceiling: `rfdetr_threshold_tuned_models_20260609/checkpoints/tenjo_standard_orig_checkpoint_epoch_009.pth`
- Inner wall: `rfdetr_threshold_tuned_models_20260609/checkpoints/inner_wall_checkpoint_epoch_026.pth`
- RC wall: `rfdetr_threshold_tuned_models_20260609/checkpoints/rc_wall_checkpoint_epoch_009.pth`
- RC column: `rfdetr_model_candidates_20260602/rc_column_epoch47/checkpoint_epoch_047.pth`

## Wall display rule

When the router detects `壁类`, the pipeline still runs both `inner_wall` and `rc_wall`.
The raw outputs are kept for audit, but `display_crack_detections` emits a single
PC-facing wall result:

| inner wall | RC wall | display |
|---|---|---|
| B | B | 壁-B |
| B | C | 壁-C |
| B | D | 壁-D |
| C | B | 壁-B |
| C | C | 壁-C |
| C | D | 壁-D |
| D | B | 壁-D |
| D | C | 壁-D |
| D | D | 壁-D |

## Validation

Completed:

```bash
python -m py_compile rfdetr_prod_pipeline/pipeline/*.py rfdetr_prod_pipeline/scripts/*.py
python rfdetr_prod_pipeline/scripts/test_pipeline_smoke.py
python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline \
  --config rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml \
  --source data/rfdetr_rc_wall_all_non_legacy_test_v1/test/images/data_add100__3-B-00009.jpg \
  --output-dir outputs/rfdetr_prod_pipeline/smoke_real \
  --device cpu \
  --skip-visualization
```

Both smoke runs completed with `error_count: 0`.
