# RF-DETR Pipeline

Role: current end-to-end production pipeline.

Current code:

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/run_full_pipeline.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/crack_detector_registry.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/rfdetr_backend.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/rfdetr_router_infer.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/wall_candidate_display.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/display_merge.py
```

Current config:

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml
```

Important docs:

```text
docs/pipeline_eval_20260615.md
docs/pipeline_visual_error_analysis_20260616.md
final_release_20260615/MANIFEST.md
```

Smoke test:

```bash
python3 -m pytest rfdetr_prod_pipeline/tests -q
```
