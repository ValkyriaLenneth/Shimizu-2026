# RF-DETR Router

Role: route an input image into structural regions before downstream B/C/D
recognition.

Current code and configs:

```text
systems/rfdetr/scripts/train_rfdetr_router.py
systems/rfdetr/scripts/rfdetr_router_callbacks.py
systems/rfdetr/scripts/check_rfdetr_router_dataset.py
systems/rfdetr/router/configs/rfdetr_router_base_aug_v2.yaml
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/rfdetr_router_infer.py
```

Release artifact paths inside the full release archive:

```text
final_release_20260615/models/rfdetr/router/checkpoint_epoch_023.pth
final_release_20260615/models/rfdetr/router/checkpoint_23.ckpt
```

Important docs:

```text
docs/rfdetr_router_training_20260602.md
docs/rfdetr_work_summary_20260609.md
```
