# RF-DETR Recognition Model - tenjo

Business label: 天井.

Configs:

```text
systems/rfdetr/recognition_models/tenjo/configs/rfdetr_tenjo_baseline.yaml
systems/rfdetr/recognition_models/tenjo/configs/rfdetr_tenjo_report_finetune.yaml
```

Release checkpoint path inside the full release archive:

```text
final_release_20260615/models/rfdetr/downstream/tenjo/tenjo_standard_orig_checkpoint_epoch_009.pth
```

Status:

```text
Report candidate exists. 2026-06-16 notes recommend adding class-aware NMS to
the report evaluator before replacing the release checkpoint.
```

Docs:

```text
docs/tenjo_rfdetr_failure_analysis_20260608.md
docs/tenjo_rfdetr_report_optimization_20260616.md
```
