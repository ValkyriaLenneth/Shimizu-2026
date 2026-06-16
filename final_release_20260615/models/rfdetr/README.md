# RF-DETR Model Directory

This directory is the final release target for RF-DETR weights.

Current status:

- `config/pipeline.rfdetr_prod.local.yaml` has been copied for reference.
- `config/pipeline.rfdetr_prod.final_release.yaml` points to the organized weights in this release directory.
- Recommended router and four downstream RF-DETR checkpoints are present.
- Reference / alternative checkpoints are retained under each component's `references/` directory where available.
- RC wall was updated on 2026-06-16 with an optimized checkpoint while keeping the stable deployment filename.

Expected layout:

```text
router/
  checkpoint_epoch_023.pth
downstream/
  tenjo/
    tenjo_standard_orig_checkpoint_epoch_009.pth
  inner_wall/
    inner_wall_checkpoint_epoch_026.pth
  rc_wall/
    rc_wall_checkpoint_epoch_009.pth
    rc_wall_checkpoint_epoch_001_optimized_20260616.pth
  rc_column/
    checkpoint_epoch_047.pth
config/
  pipeline.rfdetr_prod.local.yaml
  pipeline.rfdetr_prod.final_release.yaml
```

Recommended deployment weights:

```text
router/checkpoint_epoch_023.pth
downstream/tenjo/tenjo_standard_orig_checkpoint_epoch_009.pth
downstream/inner_wall/inner_wall_checkpoint_epoch_026.pth
downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth
downstream/rc_column/checkpoint_epoch_047.pth
```

RC wall note:

```text
downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth
```

is the 2026-06-16 optimized checkpoint copied from:

```text
outputs/rfdetr_single_crack/rc_wall_report_best_c_crop_e001_p0722_r0812/checkpoint_epoch_001.pth
```

The previous packaged RC wall checkpoint is retained at:

```text
downstream/rc_wall/references/rc_wall_20260615_pre_optimization_checkpoint_epoch_009.pth
```
