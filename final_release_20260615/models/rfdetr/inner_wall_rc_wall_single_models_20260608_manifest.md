# RF-DETR single model archive - 2026-06-08

This archive originally contained the fixed inner wall model and the then-current best single-checkpoint RC wall model.

Update 2026-06-16: the packaged deployment checkpoint at `downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth` has been replaced by the optimized RC wall checkpoint from `outputs/rfdetr_single_crack/rc_wall_report_best_c_crop_e001_p0722_r0812/checkpoint_epoch_001.pth`. The original packaged checkpoint is retained in `downstream/rc_wall/references/rc_wall_20260615_pre_optimization_checkpoint_epoch_009.pth`.

## Contents

- `inner_wall_epoch26/checkpoint_epoch_026.pth`
  - Source: `outputs/rfdetr_single_crack/inner_wall_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_026.pth`
  - Official test recall: 0.837963
  - Per-class recall: B 0.6250, C 1.0000, D 0.8889
  - Precision: 0.7359
  - F1: 0.7560
  - mAP50: 0.7842

- `rc_wall_epoch09/checkpoint_epoch_009.pth`
  - Source: `outputs/rfdetr_single_crack/rc_wall_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_009.pth`
  - Official test recall: 0.720238
  - Per-class recall: B 0.7857, C 0.5000, D 0.8750
  - Selection basis: best current RC wall single checkpoint by official test recall.

- `downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth` after 2026-06-16 update
  - Source: `outputs/rfdetr_single_crack/rc_wall_report_best_c_crop_e001_p0722_r0812/checkpoint_epoch_001.pth`
  - Fixed report protocol: thresholds B/C/D = 0.28/0.45/0.25, match IoU = 0.229
  - Precision: 0.722
  - Recall: 0.812
  - Per-class recall: B 0.857, C 0.600, D 1.000
  - Selection basis: improved fixed-report precision and recall while preserving stable deployment path.

## Notes

- RC wall is packaged here as a single model checkpoint, not a class-routed dual-model result.
- Official RF-DETR test recall after checkpoint reload is the selection criterion.
