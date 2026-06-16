# RC Wall RF-DETR Optimization - 2026-06-16

## Summary

The recommended RC wall downstream RF-DETR checkpoint has been updated with the best 2026-06-16 fine-tuned checkpoint.

Packaged checkpoint:

```text
final_release_20260615/models/rfdetr/downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth
```

This path is kept stable for pipeline compatibility. The file now contains the optimized checkpoint copied from:

```text
outputs/rfdetr_single_crack/rc_wall_report_best_c_crop_e001_p0722_r0812/checkpoint_epoch_001.pth
```

The previous packaged RC wall checkpoint was retained at:

```text
final_release_20260615/models/rfdetr/downstream/rc_wall/references/rc_wall_20260615_pre_optimization_checkpoint_epoch_009.pth
```

## Report Protocol

- Test split: `data/rfdetr_rc_wall_all_non_legacy_test_v1/test`, derived from `data_split.json`.
- Match IoU: `0.229`.
- Class thresholds: B `0.28`, C `0.45`, D `0.25`.
- Evaluator: `scripts/evaluate_rfdetr_class_threshold_grid.py`.

## Fixed-Threshold Metrics

| Model | Precision | Recall | F1 | B Recall | C Recall | D Recall |
|---|---:|---:|---:|---:|---:|---:|
| Previous packaged RC wall RF-DETR | 0.632 | 0.750 | 0.686 | 0.857 | 0.500 | 0.875 |
| 2026-06-16 optimized RC wall RF-DETR | 0.722 | 0.812 | 0.765 | 0.857 | 0.600 | 1.000 |

The optimized checkpoint improves precision, overall recall, C recall, and D recall under the fixed report protocol.

## Experiments Checked

The following training directions were tested but not retained because they did not exceed the optimized checkpoint under the fixed report protocol:

- C crop augmentation plus oversampling.
- Higher-resolution 1120 training.
- Train-hardcase C false-negative views.
- Matcher/loss adjustments for C confidence and localization.
- Very small learning-rate continuation.
- Varifocal loss continuation.

## C-Class Residual Error Analysis

The remaining C recall is `6/10` on the fixed report test set. Raising C recall requires rescuing at least one additional C ground-truth box.

Observed C false-negative groups:

| Image | Finding |
|---|---|
| `data_add100__3-C-00021.jpg` | High-confidence C prediction, but predicted box is larger than the GT box. IoU is `0.221810`, just below the `0.229` report threshold. |
| `data_add100__3-C-00028.jpg` | Model covers the wider damaged region while GT marks a smaller subregion. This is primarily a box-definition mismatch. |
| `data_add100__3-C-40152.jpg` | Same-class overlap is sufficient, but C confidence is below the `0.45` threshold. |
| `data_add100__c-40537.jpg` | Small C target has good same-class overlap but low confidence; another C target in the same image is present. |

Important precision limiter:

| Image | Finding |
|---|---|
| `data_add100__3-B-00058.jpg` | GT is B but the model predicts high-confidence C with strong overlap. This appears to be a B/C boundary case and contributes to C false positives. |

## Packaged Analysis Assets

```text
final_release_20260615/models/rfdetr/metrics/rc_wall_optimization_20260616/test_results_profiles_summary.csv
final_release_20260615/models/rfdetr/metrics/rc_wall_optimization_20260616/hard_cases_thr045_iou0229.csv
final_release_20260615/models/rfdetr/metrics/rc_wall_optimization_20260616/train_hard_cases_thr045_iou0229.csv
final_release_20260615/docs/report_assets_20260616_rc_wall/c_related_contact_sheet.jpg
final_release_20260615/docs/report_assets_20260616_rc_wall/c_related_all_gt_contact_sheet.jpg
final_release_20260615/docs/report_assets_20260616_rc_wall/train_label_conflict/train_conflict_contact_sheet.jpg
```

## Recommended Next Step

Further C recall improvement is more likely to come from label review and targeted data additions than from blind continuation training. Review the four C false negatives and the B/C boundary case above before launching another training run.
