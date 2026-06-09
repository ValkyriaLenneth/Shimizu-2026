# RF-DETR downstream progress 2026-06-08

## Evaluation policy

Selection is recall-first on the official test split reconstructed from `data_split.json`.

RF-DETR `test/recall` is the recall value from RF-DETR's F1 confidence-threshold sweep, macro-averaged over classes with ground truth. It is not a fixed `conf=0.25` number. For old YOLO comparisons, note that previous runs likely used a very low threshold around `0.01`, so low-threshold diagnostics are recorded separately and are not the official RF-DETR selection metric.

The RF-DETR dataset views use:

```text
train = all non-official-test images
valid = official test mirror, required by RF-DETR training
test  = official test
```

Because `valid` mirrors `test`, train-time validation is only a monitoring signal. Final decisions are made by checkpoint reload and explicit test evaluation.

## Fixed inner-wall candidate

Status: fixed / archived.

Candidate:

```text
outputs/rfdetr_single_crack/inner_wall_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_026.pth
```

Archive:

```text
rfdetr_model_candidates_20260608.tar.zst
```

Official test result:

| scope | target recall | candidate recall | status |
|---|---:|---:|---|
| Overall | > 0.750 | 0.8380 | pass |
| B | > 0.747 | 0.6250 | below target |
| C | > 0.773 | 1.0000 | pass |
| D | >= 0.800 | 0.8889 | pass |

Overall metrics:

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.7359 | 0.8380 | 0.7560 | 0.7842 | 0.4755 |

The follow-up run from epoch 26 with B-class oversampling did not improve the official recall-first result:

```text
outputs/rfdetr_single_crack/inner_wall_medium_best_e026_b_os3_ft_lr2e5
```

Best checked intermediate point from that run was epoch 8:

| Precision | Recall | F1 | mAP50 | B recall | C recall | D recall |
|---:|---:|---:|---:|---:|---:|---:|
| 0.7110 | 0.7963 | 0.7422 | 0.7300 | 0.6250 | 0.8750 | 0.8889 |

Conclusion: inner-wall is good enough to stop for now. It is not perfect because B remains below the prior adjusted target, but overall/C/D are strong and additional B oversampling degraded the candidate.

## Ceiling review

Targets:

| scope | target recall |
|---|---:|
| Overall | > 0.845 |
| B | > 0.750 |
| C | > 0.826 |
| D | >= 1.000 |

Current best official-test candidates remain far below target.

Baseline:

```text
outputs/rfdetr_single_crack/tenjo_medium_all_non_legacy_test_v1
```

Representative official-test result:

| epoch | Precision | Recall | mAP50 | B recall | C recall | D recall |
|---:|---:|---:|---:|---:|---:|---:|
| 6 | 0.806 | 0.568 | 0.669 | 0.4545 | 0.5833 | 0.6667 |

Tried and rejected:

| run | idea | best observed result | conclusion |
|---|---|---|---|
| `tenjo_medium_res896_bd_os3_test_v1` | resolution 896 + B/D oversampling | best selected recall 0.633; B stuck at 0.4545 | does not solve B or overall |
| `tenjo_medium_res896_bd_os3_test_v1_continue_e016` | continue from failed high-res/oversampled checkpoint | best selected recall 0.568 | degraded |
| `tenjo_medium_best_e006_res896_ft_lr2e5` | start from baseline epoch 6, no oversampling, resolution 896, lr 2e-5 | final/epoch29 recall 0.605; B 0.4545, C 0.5833, D 0.7778 | no meaningful gain |
| `tenjo_medium_e006_b_os2_res896_ft_lr1e5` | start from baseline epoch 6, B full-image oversampling x2, lr 1e-5 | best epoch 4 recall 0.6978; B 0.4545, C 0.7500, D 0.8889 | improves C/D and overall, still does not move B |
| `tenjo_medium_e004_b_crop2_res896_ft_lr5e6` | start from B-os2 epoch 4, add 614 train-only B crop images, lr 5e-6 | best epoch 7 recall 0.7071; B 0.4545, C 0.6667, D 1.0000 | crop augmentation improves D/overall but B remains fixed |

New hard-case / threshold diagnostics:

```text
outputs/rfdetr_single_crack/tenjo_medium_all_non_legacy_test_v1/hard_cases_epoch006_thr025.csv
outputs/rfdetr_single_crack/tenjo_medium_all_non_legacy_test_v1/threshold_sweep_epoch006_grid.csv
outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2_res896_ft_lr5e6/hard_cases_epoch002_thr025.csv
outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2_res896_ft_lr5e6/threshold_sweep_epoch002_grid.csv
```

Low-threshold diagnostic is different: some ceiling checkpoints can reach very high recall at `conf <= 0.01`, but precision collapses due to thousands of false positives. For example, crop epoch 2 reaches B recall 0.8182 / C 1.0000 / D 1.0000 at threshold 0.001, but with 9137 false positives and precision 0.0033. This means the model often produces candidate boxes, but score ranking and false-positive separation are poor. Further ceiling work should not be another simple oversampling or crop-only run.

Recommended next ceiling direction:

1. Stop blind full-image B oversampling and simple B crop augmentation; both failed to move official-test B above 0.4545.
2. Inspect the B false positives and low-confidence B true positives visually. The failure mode is now score separation, not absence of candidate boxes.
3. Next training should use hard-negative mining from train predictions or a curated background-heavy fine-tune, keeping the official test split fixed.
4. Only after visual review, consider class-specific threshold calibration as a secondary report, not as the official model replacement.

## RC wall review

Targets:

| scope | target recall |
|---|---:|
| Overall | > 0.720 |
| B | > 0.739 |
| C | > 0.680 |
| D | > 0.667 |

Current official recall-best checkpoint:

```text
outputs/rfdetr_single_crack/rc_wall_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_009.pth
```

| epoch | Precision | Recall | mAP50 | B recall | C recall | D recall |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 0.5772 | 0.7202 | 0.6001 | 0.7857 | 0.5000 | 0.8750 |

This is usable as the current recall-best RC wall model, but it does not satisfy the C target.

Other important baseline point:

| epoch | Recall | B recall | C recall | D recall | note |
|---:|---:|---:|---:|---:|---|
| 63 | 0.7036 | 0.7857 | 0.7000 | 0.6250 | C passes, D fails |

Tried and rejected:

| run | idea | result | conclusion |
|---|---|---|---|
| `rc_wall_medium_c_os3_test_v1_continue_e021` | C oversampling from early C-os checkpoint | best checked recall 0.646; C up to 0.600 | not enough, hurts other classes |
| `rc_wall_medium_best_e009_c_os3_ft_lr2e5` | start from epoch 9 and push C with C-os3, lr 2e-5 | best checked recall 0.702; C remains 0.500 | does not move C |
| `rc_wall_medium_best_e063_d_os2_ft_lr1e5` | start from epoch 63 and push D lightly | best checked epoch 4 recall 0.646; B 0.7143, C 0.6000, D 0.6250 | degraded from both epoch 9 and epoch 63 |
| `rc_wall_medium_e063_orig_ft_lr5e6` | low-lr continuation from epoch 63 on original data | best checked epoch 3 recall 0.6702; B 0.7857, C 0.6000, D 0.6250 | still worse than epoch 9 overall and worse than epoch 63 for C |

New RC wall analysis:

```text
outputs/rfdetr_single_crack/rc_wall_medium_all_non_legacy_test_v1/hard_cases_epoch009_thr025.csv
outputs/rfdetr_single_crack/rc_wall_medium_all_non_legacy_test_v1/hard_cases_epoch063_thr025.csv
outputs/rfdetr_single_crack/rc_wall_medium_e063_orig_ft_lr5e6/hard_cases_epoch003_thr025.csv
outputs/rfdetr_single_crack/rc_wall_medium_all_non_legacy_test_v1/class_routing_e9_b_d_e63_c_threshold_sweep.csv
```

Per-class checkpoint routing is the first RC wall result that clears all targets on an explicit fixed-threshold evaluation:

```text
class 0/B -> epoch 9
class 1/C -> epoch 63
class 2/D -> epoch 9
threshold = 0.25
```

| Precision | Recall | F1 | B recall | C recall | D recall |
|---:|---:|---:|---:|---:|---:|
| 0.4630 | 0.7813 | 0.5814 | 0.8571 | 0.7000 | 0.7500 |

This is not a single exported RF-DETR checkpoint; it is a two-checkpoint routing policy implemented for evaluation in:

```text
scripts/evaluate_rfdetr_class_routing.py
```

Recommended next RC wall direction:

1. Treat RC wall as functionally solved if a routed inference policy is acceptable.
2. If a single checkpoint is required, epoch 9 remains the best single-checkpoint official recall model, but C is below target.
3. Do not continue broad C/D oversampling without a curated hard-case plan; low-lr continuation and target-class oversampling both degraded the tradeoff.
4. Productionizing the routed policy requires adding an inference wrapper that runs both checkpoints and filters predictions by class source.
