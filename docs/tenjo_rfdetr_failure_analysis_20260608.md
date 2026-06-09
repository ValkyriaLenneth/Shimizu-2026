# Tenjo RF-DETR failure analysis 2026-06-08

## Scope

Goal: investigate why RF-DETR replacement runs for `tenjo` did not satisfy the recall target while the other component models mostly did.

Official target from `configs/rfdetr_tenjo_baseline.yaml`:

| scope | recall target |
|---|---:|
| overall | 0.845 |
| B | 0.750 |
| C | 0.826 |
| D | 1.000 |

Official RF-DETR test split:

| split | images | B boxes | C boxes | D boxes |
|---|---:|---:|---:|---:|
| train | 912 | 307 | 447 | 237 |
| test | 31 | 11 | 12 | 9 |

Because test B has only 11 boxes, each missed B instance changes B recall by 0.0909.

## Best observed official-test runs

| run | best selected epoch | overall R | B behavior | conclusion |
|---|---:|---:|---|---|
| `tenjo_medium_all_non_legacy_test_v1` | 6 | 0.5682 | B recall fixed around 0.4545 | baseline fails |
| `tenjo_medium_res896_bd_os3_test_v1` | 1 | 0.6330 | B still stuck | 896 + B/D oversampling does not solve it |
| `tenjo_medium_e006_b_os2_res896_ft_lr1e5` | 4 | 0.6978 | B still 0.4545 | improves C/D/overall only |
| `tenjo_medium_e004_b_crop2_res896_ft_lr5e6` | 7 | 0.7071 | B still 0.4545 | best official row, still below target |
| hard-case fine-tunes | varied | <= 0.6145 selected | B still around 0.4545 | hardcase duplication did not fix score ranking |

Top selected official row:

```text
outputs/rfdetr_single_crack/tenjo_medium_e004_b_crop2_res896_ft_lr5e6/test_results_selected_partial.csv
epoch 7: precision=0.5264, recall=0.7071, mAP50=0.6232
```

This remains below the official overall target 0.845 and B target 0.750.

## Threshold finding

At very low thresholds, the model often has candidate boxes for the missed objects.

Baseline epoch 6 threshold sweep:

| threshold | precision | recall | B R | C R | D R | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.0034 | 0.9688 | 0.9091 | 1.0000 | 1.0000 | 9117 | 1 |
| 0.35 | 0.6552 | 0.5938 | 0.4545 | 0.6667 | 0.6667 | 10 | 13 |

Crop epoch 2 threshold sweep:

| threshold | precision | recall | B R | C R | D R | FP | FN |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.001 | 0.0033 | 0.9375 | 0.8182 | 1.0000 | 1.0000 | 9137 | 2 |
| 0.35 | 0.5429 | 0.5938 | 0.4545 | 0.6667 | 0.6667 | 16 | 13 |

Interpretation: if the business objective is literally recall only and false positives do not matter, RF-DETR can pass by using an extremely low threshold such as 0.001. Under any usable detector setting, the failure is score separation: true boxes exist at low confidence, but background/wrong-location boxes outrank them.

## Hard-case pattern

For baseline epoch 6 at threshold 0.25:

| class | false negatives |
|---|---:|
| B | 6 |
| C | 2 |
| D | 2 |

Repeated B false negatives:

| image | low same-class confidence | low same-class IoU | reason |
|---|---:|---:|---|
| `data_add100__1-B-10086.jpg` | 0.0212 | 0.7147 | matched only below threshold |
| `data_add100__1-B-20012.jpg` | 0.2008 | 0.7071 | matched only below threshold |
| `data_add100__1-B-30009.jpg` | 0.0823 | 0.6738 | matched only below threshold |
| `data_add100__a-40251.jpg` | 0.0582 | 0.8353 | matched only below threshold |
| `data_add100__a-40309.jpg` | 0.0182 | 0.6343 | matched only below threshold |
| `data_add100__a-20009.jpg` | 0.0166 | 0.2602 | no same-class IoU match |

Most B misses are not complete localization absence. They are low-confidence correct or near-correct boxes.

## Geometry / data issue

Test B boxes are much smaller and more shape-diverse than C/D:

| class | train median area | test median area | test median width | test median height |
|---|---:|---:|---:|---:|
| B | 0.0278 | 0.0215 | 0.1627 | 0.0789 |
| C | 0.1869 | 0.1709 | 0.4057 | 0.3900 |
| D | 0.6286 | 0.3854 | 0.9171 | 0.4860 |

B is a small/thin-object regime, while C/D are large-region regimes. Simple full-image oversampling and crop duplication increased exposure but did not fix RF-DETR's score ranking for small/thin B defects.

Visual reference generated during this audit:

```text
report_assets/tenjo_b_test_gt_contact_sheet.jpg
```

## Failed training directions

The prior attempts failed for specific reasons:

1. Full-image B/D oversampling added repeats but did not change the hard B score distribution.
2. B crop augmentation improved overall recall through C/D and some localization changes, but B remained fixed at 5/11 under usable thresholds.
3. Hard-case duplication was built from FP/FN examples, but it duplicated the same ambiguous visual modes without enough true hard negatives or calibration pressure.
4. Long low-lr continuations mostly degraded or stayed flat.
5. Some overnight continuation runs ended with SIGTERM/DataLoader worker termination, but the completed checkpoints before termination still showed the same B recall plateau.

## Recommended next moves

If recall is literally the only metric:

1. Use a low fixed inference threshold, starting with `0.001`.
2. Keep `num_select/max_det` high enough, because recall at 0.001 depends on retaining many candidates.
3. Report the precision collapse explicitly: roughly 9000 false positives on 31 test images.

If the model still needs to be operational:

1. Stop more blind B oversampling/crop-only runs.
2. Build a hard-negative training set from the high-confidence false positives that outrank low-confidence true B boxes.
3. Try class-specific thresholding/calibration before new architecture work; B likely needs a much lower threshold than C/D.
4. Consider B-specific small-object training: higher resolution or tiled/crop inference paired with background-heavy negatives, not just positive crop duplication.
5. Evaluate a two-stage policy: RF-DETR high-recall low-threshold proposals followed by a lightweight verifier/classifier on B candidates.
