# Results Summary

## Dataset

- Raw valid images: 3015
- Excluded cross-class duplicate files: 20
- Final fixed dataset size: 2995
- Train: 2094
- Val: 451
- Test: 450

Test split:

```text
RC壁: 146
RC柱: 65
内壁: 128
天井: 111
```

## Runs

```text
EfficientNet-B0: outputs/runs/20260425_104726_efficientnet_b0
ResNet34:       outputs/runs/20260425_132343_resnet34
Ensemble:       outputs/runs/20260425_143017_ensemble_effb0_resnet34
```

## Test Metrics

All metrics are computed on the same fixed 450-image test split.

```text
EfficientNet-B0:
accuracy 0.7978 | macro precision 0.7936 | macro R1/recall 0.8231 | macro F1 0.8040

ResNet34:
accuracy 0.7733 | macro precision 0.7681 | macro R1/recall 0.7927 | macro F1 0.7772

Ensemble 0.5/0.5:
accuracy 0.8111 | macro precision 0.8110 | macro R1/recall 0.8320 | macro F1 0.8192
```

## Notes

- R1 is reported as Recall@1, equivalent to class recall in this single-label classification task.
- The instance has no CJK font installed, so generated PNG labels may render Japanese glyph warnings. CSV/JSON reports preserve the Japanese class names correctly.
- Raw image data is not included in the result archive by default.
