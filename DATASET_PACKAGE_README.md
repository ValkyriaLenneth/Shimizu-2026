# Processed Dataset Package

This archive contains the fixed reusable image classification dataset split.

## Contents

```text
data/processed/building_cls_v1/
  train/
  val/
  test/
data/manifests/
data/audit/
configs/dataset.yaml
```

## Class Names

```text
天井
内壁
RC壁
RC柱
```

## Split Counts

```text
train: 2094
val:   451
test:  450
total: 2995
```

Per-class split:

```text
test   RC壁 146 | RC柱 65 | 内壁 128 | 天井 111
train  RC壁 682 | RC柱 302 | 内壁 594 | 天井 516
val    RC壁 147 | RC柱 65 | 内壁 128 | 天井 111
```

## Notes

- The original valid raw image count was 3015.
- 20 cross-class duplicate files were excluded before splitting.
- The split is fixed with seed `20260425`.
- Use `data/manifests/class_to_idx.json` for class index mapping.
