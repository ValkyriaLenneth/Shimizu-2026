# RF-DETR router 5-class reviewed dedup handoff 2026-07-07

## Scope

This handoff updates the 2026-06-30 5-class router data state after manual
review of the two newly added classes:

- `ブレース`
- `柱脚`

The next handoff and next router training should use the reviewed,
deduplicated dataset below, not the earlier raw Gemini-only dataset.

## Canonical Dataset

Use this dataset for the next 5-class RF-DETR router training:

```text
data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid
```

Training config:

```text
systems/rfdetr/router/configs/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid.yaml
```

Build summary:

```text
data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid/build_summary.json
```

Dataset check:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid_dataset_check.json
```

## Manual Review Source

The reviewed dedup queue and manual annotations are local generated artifacts:

```text
outputs/gemini_new_router_classes_20260630/manual_review_dedup/dedup_items.json
outputs/gemini_new_router_classes_20260630/manual_review_dedup/review_annotations.json
outputs/gemini_new_router_classes_20260630/manual_review_dedup/dedup_groups.json
outputs/gemini_new_router_classes_20260630/manual_review_dedup/dedup_summary.json
```

These files are under ignored `outputs/` and must be included in the next
handoff archive.

## Dedup Summary

Original Gemini rows:

```text
919
```

Deduplicated unique images:

| class | unique images |
|---|---:|
| ブレース | 362 |
| 柱脚 | 323 |
| total | 685 |

Removed duplicate rows:

```text
234
```

Manual review records currently saved:

```text
451
```

Rows without a saved manual review record keep the deduplicated queue's current
Gemini boxes. Saved manual review boxes override Gemini boxes.

## Final Dataset Split

The new dataset keeps the existing 3-class router split and appends the
reviewed-deduplicated new-class samples. `valid` mirrors `test`.

| split | images | 天井 boxes | 壁类 boxes | RC柱 boxes | ブレース boxes | 柱脚 boxes |
|---|---:|---:|---:|---:|---:|---:|
| train | 4599 | 2128 | 4172 | 1800 | 326 | 264 |
| valid (= test) | 417 | 183 | 391 | 102 | 43 | 33 |
| test | 417 | 183 | 391 | 102 | 43 | 33 |

One reviewed `柱脚` candidate was excluded by `expected-only` policy because
its saved manual label contains only `壁类`:

```text
data/raw_new_classes_20260630/extracted/柱脚/f-00267.jpg
```

## Rebuild Command

```bash
python3 systems/rfdetr/scripts/build_rfdetr_router_5class_dataset.py \
  --base-yolo-dir .local_artifacts/handoff_20260526/final_download_20260526/handoff_20260519/shimizu_20260519_minimal_repro_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900_aug_v2 \
  --review-items outputs/gemini_new_router_classes_20260630/manual_review_dedup/dedup_items.json \
  --review-annotations outputs/gemini_new_router_classes_20260630/manual_review_dedup/review_annotations.json \
  --output-dir data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid \
  --new-label-policy expected-only \
  --valid-source test \
  --link-mode hardlink \
  --overwrite
```

Preflight:

```bash
python3 systems/rfdetr/scripts/check_rfdetr_router_dataset.py \
  --dataset-dir data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid \
  --write-summary outputs/rfdetr_router/medium_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid_dataset_check.json
```

## Training Note

The selected 2026-06-30 model was trained on the earlier Gemini-only dataset.
After this data update, retrain the 5-class router using the reviewed-dedup
config before reporting new model metrics.
