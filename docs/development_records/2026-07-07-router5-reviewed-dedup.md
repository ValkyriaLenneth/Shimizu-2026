# 2026-07-07 router 5-class reviewed dedup data update

## Purpose

The `ブレース` and `柱脚` images were supplied in two batches with duplicate
content across old and new batches. After manual annotation review, the router
data source was deduplicated by image SHA256 and rebuilt from reviewed labels.

## Inputs

Manual review app:

```text
tools/gemini_annotation_review_app/app.py
tools/gemini_annotation_review_app/dedup_review.py
```

Manual review artifacts:

```text
outputs/gemini_new_router_classes_20260630/manual_review_dedup/dedup_items.json
outputs/gemini_new_router_classes_20260630/manual_review_dedup/review_annotations.json
```

## Dedup Result

| item | count |
|---|---:|
| original Gemini rows | 919 |
| deduplicated unique images | 685 |
| removed duplicate rows | 234 |
| saved manual review records | 451 |

Unique images by expected class:

| class | images |
|---|---:|
| ブレース | 362 |
| 柱脚 | 323 |

## Dataset Rebuild

Output dataset:

```text
data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid
```

Policy:

```text
new_label_policy: expected-only
valid_source: test
```

The expected-only policy keeps only the expected new-class boxes from the new
images. This avoids adding unreviewed old-class Gemini boxes from the new-class
photo batches into the existing old-class router data.

## Dataset Counts

| split | images | 天井 boxes | 壁类 boxes | RC柱 boxes | ブレース boxes | 柱脚 boxes |
|---|---:|---:|---:|---:|---:|---:|
| train | 4599 | 2128 | 4172 | 1800 | 326 | 264 |
| valid (= test) | 417 | 183 | 391 | 102 | 43 | 33 |
| test | 417 | 183 | 391 | 102 | 43 | 33 |

Preflight passed with no missing labels, orphan labels, empty labels, or
malformed label lines.

## Next Step

Retrain RF-DETR router with:

```text
systems/rfdetr/router/configs/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid.yaml
```

The old 2026-06-30 selected checkpoint remains a historical model trained on
Gemini-only labels.
