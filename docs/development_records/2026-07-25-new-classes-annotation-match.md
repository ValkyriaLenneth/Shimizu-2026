# 2026-07-25 ブレース / 柱脚 annotation-to-image 1:1 match

## Purpose

Two new element categories arrived as three independent drops. Before any
RF-DETR training the delivered labels must be paired 1:1 with the raw images,
and the duplicated photographs inside the raw batches must be collapsed so that
identical pixels do not receive contradictory supervision or leak across splits.

## Inputs

```text
data/downloads/raw_extract/ブレース/                                       381 images
data/downloads/raw_extract/柱脚/                                           328 images
data/downloads/annot_extract/20260717_.../5_ブレース/obj_train_data/...     381 labels
data/downloads/annot_extract/20260717_.../6_柱脚/obj_train_data/...         328 labels
data/downloads/annot_extract/20260724_.../ブレース_追加分_JSCA講習より/      29 images, no labels
data/downloads/annot_extract/20260724_.../柱脚_追加分_JSCA講習より/          16 images, no labels
```

The `20260717` drop is CVAT YOLO 1.1 with `obj.names = B / C / D`. These are
**damage-grade** labels, so this data belongs to the downstream recognition
stage, not to the router. The router already carries `ブレース` and `柱脚` as
element classes 3 and 4 from the 2026-07-07 reviewed/deduplicated update.

## Tool

```bash
python systems/rfdetr/scripts/match_new_class_annotations.py \
  --downloads-dir data/downloads \
  --output-dir outputs/new_class_annotation_match_20260724 \
  --emit-paired-dir data/new_classes_paired_20260724
```

Artifacts:

```text
outputs/new_class_annotation_match_20260724/manifest.json            per-image records
outputs/new_class_annotation_match_20260724/image_label_pairs.csv    flat 1:1 pairing table
outputs/new_class_annotation_match_20260724/match_summary.json       counts only
outputs/new_class_annotation_match_20260724/duplicate_conflicts.json review queue
data/new_classes_paired_20260724/{brace,column_base}/{images,labels} deduplicated pairs
```

## Pairing Integrity

Filename stems pair the two drops exactly. Verified as a bijection in both
directions, with no orphan image, no orphan label, and no malformed label line.

| category | images | labels | orphans | malformed lines |
|---|---:|---:|---:|---:|
| ブレース | 381 | 381 | 0 | 0 |
| 柱脚 | 328 | 328 | 0 | 0 |

The 45 `20260724` images use fresh stems (`e-00382`..`e-00410`,
`f-00329`..`f-00344`) and carry no labels, so they are recorded as
`extra_unlabelled` and excluded from the training pairing.

## Duplicate Detection

SHA256 alone is not sufficient here. 21 duplicate pairs are the same photograph
stored at a different resolution, for example `e-00078` at 415x311 and
`e-00236` at 2048x1536. Detection therefore runs in two stages:

```text
stage 1  SHA256 equality                        -> byte-identical copies
stage 2  dHash Hamming <= 4 AND 32x32 MSE <= 2  -> rescaled / re-encoded copies
```

The MSE gate is what keeps the pass honest: one pair (`e-00046` / `e-00047`)
collides on dHash with distance 0 but has MSE 218, and is correctly kept as two
distinct images. A weaker `dHash <= 6` grouping is also recorded as
`scene_group_id`; those images are kept as separate training samples but must
stay inside a single split, because they are near-identical views of one scene.

Clustering runs inside a category only. Cross-category identical content is
reported rather than merged, since the two categories become two independent
datasets.

## Dedup Result

| item | ブレース | 柱脚 |
|---|---:|---:|
| annotated images delivered | 381 | 328 |
| unique images after dedup | 352 | 306 |
| redundant copies dropped | 29 | 22 |
| boxes delivered | 507 | 336 |
| boxes on unique images | 477 | 320 |
| empty labels delivered | 67 | 89 |
| empty labels after dedup | 59 | 82 |
| scene groups after dedup | 349 | 298 |

Grade distribution on the deduplicated set:

| category | B | C | D | total |
|---|---:|---:|---:|---:|
| ブレース | 146 | 239 | 92 | 477 |
| 柱脚 | 199 | 82 | 39 | 320 |

Representative election inside a duplicate cluster is deterministic: most boxes
first, then largest pixel area, then lowest stem. "Most boxes" is the right
default because the dominant failure mode in this delivery is a duplicate copy
that was left unannotated.

## Review Queue

45 of 52 duplicate clusters disagree across copies of the same photograph.

| severity | clusters | meaning |
|---|---:|---|
| `disagreement` | 14 | every copy annotated, different box counts |
| `coordinate_drift` | 20 | same box count, coordinates differ slightly |
| `unannotated_duplicate` | 11 | one copy annotated, its twin left empty |

`coordinate_drift` is expected when the same photo was annotated twice at
different resolutions and needs no action. The two items that do need a human
decision:

**11 clusters contradict on damage grade.** Same pixels, different grade:

```text
dup-0425  f-00047 [B]      vs  f-00131 [C]
dup-0429  f-00051 [B]      vs  f-00134 [C]
dup-0432  f-00054 [B]      vs  f-00137 [C]
dup-0433  f-00055 [B]      vs  f-00138 [C]
dup-0576  f-00211 [B]      vs  f-00303 [C]
dup-0577  f-00212 [B]      vs  f-00304 [C]
dup-0086  e-00087 [D]      vs  e-00118 [B, D]
dup-0088  e-00089 [B, C]   vs  e-00248 [C]
dup-0108  e-00109 [B, C]   vs  e-00249 [C]
dup-0118  e-00120 [B, D]   vs  e-00247 [D]
dup-0119  e-00121 [B, C]   vs  e-00254 [C]  vs  e-00274 [C]
```

The 柱脚 B-vs-C cases are systematic, which suggests a threshold ambiguity in
the grading criterion rather than isolated annotator slips.

**3 images are filed under both categories.** Byte-identical, all in the
unlabelled `20260724` drop:

```text
brace/e-00383 == column_base/f-00329
brace/e-00385 == column_base/f-00330
brace/e-00386 == column_base/f-00331
```

## Open Questions For The Annotation Team

1. Resolve the 11 grade contradictions above, especially the systematic 柱脚
   B-vs-C group.
2. Confirm the semantics of the remaining empty labels: 59/352 for ブレース and
   82/306 for 柱脚. "Inspected, no damage" makes them valuable negatives;
   "not yet annotated" makes them harmful.
3. Confirm the intended category for the 3 images filed under both categories.
4. Confirm whether the 45 `20260724` images will be annotated. They are small
   (minimum side 127-247 px) and of limited training value as-is.

## Dataset Characteristics Relevant To Training

| property | ブレース | 柱脚 |
|---|---|---|
| kept images | 352 | 306 |
| min-side median | 632 px | 480 px |
| images with min-side < 400 px | 88 | 55 |
| boxes with relative area < 0.01 | 111 / 477 | 75 / 320 |
| images with a single box | 206 | 157 |
| rarest grade | D, 92 boxes | D, 39 boxes |

Both sets are roughly an order of magnitude smaller than the existing
downstream datasets, resolution is highly heterogeneous, and grade D is scarce
in 柱脚 with 39 boxes. These three facts drive the training plan.

## Next Step

Build the two downstream RF-DETR datasets from
`data/new_classes_paired_20260724`, splitting on `scene_group_id` so that
near-identical views cannot straddle train and test, and following the existing
downstream convention of mirroring test into valid.
