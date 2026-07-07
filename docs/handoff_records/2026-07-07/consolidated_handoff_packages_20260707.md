# Consolidated handoff packages 2026-07-07

## Decision

The 2026-07-07 handoff is split into two packages:

- RF-DETR main package: continue future handoffs from this package.
- YOLO archive package: keep only for historical recovery/comparison.

## RF-DETR Main Package

Local staging directory:

```text
.local_artifacts/handoff_20260707_rfdetr_main
```

Planned archive:

```text
.local_artifacts/shimizu_20260707_rfdetr_main_handoff.tar.zst
```

Archive size:

```text
10,549,031,445 bytes
```

SHA256:

```text
e05c2470f59d35d302db6c070dbe3d987160df8c46ebeb357aa87a47f11cf9f7
```

Contents:

- 5-class router data and labels:
  `data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid`
- Reviewed/deduplicated manual annotation records:
  `outputs/gemini_new_router_classes_20260630/manual_review_dedup`
- Downstream 4-branch crack-recognition data and labels:
  `final_crack_yolo_20260519/split`
- Original final-release split index:
  `data_split.json`
- Previous 3-class RF-DETR router model:
  `checkpoint_epoch_023.pth`
- Previous 5-class RF-DETR router model:
  `selected_precision_p090_epoch049_thr069.pth`
- Four downstream RF-DETR models:
  `tenjo`, `inner_wall`, `rc_wall`, `rc_column`
- Relevant configs, scripts, and handoff notes.

Important note: the available local system is router 5 classes plus downstream
4 branches. It is not currently a single 6-class RF-DETR model head.

## YOLO Archive Package

Local staging directory:

```text
.local_artifacts/handoff_20260707_yolo_archive
```

Planned archive:

```text
.local_artifacts/shimizu_20260707_yolo_archive.tar.zst
```

Archive size:

```text
9,850,109,547 bytes
```

SHA256:

```text
3967f6e0dfb60604d9d9dbd48f9d4445f91ccd87e789070b428b5bc29b723fd6
```

Contents:

- 2026-05-26 final download archive.
- 2026-05-26 best YOLO router/eval archive.
- 2026-05-26 final synthetic router data archive.
- 2026-05-19 and 2026-05-26 historical handoff docs.

This package is archive-only. The next delivery should use the RF-DETR package.
