# Documentation Outline

This directory is organized as the project memory for Shimizu 2026. Use this
file as the first entry point, then follow the three record groups below.

## 1. Development Records

Path:

```text
docs/development_records/
```

Purpose: day-by-day engineering history, experiments, evaluations, analysis
assets, and repository organization notes.

Recommended reading order:

| period | focus | entry documents |
|---|---|---|
| 2026-05-19 | YOLO9 router, cleaned router data, first end-to-end comparison | `development_records/2026-05-19-yolo-router/` |
| 2026-05-26 | YOLO fallback/display logic and synthetic router data plan | `development_records/2026-05-26-yolo-fallback/` |
| 2026-06-02 | RF-DETR migration start, RF-DETR router, RC column downstream validation | `development_records/2026-06-02-rfdetr-migration/` |
| 2026-06-08/09 | RF-DETR downstream expansion and production pipeline creation | `development_records/2026-06-08-09-rfdetr-downstream/` |
| 2026-06-15/16 | Final release evaluation, visual error analysis, RC wall report optimization | `development_records/2026-06-15-16-final-release/` |
| 2026-06-23 | Repository cleanup and model-family reorganization | `development_records/repo_organization_20260623.md` |
| 2026-07-07 | reviewed/deduplicated 5-class router data update and RF-DETR/Yolo handoff package split | `development_records/2026-07-07-router5-reviewed-dedup.md`, `handoff_records/2026-07-07/consolidated_handoff_packages_20260707.md` |

Supporting material:

```text
development_records/assets/
development_records/legacy_classification/
development_records/misc/
```

## 2. Handoff Records

Path:

```text
docs/handoff_records/
```

Purpose: recovery instructions, weekly handoff notes, package manifests, and
final release handoff state.

Start here:

```text
handoff_records/README.md
```

Key handoff checkpoints:

| date | meaning |
|---|---|
| 2026-05-19 | first YOLO router and crack dataset handoff |
| 2026-05-26 | best YOLO router, fallback logic, final synthetic data handoff |
| 2026-06-02 | RF-DETR migration recovery notes |
| 2026-06-16 | RC wall optimization and final release update handoff |
| 2026-07-07 | reviewed/deduplicated router 5-class data update; RF-DETR main package and YOLO archive package |

The large release payload is intentionally outside git:

```text
.local_artifacts/releases/final_release_20260615.tar.zst
.local_artifacts/extracted/final_release_20260615/
.local_artifacts/shimizu_20260707_rfdetr_main_handoff.tar.zst
.local_artifacts/shimizu_20260707_yolo_archive.tar.zst
```

The tracked `final_release_20260615/` directory keeps reviewable metadata only.

## 3. Meeting Notes

Path:

```text
docs/meeting_notes/
```

Purpose: meeting narratives, customer-facing story drafts, and report notes.

Current meeting sequence:

| date | focus |
|---|---|
| 2026-05-26 | YOLO router pipeline, fallback, and synthetic data |
| 2026-06-02 | RF-DETR router migration progress |
| 2026-06-09 | RF-DETR downstream expansion |

Japanese report material is under:

```text
meeting_notes/reports/
```

## Current Project Direction

The current production direction is RF-DETR:

```text
RF-DETR router
  -> RF-DETR downstream B/C/D recognition models
  -> RF-DETR production pipeline
  -> final display merge / report artifacts
```

Code is organized under:

```text
systems/rfdetr/
systems/yolo/
```

The root symlinks remain for backward compatibility with older commands and
historical documentation.
