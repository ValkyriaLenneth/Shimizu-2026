# Handoff Consolidation Index

This index is the canonical map for the weekly handoff material. The detailed
notes remain in `docs/`, while large local archives and extracted payloads are
kept out of git under `.local_artifacts/`.

## Final Reading Order

| date | role | keep as canonical | archive status |
|---|---|---|---|
| 2026-05-19 | YOLO router, initial crack dataset, first end-to-end pipeline handoff | `docs/handoff_records/2026-05-19/final_handoff_20260519.md`, `docs/handoff_records/2026-05-19/package_README.md` | local package moved to `.local_artifacts/handoff_20260519/` |
| 2026-05-26 | YOLO best router, fallback/display logic, final synthetic router data | `docs/handoff_records/2026-05-26/final_handoff_20260526.md`, `docs/handoff_records/2026-05-26/package_README.md` | local package moved to `.local_artifacts/handoff_20260526/` |
| 2026-06-02 | RF-DETR migration start; router reaches Precision >= 0.90; RC column downstream validates RF-DETR | `docs/development_records/2026-06-02-rfdetr-migration/development_summary_20260602.md`, `docs/handoff_records/2026-06-02/handoff_restore_20260602.md` | models/data are external artifacts |
| 2026-06-09 | RF-DETR downstream expansion, threshold tuning, RF-DETR production pipeline | `docs/development_records/2026-06-08-09-rfdetr-downstream/rfdetr_work_summary_20260609.md`, `docs/meeting_notes/2026-06-09/meeting_rfdetr_downstream_expansion_20260609.md` | model packages are external artifacts |
| 2026-06-15 | Final release layout and RF-DETR pipeline evaluation | `final_release_20260615/MANIFEST.md`, `docs/development_records/2026-06-15-16-final-release/pipeline_eval_20260615.md` | tracked folder keeps metadata only; data/models stay ignored |
| 2026-06-16 | RC wall report optimization and final display/report update | `docs/handoff_records/2026-06-16/today_rc_wall_optimization_handoff_20260616.md`, `docs/development_records/2026-06-15-16-final-release/pipeline_visual_error_analysis_20260616.md` | optimized model binaries stay external/ignored |
| 2026-07-07 | Reviewed and deduplicated `ブレース` / `柱脚` router data update | `docs/handoff_records/2026-07-07/router_5class_reviewed_dedup_handoff_20260707.md`, `docs/development_records/2026-07-07-router5-reviewed-dedup.md` | reviewed data and labels are ignored local artifacts and must be included in the next handoff archive |

## Consolidation Decision

The weekly handoff folders should not remain as first-class repo roots. They
mix code snapshots, compressed archives, generated outputs, extracted datasets,
and short README files. The repo now keeps only the small, reviewable handoff
documents in git:

```text
docs/handoff_records/
docs/development_records/
docs/meeting_notes/
```

Large handoff archives, extracted payloads, old dataset zips, and result folders
are local recovery material. They belong in:

```text
.local_artifacts/
```

## Project Progress Summary

1. 2026-05-19 established the YOLO9 three-class router and connected it to the
   crack-recognition pipeline. It also froze the initial final crack dataset.
2. 2026-05-26 kept only the best YOLO router/evaluation assets and introduced
   wall fallback/display logic plus final synthetic router data.
3. 2026-06-02 shifted the technical direction to RF-DETR. The RF-DETR router met
   the customer Precision target and RC column downstream recall exceeded the
   earlier YOLO target.
4. 2026-06-09 expanded RF-DETR downstream models to the remaining categories,
   added threshold-tuned model packaging, and created the RF-DETR production
   pipeline.
5. 2026-06-15/16 turned that work into the current final release layout, then
   improved RC wall reporting metrics and updated the release metadata.
6. 2026-07-07 deduplicated the two new router classes and rebuilt the next
   training dataset from manual review labels.

## Current Code Direction

The active codebase is organized by model family and role:

```text
systems/rfdetr/router/
systems/rfdetr/recognition_models/
systems/rfdetr/pipeline/
systems/yolo/router/
systems/yolo/recognition_models/
systems/yolo/pipeline/
```

Root-level `rfdetr_prod_pipeline`, `router_crack_pipeline`,
`coarse_router_yolov9`, `scripts`, and `configs` are compatibility symlinks for
old commands and historical documents.
