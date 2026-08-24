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
| 2026-07-07 | Reviewed and deduplicated `ブレース` / `柱脚` router data update; consolidated RF-DETR/Yolo package split | `docs/handoff_records/2026-07-07/router_5class_reviewed_dedup_handoff_20260707.md`, `docs/handoff_records/2026-07-07/consolidated_handoff_packages_20260707.md`, `docs/development_records/2026-07-07-router5-reviewed-dedup.md` | RF-DETR main staging is `.local_artifacts/handoff_20260707_rfdetr_main`; YOLO archive staging is `.local_artifacts/handoff_20260707_yolo_archive` |
| 2026-08-03 | Gemini synthetic data (S1 counterfactual negatives) for `ブレース` / `柱脚` B/C/D, calibrated quality gate, and the training plan | `docs/handoff_records/2026-08-03/synthetic_data_handoff_20260803.md`, `docs/development_records/2026-08-03-new-classes-synthetic-data-plan.md`, `docs/development_records/2026-08-03-s1-pipeline-and-judge-calibration.md` | generated images stay ignored under `outputs/gemini_synth/`; small state files are tracked in `docs/development_records/assets/2026-08-03-s1/` |
| 2026-08-04 | Annotation-completeness audit finds the label noise behind the recall gap; sixteen negative results; the two methods that worked (WBF inference aggregation, BRL sparse-annotation loss) | `docs/handoff_records/2026-08-04/handoff_20260804.md`, `docs/development_records/2026-08-04-label-noise-finding-and-sixteen-negative-results.md`, `docs/development_records/2026-08-04-synthetic-negatives-and-failed-interventions.md` | models, threshold grids, audit JSON and the client report ship in `shimizu_handoff_20260804.tar.zst` (359 MB, external) |
| 2026-08-15 | `ブレース` recall freeze: the 0.723 operating point pinned with its inference parameters, plus the per-grade B/C/D breakdown the client asked for | `docs/handoff_records/2026-08-15/brace_recall_freeze_20260815.md` | checkpoints and threshold grids ship in the `handoff_20260815_brace_recall_freeze` package (261 MB, external); the machine-readable point table is tracked at `docs/development_records/assets/2026-08-15/brace_frozen_operating_points.json` |
| 2026-08-16 | `柱脚` delivery freeze: three inference-side gains (fusion parameters, horizontal-flip TTA, router spatial gate) and four training-side interventions that all measured harmful; the cross-validation measurement floor that made either verdict possible | `docs/handoff_records/2026-08-16/column_base_freeze_20260816.md`, `docs/development_records/2026-08-16-column-base-measurement-floor.md` | checkpoints and raw result JSON ship in the `handoff_20260816_column_base_freeze` package (256 MB, external); the machine-readable result table is tracked at `docs/development_records/assets/2026-08-16/column_base_20260816_results.json` |
| 2026-08-24 | Gemini router annotation JSON, representative failed fine-tune, calibrated shared-parameter experiment, reconstruction scripts, and `digitalappv4` building-structure constraints | `docs/handoff_records/2026-08-24/router_incremental_and_structure_constraint_handoff_20260824.md`, `docs/development_records/2026-08-24-router-sound-data-incremental-results.md` | compact external archive is `.local_artifacts/shimizu_20260824_router_incremental_compact_handoff.tar.zst`; it contains no images or materialized dataset, baseline stays deployed, and failed/experimental weights are explicitly non-production |

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
7. 2026-07-07 consolidated the next handoff into a RF-DETR main package and a
   YOLO archive-only package. Future model delivery should continue from the
   RF-DETR package.
8. 2026-08-04 traced the `ブレース` / `柱脚` recall gap to incomplete annotation
   rather than to tuning, and found the only two interventions that moved it:
   WBF inference aggregation and the BRL sparse-annotation loss.
9. 2026-08-15 froze the `ブレース` operating point behind the reported overall
   recall 0.723 and reported it per grade. B reaches only 0.636 there; all four
   figures clear 0.70 at a different threshold triple, at precision 0.359.
10. 2026-08-16 moved the `柱脚` delivery entirely from the inference side -
   precision 0.300 to 0.395 measured, false alarms 2.45 to 1.66 boxes per sound
   image, four-target feasibility 26.6% to 46.4%, with recall unchanged and no
   model retrained. The same day closed the training side: four interventions,
   all harmful under a cross-validation protocol whose detection floor
   (0.045 precision, 0.29 boxes/image) is 2-5x tighter than the previous one.
   Five improvements claimed earlier in the week fell below the old floor and
   were retracted as unmeasurable rather than disproved.

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
