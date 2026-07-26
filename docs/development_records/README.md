# Development Records

Engineering notes are grouped by project phase. Use these records to understand
why the current RF-DETR direction replaced the earlier YOLO router work.

## Timeline

| folder | focus |
|---|---|
| `2026-05-19-yolo-router/` | YOLO9 router data cleaning, training, and first end-to-end integration |
| `2026-05-26-yolo-fallback/` | fallback/display logic, hard cases, and synthetic router data planning |
| `2026-06-02-rfdetr-migration/` | RF-DETR router migration and first RF-DETR downstream validation |
| `2026-06-08-09-rfdetr-downstream/` | downstream model expansion, threshold tuning, and RF-DETR production pipeline notes |
| `2026-06-15-16-final-release/` | final release evaluation, visual error analysis, and RC wall optimization |
| `2026-07-25` / `2026-07-26` files | ブレース / 柱脚 downstream models: data freeze, baseline, the element-vs-damage shortcut, and background negatives |

## ブレース / 柱脚 (2026-07-25 to 07-26)

Start with **`2026-07-26-new-classes-final-state.md`** - consolidated result,
tradeoff curves, reproduction steps, and the full table of what was tried. The
others hold the detail:

| file | focus |
|---|---|
| `2026-07-25-new-classes-annotation-match.md` | delivery pairing, dedup, grade contradictions |
| `2026-07-25-new-classes-training-plan.md` | initial plan and audit of the four delivered categories |
| `2026-07-25-new-classes-divergence-audit.md` | recipe parity check, full-image inference confirmation |
| `2026-07-25-new-classes-baseline-results.md` | first-round experiments and cross-validation |
| `2026-07-25-new-classes-frozen-data-and-next-plan.md` | dataset freeze rationale and lockfile |
| `2026-07-25-new-classes-baseline-v1-results.md` | plain baseline on the frozen split |
| `2026-07-26-new-classes-shortcut-learning-finding.md` | the models detected the element, not the damage |
| `2026-07-26-new-classes-negatives-results.md` | background negatives, the one intervention that worked |

## Supporting Folders

| folder | contents |
|---|---|
| `assets/` | report images, contact sheets, comparison CSVs, and diagrams |
| `legacy_classification/` | original four-class image-classification dataset/result notes |
| `misc/` | small one-off comparison reports and exported CSVs |

Repository layout notes are kept in:

```text
repo_organization_20260623.md
```
