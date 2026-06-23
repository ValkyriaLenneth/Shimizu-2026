# RF-DETR work summary 2026-06-09

This note records the code, training, threshold tuning, packaging, and pipeline work completed on 2026-06-09.

## 1. Training-script changes

`scripts/train_rfdetr_router.py` and `scripts/rfdetr_router_callbacks.py` were extended so RF-DETR training can run with two evaluation profiles per epoch:

- `operating`: practical-use profile.
- `report`: report-oriented recall profile.

The callback now saves epoch `.pth` checkpoints and can run external threshold-sweep evaluation after each epoch. This produced per-run files such as:

- `test_results_operating.csv`
- `test_results_report.csv`
- `test_results_profiles_summary.csv`

Additional trainer options were added for resumed/tuned experiments:

- checkpoint initialization
- checkpoint interval override
- resolution override
- `num_queries`
- matcher cost overrides
- focal alpha / varifocal loss flag
- augmentation config loading

## 2. Tenjo training runs

The main 2026-06-09 tenjo runs were:

| run | output dir | key result under operating profile |
|---|---|---|
| standard dual CPU eval v2 | `outputs/rfdetr_single_crack/tenjo_medium_standard_dual_cpu_eval_v2_20260609` | best F1 at epoch 009: P 0.403, R 0.844, F1 0.545 |
| improved dual CPU eval v2 | `outputs/rfdetr_single_crack/tenjo_medium_improved_bos2_crop2_res896_lr5e6_dual_cpu_eval_v2_20260609` | best F1 at epoch 007: P 0.510, R 0.781, F1 0.617 |
| standard resume e012 | `outputs/rfdetr_single_crack/tenjo_medium_standard_dual_cpu_eval_v2_resume_e012_20260609` | best F1 at epoch 029: P 0.561, R 0.719, F1 0.630 |
| improved resume e007 | `outputs/rfdetr_single_crack/tenjo_medium_improved_bos2_crop2_res896_lr5e6_dual_cpu_eval_v2_resume_e007_20260609` | best F1 at epoch 044: P 0.610, R 0.781, F1 0.685 |

For practical use, the best balanced tenjo candidate remained the standard original epoch 009 after class-threshold tuning:

| component | checkpoint | P | R | F1 | B R | C R | D R |
|---|---|---:|---:|---:|---:|---:|---:|
| tenjo | `tenjo_standard_orig_checkpoint_epoch_009.pth` | 0.650 | 0.812 | 0.722 | 0.727 | 0.917 | 0.778 |

The recall-priority tenjo alternative was also recorded:

| component | checkpoint | P | R | F1 | B R | C R | D R |
|---|---|---:|---:|---:|---:|---:|---:|
| tenjo recall priority | `tenjo_standard_orig_checkpoint_epoch_009.pth` | 0.614 | 0.844 | 0.711 | 0.818 | 0.917 | 0.778 |

## 3. Final threshold-tuned downstream results

The selected threshold-tuned model package is:

`rfdetr_threshold_tuned_models_20260609/`

The regenerated package now keeps both the recommended deployment checkpoints and additional reference checkpoints:

- recommended: tenjo e009, RC壁 e009, 内壁 e026.
- alternatives/reference: tenjo baseline e006, tenjo standard resume e029, tenjo improved initial e007, tenjo improved resume e044, RC壁 e063 original fine-tune e003, RC壁 e063 D-oriented e002.

Final selected rows:

| component | P | R | F1 | B R | C R | D R | note |
|---|---:|---:|---:|---:|---:|---:|---|
| tenjo | 0.650 | 0.812 | 0.722 | 0.727 | 0.917 | 0.778 | balanced practical point |
| rc_wall | 0.632 | 0.750 | 0.686 | 0.857 | 0.500 | 0.875 | best practical single-checkpoint point |
| inner_wall | 0.811 | 0.909 | 0.857 | 0.875 | 1.000 | 0.889 | best F1 practical point |
| inner_wall precision priority | 0.824 | 0.848 | 0.836 | 0.750 | 1.000 | 0.889 | precision-priority report point |

The meeting report uses the precision-priority inner-wall row and the balanced tenjo / rc-wall rows.

## 4. Meeting report

Main report:

`docs/meeting_notes/2026-06-09/meeting_rfdetr_downstream_expansion_20260609.md`

Assets:

`docs/development_records/assets/2026-06-09-rfdetr-downstream/`

The report compares RF-DETR against YOLO9 using this convention:

- YOLO9 Precision: raw official split evaluation from this workspace.
- YOLO9 Recall: previous customer report adjusted/new Recall.

The report package was regenerated as:

`rfdetr_downstream_expansion_report_20260609.tar.zst`

## 5. RF-DETR production pipeline

An independent RF-DETR-oriented pipeline was created:

`rfdetr_prod_pipeline/`

It is based on the local `router_crack_pipeline/` implementation and replaces model execution with:

- RF-DETR router
- RF-DETR tenjo downstream model
- RF-DETR inner-wall downstream model
- RF-DETR RC-wall downstream model
- RF-DETR RC-column downstream model

Default config:

`rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml`

Validation completed:

| run | images | router status | warnings | errors |
|---|---:|---|---|---:|
| smoke mock | 1 | ok | none | 0 |
| smoke real | 1 | ok | none | 0 |
| wall-rule batch | 12 | ok for 12 | none | 0 |

## 6. Wall display rule

For router class `壁类`, the pipeline now calls both `inner_wall` and `rc_wall`, but PC display emits one wall result only:

| inner wall | RC wall | display |
|---|---|---|
| B | B | 壁-B |
| B | C | 壁-C |
| B | D | 壁-D |
| C | B | 壁-B |
| C | C | 壁-C |
| C | D | 壁-D |
| D | B | 壁-D |
| D | C | 壁-D |
| D | D | 壁-D |

The raw candidate details remain in JSON for audit. The report includes focused visualizations for:

- `inner_wall=C`, `rc_wall=B` -> `壁-B`
- `inner_wall=D`, `rc_wall=C` -> `壁-D`

## 7. Verification commands

Code checks:

```bash
python -m py_compile rfdetr_prod_pipeline/pipeline/*.py rfdetr_prod_pipeline/scripts/*.py rfdetr_prod_pipeline/tests/*.py
python - <<'PY'
from rfdetr_prod_pipeline.pipeline.wall_candidate_display import wall_display_grade
expected = {('B','B'):'B',('B','C'):'C',('B','D'):'D',('C','B'):'B',('C','C'):'C',('C','D'):'D',('D','B'):'D',('D','C'):'D',('D','D'):'D'}
for pair, grade in expected.items():
    assert wall_display_grade(*pair) == grade
print('wall display rule OK')
PY
```

Report asset checks:

```bash
python scripts/generate_wall_rule_report_visuals.py
python scripts/generate_downstream_expansion_report_assets.py  # only when regenerating case assets
```

## 8. Notes

- Large generated datasets, model checkpoints, downloads, logs, and compressed model packages are intentionally excluded from Git.
- The GitHub token shared in chat should be revoked and regenerated. It was not written into repository files.
