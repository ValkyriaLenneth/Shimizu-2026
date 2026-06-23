# Pipeline Evaluation 2026-06-15

## Scope

This evaluation tests the RF-DETR production pipeline using the final release model layout:

```text
final_release_20260615/models/rfdetr/config/pipeline.rfdetr_prod.final_release.yaml
```

The run uses CUDA and skips visualization to measure inference throughput without image rendering overhead.

## Test Split

Permanent split:

```text
data/pipeline_eval_official_plus_20260615/split.json
```

Policy:

- Include every `data_split.json` test stem.
- Add the same number of non-official samples per component from the remaining full dataset.
- Components: `tenjo`, `inner_wall`, `rc_wall`, `rc_column`.
- Seed: `20260615`.

Resulting sample:

| component | official test | additional | total |
|---|---:|---:|---:|
| tenjo | 31 | 31 | 62 |
| inner_wall | 31 | 31 | 62 |
| rc_wall | 31 | 31 | 62 |
| rc_column | 31 | 31 | 62 |
| total | 124 | 124 | 248 |

Files:

```text
data/pipeline_eval_official_plus_20260615/images/
data/pipeline_eval_official_plus_20260615/labels/
data/pipeline_eval_official_plus_20260615/manifest.csv
data/pipeline_eval_official_plus_20260615/split_summary.json
```

## Command

```bash
python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline \
  --config final_release_20260615/models/rfdetr/config/pipeline.rfdetr_prod.final_release.yaml \
  --source data/pipeline_eval_official_plus_20260615/images \
  --output-dir outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline \
  --device cuda:0 \
  --skip-visualization
```

## Outputs

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/results.jsonl
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/summary.json
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/analysis_summary.json
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/per_image_analysis.csv
```

## Pipeline Health

| metric | value |
|---|---:|
| images | 248 |
| errors | 0 |
| router ok | 248 |
| warnings | 1 `ambiguous_class_candidates:1` |
| internal crack detections | 699 |
| display detections | 664 |

Router primary-class hit rate:

| component | hit / total | rate |
|---|---:|---:|
| tenjo | 51 / 62 | 0.823 |
| inner_wall | 44 / 62 | 0.710 |
| rc_wall | 51 / 62 | 0.823 |
| rc_column | 41 / 62 | 0.661 |

Router any-candidate hit rate:

| component | hit / total | rate |
|---|---:|---:|
| tenjo | 61 / 62 | 0.984 |
| inner_wall | 61 / 62 | 0.984 |
| rc_wall | 56 / 62 | 0.903 |
| rc_column | 54 / 62 | 0.871 |

Interpretation: primary routing is not yet stable enough for hard top-1 routing, but keeping all router boxes recovers many cases. This supports continuing with multi-region / fallback routing rather than forcing single-class router decisions.

## Speed

Measured from per-image `elapsed_ms` in `results.jsonl`.

| metric | ms |
|---|---:|
| mean | 121.4 |
| mean excluding first 5 | 116.3 |
| p50 | 65.3 |
| p90 | 325.8 |
| p95 | 369.2 |
| p99 | 700.0 |
| min | 16.2 |
| max | 1554.8 |

Throughput:

| metric | images/sec |
|---|---:|
| all images | 8.23 |
| excluding first 5 | 8.60 |

RF-DETR emitted runtime warnings that models are not optimized for inference. A later optimization pass should test `model.optimize_for_inference()` or a warmed persistent service process. The first image is also a cold-start outlier.

## Detection Metrics

Evaluation uses YOLO labels from the permanent split, same B/C/D grade matching, and IoU >= 0.5.

### Internal `crack_detections`

| scope | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.268 | 0.706 | 0.388 | 187 | 512 | 78 |
| B | 0.208 | 0.718 | 0.322 | 94 | 358 | 37 |
| C | 0.352 | 0.667 | 0.461 | 50 | 92 | 25 |
| D | 0.410 | 0.729 | 0.524 | 43 | 62 | 16 |

By component, overall recall:

| component | precision | recall | F1 |
|---|---:|---:|---:|
| tenjo | 0.266 | 0.667 | 0.380 |
| inner_wall | 0.265 | 0.725 | 0.388 |
| rc_wall | 0.322 | 0.681 | 0.437 |
| rc_column | 0.233 | 0.750 | 0.356 |

### User-facing `display_crack_detections`

| scope | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| overall | 0.259 | 0.649 | 0.370 | 172 | 492 | 93 |
| B | 0.194 | 0.656 | 0.300 | 86 | 357 | 45 |
| C | 0.381 | 0.600 | 0.466 | 45 | 73 | 30 |
| D | 0.398 | 0.695 | 0.506 | 41 | 62 | 18 |

By component, user-facing overall recall:

| component | precision | recall | F1 |
|---|---:|---:|---:|
| tenjo | 0.278 | 0.667 | 0.393 |
| inner_wall | 0.204 | 0.536 | 0.296 |
| rc_wall | 0.328 | 0.652 | 0.437 |
| rc_column | 0.246 | 0.750 | 0.371 |

### Official vs Additional

Internal `crack_detections`:

| subset | precision | recall | F1 |
|---|---:|---:|---:|
| official_test | 0.260 | 0.672 | 0.375 |
| additional | 0.274 | 0.737 | 0.400 |

User-facing `display_crack_detections`:

| subset | precision | recall | F1 |
|---|---:|---:|---:|
| official_test | 0.257 | 0.641 | 0.367 |
| additional | 0.261 | 0.657 | 0.373 |

## Findings

1. Pipeline stability is good for this batch: all 248 images completed, no exceptions, router status `ok` for every image.
2. Recall is usable for a first full pipeline pass, but precision is low because the current pipeline keeps many router regions and downstream candidates.
3. Display rules reduce output count from 699 internal detections to 664 display detections, but they also reduce recall from 0.706 to 0.649. The wall display layer needs refinement before it is treated as final business output.
4. Router primary-class accuracy is the main bottleneck for hard routing, especially RC柱 and inner_wall. Any-candidate hit is much higher, so fallback/multi-region routing is the safer direction.
5. Speed is acceptable for batch testing, but p95 and p99 are high when many router regions trigger multiple downstream models. Reducing duplicate router regions and optimizing RF-DETR inference should be prioritized.

## Next Actions

1. Add configurable router-region pruning before downstream inference, while preserving any-candidate recall.
2. Add a speed profile split by number of router regions and downstream model calls.
3. Tune wall display grouping to reduce false positives without losing matched detections.
4. Run a threshold grid on this fixed pipeline split, not only on single-model official test views.
5. For production UX, evaluate a low-confidence/no-router fallback separately from the default path.

## 2026-06-16 Update: Display Merge Result

After manual review of the first visualization pass, the accepted latest pipeline result is:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/
```

The final release config now keeps the display-layer overlap suppression and the wall display candidate fix:

```text
final_release_20260615/models/rfdetr/config/pipeline.rfdetr_prod.final_release.yaml
```

Important terminology for this run:

- `final display`: final user-facing output after wall display rules and display merge.
- `internal pre-display`: internal `crack_detections` before the final display policy.

### Display Merge Summary

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/display_merge_summary.json
```

| item | value |
|---|---:|
| images | 248 |
| display detections before merge | 664 |
| display detections after merge | 505 |
| suppressed display detections | 236 |
| images with suppression | 133 |

### Latest Metrics

| output | precision | recall | F1 | TP | FP | FN | pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| final display | 0.325 | 0.619 | 0.426 | 164 | 341 | 101 | 505 |
| internal pre-display | 0.280 | 0.740 | 0.407 | 196 | 503 | 69 | 699 |

By component, final display:

| component | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| tenjo | 0.333 | 0.651 | 0.441 | 41 | 82 | 22 |
| inner_wall | 0.272 | 0.493 | 0.351 | 34 | 91 | 35 |
| rc_wall | 0.398 | 0.623 | 0.486 | 43 | 65 | 26 |
| rc_column | 0.309 | 0.719 | 0.432 | 46 | 103 | 18 |

Compared with the original baseline display output, the display count dropped from 664 to 505 and precision improved from 0.259 to 0.325. Recall dropped from 0.649 to 0.619. From the UX perspective this is acceptable as the current working version because it removes many visibly duplicated boxes; further recall recovery should be tested carefully against clutter.

### Applied Fixes

Applied and kept:

- Final display overlap suppression in `rfdetr_prod_pipeline/pipeline/display_merge.py`.
- `run_full_pipeline.py` applies display merge after composing the user-facing display records.
- Wall display geometry uses the paired/union region when available, so a smaller high-grade candidate does not shrink a better matched wall output.
- `wall_display.max_single_groups_per_model` is set to `4`, preventing useful wall candidates from being dropped too early.

### Rejected Fallback Experiment

The broad wall/RC sister fallback was tested here:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_wall_rc_sister_fallback_v1/
```

Result:

| run | precision | recall | F1 | TP | FP | FN | mean ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| displaymerge_v1 | 0.325 | 0.619 | 0.426 | 164 | 341 | 101 | 121 | 365 |
| wall_rc_sister_fallback_v1 | 0.242 | 0.645 | 0.351 | 171 | 537 | 94 | 155 | 458 |

The fallback found 7 more TP and reduced FN by 7, but added 196 FP and slowed inference. It is not part of the current accepted pipeline.

Strict rescue simulation was also negative:

```text
outputs/rfdetr_prod_pipeline/wall_rc_strict_rescue_sim_20260616/summary.json
```

The strictest useful-looking variant (`conf>=0.55 + shape`) kept the same recall as baseline displaymerge_v1 and still increased FP slightly. Looser variants produced more clutter. Current conclusion: router fallback is not a good next lever unless backed by better trigger logic or retraining data.

## 2026-06-16 Update: Router Severity

Router analysis artifacts:

```text
outputs/rfdetr_prod_pipeline/router_error_severity_20260616/summary.json
outputs/rfdetr_prod_pipeline/router_wall_rc_confusion_features_20260616/summary.json
outputs/rfdetr_prod_pipeline/router_per_query_ambiguity_20260616/
```

Main counts:

| item | count |
|---|---:|
| GT objects | 265 |
| expected router covers GT | 238 |
| wrong router only covers GT | 23 |
| no router covers GT | 4 |
| display-unmatched GT | 101 |
| display-unmatched GT with expected router coverage | 83 |
| display-unmatched GT with wrong-router-only coverage | 14 |
| display-unmatched GT with no router coverage | 4 |

Interpretation:

- Router errors affect some cases, but they are not the dominant cause of final user-facing misses.
- Most display FNs still have a correct-class router region covering the GT, so the next improvements should focus on downstream thresholds, display rules, and duplicate/overlap policy.
- For wall/RC confusion, the wrong router output is usually high-confidence wrong, not an uncertain top1/top2-close case. The per-query ambiguity strategy is therefore not useful on this test set.
