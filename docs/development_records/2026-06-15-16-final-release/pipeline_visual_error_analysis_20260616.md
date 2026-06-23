# Pipeline Visual Error Analysis 2026-06-16

## Purpose

This note separates the pipeline result analysis into two views:

1. Pure data / metric view.
2. User-visible experience view.

The user-visible view is more important for the next decision, because the current labels and IoU matching can be misleading for a routed multi-model pipeline.

## Inputs

Pipeline run:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/
```

Permanent test split:

```text
data/pipeline_eval_official_plus_20260615/split.json
```

Human review package:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/
```

Main visual entry:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/large_index.html
```

Each review image has three panels:

- Left: GT labels, with missed GT marked as `FN`.
- Middle: `PC DISPLAY FINAL (DEDUPED)`, the final user-facing output after wall display rules and display merge.
- Right: `INTERNAL PRE-DISPLAY`, the internal downstream candidates before final display policy.

## Visual Review Artifacts

Priority CSV:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/large_review_cases.csv
```

Summary:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/large_review_summary.json
```

Static large per-case images:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/cases/
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/priority/
```

## 1. Pure Data View

The pipeline completed all 248 images:

| item | value |
|---|---:|
| images | 248 |
| pipeline errors | 0 |
| router status ok | 248 |
| internal detections | 699 |
| final display detections | 505 |
| display detections before merge | 664 |
| suppressed display detections | 236 |
| images with suppression | 133 |

Using strict IoU >= 0.5 and same B/C/D class matching:

| output | precision | recall | F1 |
|---|---:|---:|---:|
| internal pre-display | 0.280 | 0.740 | 0.407 |
| final display | 0.325 | 0.619 | 0.426 |

By component, final display:

| component | precision | recall | F1 |
|---|---:|---:|---:|
| tenjo | 0.333 | 0.651 | 0.441 |
| inner_wall | 0.272 | 0.493 | 0.351 |
| rc_wall | 0.398 | 0.623 | 0.486 |
| rc_column | 0.309 | 0.719 | 0.432 |

Router primary hit:

| component | primary hit rate | any-candidate hit rate |
|---|---:|---:|
| tenjo | 0.823 | 0.984 |
| inner_wall | 0.710 | 0.984 |
| rc_wall | 0.823 | 0.903 |
| rc_column | 0.661 | 0.871 |

Data-view conclusion:

- The pipeline is stable: no runtime failures.
- Recall is materially higher than precision. This is expected because the current pipeline keeps many router boxes and downstream candidates.
- Primary router class is not reliable enough for hard top-1 routing. Any-candidate hit is much higher, so multi-candidate routing is currently protecting recall.
- The final display merge reduces visible clutter materially, but still needs manual confirmation because it trades some strict recall for fewer duplicate boxes.

## Why The Data Can Be Misleading

The metric view is useful but not sufficient:

1. The dataset labels are per-component damage labels. If the pipeline finds a visually plausible extra structure or damage outside that component label, the metric counts it as FP even if a user might not consider it harmful.
2. IoU >= 0.5 is strict for large wall/ceiling regions. A visually acceptable large-region result can be counted as FN/FP if the box extent differs from the label convention.
3. The wall display rule intentionally converts `inner_wall` and `rc_wall` into one PC-facing `壁-B/C/D` result. That can reduce duplicate UI clutter but can also lower strict B/C/D matching.
4. Some labels represent damage extent, while the pipeline sometimes outputs broader affected regions. This is a UX question, not only a metric question.
5. Router boxes are structural regions, not crack boxes. Multiple router regions can create repeated downstream detections that inflate FP counts.

## 2. User Experience View

The latest generated review status counts are:

| status | cases |
|---|---:|
| GOOD | 75 |
| FN | 95 |
| DEDUPED | 46 |
| Router_primary_miss | 23 |
| FP_many | 9 |

`DEDUPED` means the display merge suppressed at least one visible duplicate in that case.

UX-view issues to confirm manually:

1. **False visible clutter**
   - This improved after display merge: `FP_many` dropped from the baseline review count of 25 to 9.
   - Remaining `FP_many` cases are still high priority because they directly affect user trust.

2. **Missed visible damage**
   - FN cases are the highest priority because a user may see obvious damage that the final PC display misses.
   - Current final display has 101 unmatched GT by strict matching.

3. **Router primary mismatch but any-candidate recovery**
   - Several images have wrong primary router class, but the expected class exists as another router candidate.
   - This suggests not to simplify to hard top-1 routing yet.
   - Separate severity analysis shows router errors are real but not the main bottleneck.

4. **Wall display rule tradeoff**
   - The wall display layer now merges `inner_wall` and `rc_wall` into a user-facing wall result.
   - Manual review should check whether the final `壁-B/C/D` region is the best visual explanation, even when internal candidates include multiple possible boxes.

5. **Slow images are often multi-region images**
   - Slow cases usually have many router boxes and downstream calls.

## Manual Review Priority

Recommended order for human confirmation:

1. Open `human_review_static_large/large_index.html`.
2. Start with `priority/`, because these are the largest static case images selected for review.
3. For any questionable case, compare the middle panel `PC DISPLAY FINAL (DEDUPED)` against the right panel `INTERNAL PRE-DISPLAY`.
4. Use `large_review_cases.csv` to mark acceptable / unacceptable cases.

Suggested judgement labels:

```text
acceptable
minor_clutter
major_clutter
missed_visible_damage
wrong_structure_type
wrong_grade
label_disagreement
needs_business_rule
```

## Initial Interpretation

The main issue is not pipeline stability. It is output policy.

From the metric side, display merge improved precision and reduced visible duplicates, but recall remains below the internal candidate layer. From the user side, the key question is whether the final merged box is the right explanation for the customer, not only whether it maximizes IoU.

Current router analysis says routing mistakes are not the dominant source of misses: among 101 final-display unmatched GT, 83 already have correct-class router coverage. The next optimization should therefore focus on downstream thresholds, final display rules, and review-driven distinction between true visual mistakes and label-convention mismatches.

The visual review package is now the source of truth for deciding which of these should be optimized first.

## Router Follow-up

Router analysis outputs:

```text
outputs/rfdetr_prod_pipeline/router_error_severity_20260616/summary.json
outputs/rfdetr_prod_pipeline/router_wall_rc_confusion_features_20260616/summary.json
outputs/rfdetr_prod_pipeline/router_per_query_ambiguity_20260616/
```

Summary:

- 238 / 265 GT objects have expected-class router coverage.
- 23 / 265 GT objects are covered only by a wrong router class.
- 14 / 101 final-display unmatched GT are wrong-router-only cases.
- Wall/RC confusion is usually high-confidence wrong, not top1/top2 uncertainty.
- Broad wall/RC fallback was tested but rejected because FP and latency increased too much.
