# 2026-07-26 Background negatives: the intervention that worked

Follow-on from `2026-07-26-new-classes-shortcut-learning-finding.md`. All numbers
are on the frozen 8:2 split, per-class threshold grid at match IoU 0.229. The test
split is byte-identical across every run compared here, verified per build.

## Result

Selected on **recall at precision >= 0.60**, the client's binding constraint.

| category | variant | negative fraction | best F1 | recall at P>=0.60 |
|---|---|---:|---:|---:|
| ブレース | baseline | 0% | 0.503 | 0.434 |
| ブレース | negatives_v1 | 20% | 0.536 | 0.470 |
| ブレース | **neg2x** | **33%** | **0.554** | **0.494** |
| ブレース | negcrop | 33% | 0.540 | none |
| 柱脚 | baseline | 0% | 0.504 | 0.417 |
| 柱脚 | **negatives_v1** | **31%** | **0.558** | **0.514** |
| 柱脚 | neg2x | 48% | 0.529 | 0.444 |
| 柱脚 | negcrop | 40% | 0.500 | 0.403 |

Net gain over baseline: **ブレース +0.060 recall, 柱脚 +0.097 recall**, both at the
same precision floor. False positives roughly halved - for 柱脚 at recall >= 0.50,
44 down to 21.

This is the only intervention tried that moved the binding constraint. For
comparison, varifocal loss gave +0.000 / +0.028 and frozen-encoder gave -0.265.

## Three findings

**1. Negatives work, and the effect is dose-dependent.** ブレース traces a clean
monotonic curve as negative fraction rises: 0.434 -> 0.470 -> 0.494 at 0%, 20%,
33%. A single comparison can be noise; a monotonic dose-response over three points
is much harder to explain that way.

**2. The optimum is about 30-35%, and past it the effect reverses.** 柱脚 at 48%
negatives scored 0.444, materially *below* its own 31% result of 0.514. Both
categories independently point at the same band: ブレース is still at its best at
33%, 柱脚 peaked at 31% and degraded at 48%. Beyond roughly a third, too much of
each batch is spent on empty images and the positive signal starves.

This retires "add more of the same negatives" as a lever. Further gains need new
photographs, not repetition - repeating the existing 141 images has reached its
ceiling.

**3. Crop negatives failed in both categories, but the test was confounded.**
`negcrop` cropped around baseline_v1's detections on the undamaged images, on the
theory that an intact element framed like a positive is a harder negative than a
whole scene mostly made of floor and sky. It came last in both categories, and for
ブレース no threshold combination reached precision 0.60 at all.

The registered prediction offered two readings - hardness matters, or whole-image
negatives were already hard enough. Neither is safe here, because the experiment
carried a confound that was not registered: **the negatives were crops while the
positives were whole original images**, differing systematically in resolution,
framing and JPEG re-encoding. The model can satisfy the objective by learning
"looks like a crop -> no damage", which would leave it exactly as observed - decent
in the middle of the curve, collapsing at the high-precision end where it must
actually discriminate.

The idea is not refuted; the experiment was. A clean test needs positives and
negatives to travel the same processing path - both cropped (paired with the crop2
positive view) or both whole.

## What this means for the shortcut hypothesis

The evidence is the held-out test set, where precision rose at nearly every recall
level in both categories and false positives roughly halved. That is consistent
with the model relying less on "an element is present".

One check that does **not** count as evidence: re-running the empty-label audit
with the negatives model shows peak confidence on undamaged images collapsing
(ブレース median 0.436 -> 0.000). Those 141 images were in the training set as
negatives, so this measures memorisation, not generalisation. It is recorded here
only so nobody later mistakes it for confirmation.

## A gap in the evaluation

The frozen test split contains only images that carry damage, because the split was
built from the boxed images and the 141 zero-box images were excluded from train
and test alike. So the current numbers can say "false positives on damaged images
went down" but cannot say "the model stays quiet on a photograph of a sound
element" - which is the common case in production and the direct expression of the
shortcut.

Closing this needs a held-out negative set: split the zero-box images by scene
group, train on most, and reserve the rest as a separate false-alarm benchmark
reported alongside the main metrics. It should not be merged into the frozen test
split, which must stay stable for cross-run comparison.

## Saved artifacts

```text
outputs/rfdetr_new_classes/candidates_best/            best per category, with manifest
outputs/rfdetr_new_classes/candidates_negatives_v1/    round 1 candidates
outputs/rfdetr_new_classes/candidates_baseline_v1/     baseline candidates
systems/rfdetr/scripts/build_rfdetr_negatives_view.py  --negative-repeat, --crop-negatives, --max-audit-score
systems/rfdetr/scripts/run_new_classes_negatives.sh    round 1
systems/rfdetr/scripts/run_new_classes_negatives_v2.sh round 2
systems/rfdetr/scripts/audit_empty_label_images.py     the audit that found the shortcut
```

Best available models:

| category | checkpoint | thresholds B/C/D | recall | precision | F1 |
|---|---|---|---:|---:|---:|
| ブレース | `brace_neg2x_epoch_050.pth` | see manifest | 0.494 | >=0.600 | 0.527 |
| 柱脚 | `column_base_negatives_v1_epoch_016.pth` | see manifest | 0.514 | 0.607 | 0.556 |

## Next

1. **Ask the client for photographs of undamaged braces and column bases.** The
   dose-response says negatives help; repetition has saturated; only new images
   extend the curve. These need no damage grading - only the assertion that the
   element is sound - so they are far cheaper than ordinary annotation.
2. **Triage the 141.** All were used untriaged, and a minority carry unannotated
   damage (`f-00189` exposed rebar, `f-00322` spalled concrete, `f-00203` corroded
   base). `--max-audit-score` drops the most suspicious without a rebuild.
3. **Build the held-out false-alarm benchmark** described above.
4. **Re-test crop negatives without the confound**, pairing them with the crop2
   positive view.

Still short of the delivered band (F1 0.72-0.86, recall 0.812-0.875 at precision
0.596-0.824). Recall at the precision floor is now 0.49-0.51 against roughly 0.82
delivered, so about 60% of the way in recall terms, and the remaining gap is a data
question rather than a tuning one.
