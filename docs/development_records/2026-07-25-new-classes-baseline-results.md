# 2026-07-25 ブレース / 柱脚 baseline results

## Verdict

Current state after three experiments, judged on overall performance:

| category | best F1 | mAP50 | delivered band |
|---|---:|---:|---|
| ブレース | 0.613 | 0.366 | F1 0.72-0.86, mAP50 0.73-0.78 |
| 柱脚 | 0.548 | 0.342 | same |

Crop augmentation closed a meaningful part of the gap - ブレース F1 0.565 -> 0.613
and 柱脚 0.411 -> 0.548 - and ブレース now sits 0.11 below 天井's shipped 0.722.
Joint pretraining across the two categories was tried and made things worse.

The F1 gap is now within reach of tuning. The mAP50 gap is still about a factor of
two, which is a statement about the detector's ranking quality rather than about
threshold placement, and no amount of threshold work will close it.

Training data volume remains the binding constraint: 264 and 203 distinct training
images against 605 for RC柱 and 966 for RC壁. Crop augmentation multiplies samples
but not information - see the peak-step analysis in the experiment log.

The original verdict recorded here, before crop augmentation, was that recall 0.80
was only reachable at precision 0.15 and 0.09. That still holds for the base
datasets and is kept below for the record.

## What Was Run

Recipe identical to the audited 天井 / 内壁 / RC壁 / RC柱 baseline: RFDETRMedium,
80 epochs, batch 28, grad accum 1, lr 1e-4, 16-mixed, default resolution, one
category per GPU.

```bash
systems/rfdetr/scripts/run_new_classes_baseline_comparison.sh   # train + sweep
systems/rfdetr/scripts/run_new_classes_threshold_grid.sh        # per-class grid
python systems/rfdetr/scripts/summarize_new_class_results.py    # report
```

Wall clock: ブレース 27 min, 柱脚 42 min (it shared GPU time with the ブレース
sweep), 80-checkpoint sweep ~15-20 min each, threshold grid ~35 s per checkpoint.

## Epoch Sweep (single default threshold, official test)

All 80 epoch checkpoints reloaded and force-evaluated, because
`checkpoint_best_total.pth` is selected by mAP rather than recall.

ブレース, recall-first top 5:

| epoch | recall | precision | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 33 | 0.505 | 0.327 | 0.343 | 0.156 |
| 23 | 0.491 | 0.295 | 0.306 | 0.135 |
| 31 | 0.478 | 0.372 | 0.377 | 0.172 |
| 19 | 0.462 | 0.307 | 0.277 | 0.133 |
| 32 | 0.449 | 0.385 | 0.376 | 0.168 |

柱脚, recall-first top 5:

| epoch | recall | precision | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 14 | 0.391 | 0.252 | 0.208 | 0.130 |
| 21 | 0.369 | 0.279 | 0.209 | 0.121 |
| 34 | 0.362 | 0.407 | 0.279 | 0.160 |
| 37 | 0.355 | 0.439 | 0.271 | 0.145 |
| 11 | 0.355 | 0.267 | 0.185 | 0.108 |

Peak epochs are 19-34 for both. The remaining 45-60 epochs contribute nothing,
which matches the delivered categories being selected at epochs 9, 26, 9 and 47.

## Per-Class Threshold Grid (match IoU 0.229, 3375 points per checkpoint)

Three operating points per checkpoint, to show the whole trade-off rather than
one cherry-picked row.

ブレース:

| checkpoint | point | B/C/D thr | recall | precision | F1 | B | C | D |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ep023 | recall-first | 0.10/0.12/0.18 | 0.807 | 0.152 | 0.256 | 0.773 | 0.844 | 0.750 |
| ep031 | best F1 | 0.35/0.28/0.40 | 0.554 | 0.523 | 0.538 | 0.55 | 0.58 | 0.50 |
| ep031 | max R at P>=0.60 | 0.50/0.45/0.45 | 0.434 | 0.600 | 0.503 | 0.41 | 0.44 | 0.44 |

柱脚:

| checkpoint | point | B/C/D thr | recall | precision | F1 | B | C | D |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ep014 | recall-first | 0.05/0.07/0.05 | 0.792 | 0.027 | 0.052 | 0.851 | 0.733 | 0.600 |
| ep034 | best F1 | 0.45/0.18/0.45 | 0.431 | 0.554 | 0.484 | 0.40 | 0.60 | 0.30 |
| ep034 | max R at P>=0.60 | 0.45/0.35/0.45 | 0.403 | 0.604 | 0.483 | 0.40 | 0.47 | 0.30 |

No 柱脚 grid point reaches both recall 0.80 and precision 0.60; the maximum
recall at or above precision 0.60 is 0.403.

Comparison with the delivered models:

| 部材 | Precision | Recall |
|---|---:|---:|
| 天井 | 0.596 | 0.875 |
| RC壁 | 0.722 | 0.812 |
| 内壁 | 0.824 | 0.848 |
| RC柱 | 0.661 | 0.826 |
| **ブレース (new)** | **0.600** | **0.434** |
| **柱脚 (new)** | **0.604** | **0.403** |

## Why

Training data is roughly a third of the closest comparable category:

| category | train images | train boxes | D boxes |
|---|---:|---:|---:|
| RC柱 (delivered) | 605 | 648 | 145 |
| RC壁 (delivered) | 966 | 1235 | 177 |
| ブレース (new) | 235 | 394 | 76 |
| 柱脚 (new) | 179 | 248 | 29 |

柱脚 is the weakest on every axis and has only 29 D boxes across 26 images, which
shows up directly: D recall is 0.30 at any usable threshold, against 1.000 for
delivered RC柱.

The threshold the model needs in order to reach recall 0.80 is also diagnostic.
The delivered models operate at B/C/D thresholds around 0.25-0.45; these two need
0.05-0.18. The model's confident predictions are too few, which is what
insufficient training data looks like rather than a tuning error.

## Bug Fixed Along The Way

`evaluate_rfdetr_class_threshold_grid.py` called `match_counts()` with three
arguments after that function gained a required `num_classes` parameter in
`evaluate_rfdetr_threshold_sweep.py`. Every per-class threshold grid raised
`TypeError` before writing a row. Fixed at the call site. This is a pre-existing
repo bug and it also affects the `evaluator: class_threshold_grid` external eval
profiles in the tenjo and rc_wall report-finetune configs.

## Experiment Log (overall-performance lens)

Goal changed on 2026-07-25 to overall performance first, so the table below leads
with best F1 and mAP50 rather than a recall-first operating point. All rows within
a category share the same 9:1 test split and are directly comparable; crop
augmentation only alters train.

| variant | ブレース F1 | ブレース mAP50 | 柱脚 F1 | 柱脚 mAP50 |
|---|---:|---:|---:|---:|
| 9:1 base | 0.565 | 0.307 | 0.411 | 0.266 |
| 9:1 crop2 | **0.613** | **0.366** | **0.548** | **0.342** |
| 9:1 crop2 + joint pretrain/ft | 0.571 | 0.332 | 0.431 | 0.269 |
| delivered band | 0.72 - 0.86 | 0.73 - 0.78 | 0.72 - 0.86 | 0.73 - 0.78 |

Current best candidates:

| category | checkpoint | B/C/D thresholds | F1 | recall | precision |
|---|---|---|---:|---:|---:|
| ブレース | crop2 ep24 | 0.30/0.45/0.30 | 0.613 | 0.590 | 0.639 |
| ブレース, usable recall point | crop2 ep24 | 0.30/0.35/0.30 | 0.608 | 0.615 | 0.600 |
| 柱脚 | crop2 ep32 | 0.20/0.20/0.45 | 0.548 | 0.526 | 0.571 |

### What worked: crop augmentation

`build_rfdetr_crop_aug_view.py`, crops 3x context around every box with jittered
variants, train-only. Gains on the same test split:

| category | F1 | mAP50 | max recall at precision 0.60 |
|---|---|---|---|
| ブレース | 0.565 -> 0.613 | 0.307 -> 0.366 | 0.487 -> 0.615 (+26%) |
| 柱脚 | 0.411 -> 0.548 | 0.266 -> 0.342 | 0.237 -> 0.447 (+89%) |

柱脚 gains far more, matching the diagnosis that it is volume-limited while
ブレース is scale-limited.

One correction to the original reasoning for this lever: the benefit is **scale
normalisation, not data volume**. The 1140 crop images come from only 264 distinct
source images, so the samples are strongly correlated and add no new information.
The evidence is the peak position - measured in optimizer steps, every run peaks
at roughly the same place:

| run | steps/epoch | peak epoch | peak step |
|---|---:|---:|---:|
| ブレース base | 9 | 22-29 | ~200-260 |
| ブレース crop2 | 41 | 3-4 | ~165 |

Crop augmentation reaches the peak sooner; it does not move it. That is why 45
epochs is wasteful for a crop view, and it is the reason the follow-up experiment
lowers the learning rate instead of enlarging the corpus further.

### What did not work: joint pretrain, per-category fine-tune

Hypothesis: both categories share B/C/D grading semantics, so a model pretrained on
their union (1907 crop images, 3564 boxes, D 463) should be a better
initialization than COCO for each per-category fine-tune, without changing the
one-model-per-category deployment shape.

Result: worse than crop2 alone on both categories.

| | crop2 | crop2 + joint-ft | delta |
|---|---:|---:|---:|
| ブレース F1 | 0.613 | 0.571 | -0.042 |
| ブレース mAP50 | 0.366 | 0.332 | -9% |
| 柱脚 F1 | 0.548 | 0.431 | -0.117 |
| 柱脚 mAP50 | 0.342 | 0.269 | -21% |

Likely mechanism: sharing the B/C/D label space across element types asks the model
to judge damage grade without knowing which element it is looking at, but grade
appearance is element-specific - a D brace is a buckled steel member, a D column
base is spalled concrete with exposed rebar. The shared label space collapses two
visually different tasks into one and introduces label ambiguity. 柱脚 degrades
more, consistent with it being the smaller dataset and therefore more dependent on
initialization quality, and more easily dominated by ブレース's steel appearance.

Caveat, stated so the result is not overclaimed: the fine-tune ran 15 epochs at
lr 5e-5, while crop2 ran 45 epochs at lr 1e-4 with its best at ep24/ep32.
ブレース's joint-ft best was ep11 of 15, right against the ceiling, so it may be
under-trained. There is no trend suggesting it would pass 0.613, and the label
ambiguity is a mechanism that more epochs would not fix, so this was not retried
with a larger budget.

Artifacts kept: `data/rfdetr_joint_bcd_20260725_split91_crop2_test_as_valid`,
`systems/rfdetr/recognition_models/joint_bcd/configs/rfdetr_joint_bcd_pretrain.yaml`,
`systems/rfdetr/scripts/run_new_classes_joint_pretrain_finetune.sh`. The joint
model is never deployed.

### Partly worked: lower lr on a tighter crop view (lr3e5 + crop3)

crop3 is `crops-per-box 3, context 2.0` - more variants and a tighter window than
crop2's `2 / 3.0`. Combined with lr 3e-5 over 30 epochs:

| category | metric | crop2 | lr3e5 + crop3 |
|---|---|---:|---:|
| ブレース | mAP50 | 0.366 | **0.397** |
| ブレース | mAP@.5:.95 | 0.145 | **0.213** |
| ブレース | best F1 | **0.613** | 0.595 |
| 柱脚 | mAP50 | 0.342 | **0.352** |
| 柱脚 | mAP@.5:.95 | **0.207** | 0.192 |
| 柱脚 | best F1 | **0.548** | 0.529 |

mAP improved and F1 slightly declined. The mAP@.5:.95 gain of 47% on ブレース is
large and points at better localisation, which is what a tighter crop window and a
gentler learning rate would be expected to buy.

### The F1 noise floor on these test splits

The F1 differences above are not resolvable. ブレース test holds 39 GT boxes, so a
single true positive flipping changes recall by 1/39 = 0.026, while the F1 gap
between crop2 and lr3e5+crop3 is 0.018 - less than one box. 柱脚 has 38 boxes and
the same arithmetic.

Practical consequence for the rest of this work: **F1 differences below about 0.05
on these splits are noise, and mAP is the metric to steer by.** mAP integrates over
all thresholds and over all detections rather than resting on one operating point,
so it resolves changes that F1 cannot. By that measure lr3e5+crop3 is the better
model and crop2 is tied with it on F1, not better.

This also retrospectively weakens any conclusion drawn from small F1 deltas earlier
in this log, including the joint-pretrain comparison for ブレース (0.613 vs 0.571,
about 1.6 boxes). The 柱脚 joint-pretrain gap of 0.117, roughly 4.5 boxes, is
outside the noise floor and stands.

### Methodological note on lr3e5 + crop3

Two variables were changed at once - learning rate and crop view - so the mAP gain
cannot be attributed to either. Round 2 disentangles it:

```text
A: crop2 + lr 3e-5   isolates the lr effect at a fixed crop view
B: crop3 + lr 1e-5   pushes lr further on the current best-mAP view
```

### Round 2: disentangling lr from crop view

| category | variant | mAP50 | mAP@.5:.95 | best F1 | max R at P>=0.60 |
|---|---|---:|---:|---:|---:|
| ブレース | crop2 (lr 1e-4) | 0.366 | 0.145 | 0.613 | 0.615 |
| ブレース | crop2 + lr 3e-5 | 0.363 | 0.185 | **0.635** | **0.667** |
| ブレース | crop3 + lr 3e-5 | **0.397** | **0.213** | 0.595 | 0.564 |
| ブレース | crop3 + lr 1e-5 | 0.338 | 0.190 | 0.617 | 0.615 |
| 柱脚 | crop2 (lr 1e-4) | 0.342 | 0.207 | **0.548** | 0.447 |
| 柱脚 | crop2 + lr 3e-5 | 0.325 | 0.183 | 0.507 | 0.421 |
| 柱脚 | crop3 + lr 3e-5 | 0.352 | 0.192 | 0.529 | **0.474** |
| 柱脚 | crop3 + lr 1e-5 | **0.385** | **0.223** | 0.464 | 0.342 |

Attribution, from holding one variable fixed:

* lr 1e-4 -> 3e-5 at fixed crop2 raises ブレース mAP@.5:.95 from 0.145 to 0.185,
  a 28% gain in localisation quality.
* crop2 -> crop3 at fixed lr 3e-5 raises ブレース mAP50 from 0.363 to 0.397.

The two act on different axes and are not redundant.

The two categories prefer different crop context, which is worth keeping: ブレース
does best at context 3.0 and 柱脚 at 2.0. That fits the imagery - ブレース photos are
wide truss and ceiling scenes where judging a buckled member needs surrounding
structure, while 柱脚 photos are closer framed and benefit from a tighter window.
It is also an argument for the one-model-per-category architecture, since a shared
preprocessing choice would be wrong for one of them.

Everything past those two findings is inside the noise floor. lr 3e-5 -> 1e-5 moves
the two categories in opposite directions on mAP50 (ブレース down to 0.338, 柱脚 up
to its best 0.385), and the F1 and mAP rankings contradict each other. Those are
not real differences.

## Cross-Validation: How Much To Trust Any Of This

The numbers above are the maximum over roughly 60000 (checkpoint, threshold)
configurations evaluated on 39 boxes for ブレース and 38 for 柱脚, with valid
mirroring test. The maximum of that many noisy estimates is optimistically biased
by construction. Continuing to tune against it would inflate the bias without
improving the model.

The same caution applies to the delivered reference band. Each delivered model's
test split holds about 32 boxes - recoverable from tp+fn in
`final_release_20260615/models/rfdetr/metrics/selected_thresholds.csv` - and they
were selected under the same valid==test convention. Part of the apparent gap
between our models and theirs may be a difference in how many configurations were
searched, not in model quality. Only a like-for-like unbiased measurement can
separate those.

### Protocol

```text
systems/rfdetr/scripts/build_rfdetr_new_class_cv_folds.py   5 folds, scene-group aware
systems/rfdetr/scripts/run_new_classes_cv.sh                train + fixed-threshold eval
systems/rfdetr/scripts/report_new_class_cv.py               pooled out-of-fold report
```

| element | choice | why |
|---|---|---|
| folds | 5, split on `scene_group_id`, stratified by rarest grade | same leakage guard as the main split; keeps D present in every fold |
| recipe | medium, crop2 view, lr 3e-5, 30 epochs, batch 28 | fixed, not tuned per fold |
| epoch | the final epoch | selecting the epoch on the fold's own test is the bias being avoided |
| thresholds | fixed per-category triple carried over from the 9:1 runs | not re-tuned per fold |
| scoring | tp/fp/fn summed across folds | rests on 477 / 320 boxes instead of 39 / 38 |

Pooled test is 293 images / 477 boxes for ブレース and 224 / 320 for 柱脚, twelve
times the single split, with zero scene-group leakage verified per fold.

### Two limitations, stated rather than buried

1. **Residual threshold leakage.** The fixed thresholds were originally chosen on
   the 9:1 test split, whose images also appear in the CV folds. This is one
   configuration carried over instead of 60000 selected in place, so the bias is
   drastically reduced but not zero.
2. **The final-epoch rule makes this a conservative bound.** The single-split
   number used the best checkpoint and the best threshold; CV uses the final
   checkpoint and a fixed threshold. crop2 folds give about 35 optimizer steps per
   epoch, so 30 epochs is roughly 1050 steps, well past the ~200-step peak seen in
   every run. The final checkpoint is therefore likely past its best.

   The consequence is that the pooled CV figure is a **lower bound** and the
   single-split figure is an **upper bound**; the honest value sits between them.
   The difference between the two conflates selection optimism with a
   final-versus-best-epoch penalty. If the gap turns out large, a second CV pass
   with an a-priori fixed early epoch - around epoch 6, which is where the peak has
   landed in every run so far - would separate the two effects.

### Interpretation criteria, fixed before the CV results were seen

Written in advance because two conclusions earlier in this log had to be walked
back after the fact - crop's benefit was first attributed to data volume when it is
scale normalisation, and the first CV attempt measured epoch decay rather than
selection bias. Fixing the reading rules up front is the cheapest guard against a
third.

**On the D-boost experiment** (crop2 + D-targeted crops, everything else held):

| observation | conclusion |
|---|---|
| pooled D recall rises and overall F1 does not fall | the D:B imbalance was a real cause; keep the boosted view |
| D recall rises but overall F1 falls by more than 0.05 | traded C/B away for D; revisit the boost factor rather than keep it |
| D recall moves by less than 0.05 | class balance was not the binding constraint for D; stop pursuing it |

**On the grade-confusion diagnostic:**

| observation | conclusion |
|---|---|
| most missed D boxes are `grade_confused` | a discrimination problem; more detection data will not fix it, and the label review of the 11 grade contradictions becomes the priority |
| most missed D boxes are `missed` outright | a detection problem; crop and data volume remain the right levers |
| roughly even | both, and neither lever alone will close the gap |

**On the CV numbers themselves:**

* The pooled out-of-fold figure is the number to quote internally and to the
  client. The single-split figure is not.
* The delivered reference band was produced under the same valid==test convention
  on ~32-box test splits, so it carries optimism of the same kind. The measured
  optimism on our own data is the best available estimate of its size. The gap to
  the delivered band should therefore be read as an upper bound on the true gap,
  not as the true gap.
* Any per-fold F1 spread of the same order as the differences between our
  experiment variants means those variants were never distinguishable, and the
  experiment log above should be read with that in mind.

### Diagnostic: the failure is detection, not grade discrimination

`analyze_new_class_grade_confusion.py` classifies every ground-truth box at the
operating thresholds into one of three outcomes - matched by a same-grade
prediction, overlapped only by predictions of other grades (grade confusion), or
not overlapped at all (detection miss).

ブレース, checkpoint ep017 of crop2 + lr3e-5, thresholds 0.30/0.35/0.40, IoU 0.229,
29 images and 39 boxes:

| grade | GT | matched | grade confused | missed | recall |
|---|---:|---:|---:|---:|---:|
| B | 13 | 9 | **0** | 4 | 0.692 |
| C | 18 | 10 | **0** | 8 | 0.556 |
| D | 8 | 3 | **0** | 5 | 0.375 |
| all | 39 | 22 | **0** | 17 | 0.564 |

**Not one grade confusion.** Every missed box is missed outright, with nothing
overlapping it at all. The model is not hesitating between B, C and D - it fails to
find the damage region in the first place.

This retires a concern raised in the annotation analysis. That note observed that
6 of the 11 documented grade contradictions are systematic 柱脚 B-vs-C pairs and
warned that "a B/C confusion in the results may reflect the label set rather than
the model". At the operating point that is not happening, so **label review is not
the current bottleneck and fixing the contradictions would not move these
numbers.** It remains worth doing for data hygiene, not for performance.

Consequences for the lever list:

| lever | verdict |
|---|---|
| label review of the 11 grade contradictions | not the bottleneck, deprioritise |
| D-targeted crop boost | well aimed - D recall 0.375 is the lowest of the three |
| crop augmentation, more data | still the right levers |
| classification head or grade-discrimination work | not needed |

Limitation: measured at one checkpoint, one fixed threshold triple, on 39 boxes.
At lower thresholds some misses could become confusions. Zero confusion at the
operating point is a strong signal nonetheless. The 柱脚 equivalent is queued.

### Cross-validation results - the only numbers to quote

Recipe: medium, crop2 view, lr 3e-5, 12 epochs, batch 28. Fixed epoch 6, fixed
per-category thresholds, tp/fp/fn pooled over five folds.

| category | precision | recall | **F1 (unbiased)** | biased upper bound | per-fold sd |
|---|---:|---:|---:|---:|---:|
| ブレース | 0.382 | 0.356 | **0.369** | 0.413 | 0.070 |
| 柱脚 | 0.265 | 0.388 | **0.315** | 0.382 | 0.065 |

The honest value sits between the unbiased and biased columns: ブレース 0.369-0.413,
柱脚 0.315-0.382.

Per-fold F1:

```text
ブレース  0.470  0.406  0.316  0.346  0.299
柱脚     0.344  0.244  0.239  0.368  0.365
```

Per-grade, pooled:

| category | grade | tp | fp | fn | precision | recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| ブレース | B | 58 | **159** | 88 | 0.267 | 0.397 | 0.320 |
| ブレース | C | 92 | 100 | 147 | 0.479 | 0.385 | 0.427 |
| ブレース | D | 20 | 16 | 72 | 0.556 | 0.217 | 0.312 |
| 柱脚 | B | 95 | **321** | 104 | 0.228 | 0.477 | 0.309 |
| 柱脚 | C | 21 | 17 | 61 | 0.553 | 0.256 | 0.350 |
| 柱脚 | D | 8 | 6 | 31 | 0.571 | 0.205 | 0.302 |

The two categories fail the same way, which makes it a property of the task rather
than of one dataset: **B floods the image with false positives** while **C and D
are found less than a quarter to a third of the time**.

### How wrong the single-split numbers were

| category | single split | pooled CV | optimism |
|---|---:|---:|---:|
| ブレース | 0.635 | 0.369 | **+0.266** |
| 柱脚 | 0.507 | 0.315 | +0.192 |

The ブレース single-split figure overstated performance by 72%. Both categories are
inflated by roughly the same amount, which is what a shared cause - selecting the
maximum over ~60000 configurations on 39 boxes - predicts.

### Which of this log's conclusions survive

Per-fold spread is 0.070 and 0.065. Every hyperparameter difference reported
earlier is smaller than that:

| comparison | difference | vs fold spread |
|---|---:|---|
| crop2 -> crop2 + lr 3e-5 | 0.022 | 0.3 sd |
| crop2 -> crop3 + lr 3e-5 | 0.018 | 0.3 sd |
| base -> crop2 | 0.048 | 0.7 sd |

**None of the hyperparameter comparisons in this log are statistically supported.**
The claims that crop2+lr3e-5 is best for ブレース, that crop3 suits 柱脚, and that the
two categories prefer different crop context were all read off differences smaller
than the noise. They are withdrawn.

The base-versus-crop comparison at 0.7 sd is the largest and the only one worth
re-testing properly, which is why a plain-baseline CV under the identical protocol
is the next run. Until it finishes, **there is no unbiased evidence that crop
augmentation helps at all**; the mAP50 gain of 19-29% is a larger effect but rests
on the same single split.

### Operational note

The chain script that queued the baseline CV deadlocked. It waited on
`pgrep -f "run_new_classes[_]cv.sh"`; the bracket trick stops the pattern matching
the chain script itself, but the shell that *created* the script via heredoc still
carried the full text - including the literal invocation line - in its command
line, so the chain matched its own parent and waited forever with both GPUs idle.
When guarding on a process name, match on something that cannot appear in the
creating shell's command line, or check for the actual worker processes instead.

### Tooling added

```text
systems/rfdetr/scripts/compare_new_class_experiments.py   one table over all runs
systems/rfdetr/scripts/summarize_new_class_results.py     three operating points per checkpoint
systems/rfdetr/scripts/build_rfdetr_joint_bcd_view.py     joint corpus builder
systems/rfdetr/scripts/build_rfdetr_new_class_downstream_datasets.py
systems/rfdetr/scripts/visualize_new_class_dataset_samples.py
```

`run_new_classes_baseline_comparison.sh` is now a general experiment runner:
`EXPERIMENT`, `SUFFIX`, `EPOCHS`, `LR` and `TAG` are all overridable, and `TAG` is
appended to the run directory so variants never overwrite each other.

## Root Cause: Not Simply A Shortage Of Data

The obvious explanation - these categories have a third of the training data of the
delivered ones - turns out to be incomplete. If volume were the binding constraint,
per-class training box count should predict per-class performance. It does not.

Cross-validated per-class results against training box counts and median box area:

| category | grade | train boxes | median area | CV F1 | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| ブレース | B | 133 | 0.0076 | 0.363 | 0.296 | 0.468 |
| ブレース | C | 221 | 0.0648 | 0.441 | 0.500 | 0.395 |
| ブレース | D | 84 | 0.1636 | 0.350 | 0.609 | 0.246 |
| 柱脚 | B | **173** | 0.0235 | **0.269** | **0.203** | 0.398 |
| 柱脚 | C | 74 | 0.1000 | 0.314 | 0.611 | 0.212 |
| 柱脚 | D | 35 | 0.0676 | 0.286 | 0.556 | 0.192 |

柱脚 refutes the volume story outright: **B has the most training boxes of any
grade and the worst F1 and the worst precision.** More data for that class did not
buy anything.

Box area predicts precision far better, and it holds in both categories - sort the
grades by median area and precision rises monotonically:

```text
ブレース  B(0.008, P 0.296) -> C(0.065, P 0.500) -> D(0.164, P 0.609)
柱脚     B(0.024, P 0.203) -> D(0.068, P 0.556) -> C(0.100, P 0.611)
```

Note the 柱脚 ordering: D sits between B and C on area and also on precision, out of
order with its box count. Area tracks the metric; volume does not.

### The three separable causes

**1. Small objects are intrinsically hard here - the largest cause, and not a data
problem.** B damage occupies 0.8-2.4% of the frame. At that scale a hairline crack
or bolt corrosion is hard to separate from ordinary texture, shadow, rust and
joints, and the model floods the image with false positives: 88 FP against 37 TP
for ブレース B, 185 FP against 47 TP for 柱脚 B.

The project's own history is the strongest evidence that data will not fix this.
天井 has a full dataset and its B class still sat at recall 0.4545 through
oversampling, crop augmentation and resolution increases, recorded as "B remains
fixed". More data did not solve B for a delivered category.

**2. Scale span - specific to ブレース.** Box area covers a 332x range, D boxes
being 21x the area of B boxes, and one single-resolution detector has to span it.
This one has direct experimental support: crop augmentation, which is exactly scale
normalisation, produced the single largest gain of any intervention at +19-29%
mAP50.

**3. Data volume - real, but it only binds on D.** D has 84 and 35 boxes. Its
signature is high precision with low recall - 0.609/0.246 and 0.556/0.192 - the
model is simply unwilling to predict D. This is the one place where more examples
should help directly, and it is what the D-boost experiment tests.

### What this implies

Adding annotation would most likely improve D and leave B roughly where it is - and
B is the larger drag, because it contributes the bulk of the false positives.

The productive directions are the ones aimed at small objects: higher inference
resolution, tiled inference, or a conversation with the client about capture
distance. ブレース photographs are wide truss and ceiling scenes, and a crack at 0.8%
of the frame may simply not carry enough pixels to support a grade judgement at
all. That last point is worth settling with real images and the client's capture
guidance rather than with more modelling.

The grade-confusion diagnostic supports this reading: zero confusions means the
model is not mis-grading anything, it is failing to find the damage.

## Recommended Next Steps

In order of expected value per unit of effort:

1. **More annotated data.** This is the actual bottleneck and the only lever that
   closes a gap this size. Two concrete sources:
   - the 45 unlabelled `20260724` images, currently excluded
   - the 141 dropped empty-label images (59 ブレース, 82 柱脚). The client
     instruction was to delete them on the basis that every image contains
     damage. Worth re-confirming: if a meaningful share are simply *not yet
     annotated* rather than damage-free, annotating them would add about 48% more
     ブレース and 46% more 柱脚 training images.
2. **Initialize from the released `rc_column` checkpoint instead of COCO.** 柱脚 is
   the base of an RC column, the closest existing domain, and RC柱 trained on
   3.4x the data. Requires restoring the release archive - `.local_artifacts/` is
   empty in this checkout, so no project checkpoint is available locally.
3. **Oversample D and C** for 柱脚 (`build_rfdetr_oversampled_view.py` pattern).
   Note the delivered history is not encouraging here: repeated oversampling runs
   for tenjo B and rc_wall C did not move those classes.
4. **Low-lr fine-tune from the best epoch** (lr 2e-6, batch 16, grad accum 2,
   ~5 epochs), which is how RC壁 reached its shipped point.
5. Light augmentation presets (`aug_tenjo_light_geo` / `aug_tenjo_light_pixel`).

A resolution increase is explicitly not recommended: it was already tried for
tenjo and rc_wall and recorded as giving no meaningful gain.

## Honest Framing For The Client

Reporting these two categories next to the four delivered ones at the same
precision would show roughly half the recall. The cause is data volume, and it is
measurable rather than speculative. The useful message is that ブレース and 柱脚
need annotation on the order of the other categories - about 600 training images
each - before a comparable model is realistic.
