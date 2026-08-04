# 2026-08-03 S1 pipeline, and what calibrating the quality judge changed

Companion to `2026-08-03-new-classes-synthetic-data-plan.md`. That document set
the direction; this one records what was built, what broke, and the measurements
that settled each question.

## 1. What exists now

| script | role |
|---|---|
| `systems/gemini/scripts/synth_common.py` | shared geometry, compositing, texture matching, transport |
| `systems/gemini/scripts/extract_grade_reference_crops.py` | per-grade crop bank + contact sheets |
| `systems/gemini/scripts/probe_damage_removal.py` | first feasibility probe |
| `systems/gemini/scripts/build_counterfactual_negatives.py` | S1 generator |
| `systems/gemini/scripts/qc_synthetic_negatives.py` | quality gate |
| `systems/gemini/scripts/calibrate_qc_judge.py` | control experiments on the gate |

Models: `gemini-3-pro-image` for editing, `gemini-3.1-pro-preview` for vision
judging and damage inventory.

## 2. The generator, and why each part is there

Every design element below replaced something that was measured to fail.

**Masked paste-back.** The model sees a wide context window; only the damage
boxes are composited back. The first probe, which pasted the whole window,
erased rust staining, ageing and the surveyor's red inspection circles, turning
a weathered column into a new-looking one. Restricting the write to the boxes
bounds over-cleaning by construction: everything else keeps original pixels.

**Real negatives as visual references.** Text instructions did not hold the
model to "sound but aged" - the prompt already said to preserve staining and
dirt, and it cleaned anyway. Three real empty-label photographs from the same
category are supplied as references instead.

**2K generation, downscaled.** `gemini-3-pro-image` returns 1196x896 by default;
`imageConfig.imageSize: "2K"` returns 2392x1792. Generating above the paste-back
window and downscaling supersamples the edit. Upscaling a 1K generation into a
1280px window cannot produce texture that was never generated.

**Best-of-N by least change.** Three candidates per cluster, scored to prefer
the smallest edit that still changes something. One single-sample edit replaced
a band of concrete with pipes and a blue tarpaulin (`core_diff` 81 against a
typical 4-20); such failures are loud in that metric, so scoring selects them
away without a second model to adjudicate.

**Damage inventory pre-screen.** The delivered labels cover only part of what a
strict reader calls damage. Images whose annotated crack was correctly repaired
were still rejected for corroded anchor bolts, rust scaling on the base plate,
and corroded adjacent pipework - none of which carry a box. The inventory pass
finds them so they can be repaired too.

**Minimum window 768px.** A first attempt at refine rounds made things worse:
the inventory returns many small regions, each became its own cluster, each got
a tight window, and the model - starved of context - returned flat grey patches
where a steel flange had been. Larger minimum windows and more aggressive
cluster merging fixed it.

## 3. Calibrating the judge, which changed the gate twice

The gate rejected 6/6 in its first form. Before tuning generation against those
verdicts the instrument was measured, by running the same prompt on real
photographs where the correct answer is known.

### Control 1 - unedited real photographs, passed as both images

16 real negatives and 16 real positives, `column_base`.

| question | false positives | verdict |
|---|---|---|
| `looks_edited` | 0% | looked trustworthy |
| `authentic_survey_photo` | 0% | looked trustworthy |
| `only_change_is_damage_removal` | **69% / 100%** | **removed from the gate** |
| `damage_visible` | **69% of client-clean negatives** | **unusable as an absolute gate** |

Two consequences.

`only_change_is_damage_removal` reported collateral changes between two
*identical* images. It was scoring nothing.

`damage_visible` flagged 11 of 16 images the client labelled empty. Its grade
distribution on real negatives (B 3, C 7, D 1, none 5) is barely distinguishable
from real positives (B 5, C 9, D 2). Whether the judge over-reads or the
empty-label pool is contaminated, an absolute cleanliness gate is one the
corpus's own real negatives fail, so the damage question was rewritten to ask
comparatively about the repaired sites only.

### Control 2 - reversed pairs

`looks_edited` survived control 1 and became the dominant rejection reason, at
5/6. So the pair was reversed: the synthetic shown first, the genuine original
second, asking the same question about "IMAGE 2".

```text
looks_edited on the SYNTHETIC image : 5/6
looks_edited on the REAL photograph : 5/6
```

Identical. Shown a pair, the judge reports editing from the mere existence of a
difference - it described "a blurry smooth vertical band where the crack was"
while looking at the untouched original. In a paired setting the question has a
100% false-positive rate and cannot be used.

Edit detection was therefore split into a **single-image** question with no
reference. Re-calibrated on 24 real crops at the same window scale: **8% false
positives**, which is usable.

### Where the gate landed

| question | form | calibration |
|---|---|---|
| damage removed / residual at repair site | paired, cropped to the repaired window | comparative, so the contaminated-negative problem does not apply |
| looks edited / authentic | single image, no reference | 8% FP on real crops |
| pixel metrics | local, no API | outside-mask identity, over-smooth, hallucination, brightness, saturation |

One more instrument fault was found and fixed along the way: judging the whole
frame downscaled to 1024 made repairs invisible. A hairline crack in a 2816px
photograph does not survive the downscale in *either* image, so two genuinely
repaired images came back as "identical to the original" while local metrics
measured real change. The judge now crops to the repaired window at native
resolution.

## 4. Where the gate agrees with a human reading

On the six-image development set the calibrated gate passes `f-00327` - three
cracks removed, concrete texture, form-tie holes, staining and ground debris all
preserved, which independent visual reading also rated the best of the batch -
and rejects `f-00254` with `looks_edited`, which visual reading independently
rated the worst: the repair *added* a pool of orange rust and replaced a
stiffener edge with an implausible grey wedge.

## 5. Measured on 22 柱脚 images, and what the second pass changed

The first batch through the calibrated gate, and the same images after three
targeted fixes.

| | first batch | after fixes |
|---|---:|---:|
| pass | 5 (23%) | 6 (29%) |
| review | 7 (32%) | 10 (48%) |
| **reject** | 10 (45%) | **5 (24%)** |
| usable (pass + review) | 55% | **76%** |

Rejection reasons, first batch to second: `not_repaired` 7 -> 3,
`residual_severe` 6 -> 3, `brighter` 4 -> 2.

The three fixes, each aimed at a counted reason:

**A corrosion-specific instruction.** Naming the finished surface concretely -
"an even mill or painted finish, no orange product anywhere, bolt threads and
plate edges complete and sharply defined, do not redraw the rust in a tidier
form" - instead of asking for corrosion to be removed. `f-00254`, which the
first pass rejected for *adding* a pool of orange rust and replacing a stiffener
edge with a grey wedge, passes on the second.

**Photometric matching.** The model lightens shadowed concrete; four images came
back 29-57 levels brighter inside the mask than the ring around it. The edited
region's per-channel median is pulled back to the original region's median. The
statistic is a median rather than a mean because the original region still
contains the crack, and a mean would drag the correction dark by exactly the
thing that was repaired.

**Texture gain capped at 2.0.** The amplify-then-top-up texture matcher was
overshooting on smooth surfaces and inventing detail; two images tripped the
`invented_texture` guard.

### Yield by damage type

| damage type | pass | review | reject | n | usable |
|---|---:|---:|---:|---:|---:|
| concrete (cracking, spalling) | 5 | 6 | 6 | 17 | 65% |
| steel (corrosion) | 0 | 1 | 2 | 3 | 33% |

Measured before the corrosion instruction existed. The steel sample is three
images and settles nothing on its own, but it agrees with the visual evidence,
and the fix aimed at it moved the one case that had failed worst.

## 6. The accepted pool, and split isolation

`build_synthetic_negative_pool.py` carries accepted images into a training-ready
pool. It enforces one thing that a copy command would not: **a synthetic
negative inherits its source photograph's scene, element and camera position**,
so one derived from a frozen-test image would leak that test scene into training
as surely as copying the image itself. Sources are intersected with the frozen
train split. On this batch that dropped 1 image from the strict pool and 2 from
the lenient one - small, and exactly the kind of quiet contamination that would
otherwise have been invisible in the results.

Two arms are built, because the review bucket is dominated by the single-image
edit question whose false-positive rate on genuine photographs is 8%:

```text
outputs/gemini_synth/s1_accepted/           pass only          5 images
outputs/gemini_synth/s1_accepted_lenient/   pass + review     14 images
```

## 7. Standing observation to test on the batch

The failures cluster by damage type rather than by image. Concrete cracking and
spalling repair cleanly; steel corrosion does not - on corroded bolts, base
plates and adjacent pipework the model tends to re-render rust rather than
remove it, sometimes worse than the original. If that holds on a larger sample,
S1's usable yield is the concrete-damage subset, and the steel-corrosion subset
needs either a different instruction or exclusion.

This has a companion in the plan document's finding that ブレース C is
member-level deformation boxed over the whole brace. Both point the same way:
S1 suits localised material defects on concrete, and not damage that is
constituted by the geometry or condition of a steel member.
