# 2026-07-26 The models detect the element, not the damage

## What was measured

`systems/rfdetr/scripts/audit_empty_label_images.py` ran the baseline_v1 candidate
checkpoints (ブレース ep33, 柱脚 ep35) over the 141 empty-label images that were
excluded from training, and rendered a contact sheet sorted by peak confidence.

The intent was narrow: settle whether those images are "inspected, no damage"
(usable as hard negatives) or "not yet annotated" (poison). The sheets answered
that, and also exposed something larger.

| category | images | any detection >=0.10 | peak >=0.30 | peak >=0.50 | median peak | max |
|---|---:|---:|---:|---:|---:|---:|
| ブレース | 59 | 59 (100%) | 46 (78%) | 26 (44%) | 0.436 | 0.952 |
| 柱脚 | 82 | 79 (96%) | 48 (59%) | 21 (26%) | 0.351 | 0.913 |

## The finding

Looking at the sheets rather than the numbers: **the high-confidence detections
sit on intact, undamaged elements.**

For ブレース, the top-confidence images are building facades and warehouse
interiors showing X-shaped cross-bracing in good condition, and the model draws a
box around the brace itself at 0.86-0.95.

For 柱脚 the pattern is even cleaner: image after image of a plain, sound concrete
column base, with the model boxing the concrete pedestal at 0.60-0.91.

The models have not learned to detect damage. They have learned to detect **the
element that damage appears on.**

## Why this happened

It follows directly from a data decision recorded in
`2026-07-25-new-classes-annotation-match.md`: empty-label images were dropped
because "every image contains damage". The consequence is that in the training
corpus, *element present* and *damage present* are perfectly correlated - there is
no image containing a brace without brace damage, or a column base without column
base damage.

Given that correlation, "find the brace" is a strictly easier hypothesis than
"find the damage on the brace", and it achieves identical training loss. The model
took the shortcut. This is textbook shortcut learning, caused by the absence of
negatives rather than by anything about RF-DETR.

## What it explains

This single mechanism accounts for results that previously looked unrelated:

* **The false-positive flood.** 159 / 321 B-grade false positives, 19 predictions
  per image at low threshold, precision 0.23-0.27. Of course - the model fires on
  every brace and every column base it sees, damaged or not.
* **Why per-class thresholds barely help.** Thresholding a score that means
  "an element is here" cannot recover a score that means "damage is here".
* **Why varifocal loss did nothing** (+0.031 / +0.012 best F1, 0.000 / +0.028 at
  the precision floor, all inside the noise band). Varifocal reshapes how a score
  is calibrated. It cannot change *what the score is a detector of*. The null
  result is now expected rather than disappointing.
* **Why crop augmentation was never reproducible.** Cropping around ground-truth
  boxes produces more images in which element and damage still co-occur perfectly.
  It amplifies the shortcut instead of breaking it.
* **The measured ceiling.** At floored thresholds recall reaches 0.98 / 0.88
  because every element gets a box and the damage is on an element. That was read
  as "the model sees the damage". It is better read as "the model sees the element,
  and the damage happens to be inside it".

## Consequence for the excluded images

The 141 images are **a mix**, and blanket inclusion would be as wrong as blanket
exclusion.

Most of the top-confidence images are genuinely damage-free - sound braces, sound
column bases - and are exactly the negatives the corpus lacks. But a minority
carry real, unannotated damage. `f-00189` shows clearly exposed rebar, which is
unambiguous D-grade; `f-00322` shows spalled concrete with debris; `f-00203` shows
a corroded steel column base. Training on those as background would teach the model
to suppress genuine damage.

So the answer to the question left open with the annotation team is "both", and the
action is triage rather than a yes/no decision.

## Revised direction

The binding problem is not score calibration and not data volume. It is that the
training corpus cannot distinguish the two hypotheses the model has to choose
between. Everything follows from fixing that.

1. **Triage the 141 excluded images into damage-free and unannotated.** The contact
   sheets in `outputs/rfdetr_new_classes/empty_label_audit/` are built for exactly
   this and are the artifact to send the annotation team. This is the prerequisite
   for everything below.
2. **Train with the confirmed damage-free images as background samples.** RF-DETR's
   YOLO loader supports this natively - an image with no matching label file is
   loaded as a background sample with empty detections. This is the only
   intervention that directly breaks the element/damage correlation.
3. **Re-examine what a "negative" needs to be.** Whole-image negatives may be too
   easy. The sharper negative is a *crop of an undamaged element* - the same framing
   the positives get. That is buildable from the damage-free images with the
   existing crop-view tooling.
4. **Ask the client for undamaged-element photographs.** If the 141 do not yield
   enough confirmed negatives, this is a cheap capture request: photographs of sound
   braces and sound column bases need no damage grading, only the assertion that
   they are undamaged.

Deprioritised as a result: further loss-function work, further crop-view variants,
and additional *damage* annotation. More positives do not break a correlation that
exists because there are no negatives.

## Caveat

The evidence is a visual reading of two 24-image contact sheets by one reviewer,
using a model with a known false-positive problem. A confident detection on an
undamaged element is consistent with the shortcut hypothesis, but so is an ordinary
false positive; what makes the reading strong is that the pattern is systematic and
appears in both categories independently, and that it explains four previously
separate null results. It should still be confirmed by the annotation team's triage
before being reported to the client as fact.
