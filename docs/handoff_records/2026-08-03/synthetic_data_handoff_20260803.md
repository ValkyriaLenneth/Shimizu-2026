# 2026-08-03 handoff - ブレース / 柱脚 synthetic data (S1) and the training plan

Read this first if you are picking the work up on the training host. It is
written to be self-contained: it assumes no knowledge of the session that
produced it, and it names what is in git, what is not, and what to do first.

Companion documents, in reading order:

| document | what it holds |
|---|---|
| `docs/development_records/2026-07-26-new-classes-final-state.md` | where the B/C/D models stand and every intervention already tried |
| `docs/development_records/2026-08-03-new-classes-synthetic-data-plan.md` | why synthetic data, which of three lines to run, and the constraints |
| `docs/development_records/2026-08-03-s1-pipeline-and-judge-calibration.md` | how the S1 pipeline works and the control experiments that shaped it |
| this file | state, transfer, gotchas, and the ordered next steps |

## 1. Where the project actually is

The two new categories are stuck well below the delivered band:

| category | recall | precision | F1 | train images |
|---|---:|---:|---:|---:|
| ブレース | 0.590 | 0.521 | 0.554 | 235 |
| 柱脚 | 0.514 | 0.607 | 0.556 | 179 |
| delivered four categories | - | - | 0.72-0.86 | 605-966 |

The cause is diagnosed and is not a tuning problem. Every training image
contains damage, so "element present" and "damage present" are perfectly
correlated and the models learned to find the element: they fire at 0.86-0.95 on
intact braces and sound column bases. Six training-side interventions (loss
shape, encoder learning rate, encoder freezing, augmentation strength, inference
tiling, crop negatives) returned zero or negative. One data-side intervention
worked - adding the 141 empty-label images as background negatives, +0.060
recall on ブレース and +0.097 on 柱脚 - with a dose-response that saturates at
around 31-35% and turns harmful by 48%.

Repetition of those 141 is saturated. Only new negatives extend the curve. That
is what S1 produces.

## 2. What was built on 2026-08-03

**S1 - counterfactual negatives.** A real damaged photograph has every damage
region repaired back to a sound state and is emitted with an empty label. The
result shares its scene, element, framing, lighting and camera with a real
positive, so the pair differs only in the damage. That is the most direct attack
available on the element/damage correlation, and it needs no boxes at all, which
sidesteps the annotation noise that killed the 2026-05-26 synthetic router
experiment.

```text
systems/gemini/scripts/
  synth_common.py                     shared geometry, compositing, texture/tone matching, transport
  extract_grade_reference_crops.py    per-grade crop bank + contact sheets (no API)
  probe_damage_removal.py             the original feasibility probe, kept for provenance
  build_counterfactual_negatives.py   S1 generator
  qc_synthetic_negatives.py           calibrated quality gate
  calibrate_qc_judge.py               control experiments on the gate
  build_synthetic_negative_pool.py    split-isolated accepted pool (no API)
```

Models used: `gemini-3-pro-image` for editing, `gemini-3.1-pro-preview` for the
vision judge and the damage inventory.

### Measured yield, 柱脚, 22 images

| | first batch | after three fixes |
|---|---:|---:|
| pass | 5 (23%) | 6 (29%) |
| review | 7 (32%) | 10 (48%) |
| reject | 10 (45%) | **5 (24%)** |
| usable | 55% | **76%** |

By damage type, measured before the corrosion fix existed: concrete cracking and
spalling 65% usable (17 images), steel corrosion 33% (3 images). The corrosion
fix moved the worst steel case from reject to pass, but the steel sample is too
small to call.

## 3. What is in git, and what is not

**In git.** All seven scripts, the two development records, this handoff, and
small state files under `docs/development_records/assets/2026-08-03-s1/`:

```text
column_base_qc_results.json                 per-image verdicts, reasons, metrics, judge output
column_base_generation_results.jsonl        per-image generation record, boxes repaired, candidate diffs
column_base_judge_calibration.json          the control experiment raw results
column_base_accepted_strict_manifest.json   pass-only pool, with provenance
column_base_accepted_lenient_manifest.json  pass+review pool
example_pass_f-00327_before_after.jpg       what an accepted repair looks like
example_reject_f-00254_before_after.jpg     what a rejected repair looks like
```

Those let you read the state without moving any image data.

**NOT in git.** `outputs/` is ignored, so the generated images do not arrive with
a clone. 222 MB total, of which the part that matters is small:

```text
outputs/gemini_synth/s1_accepted/            pass only        5 images   4.5 MB
outputs/gemini_synth/s1_accepted_lenient/    pass + review   14 images    11 MB
outputs/gemini_synth/s1_counterfactual_negatives/  all generated + compare  29 MB
outputs/gemini_synth/grade_references/       797 grade crops + contact sheets
```

Transfer the pool from the machine that generated it:

```bash
rsync -avz --progress \
  <local>/Shimizu-2026/outputs/gemini_synth/s1_accepted_lenient/ \
  <host>:/workspace/Shimizu-2026/outputs/gemini_synth/s1_accepted_lenient/
```

Or regenerate from scratch - the scripts are deterministic in their inputs and
resume from `generation_results.jsonl`, so a rerun only fills gaps. Regeneration
needs the API key (section 5).

## 4. Host gotchas that will waste your time otherwise

**Every command needs `taskset -c 0,2-63`, not only python.** CPU 1 is
advertised online but nothing scheduled onto it ever runs; a bare `grep` over
site-packages has wedged a session in state R, unkillable. The launcher scripts
already apply the mask - ad-hoc shell commands are the exposure.

```bash
taskset -c 0,2-63 /bin/true || echo "check /sys/devices/system/cpu/online"
```

**The dataset split is frozen and must be verified before any training.**

```bash
taskset -c 0,2-63 .venv/bin/python systems/rfdetr/scripts/freeze_new_class_datasets.py --check
```

Lockfile `data/frozen/new_classes_20260725.lock.json`, fingerprinted over label
*content*. Fold 3 is the designated dev fold and is selection-contaminated;
state whether it is included whenever a pooled CV figure is quoted.

**`lr_encoder` has never been set.** `train_rfdetr_router.py` exposes `--lr` and
passes it to the decoder, but never sets `lr_encoder`, so RF-DETR's default
1.5e-4 applied to the DINOv2 backbone in every run. Every "low learning rate"
experiment in the log - lr 3e-5, lr 1e-5, the whole CV protocol - actually
tested a gentler decoder against an unchanged backbone. Fix before drawing any
further conclusion about encoder behaviour.

**`official_eval_dataset_dir` ignores `--dataset-dir`.** Harmless today because
the new-class runs disable `test_each_epoch` and `external_eval_profiles`, but
the first run that enables per-epoch testing would evaluate on trained-on
images.

**Synthetic negatives must not cross the frozen split.** A counterfactual
inherits its source photograph's scene, element and camera position, so one
derived from a test image leaks that scene into training as surely as copying
the image would. `build_synthetic_negative_pool.py` enforces this against
`.local_artifacts/handoff_20260726/split/<category>_split.json`; on the first
batch it dropped 1-2 images. Do not bypass it.

## 5. API key

The scripts read `GEMINI_API_KEY`. It is exported in the operator's local
`~/.zshenv`; **it is not in the repo and will not be present on the host**.
Export it there before running any generation or QC. Sparticle rotates these
keys roughly every three months and an expired one returns `API_KEY_INVALID`.
The endpoint is ordinary Google AI Studio,
`https://generativelanguage.googleapis.com/v1beta`, auth by `x-goog-api-key`
header or `?key=`; verify with a `models` list call before assuming anything
else is broken.

## 6. Two findings that change how the data should be read

**Grade correlates with box scale.** Box area as a fraction of the image:

| | B | C | D |
|---|---:|---:|---:|
| ブレース median | 0.87% | 6.93% | 16.4% |
| ブレース boxes over 25% of frame | 3.4% | 19.2% | 32.6% |
| 柱脚 median | 2.14% | 9.54% | 5.85% |

Box scale is close to a proxy for grade. Any synthetic positive must respect the
per-grade scale convention or the label is wrong even when the appearance is
right.

**ブレース grade C is largely member-level deformation, boxed over the whole
brace bay** - slackness and distortion, not a localised defect texture. Two
consequences. First, for C the annotation itself makes "detect the element" and
"detect the damage" nearly the same task at box level, which is part of why the
shortcut formed and why negatives were the only thing that helped. Second, C
cannot be synthesised by painting damage into a small region; it would need
geometry editing. S1 is expected to suit 柱脚 concrete damage best, and ブレース
C not at all. **ブレース has not been run yet** - the scripts accept
`--category brace` but no batch exists.

There is no B/C/D grading rubric anywhere in the repo. The labels came from the
client's CVAT export with `obj.names = B / C / D`, so grade semantics live only
in the real crops. That is why generation is conditioned on real reference crops
rather than on a written description of each grade.

## 7. The quality gate, and why you should not trust it naively

The gate rejected 6/6 in its first form. Before tuning generation against it,
the instrument was measured against real photographs where the answer is known.
Two of four questions failed and were changed. Full detail in
`2026-08-03-s1-pipeline-and-judge-calibration.md`; the short version:

| question | outcome |
|---|---|
| `only_change_is_damage_removal` | 69-100% false positives on *identical* images - removed |
| `damage_visible` (absolute) | flagged 69% of client-clean negatives - replaced by a comparative question about the repaired sites only |
| `looks_edited` **when asked on a pair** | 100% false positive: a reversed-pair control fired 5/6 in *both* directions, describing the genuine photograph as edited - moved to a single-image question, 8% FP |
| whole-frame judging at 1024px | hairline cracks vanish in both images, so real repairs read as "identical" - judge now crops to the repaired window at native resolution |

The lesson generalises: **calibrate any model-based gate against known-answer
controls before optimising against its verdicts.** `calibrate_qc_judge.py` is
the harness; rerun it if you change the prompt.

A side finding worth carrying to the client conversation: the judge finds damage
on 69% of the 82 column_base images the client delivered with empty labels, and
its grade distribution there (B 3, C 7, D 1, none 5) is barely distinguishable
from real positives (B 5, C 9, D 2). The repo previously assumed only a minority
were contaminated. Whether the judge over-reads or the pool is genuinely dirty,
the 141 empty-label images deserve triage before they carry more weight.

## 8. Training plan - recommended order

Self-supervised pretraining was considered as the main line and is **not
recommended**, for three reasons grounded in this project's own measurements.
The failure is an ambiguous supervision signal, not weak features - the grade
confusion matrix shows *zero* confusions, so the model never mis-grades, it
fails to find the damage; better representations make the element easier to find
and reinforce the shortcut. Freezing the DINOv2 encoder, which is leaning harder
on generic self-supervised features, produced -0.265 recall, the worst entry in
the intervention table. And the in-domain corpus across all six categories is
about 4000 images against DINOv2's 142M, two to three orders of magnitude short
of where self-supervised pretraining pays. This changes if the client can supply
tens of thousands of unlabelled survey photographs, which cost them far less
than annotation and are worth asking for.

Run in this order:

**1. Build the false-alarm benchmark. Prerequisite, no GPU needed.** The frozen
test split contains only images that carry damage, so "does the model stay quiet
on a sound element" - the thing negatives are supposed to fix - is unmeasured.
Without this, S1's main effect is invisible and a null result would be
misinterpreted as "synthetic data does not work".

**2. Fix `lr_encoder` and finish the incomplete `cv/base` run** (2/10 folds).
Cheap, repairs the confound in section 4, and settles the still-open question of
whether crop augmentation helps at all.

**3. Sweep the synthetic-negative ratio.** Pure data change, reusing
`build_rfdetr_negatives_view.py` - the exact path that produced the only
positive result to date. Real negatives showed 31-35% effective and 48% harmful,
so the synthetic ratio must be swept, not assumed. Report against the frozen
5-fold protocol, not a single split: at 83 and 72 test boxes one true positive
is worth 0.012-0.014 recall and F1 differences under 0.03 are not resolvable.

**4. Single-class detection plus a separate grading head.** The strongest untried
idea, and in my view ahead of any representation work. Zero grade confusions
means splitting scarce boxes three ways costs detection for nothing: merging
B/C/D gives the detector 477 and 320 positives instead of 146/239/92 and
199/82/39, with grading done afterwards on the detected crop.

**5. Counterfactual-pair auxiliary loss.** Only if 1-4 plateau. S1 pairs differ
*only* in the damage, so any feature that separates a pair must be a damage
feature - a far sharper version of the self-supervised idea than generic
pretraining. Costs training-code changes, which is why it comes last: six clever
training-side interventions have already returned nothing, and the cheap data
path should be exhausted first.

## 9. Immediate next actions

```bash
# 0. orient
taskset -c 0,2-63 /bin/true
git log --oneline -3
taskset -c 0,2-63 .venv/bin/python systems/rfdetr/scripts/freeze_new_class_datasets.py --check

# 1. bring the synthetic pool over (see section 3), then confirm it landed
ls outputs/gemini_synth/s1_accepted_lenient/column_base/images | wc -l

# 2. run ブレース, which has never been generated (needs GEMINI_API_KEY)
python systems/gemini/scripts/build_counterfactual_negatives.py \
  --category brace --limit 24 --prescreen --concurrency 4
python systems/gemini/scripts/qc_synthetic_negatives.py --category brace
```

Expect ブレース to yield worse than 柱脚 for the reason in section 6. If it does,
that is a result worth reporting rather than a bug: it says the remaining ブレース
gap needs real photographs, not synthesis, which is a concrete and cheap ask to
put to the client alongside the standing request for photographs of undamaged
braces and column bases.
