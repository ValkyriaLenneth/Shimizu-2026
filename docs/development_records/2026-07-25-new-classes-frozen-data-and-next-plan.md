# 2026-07-25 ブレース / 柱脚 frozen datasets and the next training plan

Supersedes the "Recommended Next Steps" section of
`2026-07-25-new-classes-baseline-results.md` as the working plan. The results,
diagnostics and root-cause analysis in that document all stand.

## 1. Why the data had to be frozen

36 dataset directories accumulated for these two categories across three
generations of split policy and four train views:

| generation | directories | status |
|---|---|---|
| 8:2 scene-group split (`bcd_20260725`) | 2 | superseded, kept for provenance |
| 9:1 split (`bcd_20260725_split91`) + crop2 / crop3 / crop2dboost | 8 | superseded as an evaluation surface |
| 5-fold CV (`cv5_20260725`) + crop2 / dboost views | 25 | **canonical** |
| joint brace+column_base corpus | 1 | negative result, never deployed |

Numbers from different generations appear side by side in the experiment log, and
the fold assignment is produced by a script that accepts `--overwrite`. Nothing on
disk recorded which split a stored result belonged to, so a rebuild with a
different seed or a changed upstream manifest would have invalidated every number
without changing a single count.

## 2. What is now frozen

```bash
systems/rfdetr/scripts/freeze_new_class_datasets.py --write   # create the lockfile
systems/rfdetr/scripts/freeze_new_class_datasets.py --check   # verify, exit 1 on drift
```

Lockfile: `data/frozen/new_classes_20260725.lock.json`

The fingerprint is over label *content* - stem plus the exact box lines at fixed
precision - not over filenames or counts. A reshuffled fold, a dropped image or a
rescaled coordinate changes the digest. Image identity is covered by a
`(stem, extension, byte size)` digest, which catches a re-export at a different
resolution without hashing every pixel.

### The canonical evaluation surface

`cv5_20260725`, 5 folds per category, seed 20260725, split on `scene_group_id`,
stratified by the rarest grade present in the group.

| category | pooled test images | boxes | B | C | D |
|---|---:|---:|---:|---:|---:|
| ブレース | 293 | 477 | 146 | 239 | 92 |
| 柱脚 | 224 | 320 | 199 | 82 | 39 |

Verified at freeze time, and re-verified by `--check`:

* The five folds are an exact partition - pooled test covers every usable source
  image once and exactly the full box count of the deduplicated corpus.
* Zero scene-group leakage in all ten folds, and zero stem overlap between train
  and test.
* D is present in every fold: 17-20 boxes per fold for ブレース, 6-11 for 柱脚.

### Derived train views are tied to the same test splits

Checked byte-for-byte rather than assumed. For all ten folds, the `test` split of
the `cv5crop2` and `cv5dboost` views is **identical** to the `cv5` base view:

```text
brace       fold0-4  test IDENTICAL
column_base fold0-4  test IDENTICAL
```

Crop and D-boost alter `train` only, so base vs crop2 vs dboost on a given fold
are directly comparable. `cv5dboost` roughly doubles D against crop2 (ブレース
326 -> 633, 柱脚 137 -> 308).

### Fold 3 is the designated dev fold

The iteration bench (`run_new_class_bench.sh`) runs on fold 3. That fold is
therefore selection-contaminated and the lockfile records it as `dev_fold`. When a
pooled CV figure is quoted, state whether fold 3 is included; the cleanest
estimate pools folds 0, 1, 2 and 4.

## 3. Status of work in flight

The two training jobs running at the time of this note were stopped
(`auglight` bench, ブレース on GPU 0 and 柱脚 on GPU 1, 12 epochs at lr 3e-5).
Both GPUs are idle. Nothing else is queued.

| run | state |
|---|---|
| `cv/crop2` | complete, 10/10 folds - the source of the quoted 0.369 / 0.315 |
| `cv/base` | **incomplete, 2/10 folds** (fold 0 only, both categories) |
| `bench/{crop2_ref,nocrop,auglight}` | fold-3 single-split iteration, auglight stopped mid-run |

The incomplete `cv/base` run is the one the results document names as decisive:
until it finishes there is no unbiased evidence that crop augmentation helps at
all. Its fold-0 results survive; folds 1-4 were never trained.

## 4. Two configuration defects found while reading the code

### 4.1 The encoder learning rate was never lowered

RF-DETR's `TrainConfig` carries two learning rates:

```python
lr: float = 1e-4            # transformer / decoder
lr_encoder: float = 1.5e-4  # DINOv2 backbone
```

`train_rfdetr_router.py` exposes `--lr` and passes it through, but never sets
`lr_encoder`. Confirmed against a real run's `train_options.json`: the key is
absent, so the default applies.

Consequence: in every "low learning rate" experiment in the log - `lr 3e-5`,
`lr 1e-5`, and the whole CV protocol at 3e-5 - **the backbone continued to train
at 1.5e-4, five times the decoder rate.** `lr_vit_layer_decay: 0.8` tapers the
lower layers but the top of the encoder still ran at 1.5e-4.

This confounds the entire learning-rate ablation: those runs did not test "a
gentler learning rate", they tested "a gentler decoder against an unchanged
backbone". On 236 training images that is also the most plausible mechanism for
the peak-then-decay at ~200 optimizer steps that every run shows.

### 4.2 `official_eval_dataset_dir` ignores `--dataset-dir`

```python
"official_eval_dataset_dir": str(resolve_path(dataset_cfg.get("dir", dataset_dir), repo)),
```

The fallback only fires when the config omits `dataset.dir`, and every config
defines it. So `--dataset-dir` overrides the training data but not the official
eval data. A fold-3 crop2 run records:

```text
dataset_dir              .../rfdetr_brace_cv5crop2_20260725_fold3_test_as_valid
official_eval_dataset_dir .../rfdetr_brace_bcd_20260725_split91_test_as_valid
```

**This did not corrupt any result to date.** The value is only read when
`test_each_epoch` or `external_eval_profiles` is active, and the new-class runs
disable both; the CV runner passes `--dataset-dir` explicitly to the evaluator.
But the split91 test images sit inside the CV folds' *train* sets, so the first
run that enables per-epoch testing would evaluate on trained-on images. Worth
fixing before that happens.

## 5. The strongest untried lever: tiled inference

`outputs/rfdetr_new_classes/brace_tiled_2x2.json` records a result that is not
written up anywhere. ブレース, fold 3, crop2 checkpoint epoch 6, thresholds
0.30/0.35/0.40, IoU 0.229, 2x2 tiles at 0.25 overlap merged with the full image:

| | recall | precision | F1 | B recall | C recall | D recall |
|---|---:|---:|---:|---:|---:|---:|
| whole image | 0.291 | 0.347 | 0.316 | 0.250 | 0.353 | 0.222 |
| tiled 2x2 | **0.556** | 0.254 | 0.349 | **0.500** | **0.667** | **0.389** |

Recall nearly doubled, and every grade improved - B and C both roughly doubled.
No retraining was involved; this is a test-time change to an existing checkpoint.

This is the intervention the root-cause analysis predicts should work, and it is
much larger than any training-side change measured so far. The diagnostic found
that missed boxes are missed *outright* rather than mis-graded, which is a
detection-scale failure, and tiling is scale correction applied to the half of the
problem crop augmentation cannot reach: crops zoom the training side, while
inference still ran on whole images.

It also does not contradict the standing "do not raise resolution" finding. That
finding is about the *training* resolution of a full-size dataset. Tiling keeps
native pixels and multiplies the effective inference resolution instead of
upsampling.

The caveat is precision: false positives rose 64 -> 191 and precision fell from
0.347 to 0.254. The thresholds were **not** re-tuned - they were tuned for
whole-image inference, and tiling produces roughly four times the proposals, so
they are now too low by construction. Whether tiling is a net win depends entirely
on re-tuning, which has not been done.
`evaluate_new_class_tiled_inference.py` currently accepts a single threshold
triple and would need a grid to answer this.

## 6. Plan, in priority order

Ranked by evidence strength times cost. Everything is measured on the frozen folds
so results stay comparable.

**1. Re-tune thresholds on tiled inference, then validate across folds.**
Cheapest and best-evidenced. Add a threshold grid to the tiled evaluator, re-tune
on the dev fold, then run the frozen 5-fold protocol on folds 0/1/2/4 for both
categories. 柱脚 has no tiled measurement at all yet. Decision rule fixed in
advance: keep tiling if pooled F1 rises, or if recall rises at equal precision -
recall is the client-facing metric.

**2. Fix `lr_encoder` and re-run the CV baseline.** Sets `lr_encoder` alongside
`lr` (start with encoder = decoder rate, and a frozen-encoder arm). This both
repairs the confound in 4.1 and finishes the incomplete `cv/base` run, which
settles the crop-augmentation question. Freezing the DINOv2 encoder outright is
the standard few-shot recipe at this data scale and has never been tried here -
`freeze_encoder` is an RF-DETR config flag.

**3. Reduce `num_queries`.** Default is 300, with `group_detr: 13` giving 3900
query slots per image trained against roughly 1.5 ground-truth boxes. RF-DETR's
own nano and small variants ship 100-200 queries, and the config explicitly
permits decreasing it when loading pretrained weights. This targets the observed
B false-positive flood - 159 FP against 58 TP for ブレース B, 321 against 95 for
柱脚 B - directly. `--num-queries` is already wired through the trainer.

**4. Single-class detection plus a separate grading head.** The grade-confusion
diagnostic found *zero* confusions: the model never mis-grades, it fails to find
the damage. If that holds on 柱脚 too, then splitting scarce boxes three ways
costs detection performance for nothing. Merging B/C/D into one "damage" class
gives the detector 477 and 320 positives instead of 146/239/92 and 199/82/39, with
grading done afterwards on the detected crop. This restructures the task around
the measured failure mode rather than tuning around it. Confirm the 柱脚
grade-confusion result first - it is still queued.

**5. Calibrate the delivered band.** The comparison driving "we are half as good
as the delivered models" is not like-for-like: 0.72-0.86 was selected on ~32-box
splits with valid mirroring test, while 0.369/0.315 is pooled out-of-fold. Running
the identical 5-fold protocol on one delivered category would measure how much of
the gap is real and how much is measurement. This is the single most useful number
for the client conversation.

**Blocked, and sharing one blocker.** Item 5, and warm-starting 柱脚 from the
released `rc_column` checkpoint, both need the release archive restored.
`.local_artifacts/` is absent, `final_release_20260615/models/rfdetr/` holds only
`metrics/` and manifests, and no delivered-category dataset is present in `data/`.
Restoring that archive unblocks both at once and is worth asking for.

**Not pursued**, per prior findings: training-resolution increase, RFDETRLarge,
joint brace+column_base pretraining, and label review of the 11 grade
contradictions (worth doing for hygiene, but measured not to be the bottleneck).

## 7. Standing note on this host

Every command, not only python, must run under `taskset -c 0,2-63`. CPU 1 is
advertised online but nothing scheduled onto it ever runs; a bare `grep` over the
site-packages tree wedged during this session for exactly that reason. The
launcher scripts already apply the mask - ad-hoc shell commands are the exposure.
