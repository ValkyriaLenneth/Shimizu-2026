# 2026-07-25 ブレース / 柱脚 baseline v1 - the reference point

The plain baseline on the frozen single train/test split. Everything measured
later is compared against this table.

## Protocol

Recipe is the audited 天井 / 内壁 / RC壁 / RC柱 baseline verbatim: RFDETRMedium,
80 epochs, batch 28, grad accum 1, lr 1e-4, 16-mixed, default resolution (576),
valid mirroring test. No crop view, no augmentation preset, no `lr_encoder` or
`num_queries` override, no warm start, no tiled inference.

Data is the frozen 8:2 split (`freeze_new_class_datasets.py --check` passes):

| category | train | test |
|---|---|---|
| ブレース | 235 imgs / 394 boxes (B124 C194 D76) | 58 imgs / 83 boxes (B22 C45 D16) |
| 柱脚 | 179 imgs / 248 boxes (B152 C67 D29) | 45 imgs / 72 boxes (B47 C15 D10) |

Selection followed the established two-step protocol: sweep all 80 epoch
checkpoints on test, then a per-class threshold grid at match IoU 0.229
(10 thresholds^3 x 4 checkpoints = 4000 combinations per category).

Batch 28 was confirmed to saturate the cards: memory settles at ~31.5 GB of
32.6 GB on both. Raising it further would OOM at the largest multi-scale
resolution, and would also cut the already small number of optimizer steps per
epoch (9 for ブレース, 7 for 柱脚).

## Headline result

| category | epoch | thresholds B/C/D | recall | precision | F1 |
|---|---:|---|---:|---:|---:|
| ブレース | 33 | 0.50/0.45/0.40 | 0.434 | 0.600 | **0.503** |
| 柱脚 | 35 | 0.35/0.30/0.50 | 0.444 | 0.582 | **0.504** |
| 柱脚, at P>=0.60 | 35 | 0.40/0.30/0.50 | 0.417 | 0.600 | 0.492 |
| delivered band | | | 0.812-0.875 | 0.596-0.824 | 0.72-0.86 |

**The client target - recall >= 0.80 at precision >= 0.60 - is not reachable for
either category.** For ブレース the best precision at recall >= 0.80 is 0.101; for
柱脚 no threshold combination reaches recall 0.80 at all.

Per-class threshold tuning is worth having: it lifted ブレース F1 from 0.410
(single threshold) to 0.503.

Note that the grid re-ranked the checkpoints. 柱脚's best mAP50 epoch was 72, but
its best tuned operating point comes from epoch 35 - another reason the automatic
`checkpoint_best_total.pth` is not the one to ship.

## The two categories fail differently

This is the main finding, and it means they should not receive the same treatment.

Behaviour at the lowest threshold (0.05/0.05/0.05), where the model is asked to
report everything it has:

| category | max recall | B | C | D | total FP |
|---|---:|---:|---:|---:|---:|
| ブレース | 0.843 | 0.773 | 0.844 | **0.938** | 1101 |
| 柱脚 | 0.625 | 0.660 | 0.733 | **0.300** | 607 |

**ブレース is a ranking problem.** It finds 84% of the damage, including 94% of
the scarce D grade. It simply cannot score those above background texture, rust,
bolts and joints - at that operating point it emits about 19 false positives per
image. The damage is visible to the model; the scores are not separable.

This refines the earlier diagnosis. `analyze_new_class_grade_confusion.py` found
that missed boxes are "missed outright" rather than mis-graded, and that was read
as a detection failure. It is true at the *operating* threshold, but at a low
threshold those same boxes are found. The failure is score calibration, not
blindness.

**Correction, from a follow-up probe.** An earlier draft of this section claimed
柱脚 grade D was a genuine detection failure, because D recall topped out at 0.300.
That was an artifact of the grid's lower bound of 0.05, not a property of the
model. Re-probing at thresholds down to 0.001 shows D recall reaching 0.800. The
D detections exist; they are scored very low. There is no detection-capability
failure in either category, and D-targeted annotation is **not** the indicated fix.

The full curve is what matters, and the extreme is degenerate - at threshold 0.001
the model emits ~297 boxes per image out of 300 query slots, so that point measures
query blanket coverage rather than capability:

| threshold | ブレース recall | preds/image | 柱脚 recall | preds/image |
|---|---:|---:|---:|---:|
| 0.001 | 0.976 | 297 (degenerate) | 0.875 | 296 (degenerate) |
| 0.02 | 0.928 | 97 | 0.806 | 92 |
| 0.05 | 0.819 | 21 | 0.625 | 14.5 |

The client target - recall 0.80 at precision 0.60 - corresponds to roughly **2
predictions per image** (ブレース: 66 tp + 44 fp over 58 images; 柱脚: 58 tp + 38 fp
over 45 images). Both models can already reach recall 0.80, but only while emitting
21-97 boxes per image.

**So the gap is score separation, and it is quantified: the true boxes sit at rank
20-90 per image and need to move to rank 2.** Roughly a 50x improvement in ranking,
for both categories. This is not a data-volume problem and not a detection problem.

Per-class AP at the best-mAP epoch shows the same split:

| category | B | C | D |
|---|---:|---:|---:|
| ブレース (ep39) | **0.065** | 0.228 | 0.107 |
| 柱脚 (ep72) | 0.170 | 0.220 | 0.139 |

ブレース has one collapsed class carrying the whole average down; 柱脚 is
uniformly mediocre.

## 柱脚 had not finished training at 80 epochs

The recorded project rule is that peak performance lands at epoch 19-34 and the
remaining epochs contribute nothing. ブレース obeys it - peaks at ep31-39, decays
after. 柱脚 does not:

| category | best mAP50 epochs | best F1 epochs |
|---|---|---|
| ブレース | 39, 35, 42, 36 | 31, 30, 27, 33 |
| 柱脚 | **72, 78, 77, 75** | **76, 77, 74, 75** |

Every top epoch for 柱脚 sits against the 80-epoch ceiling, so it was still
improving when the run ended. This was never visible before: the CV protocol
evaluated a fixed epoch 6, and the single-split experiments ran 30-45 epochs.
Extending 柱脚's schedule is a free and untried lever.

The caveat is that the tuned best operating point still came from epoch 35, so the
late-epoch mAP gain did not translate into a better tuned point. Extending the
schedule is worth one run, not an assumption.

## Comparison with the earlier crop-augmentation results

mAP is threshold-free, so it compares across experiments even though the test
splits differ.

| configuration | ブレース mAP50 | 柱脚 mAP50 |
|---|---:|---:|
| old 9:1, plain baseline | 0.307 | 0.266 |
| old 9:1, crop2 augmentation | 0.366 | 0.342 |
| **frozen 8:2, plain baseline (this run)** | **0.358** | **0.346** |

The plain baseline now matches what crop augmentation previously achieved. This is
a third piece of evidence against crop augmentation being a real effect - the CV
analysis had already found the base-to-crop2 difference (0.048) smaller than the
per-fold spread (0.070), and withdrew the claim.

It is suggestive, not conclusive: the test splits differ (83/72 boxes here versus
39/38 before), so this is not a like-for-like comparison. Re-running crop2 on the
frozen split would settle it, and would retire or confirm a large amount of
downstream work.

## The optimization target, stated precisely

Everything above converges on one objective: **raise the classification score of
true damage boxes relative to background, by roughly 50x in rank terms.** Recall is
already there; it is unusable because it costs 20-90 false positives per image.

That objective selects the levers. Anything aimed at *finding more damage* -
additional annotation, D oversampling, longer schedules, larger backbones - is
aimed at a problem this model does not have.

1. **Varifocal loss / focal loss tuning.** `--use-varifocal-loss` and
   `--focal-alpha` are already wired through the trainer and have never been used.
   Varifocal loss exists specifically to make the classification score
   IoU-aware, which is the definition of the ranking defect measured here. This is
   the most directly targeted lever available and it costs one flag.
2. **Freeze or sharply lower the encoder learning rate.** RF-DETR trains the
   DINOv2 backbone at `lr_encoder: 1.5e-4` by default, never overridden. On 235
   images that plausibly degrades the pretrained features that separability depends
   on. `freeze_encoder` is a config flag; standard few-shot recipe, never tried here.
3. **Reduce `num_queries` from 300.** With `group_detr: 13` that is 3900 query
   slots per image against roughly 1.5 ground-truth boxes. Fewer slots means fewer
   low-quality proposals competing for rank. RF-DETR's own nano and small variants
   ship 100-200 queries and the config explicitly permits decreasing it.
4. **Matcher cost rebalancing** via `--set-cost-class` / `--set-cost-bbox` /
   `--set-cost-giou`, which control how much classification confidence weighs in
   Hungarian assignment and therefore what the score is trained to mean.
5. **Tiled inference plus threshold re-tuning.** Still worth testing, but its role
   is now narrower: it raises scores for small objects by giving them more pixels,
   which is score improvement, not extra coverage. It also multiplies proposals,
   which works against the objective, so it can only be judged after re-tuning.
6. **Re-run crop2 on the frozen split** to settle the crop question and retire or
   confirm the downstream work that depends on it.

Explicitly de-prioritised by this analysis, reversing earlier plans: more
annotation, D-targeted oversampling, and extending 柱脚's schedule. The late-epoch
mAP gain for 柱脚 is still worth one confirming run, but it is no longer a
front-line lever.

## Artifacts

```text
outputs/rfdetr_single_crack/baseline_v1/{brace,column_base}/
    test_results.csv          80-epoch sweep
    grid_ep*.csv              per-class threshold grid, 1000 rows each
    metrics.csv               per-epoch training metrics
    epoch_pth/*.pth           all 80 epoch checkpoints
systems/rfdetr/scripts/run_new_classes_baseline_v1.sh
```

## Operational note

A wait loop guarding on `pgrep -f "sweep_rfdetr_router_test.py"` deadlocked
against its own command line - the same self-match failure already recorded in
`2026-07-25-new-classes-baseline-results.md`. Guard on the output artifact
(`test_results.csv` row count) or on a PID captured at launch, never on a pattern
that appears in the watching shell's own arguments.
