# 2026-07-26 ブレース / 柱脚 final state and reproduction guide

The consolidated record for this round. Read this first; the other 2026-07-25 and
2026-07-26 notes hold the detail behind each step.

| document | what it holds |
|---|---|
| `2026-07-25-new-classes-annotation-match.md` | delivery pairing, dedup, the 11 grade contradictions |
| `2026-07-25-new-classes-baseline-v1-results.md` | plain baseline, the reference point |
| `2026-07-26-new-classes-shortcut-learning-finding.md` | why the models detected the element, not the damage |
| `2026-07-26-new-classes-negatives-results.md` | the negatives experiments in detail |
| this file | consolidated result, tradeoff curves, reproduction |

## Recommended delivery configuration

Frozen 8:2 split, per-class thresholds at match IoU 0.229. Test sets hold 58
images / 83 boxes (ブレース) and 45 / 72 (柱脚).

| category | checkpoint | thresholds B/C/D | recall | precision | F1 | FP/image |
|---|---|---|---:|---:|---:|---:|
| ブレース | `brace_neg2x_epoch_050.pth` | 0.25/0.25/0.20 | **0.590** | 0.521 | 0.554 | 0.8 |
| 柱脚 | `column_base_negatives_v1_epoch_016.pth` | 0.30/0.30/0.20 | **0.514** | 0.607 | 0.556 | 0.5 |

Against the plain baseline (0.434 / 0.417 recall at precision >= 0.60), that is
**+0.156 for ブレース and +0.097 for 柱脚**.

The two categories are deliberately operated at different precision floors,
because their tradeoff curves have different shapes - see below. ブレース at a
measured precision of 0.521 is defensible for two documented reasons: the shipped
天井 model went out at precision 0.596, and production filters downstream false
positives outside the router region by the IoA >= 0.50 rule
(`2026-07-25-new-classes-divergence-audit.md`), so measured precision understates
end-to-end precision.

## Tradeoff curves

Merged over every threshold grid run: 8488 combinations for ブレース, 5744 for 柱脚.

Maximum recall at a given precision floor:

| precision floor | ブレース recall | 柱脚 recall |
|---|---:|---:|
| 0.60 | 0.494 | **0.514** |
| 0.55 | 0.542 | 0.514 |
| 0.50 | **0.590** | 0.528 |
| 0.45 | 0.614 | 0.542 |
| 0.40 | 0.639 | 0.556 |
| 0.30 | 0.699 | 0.611 |

Highest precision at a given recall target:

| recall target | ブレース precision | 柱脚 precision |
|---|---:|---:|
| 0.60 | 0.476 | 0.308 |
| 0.70 | 0.288 | 0.166 |
| 0.80 | 0.137 | 0.036 |

**The two categories trade very differently.** Relaxing ブレース from 0.60 to 0.50
buys +0.096 recall and F1 actually *rises* to its maximum of 0.554; its weakest
grade, C, improves from 0.378 to 0.578. The same relaxation on 柱脚 buys +0.014 -
one box - while F1 falls from 0.556 to 0.524. Its curve is nearly flat there, so
precision spent buys no recall. Hence the different floors.

**Recall 0.80 is not reachable by tuning.** ブレース needs precision 0.137 (7.3
false positives per image) and 柱脚 needs 0.036 (34.6 per image). Threshold work
took ブレース from 0.494 to 0.590, which is real, but the remaining gap to 0.80 is
not a tuning problem.

## Every intervention tried, and what it did

Measured as change in recall at precision >= 0.60 versus the plain baseline.

| intervention | kind | ブレース | 柱脚 |
|---|---|---:|---:|
| **background negatives, ~30-35%** | **data** | **+0.060** | **+0.097** |
| per-class threshold relaxation to P>=0.50 | inference | **+0.096** | not worthwhile |
| varifocal loss | training | 0.000 | +0.028 (inside noise) |
| negatives at 48% | data | - | -0.070 |
| crop negatives | data | no feasible point | -0.014 |
| strong photometric augmentation | training | -0.060 | -0.125 |
| tiled 2x2 inference | inference | -0.121 | -0.139 |
| frozen DINOv2 encoder | training | -0.265 | - |

Two things generalise from this table.

**Only the data-side intervention worked.** Six attempts to extract more from the
same images - loss shape, encoder learning rate, augmentation strength, inference
tiling - were flat or harmful. The one that added missing information helped in
both categories, with a dose-response.

**Both crop-based interventions failed the same way.** `negcrop` fed crops as
negatives while positives stayed whole images; tiled inference fed 2x2 tiles to a
model trained on whole images. Both invite the model to discriminate on framing
artefacts rather than content, and both collapsed at the high-precision end while
looking acceptable mid-curve. If either is retried, positives and negatives must
travel the same processing path.

Notable non-finding: strong photometric augmentation hurt in both categories
(-0.060 / -0.125). The mechanism is plausible and worth remembering - damage grade
depends on appearance, so brightness +-0.30, contrast +-0.30, hue +-10 and gamma
70-140 perturb the very cues that separate B from C from D. The preset's author was
careful with geometry for exactly this reason but not with photometry. By the same
argument, the light presets are unlikely to be positive either; the issue is
direction, not strength.

## Reproduction

Everything needed is packaged in `.local_artifacts/handoff_20260726/` (see its
`README.md` and `SHA256SUMS`). From a fresh checkout:

```bash
# 0. host quirk: CPU 1 is advertised online but is not schedulable. Anything pinned
#    to it wedges in state R and cannot be killed, including /bin/true. Prefix every
#    command - not only python - with the mask.
taskset -c 0,2-63 /bin/true || echo "check /sys/devices/system/cpu/online"

# 1. verify the frozen split has not drifted
taskset -c 0,2-63 .venv/bin/python systems/rfdetr/scripts/freeze_new_class_datasets.py --check

# 2. rebuild the negatives training view (train-only change; test stays frozen)
taskset -c 0,2-63 .venv/bin/python systems/rfdetr/scripts/build_rfdetr_negatives_view.py --overwrite
taskset -c 0,2-63 .venv/bin/python systems/rfdetr/scripts/build_rfdetr_negatives_view.py \
  --out-suffix bcd_20260725_neg2x_test_as_valid --negative-repeat 2 --overwrite

# 3. train (ブレース on the neg2x view, 柱脚 on the neg view - their measured optima)
systems/rfdetr/scripts/run_new_classes_negatives.sh      # 柱脚 config, 31% negatives
systems/rfdetr/scripts/run_new_classes_negatives_v2.sh   # ブレース neg2x, 33% negatives

# 4. score at the delivery thresholds
taskset -c 0,2-63 .venv/bin/python scripts/evaluate_rfdetr_class_threshold_grid.py \
  --checkpoint <ckpt> --dataset-dir data/rfdetr_brace_bcd_20260725_test_as_valid \
  --split test --threshold-grid 0.20,0.25,0.30 --iou-threshold 0.229 --num-classes 3 \
  --output-csv /tmp/check.csv
```

Determinism caveats: the split is fixed by content digest and reproduces exactly.
Training does not - seed 20260602 is set but cuDNN nondeterminism and the
`batch_size: auto` probe are not pinned, so a rebuild lands within roughly one
noise band (F1 +-0.03 on these test sizes) rather than on the same number.

## Known limitations, carried forward

1. **Small test sets.** 83 and 72 boxes; one true positive is worth 0.012-0.014
   recall, so F1 differences below about 0.03 are not individually resolvable. The
   negatives result rests on the whole PR curve moving in both categories plus a
   monotonic dose-response, not on a single point.
2. **No false-alarm measurement.** The frozen test split contains only images that
   carry damage, because the 141 zero-box images were excluded from train and test
   alike. So "does the model stay quiet on a photograph of a sound element" - the
   common production case and the direct expression of the shortcut - is unmeasured.
   Closing this needs a held-out negative benchmark reported alongside, not merged
   into the frozen split.
3. **Untriaged negatives.** All 141 were used on the 2026-07-26 decision to proceed
   without annotator triage. A minority carry real but unannotated damage
   (`f-00189` exposed rebar, `f-00322` spalled concrete, `f-00203` corroded base),
   which trains suppression of genuine damage. `--max-audit-score` drops the most
   suspicious without a rebuild.
4. **Per-grade weak spots differ.** ブレース is weakest on C (0.578 at the delivery
   point); 柱脚 on D (0.400, from only 10 test D boxes, so each box moves it 0.1 -
   indicative only).
5. **Delivered band is not a like-for-like comparison.** The 0.72-0.86 F1 band was
   selected on ~32-box splits under the same valid==test convention, so it carries
   optimism of the same kind. Whether our gap is as large as it looks would need the
   delivered categories re-measured under this protocol, which needs their data
   restored - it is absent from this checkout.

## Next, in priority order

1. **Ask the client for photographs of undamaged braces and column bases.** The
   dose-response says negatives work and that repetition of the existing 141 has
   saturated; only new images extend the curve. These need no damage grading, only
   the assertion that the element is sound, so they are far cheaper than ordinary
   annotation. This is the single highest-value action available.
2. **Triage the 141.** Contact sheets for this are in
   `outputs/rfdetr_new_classes/empty_label_audit/` and are the artifact to send.
3. **Build the held-out false-alarm benchmark** (limitation 2).
4. **Retry crops without the confound** - crop positives paired with crop negatives,
   evaluated with tiled inference, so training and inference share framing.

Not recommended: further loss-function, learning-rate, encoder-freezing or
augmentation work. Six such attempts produced nothing.
