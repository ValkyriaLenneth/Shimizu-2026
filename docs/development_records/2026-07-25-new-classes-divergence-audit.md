# 2026-07-25 ブレース / 柱脚 divergence audit and diagnostics

Purpose: confirm the two new categories are being trained the same way as the
four delivered ones, and characterise why overall performance is short. Written
after the first baseline results came in well below the delivered band.

## Divergence Audit

Every item checked against the 天井 / 内壁 / RC壁 / RC柱 recipe.

| item | delivered categories | new categories | verdict |
|---|---|---|---|
| model | RFDETRMedium | RFDETRMedium | match |
| epochs / batch / grad accum | 80 / 28 / 1 | 80 / 28 / 1 | match |
| lr / precision | 1e-4 / 16-mixed | 1e-4 / 16-mixed | match |
| resolution | default, never overridden | default | match |
| valid split | mirrors the official test split | mirrors test | match |
| train:test ratio | ~9:1, fixed test stem list, valid folded into train | 9:1 | match after fix |
| empty labels in train | RC壁 preflight shows 0 | 0 | match |
| training / inference framing | full image | full image | match |
| selection protocol | epoch sweep, then per-class threshold grid at IoU 0.229 | same | match |

Two early deviations were found and corrected:

1. **8:2 split.** The first datasets used a 20% test split, chosen to stabilise
   per-class D recall. The project convention is a ~10% fixed test list with valid
   mirrored from it, so 12% of the scarcest resource - training data - was being
   given away. Rebuilt at 9:1 as
   `data/rfdetr_{brace,column_base}_bcd_20260725_split91_test_as_valid`.
2. **Inherited CPU external eval.** The 天井 baseline config is the only one of the
   four that defines `external_eval_profiles`, and those pin `device: cpu` and run
   after every epoch. Copying that config cost ~22 minutes per epoch with both
   GPUs idle. The 内壁 / RC壁 / RC柱 baselines define none; the new configs now
   define none either, and threshold work runs once after training.

### Full-image inference, confirmed

An earlier reading of `run_full_pipeline.py` suggested downstream models receive
router-cropped regions via `make_region_view`. That is the pre-2026-06-30 path.
`docs/2026-06-30/pipeline_rulebase_update.md` records the switch:

```text
region_transport: full_image_filter
region_filter_mode: center_or_ioa
region_filter_ioa_threshold: 0.50
```

and states plainly that the main path is now full-image-filter - the downstream
model infers on the whole image and the router region filters the results. So
full-image training matches production. `pipeline.rfdetr_prod.local.yaml` still
carries `ndarray_slice` and is the older configuration.

A corollary worth remembering when reading our numbers: in production, false
positives outside the router region are filtered out by the IoA >= 0.50 rule. Our
test evaluation scores the downstream model alone, with no router filter - which
is also how the four delivered models were evaluated, so the comparison is fair,
but the raw false-positive counts overstate end-to-end error.

## Where The Gap Actually Is

The delivered models' own numbers, from
`final_release_20260615/models/rfdetr/metrics/selected_thresholds.csv`:

| model | precision | recall | F1 | tp | fp | fn | GT boxes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 天井 | 0.650 | 0.812 | 0.722 | 26 | 14 | 6 | 32 |
| RC壁 | 0.722 | 0.812 | 0.765 | 26 | 10 | 6 | 32 |
| 内壁 | 0.811 | 0.909 | 0.857 | 30 | 7 | 3 | 33 |
| RC柱 | 0.661 | 0.826 | 0.735 | - | - | - | - |

Their test splits hold about 32 GT boxes - **fewer than ours** (ブレース 39,
柱脚 38). Test-set size is therefore not the explanation.

Our 9:1 baseline, same protocol:

| category | best F1 | best mAP50 | operating point |
|---|---:|---:|---|
| ブレース | 0.565 | 0.307 | ep29, thr 0.40/0.30/0.25, R 0.615 / P 0.522 |
| 柱脚 | 0.411 | 0.266 | ep17, thr 0.30/0.35/0.35, R 0.395 / P 0.429 |
| delivered band | 0.72 - 0.86 | 0.73 - 0.78 | |

F1 is short by 0.16 - 0.31, but **mAP50 is short by a factor of 2.4 - 2.7**. mAP is
threshold-free, so this is a statement about the detector's ranking quality, not
about where the operating point was placed. That is the number to move.

## Two Different Root Causes

The two categories do not fail for the same reason, which matters for choosing
levers.

Box relative-area distribution on the 9:1 train splits:

| category | boxes | p5 | p50 | p95 | p95/p5 | >25% of image | <1% of image |
|---|---:|---:|---:|---:|---:|---:|---:|
| ブレース | 438 | 0.0014 | 0.0532 | 0.4510 | **332x** | 74 | 104 |
| 柱脚 | 282 | 0.0034 | 0.0366 | 0.2967 | 88x | 21 | 58 |

Median box area by grade:

| category | B | C | D |
|---|---:|---:|---:|
| ブレース | 0.0076 | 0.0648 | **0.1636** |
| 柱脚 | 0.0235 | 0.1000 | 0.0676 |

**ブレース is a scale problem.** Box area spans 332x, and area tracks grade almost
monotonically - a D brace is buckled or fractured and fills the frame, a B is a
bolt or hairline crack at 0.76% of the image. A single-resolution detector has to
cover an 18x linear scale range. The images are also wide scene shots of trusses
and ceilings rather than element close-ups.

**柱脚 is a volume problem.** Its scale span is a much tamer 88x, yet it scores
worse, on 203 train images with only 35 D boxes.

This also puts our results in context rather than making them anomalous: 天井's
chronic failure in this project was its B class, recorded as "B stuck at 0.4545"
across oversampling, crop and resolution attempts. B is the smallest damage and
the hardest everywhere. We are seeing the project's known failure mode, amplified
by having a third of the data.

## Levers, In Order

1. **Crop augmentation** - `build_rfdetr_crop_aug_view.py`, crops `context`x around
   each box with jittered variants, train-only. Addresses both root causes at once:
   normalises scale and multiplies data.

   | | train images | boxes | B | C | D |
   |---|---:|---:|---:|---:|---:|
   | ブレース base | 264 | 438 | 133 | 221 | 84 |
   | ブレース crop2 | 1140 | 2412 | 818 | 1268 | 326 |
   | 柱脚 base | 203 | 282 | 173 | 74 | 35 |
   | 柱脚 crop2 | 767 | 1152 | 717 | 298 | 137 |

   Early in-training signal was positive on exactly the target metric - best val
   mAP50 0.372 for ブレース against a baseline whole-run best of 0.315, and 0.308
   for 柱脚 against 0.252.

2. **Joint pretrain, per-category fine-tune** - `build_rfdetr_joint_bcd_view.py`
   merges both crop2 corpora into 1907 train images / 3564 boxes (B 1535, C 1566,
   D 463), which exceeds RC壁's 966 images / 1235 boxes and has 2.6x its D boxes.
   Pretrain one model on the union, then fine-tune two per-category models from it.
   Deployment shape is unchanged - still one model per category; only the
   initialization is shared. Legitimate because B/C/D grading semantics are shared
   across elements.

3. **D oversampling** for 柱脚 - available, but the project's record is discouraging:
   repeated oversampling for 天井 B and RC壁 C never moved those classes.

4. **Low-lr fine-tune from the best epoch**, which is how RC壁 reached its shipped
   point (lr 2e-6, batch 16, grad accum 2, 5 epochs).

5. **More annotation** - the 141 dropped empty-label images and 45 unlabelled ones.
   Still the root fix, still needs the client.

Explicitly not pursued: resolution increase (tried for 天井 and RC壁, recorded as
no meaningful gain) and a larger backbone (RFDETRLarge was level or slightly worse
in early epochs and also brings resolution 864 automatically).

## Operational Notes For This Host

* CPU 1 is advertised online but is not schedulable. Anything pinned to it wedges
  in state R and cannot be killed, including `/bin/true`. torch's cpuinfo probe
  pins itself to every core during init, so a bare `import torch` hangs forever.
  Every command must run under `taskset -c 0,2-63`.
* `checkpoint_interval: 1` in the baseline configs makes RF-DETR write a 511 MB
  Lightning `.ckpt` every epoch - about 41 GB per 80-epoch run. Four runs filled
  the 199 GB disk and the sweep failed with "checkpoint_epoch_043.pth cannot be
  opened". Pass `--checkpoint-interval 999`, as the project's own overnight scripts
  do. Only `epoch_pth/*.pth` (~134 MB) is needed for evaluation.
* `evaluate_rfdetr_class_threshold_grid.py` called `match_counts()` with three
  arguments after that function gained a required `num_classes` parameter. Every
  per-class threshold grid raised `TypeError` before writing a row. Fixed at the
  call site; this also affected the `class_threshold_grid` external eval profiles
  in the 天井 and RC壁 report-finetune configs.
* `--experiment` was hardcoded to `choices=["small", "medium"]`, which made every
  other key in a config's `experiments:` block unreachable - including `large` and
  the `alt` experiment already present in the RC壁 report-finetune config. Now
  validated against the config instead, so a typo still fails loudly.
