# 2026-08-24 Router incremental training with sound-element images

## Decision

Keep the existing production Router baseline. The one-epoch incremental model
improves recall on the new Gemini-labelled brace domain, but regresses the frozen
delivery test. Neither the full fine-tune nor a conservative weight interpolation
meets the existing production operating point.

Production baseline:

```text
models/rfdetr/router_5class/selected_precision_p090_classwise_epoch004_brace_balanced_v2.pth
SHA256 48486312670c2f09343254176ea79f2364e77210e8cccd2097acf5b9282c81b6
```

## New source and annotation

Source archive: `20260807_学習用データ_ブレース,柱脚_無損傷.zip`.

For the Router, sound images are positive element examples, not empty-label
negatives. Exact SHA256 deduplication leaves 456 images from the supplied 497:

| class | unique | exact duplicates |
|---|---:|---:|
| ブレース | 427 | 40 |
| 柱脚 | 29 | 1 |

Gemini `gemini-3.1-pro-preview` annotated all 456 images through the Interactions
API. A general first pass omitted the expected class on 177 images, so those were
retried with an expected-class-focused prompt. The focused pass recovered 173;
four images were explicitly judged not to contain a structural brace and were
excluded instead of receiving fabricated boxes.

Final annotation artifacts:

```text
outputs/gemini_router_20260807/final_results.jsonl
outputs/gemini_router_20260807/excluded_no_expected_box.json
outputs/gemini_router_20260807/train_contact_sheet.jpg
outputs/gemini_router_20260807/holdout_contact_sheet.jpg
```

## Dataset

Fourteen labelled images were exact duplicates of images already in the baseline
dataset and were excluded. New images were split by ten-image filename sequence
groups so adjacent frames do not cross train/holdout boundaries.

```text
train view: data/rfdetr_router_5class_20260824_gemini_incremental
new holdout: data/rfdetr_router_5class_20260824_gemini_holdout
```

| split | brace images | brace boxes | column-base images | column-base boxes |
|---|---:|---:|---:|---:|
| appended train | 306 | 754 | 19 | 20 |
| new holdout | 104 | 269 | 9 | 10 |

The frozen baseline valid/test remains byte-linked and unchanged: 417 images,
752 boxes per split. Dataset preflight found no missing/orphan/empty/malformed
labels.

## Training

The production baseline was fine-tuned for one epoch with RF-DETR Medium:

```text
effective batch: 28 (14 x accumulation 2)
decoder lr: 1e-5
encoder lr: 1e-5
precision: 16-mixed
seed: 20260602
```

Checkpoint:

```text
outputs/rfdetr_router/medium_5class_20260824_gemini_incremental_ft/epoch_pth/checkpoint_epoch_000.pth
SHA256 2c9d7a4daef1fb36433a6455dec2b25cd09cd74e34806ab6318ef9bc4e1ae2bc
```

## Evaluation

All table values use the original production thresholds
`0.90/0.66/0.76/0.34/0.52` and IoU 0.50.

### Frozen delivery test

| model | precision | recall | F1 | brace recall | column-base recall |
|---|---:|---:|---:|---:|---:|
| production baseline | **0.9003** | **0.7327** | **0.8079** | **0.7442** | **0.8485** |
| incremental epoch 0 | 0.8742 | 0.6928 | 0.7730 | 0.6744 | 0.7273 |
| 25% interpolated | 0.8902 | 0.7221 | 0.7974 | 0.6977 | **0.8485** |

Recalibrating the incremental checkpoint under the original overall precision
constraint does not close the gap. Its best recall at precision >= 0.90 is
0.7114, below the baseline 0.7327.

### New Gemini-labelled holdout

| model | precision | recall | F1 | brace recall | column-base recall |
|---|---:|---:|---:|---:|---:|
| production baseline | 0.3455 | 0.1362 | 0.1954 | 0.1115 | **0.8000** |
| incremental epoch 0 | **0.4264** | **0.1971** | **0.2696** | **0.1822** | 0.6000 |
| 25% interpolated | 0.3661 | 0.1470 | 0.2097 | 0.1301 | 0.6000 |

The new brace domain improves, but not enough to justify the frozen-set
regression. The new annotation is also substantially denser: it treats multiple
thin braces in one image as separate instances, whereas the historical Router
often uses a larger coarse structural-region box. Before another training round,
the new labels should be manually reconciled to the historical Router box
granularity. More epochs on the current labels are not recommended.

## Strict old-class freeze experiment

An attempted classifier-row-only RF-DETR training run was rejected after a
parameter-level audit. RF-DETR rebuilt the Lightning model after the initial
freeze hook, and 508 of 509 model tensors changed. Its metrics therefore cannot
be treated as a frozen-old-class result.

To enforce the constraint independently of the trainer, four checkpoints were
rebuilt from the production baseline. Only rows 3 and 4 of all classifier heads
were interpolated from the trained checkpoint; every shared tensor, rows 0-2,
and the background row remained byte-identical to the baseline. Interpolation
strengths of 2%, 5%, 10%, and 15% were evaluated.

| strength | frozen P | frozen R | frozen F1 | brace TP/FP | column-base TP/FP | new brace TP/FP | new column-base TP/FP |
|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.9003 | 0.7327 | 0.8079 | 32/8 | 28/0 | 30/57 | 8/1 |
| 2% | 0.9002 | 0.7314 | 0.8070 | 31/8 | 28/0 | 30/58 | 8/1 |
| 5% | 0.9015 | 0.7301 | 0.8068 | 31/7 | 27/0 | 30/60 | 8/1 |
| 10% | 0.8997 | 0.7274 | 0.8044 | 30/8 | 27/0 | 30/60 | 8/1 |
| 15% | 0.9016 | 0.7314 | 0.8076 | 31/7 | 28/0 | 31/59 | 8/1 |

No candidate dominates the baseline. In particular, the new column-base
holdout is invariant at 8 TP, 1 FP, and 2 FN for all candidates. The frozen
delivery set already has column-base precision 1.0, so a strict improvement in
all three column-base metrics is mathematically impossible on that set.

Decision: retain the production baseline. A defensible second attempt requires
additional diverse column-base examples and manual label reconciliation; the
current 19-image column-base training addition does not support the requested
two-class Pareto improvement.

## Shared-parameter incremental adaptation candidate

To measure the expected shared-feature tradeoff, a second experiment trained on
only the 306 new brace images and 19 new column-base images. Column-base images
were repeated 16 times (610 training appearances total), and all shared RF-DETR
parameters were updated for one epoch at `1e-5`. A 10% linear interpolation from
the production baseline to this checkpoint was selected to keep the adaptation
small.

Selected checkpoint:

```text
models/rfdetr/router_5class/router_5class_incremental_balanced_shared_ft_a010_20260824.pth
SHA256 b9f915217e385846ec3501828357113dcb865f3347f10a16b36a1b998db38c7e
thresholds 0.90/0.66/0.76/0.40/0.52
```

At the calibrated thresholds, the new holdout improves slightly as a whole:

| model | precision | recall | F1 | brace P/R/F1 | column-base P/R/F1 |
|---|---:|---:|---:|---:|---:|
| baseline | 0.3636 | 0.1290 | 0.1905 | 0.3590/0.1041/0.1614 | 0.8889/0.8000/0.8421 |
| shared FT 10% | **0.3700** | **0.1326** | **0.1953** | **0.3625/0.1078/0.1662** | 0.8889/0.8000/0.8421 |

The fixed 417-image delivery test moves in the opposite direction. Precision is
kept above the hard 0.90 gate by calibration, while recall and F1 decline:

| model | precision | recall | F1 |
|---|---:|---:|---:|
| production baseline | 0.9003 | 0.7327 | 0.8079 |
| shared FT 10% | 0.9008 | 0.7247 | 0.8032 |

This candidate supports the shared-parameter interference story at the aggregate
new-domain level. It does not support a claim that column-base improved by
itself: column-base is unchanged on the independent new holdout. Keep the
production baseline as the deployment default unless that distinction is
acceptable for the intended report.
