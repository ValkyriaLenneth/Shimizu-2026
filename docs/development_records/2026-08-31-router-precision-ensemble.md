# 2026-08-31 Router per-class precision ensemble

## Goal

Raise every five-class Router category to precision > 0.90. Recall may be
traded away, but the selected point should preserve as much recall as possible.

## Protocol

- Frozen 417-image delivery test, 752 boxes, match IoU 0.50.
- Production verification uses OpenCV decoding, including JPEG EXIF orientation,
  because the pipeline receives BGR ndarrays from `cv2.imread`.
- Production five-class checkpoint remains the source of every output box.
- The historical three-class checkpoint may only confirm `天井`, `壁类`, and
  `RC柱`; the 2026-06-30 five-class checkpoint confirms `ブレース`. Neither
  confirmation model can introduce a box by itself.
- Search compares a five-class threshold, strict/split confirmation gates, and
  weighted confidence blending.
- Candidate selection is constrained per class, not by aggregate precision.

Search implementation:

```text
systems/rfdetr/scripts/search_router_precision_ensemble.py
```

Machine-readable operating point:

```text
systems/rfdetr/router/configs/router_5class_precision_ensemble_20260831.yaml
```

## Selected point

The selected point maximizes recall subject to every class remaining above the
literal 0.90 precision requirement. This is the preferred point after the user
clarified that a large recall reduction is unacceptable.

| class | same-path baseline P | selected P | same-path baseline R | selected R | selected TP/FP |
|---|---:|---:|---:|---:|---:|
| 天井 | 0.9592 | 0.9032 | 0.7705 | 0.9180 | 168 / 18 |
| 壁类 | 0.8882 | 0.9195 | 0.7315 | 0.7596 | 297 / 26 |
| RC柱 | 0.9048 | 0.9053 | 0.7451 | 0.8431 | 86 / 9 |
| ブレース | 0.8000 | 0.9688 | 0.7442 | 0.7209 | 31 / 1 |
| 柱脚 | 1.0000 | 1.0000 | 0.8485 | 0.8485 | 28 / 0 |
| overall | 0.9066 | 0.9187 | 0.7487 | 0.8112 | 610 / 54 |

Compared with the same OpenCV production path, the selected point adds 47 true
positives, removes 4 false positives, and reduces false negatives by 47.
`ブレース` keeps its
primary threshold at 0.34; confirmation by the older five-class checkpoint
removes seven false positives while losing only one true positive.

## Production integration

The ensemble is implemented in the RF-DETR production pipeline rather than
remaining an offline score table:

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/rfdetr_router_infer.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_precision_ensemble.yaml
systems/rfdetr/scripts/verify_router_precision_ensemble_pipeline.py
```

The primary model runs on `cuda:0`; the three-class and brace confirmation
models live on `cuda:1`. Primary and three-class inference run concurrently.
The brace model is invoked only when a primary brace candidate is below the
bypass score and actually needs confirmation. Confirmation models cannot add
boxes, and the ensemble has no low-confidence fallback that can bypass its
precision gates.

Full 417-image pipeline verification produced the selected metrics exactly.
Lazy brace confirmation reduced mean latency from 46.7 ms to 34.3 ms and p50
from 25.8 ms to 17.4 ms without changing any prediction. The same-path single
model measured 23.5 ms mean and 9.2 ms p50. These are model-stage timings on the
two RTX 5090 GPUs, excluding downstream damage models.

Verification command:

```bash
python systems/rfdetr/scripts/verify_router_precision_ensemble_pipeline.py \
  --pipeline-config systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_precision_ensemble.yaml \
  --dataset-dir handoff_20260707_rfdetr_main/data/router_5class_reviewed_dedup_test_as_valid \
  --split test \
  --output-json outputs/router_precision_20260831/pipeline_ensemble_verification_balanced_lazy_cv2.json
```

An audit found that the original offline search used Pillow dimensions while
production uses OpenCV, which applies JPEG EXIF orientation. The search tool now
records the decoder and both model devices in cache metadata and uses OpenCV
dimensions for GT conversion. The production-path figures above supersede the
earlier Pillow-only operating-point figures.

## Alternative points

| point | overall precision | overall recall | overall F1 |
|---|---:|---:|---:|
| same-path single model | 0.9066 | 0.7487 | 0.8201 |
| high-margin ensemble | 0.9263 | 0.8019 | 0.8596 |
| **balanced selected** | **0.9187** | **0.8112** | **0.8616** |
| literal max-recall | 0.9057 | 0.8178 | 0.8595 |

The literal max-recall point puts wall precision at 0.9012 and column-base at
0.9063. The selected point keeps more margin and has the highest overall F1;
only ceiling and RC-column remain close to the requirement.

## Limitation

The project intentionally mirrors `test` into `valid`; therefore this search
and the reported result use the same frozen set. The result proves the operating
point on that delivery set, not a 0.90 lower bound on unseen data. Run an
independent acceptance set before replacing production.

A 10,000-resample image-level bootstrap gives the following descriptive 95%
intervals for the selected point. These are uncertainty diagnostics, not a
correction for tuning and testing on the same images.

| class | precision 95% interval | recall 95% interval |
|---|---:|---:|
| 天井 | 0.8629-0.9430 | 0.8681-0.9607 |
| 壁类 | 0.8889-0.9481 | 0.7124-0.8059 |
| RC柱 | 0.8409-0.9596 | 0.7714-0.9101 |
| ブレース | 0.8947-1.0000 | 0.5957-0.8485 |
| 柱脚 | 1.0000-1.0000 | 0.7143-0.9643 |

The wide intervals are why an independent acceptance set remains required;
they also show that the selected point has limited unseen-data precision margin.

## Sound-data stress test

The restored 2026-08-07 sound-element annotations were rebuilt with the
original grouped split: 325 new training images and an independent 113-image
holdout. The holdout contains 269 `ブレース` boxes and 10 `柱脚` boxes.

This is not an annotation-compatible replacement for the delivery test. The
new labels contain 2.59 brace boxes per positive image and have median normalized
box area 0.080. The historical frozen set contains 1.19 brace boxes per positive
image with median area 0.423. The new set usually labels individual thin members;
the historical Router usually labels a coarse structural region.

At the balanced frozen-test operating point, the incompatible holdout scores are:

| class | TP / FP / FN | precision | recall |
|---|---:|---:|---:|
| ブレース | 27 / 43 / 242 | 0.3857 | 0.1004 |
| 柱脚 | 8 / 1 / 2 | 0.8889 | 0.8000 |

A strict point was tested and rejected. It can clear precision 0.90 on both
datasets, but only at substantial recall cost:

| class | strict rule | frozen P / R | sound holdout P / R |
|---|---|---:|---:|
| ブレース | primary 0.38, confirmation 0.65, IoU 0.525, bypass 0.98 | 1.0000 / 0.3953 | 0.9048 / 0.0706 |
| 柱脚 | primary threshold 0.96 | 1.0000 / 0.3939 | 1.0000 / 0.3000 |

The sound holdout was used to select this strict point and is therefore a
development set from this point onward. It demonstrates a conservative fallback,
not unseen-data acceptance. A fresh manually reconciled set is still required.

## Rejected sound-data calibration

Direct detector fine-tuning had already regressed the frozen delivery set. A
second experiment used only the 325 new training images to fit candidate-box
calibrators from primary confidence, historical-model support, overlap, and box
geometry. Grouped out-of-fold precision did not transfer across annotation
domains: the brace logistic calibrator fell to 0.8889 precision on the frozen
set, and column-base calibrators were worse. Training-derived confirmation gates
also missed 0.90 on the independent sound holdout. These candidates are rejected.
