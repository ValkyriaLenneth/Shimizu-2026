# 2026-07-25 ブレース / 柱脚 downstream training plan

## Scope

Two new downstream B/C/D recognition models, one per element category, matching
the existing architecture. The router is **not** retrained: `ブレース` and
`柱脚` already exist as router element classes from the 2026-07-07 reviewed and
deduplicated update.

```text
RF-DETR router (5 classes, unchanged)
  -> 天井 / 内壁 / RC壁 / RC柱 / ブレース / 柱脚 downstream B/C/D models
```

## Client Data Decisions (2026-07-25)

| decision | effect |
|---|---|
| keep only images with B/C/D boxes | 59 ブレース + 82 柱脚 empty-label images dropped |
| every training image contains damage | no background negatives in either dataset |
| grade contradictions: document, then take max | duplicate representative = copy with most boxes |
| no separate valid split | train/test only; valid mirrors test for the RF-DETR loader |
| one model per category | no joint brace+column_base model |

The 11 grade contradictions are recorded in
`docs/development_records/2026-07-25-new-classes-annotation-match.md`. Taking the
max-box copy is what the matching step already elects, so the datasets implement
this decision directly.

## Datasets

Built by:

```bash
.venv/bin/python systems/rfdetr/scripts/build_rfdetr_new_class_downstream_datasets.py --overwrite
```

Layout is the standard RF-DETR YOLO view (`dataset_file: yolo`):

```text
data/rfdetr_brace_bcd_20260725_test_as_valid/{train,valid,test}/{images,labels}
data/rfdetr_column_base_bcd_20260725_test_as_valid/{train,valid,test}/{images,labels}
```

Split policy: unit is `scene_group_id`, not the individual file. Near-identical
views of one scene therefore cannot straddle train and test. Groups are
stratified by the rarest grade they contain, which is what keeps D present in
test. Test ratio is 0.2 rather than the project's usual 0.1: at 0.1 the 柱脚
test split would hold about 4 D boxes, which cannot support a per-class recall
number.

### ブレース

| split | images | boxes | B | C | D | scene groups |
|---|---:|---:|---:|---:|---:|---:|
| train | 235 | 394 | 124 | 194 | 76 | 226 |
| test | 58 | 83 | 22 | 45 | 16 | 57 |
| valid | 58 | 83 | 22 | 45 | 16 | 57 |

### 柱脚

| split | images | boxes | B | C | D | scene groups |
|---|---:|---:|---:|---:|---:|---:|
| train | 179 | 248 | 152 | 67 | 29 | 172 |
| test | 45 | 72 | 47 | 15 | 10 | 44 |
| valid | 45 | 72 | 47 | 15 | 10 | 44 |

Preflight is clean for both, on every split: no missing labels, no orphan
labels, no empty labels, no malformed lines.

```bash
python systems/rfdetr/scripts/check_rfdetr_router_dataset.py \
  --dataset-dir data/rfdetr_brace_bcd_20260725_test_as_valid \
  --write-summary outputs/rfdetr_new_classes/brace_dataset_check.json
```

## Environment

| component | state |
|---|---|
| interpreter | `/workspace/Shimizu-2026/.venv/bin/python` (Python 3.12) |
| torch | 2.9.1+cu128 |
| rfdetr | 1.7.1 with `[train]` extras |
| GPUs | 2x RTX 5090, 32 GB each, sm_120 (the docs' 4090 is superseded) |

### CPU 1 is unschedulable and hangs every torch import

This cost most of the setup time and will bite anyone who uses this host, so it
is worth stating precisely.

`import torch` hung forever. The process sat in state `R`, consumed 0.4 s of CPU
over 28 minutes, and could not be killed - `SIGKILL` stayed pending and
undelivered.

Diagnosis path:

```text
faulthandler   -> hang is inside dlopen at torch/__init__.py:442
LD_DEBUG=libs  -> every .so loads fine; hang is inside _C's module init
dd             -> libtorch_cuda.so reads at 9 GB/s, so not I/O
sandbox off    -> identical hang, so not the sandbox
same cgroup as a working shell, 278 GB RAM free, both GPUs idle
strace         -> last syscalls are sched_setaffinity(0, 8, [0]) = 0
                  then clock_nanosleep, then sched_setaffinity(0, 8, [1]) - stop
```

torch's `cpuinfo` topology probe pins itself to each CPU in its affinity mask in
turn. CPU 1 on this host is advertised as online (`/sys/devices/system/cpu/online`
reports `0-63`, `offline` is empty, the cpuset allows `0-63`) but nothing pinned
to it ever gets scheduled. `taskset -c 1 /bin/true` hangs the same way. A sweep
of all 64 CPUs found exactly one bad core:

```text
GOOD (63): 0 2 3 4 ... 63
BAD  (1):  1
```

So this was never a torch bug. Any process pinned to CPU 1 wedges, and torch
just happens to pin itself to every core during init. The fix is an affinity mask
that excludes CPU 1:

```bash
taskset -c 0,2-63 .venv/bin/python ...
```

With that mask, `import torch` completes in 1.3 s and both 5090s pass a matmul.
The launcher applies the mask itself and refuses to start if the mask is
unusable. `CPU_LIST` overrides it; drop the mask only after `taskset -c 1
/bin/true` returns.

One hypothesis was wrong along the way and is recorded so it is not retried:
torch 2.11 is the first release shipping `libtorch_nvshmem.so`, and NVSHMEM
probing for absent InfiniBand looked like a plausible cause. It was not - torch
2.9.1 hung identically, and the original torch 2.11 in `/venv/main` works fine
once CPU 1 is excluded. The clean `.venv` is kept because it is writable and
self-contained, not because 2.11 was broken.

## Audit Of The Four Existing Categories

Before training the new categories, the existing downstream recipe was audited so
the new models are built the same way rather than from the config template.

Every documented downstream training command in this project uses
`--experiment medium`. `small` has never been used - it is an unused branch in
the config template. All four released checkpoints come from the `*_medium_*`
runs:

| category | released checkpoint | epoch |
|---|---|---:|
| 天井 | `tenjo_standard_orig_checkpoint_epoch_009.pth` | 9 |
| 内壁 | `inner_wall_checkpoint_epoch_026.pth` | 26 |
| RC壁 | `rc_wall_checkpoint_epoch_009.pth` | 9 |
| RC柱 | `checkpoint_epoch_047.pth` | 47 |

The selected epochs are 9, 26, 9 and 47, so useful checkpoints appear early and
running to the epoch ceiling is not the point.

The baseline recipe is identical across 天井 / 内壁 / RC壁 / RC柱:

| setting | value |
|---|---|
| model | RFDETRMedium |
| epochs | 80 |
| batch size | 28 |
| grad accum | 1 |
| lr | 1e-4 |
| precision | 16-mixed |
| resolution | default, not overridden |
| valid | = test (official split used as both) |
| save_epoch_pth | yes |
| test_each_epoch | no |
| seed | 20260602 |

Two procedural steps matter and were initially missing from this plan:

1. **The automatic best checkpoint is not the one to ship.**
   `rfdetr_single_crack_training_20260602.md` states that
   `checkpoint_best_total.pth` "was selected by mAP, not recall-first business
   targets". Selection instead runs `sweep_rfdetr_router_test.py` over the saved
   epoch checkpoints, reloading each and force-evaluating it on the official test
   split. For RC柱 this is how epoch 47 (recall 0.826) was found to beat
   `best_total` (recall 0.796, C below target).
2. **The delivered client numbers are per-class threshold-tuned at match IoU
   0.229**, not single-threshold outputs - 天井 used B/C/D = 0.25/0.35/0.35
   (0.20/0.35/0.35 for the recall-priority point), RC壁 used 0.28/0.45/0.25. Only
   `evaluate_rfdetr_class_threshold_grid.py --iou-threshold 0.229` reproduces that
   protocol.

Delivered reference numbers, which the new categories are compared against:

| 部材 | Precision | Recall | B Recall | C Recall | D Recall |
|---|---:|---:|---:|---:|---:|
| 天井 | 0.596 | 0.875 | 0.818 | 0.917 | 0.889 |
| RC壁 | 0.722 | 0.812 | 0.857 | 0.600 | 1.000 |
| 内壁 | 0.824 | 0.848 | 0.750 | 1.000 | 0.889 |
| RC柱 | 0.661 | 0.826 | 0.750 | 0.727 | 1.000 |

Three of these four rows match documented rows exactly (内壁 = the
precision-priority point, RC壁 = the 2026-06-16 optimized checkpoint, RC柱 =
epoch 47). 天井 differs slightly from the recorded 0.614 / 0.844, so its shipped
point comes from a later threshold re-tune.

Recall of the delivered models spans 0.812-0.875 and precision 0.596-0.824, which
is what makes recall >= 0.80 with a 0.60 precision floor the right bar for the two
new categories.

## Training Setup

One `medium` run per category, the two categories concurrently on separate GPUs:

```text
GPU 0: brace medium
GPU 1: column_base medium
```

Settings are the audited baseline recipe verbatim - RFDETRMedium, 80 epochs,
batch 28, grad accum 1, lr 1e-4, 16-mixed, default resolution. Batch 28 also
fills the 32 GB cards, reaching about 31 GB and 99-100% utilization.

Resolution is deliberately left at the default. Raising it to 896 was already
tried on this project and is recorded as not helping:
`tenjo_medium_res896_bd_os3_test_v1` "does not solve B or overall",
`tenjo_medium_best_e006_res896_ft_lr2e5` "no meaningful gain", and the RC壁
handoff concludes that "resolution increase ... did not exceed the retained
best".

Configs:

```text
systems/rfdetr/recognition_models/brace/configs/rfdetr_brace_baseline.yaml
systems/rfdetr/recognition_models/column_base/configs/rfdetr_column_base_baseline.yaml
```

Launcher, which also runs both post-training evaluation steps per category:

```bash
mkdir -p outputs/rfdetr_new_classes/logs
systems/rfdetr/scripts/run_new_classes_baseline_comparison.sh
```

No `external_eval_profiles` are configured. The 内壁 / RC壁 / RC柱 baselines carry
none; only the 天井 baseline defines them, and those pin `device: cpu` and run
after every epoch. Inheriting them cost about 22 minutes per epoch here with both
GPUs idle. Removing them brought epoch time to roughly 35 s at batch 28, so an
80-epoch run takes about 45 minutes and both categories finish together.

Progress and selection view:

```bash
python systems/rfdetr/scripts/report_new_class_training_status.py
```

It prefers `test_results.csv` from the sweep when present, falls back to
per-epoch `val` metrics during training, flags a stalled run, and with
`--list-top-checkpoints RUN_DIR` emits the top recall candidates for the
threshold grid.

Dataset QA sheets, rendered from the built datasets so a misplaced box would mean
the pairing is wrong rather than the model:

```bash
python systems/rfdetr/scripts/visualize_new_class_dataset_samples.py \
  --dataset-dir data/rfdetr_brace_bcd_20260725_test_as_valid \
  --split train --limit 18 --output outputs/rfdetr_new_classes/brace_train_samples.jpg
```

Both sheets were reviewed. Boxes sit on the damage and the grades are visually
coherent: for 柱脚, D is exposed rebar and severe spalling, C is cracking or
partial spalling, B is a hairline crack; for ブレース, D is a buckled or
fractured member, C is a deformed member, B is bolt and connection damage. The
ブレース images are noticeably wider scene shots than 柱脚, which is consistent
with the small relative box areas noted above.

## Selection And Reporting

Primary metric is `test/recall`, recall-first, consistent with the other
downstream models. Guardrails are mAP@50, mAP@50-95, precision, per-class B/C/D
recall, and false negative count. Target is overall recall >= 0.80 with precision
not falling below 0.60.

Selection order:

1. train to 80 epochs, or stop early once recall plateaus
2. `sweep_rfdetr_router_test.py` over all saved epoch checkpoints
3. `evaluate_rfdetr_class_threshold_grid.py --iou-threshold 0.229` on the top
   recall candidates, to land per-class B/C/D thresholds
4. pick recall-first, and report the per-class thresholds with the numbers

One caveat travels with every number: **柱脚 B-vs-C is partly a label question.**
6 of the 11 documented grade contradictions are 柱脚 B-vs-C pairs on identical
pixels, so B/C confusion may reflect the grading criterion rather than the model.

An earlier concern in this plan - that the new test splits are too small - was
overstated and is withdrawn. RC柱 shipped on a 31-image test split and reported
D recall 1.000 from 8 D boxes; RC壁 test is 126 images. At 58 and 45 images, the
new splits sit inside the project's normal range. What remains is simply to
report the sample size next to per-class recall.

## Expected Bottleneck And Next Levers

柱脚 grade D has 29 training boxes across 26 images. If recall falls short of
0.80, apply in this order:

1. per-class threshold tuning - the cheapest lever, and the one that produced the
   delivered numbers for all four existing categories
2. oversample D-bearing images (`build_rfdetr_oversampled_view.py` pattern)
3. low-lr fine-tune from the best epoch, which is how RC壁 reached its shipped
   point (lr 2e-6, batch 16, grad accum 2, 5 epochs)
4. reuse the light augmentation presets
   (`aug_tenjo_light_geo` / `aug_tenjo_light_pixel`)
5. initialize from the released `rc_column` checkpoint rather than COCO - 柱脚 is
   the base of an RC column and the closest existing domain. This needs the
   release archive restored; `.local_artifacts/` is empty in this checkout, so no
   project checkpoint is available locally.

A resolution increase is explicitly *not* on this list, per the audit above.

## Still Outstanding With The Client

The 45 unlabelled `20260724` images (minimum side 127-247 px) and the 3 images
filed under both categories remain unresolved. Neither blocks this training
round; they are excluded from both datasets.
