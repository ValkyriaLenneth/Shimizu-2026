# RF-DETR router 5-class incremental handoff 2026-06-30

## Scope

This handoff contains the 2026-06-30 incremental router work:

- Gemini annotation outputs for the two new router classes.
- RF-DETR 5-class router dataset split.
- Selected 5-class router model checkpoint.
- Training config, helper scripts, metrics, and development record.

The original gdown-downloaded new-class image archives and extracted raw image
directory are intentionally excluded. They are already preserved separately.

## Classes

```text
0: 天井
1: 壁类
2: RC柱
3: ブレース
4: 柱脚
```

## Dataset

Dataset path:

```text
data/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid
```

This is the historical Gemini-only 2026-06-30 dataset. After manual review and
deduplication on 2026-07-07, the next handoff/training should use:

```text
data/rfdetr_router_5class_brace_columnbase_20260707_reviewed_dedup_test_as_valid
```

See:

```text
docs/handoff_records/2026-07-07/router_5class_reviewed_dedup_handoff_20260707.md
```

RF-DETR requires a `valid` split during training. For this delivery, `valid`
mirrors `test`; there is no separate validation set. Training-time metrics
labelled as validation metrics are therefore test metrics.

Split summary:

| split | images | 天井 boxes | 壁类 boxes | RC柱 boxes | ブレース boxes | 柱脚 boxes |
|---|---:|---:|---:|---:|---:|---:|
| train | 4635 | 2128 | 4172 | 1800 | 472 | 252 |
| valid (= test) | 415 | 183 | 391 | 102 | 60 | 25 |
| test | 415 | 183 | 391 | 102 | 60 | 25 |

Build summary:

```text
data/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid/build_summary.json
```

## Gemini annotations

Annotation output:

```text
outputs/gemini_new_router_classes_20260630/results.jsonl
outputs/gemini_new_router_classes_20260630/sample_plan.jsonl
outputs/gemini_new_router_classes_20260630/summary.json
```

Summary:

```text
total: 919
ok: 899
errors: 20
model: gemini-3.1-pro-preview
api_mode: interactions
```

Failed invalid-key trial logs are not included in the handoff package.

## Selected model

Selected checkpoint:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid/selected_precision_p090_epoch049_thr069.pth
```

Use this confidence threshold:

```text
0.69
```

Overall test metrics at the selected operating point:

| precision | recall | F1 |
|---:|---:|---:|
| 0.9039 | 0.7293 | 0.8073 |

The target is overall precision >= 0.90. This target is met. It is not a
per-class precision guarantee.

Selection manifest:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid/selected_precision_p090_manifest.json
```

## Full local archive

The full incremental handoff archive is stored locally at:

```text
.local_artifacts/handoff_20260630/shimizu_20260630_rfdetr_router5_incremental.tar.zst
```

SHA256:

```text
21729a910a79eaa17e48dbf14b2d3d58c0511135eded949a39076642190e694c
```

This archive is about 4.3 GiB and is intentionally not committed to GitHub.
The Git commit contains the code, configs, documentation, lightweight metadata,
and package manifest required to identify and validate the archive.

## Config and scripts

Training config:

```text
systems/rfdetr/router/configs/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid.yaml
```

Relevant scripts:

```text
systems/gemini/scripts/annotate_new_router_classes_with_gemini.py
systems/rfdetr/scripts/build_rfdetr_router_5class_dataset.py
systems/rfdetr/scripts/evaluate_rfdetr_threshold_sweep.py
```

## Development record

Detailed record:

```text
docs/development_records/2026-06-30-rfdetr-router-5class-training.md
```

## Integration note

This package is intended as an incremental overlay on top of the 2026-06-15
release package. Extract it at the repository or release root so the relative
paths above remain unchanged.
