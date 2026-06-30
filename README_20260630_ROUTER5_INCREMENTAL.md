# 2026-06-30 RF-DETR router 5-class incremental package

This archive is an overlay package for the 2026-06-30 5-class RF-DETR router
update.

It includes:

- Gemini-generated annotation outputs for the two new classes.
- The RF-DETR 5-class router dataset split.
- The selected 5-class router checkpoint and selection metrics.
- Training config, helper scripts, and handoff/development documentation.

It intentionally excludes:

- `data/raw_new_classes_20260630/`
- Original gdown-downloaded archives and extracted raw new-class images.
- Failed invalid-key Gemini trial logs.
- Full 50-epoch training checkpoint history.

Primary model:

```text
outputs/rfdetr_router/medium_5class_brace_columnbase_20260630_test_as_valid/selected_precision_p090_epoch049_thr069.pth
```

Deployment confidence threshold:

```text
0.69
```

Class map:

```text
0: 天井
1: 壁类
2: RC柱
3: ブレース
4: 柱脚
```

Dataset:

```text
data/rfdetr_router_5class_brace_columnbase_20260630_test_as_valid
```

There is no independent validation split. The `valid` directory mirrors `test`
because RF-DETR expects a `valid` split during training.

Detailed handoff:

```text
docs/handoff_records/2026-06-30/router_5class_incremental_handoff_20260630.md
```

Detailed training record:

```text
docs/development_records/2026-06-30-rfdetr-router-5class-training.md
```

Full local handoff archive:

```text
shimizu_20260630_rfdetr_router5_incremental.tar.zst
SHA256: 21729a910a79eaa17e48dbf14b2d3d58c0511135eded949a39076642190e694c
```

The archive is about 4.3 GiB and is intentionally not committed to GitHub.
Git tracks the code, configs, documentation, small metadata files, and package
manifest needed to identify and validate the archive.
