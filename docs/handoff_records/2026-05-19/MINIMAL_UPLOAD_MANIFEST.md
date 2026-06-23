# Minimal Upload Manifest

Generated: 2026-05-26

This manifest keeps only the artifacts needed to restore the best router model and dataset labels while uploading the shared images once.

Important: the YOLO-ready crack dataset and router dataset both contain copied image files. Treat those `images/` folders as rebuildable caches, not as upload sources.

## Required

### Best router model

```text
shimizu_20260519_router_models_and_results/coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_ft_from_d900_imgw_rc_os900_e50_ddp/weights/best.pt
```

Approx size: 195 MB.

### Canonical image pool

Upload the source images once:

```text
shimizu_20260519_data_package/data/unzip
shimizu_20260519_data_package/additional_data_2026-05-19/unpacked/data_add100
```

Approx size: 4.9 GB.

### Crack detection annotations

Upload labels and metadata, but do not upload the copied images under `split/**/images`:

```text
shimizu_20260519_data_package/additional_data_2026-05-19/unpacked/labels_20251107
shimizu_20260519_data_package/data/final_crack_yolo_20260519/split/**/labels
shimizu_20260519_data_package/data/final_crack_yolo_20260519/split/**/data.yaml
shimizu_20260519_data_package/data/final_crack_yolo_20260519/README.md
shimizu_20260519_data_package/data/final_crack_yolo_20260519/manifest.csv
shimizu_20260519_data_package/data/final_crack_yolo_20260519/summary.json
shimizu_20260519_data_package/data/final_crack_yolo_20260519/raw_sources
```

The `manifest.csv` maps each final split sample back to its canonical source image.

### Router dataset used by the best model

Upload the router labels, config, and summary, but not the copied images under `images/`:

```text
shimizu_20260519_data_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900/labels
shimizu_20260519_data_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900/data.yaml
shimizu_20260519_data_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900/summary.json
```

This is the dataset variant paired with the best `merged4219 ... rc_os900 ... ddp` model.

### Gemini router annotations

Minimum final merged annotation source:

```text
shimizu_20260519_data_package/outputs/gemini_merged_4219_3_1_pro_preview_2026-05-19
```

Approx size: 28 MB.

Optional historical/source annotation runs, small enough to keep if traceability matters:

```text
shimizu_20260519_data_package/outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19
shimizu_20260519_data_package/outputs/gemini_data_add100_3_1_pro_preview_2026-05-19
```

Combined outputs size: 95 MB.

## Exclude

```text
shimizu_20260519_data_package.zip
shimizu_20260519_router_models_and_results.zip
shimizu_20260519_data_package/data/final_crack_yolo_20260519/all
shimizu_20260519_data_package/data/final_crack_yolo_20260519/split/**/images
shimizu_20260519_data_package/coarse_router_yolov9/datasets/coarse_router_3class_full_merged_4219
shimizu_20260519_data_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219
shimizu_20260519_data_package/coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900/images
shimizu_20260519_router_models_and_results/coarse_router_yolov9/runs
```

Exception: keep the single `best.pt` listed above from `runs`.

Also exclude `.DS_Store` and cache files such as `labels/*.cache`.
