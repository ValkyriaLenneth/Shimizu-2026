# Final Release 2026-06-15 Manifest

Status: first release layout complete. Data and RF-DETR model archives have been validated, extracted, and organized into the final release directory.

Update 2026-06-16: RC wall downstream RF-DETR was replaced with the optimized checkpoint from `outputs/rfdetr_single_crack/rc_wall_report_best_c_crop_e001_p0722_r0812/checkpoint_epoch_001.pth`. The deployment path remains `final_release_20260615/models/rfdetr/downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth`; the previous packaged checkpoint is retained under `downstream/rc_wall/references/`.

## Data

Extracted data package:

```text
final_release_20260615/data/final_download_20260526/
```

Source archive:

```text
final_download_20260526.tar.zst
sha256: 7ddb1ce36196c8ed137578047693c1a4f32c964aec3d7fe8dae63614e0053a66
```

Optional split reference copied from repo root:

```text
final_release_20260615/data/data_split.json
```

Primary downstream crack dataset:

```text
final_release_20260615/data/final_download_20260526/handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split
```

Available components:

```text
tenjo
inner_wall
rc_wall
rc_column
```

Split summary:

| component | split | images | labels |
|---|---|---:|---:|
| tenjo | train | 750 | 750 |
| tenjo | valid | 97 | 97 |
| tenjo | test | 96 | 96 |
| inner_wall | train | 840 | 840 |
| inner_wall | valid | 114 | 114 |
| inner_wall | test | 104 | 104 |
| rc_wall | train | 966 | 966 |
| rc_wall | valid | 90 | 90 |
| rc_wall | test | 126 | 126 |
| rc_column | train | 498 | 498 |
| rc_column | valid | 71 | 71 |
| rc_column | test | 67 | 67 |

Total under `final_crack_yolo_20260519/split`: 3819 images and 3819 label files.

Detailed class-count summary:

```text
final_release_20260615/docs/source_manifests/final_crack_yolo_split_summary.json
```

Full archive file list:

```text
final_release_20260615/docs/source_manifests/final_download_20260526.files.txt
```

## Pipeline Eval Sample

A fixed deterministic pipeline evaluation sample has been created from the full four-component dataset without using `data_split.json`:

```text
data/pipeline_eval_20260615/
```

Sampling policy:

```text
seed = 20260615
target = 50 images per component
components = tenjo, inner_wall, rc_wall, rc_column
source pool = all available train/valid/test images under final_crack_yolo_20260519/split
```

Outputs:

```text
data/pipeline_eval_20260615/images/
data/pipeline_eval_20260615/labels/
data/pipeline_eval_20260615/manifest.csv
data/pipeline_eval_20260615/split_summary.json
data/pipeline_eval_20260615/README.md
```

Selected sample size:

| component | selected images |
|---|---:|
| tenjo | 50 |
| inner_wall | 50 |
| rc_wall | 50 |
| rc_column | 50 |

Total: 200 images.

## Pipeline Eval Official Plus Sample

A second permanent evaluation split has been created for pipeline testing and performance analysis:

```text
data/pipeline_eval_official_plus_20260615/
```

Policy:

```text
include all data_split.json test stems
add the same number of non-official samples per component
seed = 20260615
```

Selected sample size:

| component | official test | additional | total |
|---|---:|---:|---:|
| tenjo | 31 | 31 | 62 |
| inner_wall | 31 | 31 | 62 |
| rc_wall | 31 | 31 | 62 |
| rc_column | 31 | 31 | 62 |

Total: 248 images.

Permanent split JSON:

```text
data/pipeline_eval_official_plus_20260615/split.json
```

Baseline pipeline result:

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/
```

Report:

```text
docs/development_records/2026-06-15-16-final-release/pipeline_eval_20260615.md
```

## RF-DETR Models

Organized model directory:

```text
final_release_20260615/models/rfdetr/
  router/
  downstream/
  config/
```

Original pipeline config copied for reference:

```text
final_release_20260615/models/rfdetr/config/pipeline.rfdetr_prod.local.yaml
```

Final-release pipeline config with paths rewritten to this release directory:

```text
final_release_20260615/models/rfdetr/config/pipeline.rfdetr_prod.final_release.yaml
```

Recommended deployment weights:

```text
final_release_20260615/models/rfdetr/router/checkpoint_epoch_023.pth
final_release_20260615/models/rfdetr/downstream/tenjo/tenjo_standard_orig_checkpoint_epoch_009.pth
final_release_20260615/models/rfdetr/downstream/inner_wall/inner_wall_checkpoint_epoch_026.pth
final_release_20260615/models/rfdetr/downstream/rc_wall/rc_wall_checkpoint_epoch_009.pth
final_release_20260615/models/rfdetr/downstream/rc_column/checkpoint_epoch_047.pth
```

RC wall optimized report metrics under the fixed report protocol:

| component | precision | recall | B recall | C recall | D recall |
|---|---:|---:|---:|---:|---:|
| rc_wall | 0.722 | 0.812 | 0.857 | 0.600 | 1.000 |

Optimization report:

```text
final_release_20260615/docs/rc_wall_optimization_20260616.md
```

Additional retained files:

```text
final_release_20260615/models/rfdetr/router/checkpoint_23.ckpt
final_release_20260615/models/rfdetr/downstream/rc_column/checkpoint_47.ckpt
final_release_20260615/models/rfdetr/downstream/tenjo/references/
final_release_20260615/models/rfdetr/downstream/inner_wall/references/
final_release_20260615/models/rfdetr/downstream/rc_wall/references/
final_release_20260615/models/rfdetr/metrics/
final_release_20260615/models/rfdetr/scripts/
```

Source model archives validated by `tar --zstd -tf`:

```text
rfdetr_model_candidates_20260602.tar.zst
rfdetr_threshold_tuned_models_20260609.tar.zst
rfdetr_inner_wall_rc_wall_single_models_20260608.tar.zst
```

Source archive manifests:

```text
final_release_20260615/docs/source_manifests/rfdetr_model_candidates_20260602.files.txt
final_release_20260615/docs/source_manifests/rfdetr_threshold_tuned_models_20260609.files.txt
final_release_20260615/docs/source_manifests/rfdetr_inner_wall_rc_wall_single_models_20260608.files.txt
```

## Checksums

Uploaded archive checksums recorded at:

```text
final_release_20260615/docs/checksums/SHA256SUMS_uploaded_archives.txt
```

Model file checksums recorded at:

```text
final_release_20260615/docs/checksums/SHA256SUMS_rfdetr_models.txt
```

Validated data archive checksum recorded separately at:

```text
final_release_20260615/docs/checksums/SHA256SUMS_validated_data_archives.txt
```
