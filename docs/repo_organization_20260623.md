# Repository Organization - 2026-06-23

## Goal

This repo has accumulated several research and handoff phases. The next
organization pass should make it searchable by:

```text
model family -> role -> category
```

Model families:

```text
yolo
rfdetr
```

Roles:

```text
router
recognition_models
pipeline
```

Recognition categories:

```text
tenjo
inner_wall
rc_wall
rc_column
```

## Current Organization Layer

The new navigation entry is:

```text
systems/
```

It is now the primary code organization layer. Existing root-level code paths
are kept as compatibility symlinks so older commands and imports continue to
work.

```text
systems/
  README.md
  yolo/
    README.md
    router/
    recognition_models/
      tenjo/
      inner_wall/
      rc_wall/
      rc_column/
    pipeline/
  rfdetr/
    README.md
    router/
    recognition_models/
      tenjo/
      inner_wall/
      rc_wall/
      rc_column/
    pipeline/
```

Use these files first when looking for a router, recognition model, pipeline
implementation, config, training script, or report.

## Compatibility Links

The old root paths remain as links:

```text
rfdetr_prod_pipeline -> systems/rfdetr/pipeline/rfdetr_prod_pipeline
router_crack_pipeline -> systems/yolo/pipeline/router_crack_pipeline
coarse_router_yolov9 -> systems/yolo/router/coarse_router_yolov9
```

The old flat script/config entrypoints are also links:

```text
scripts/*.py
scripts/*.sh
configs/*.yaml
```

This lets old commands keep working while the real files live under
`systems/`.

## Why Not Move Everything Immediately

Many paths are referenced by configs, scripts, docs, and final-release
manifests. A large mechanical move would make the repo look cleaner, but it
would also break reproducibility unless every reference is updated and tested.

The safer sequence is:

1. Add a clear index layer.
2. Mark active vs legacy components.
3. Update README and handoff docs.
4. Move code only when a module has a testable import boundary.
5. Keep data, checkpoints, and compressed releases outside git.

## Active Code Map

### RF-DETR Current Line

| role | location | notes |
|---|---|---|
| router training/eval | `systems/rfdetr/router/configs/`, `systems/rfdetr/scripts/train_rfdetr_router.py`, `systems/rfdetr/scripts/rfdetr_router_callbacks.py` | RF-DETR router training assets |
| downstream recognition training/eval | `systems/rfdetr/recognition_models/*/configs/`, `systems/rfdetr/scripts/build_rfdetr_single_crack_views.py`, `systems/rfdetr/scripts/evaluate_rfdetr_class_threshold_grid.py` | category-specific B/C/D models |
| production pipeline | `systems/rfdetr/pipeline/rfdetr_prod_pipeline/` | current end-to-end direction |
| release metadata | `final_release_20260615/` | git-tracked docs/metrics only; large weights are external |

### YOLO Legacy Line

| role | location | notes |
|---|---|---|
| router training | `systems/yolo/router/coarse_router_yolov9/` | legacy router training and YOLO9 source copy |
| pipeline | `systems/yolo/pipeline/router_crack_pipeline/` | earlier end-to-end pipeline |
| docs and reports | `docs/*20260519*`, `docs/*20260526*`, `docs/yolo9_single_crack_model_baseline_20260602.md` | historical baseline and handoff context |

## Category Map

| category | Japanese/business label | RF-DETR status | YOLO status |
|---|---|---|---|
| `tenjo` | 天井 | release checkpoint exists; report optimization candidate needs NMS protocol follow-up | baseline/reference |
| `inner_wall` | 内壁 | release checkpoint exists | baseline/reference |
| `rc_wall` | RC壁 | optimized 2026-06-16 release checkpoint | baseline/reference |
| `rc_column` | RC柱 | release checkpoint exists | baseline/reference |

## External Artifact Policy

Do not track large artifacts in git:

```text
*.pth
*.ckpt
*.pt
*.tar.zst
outputs/
data/rfdetr_*/
final_release_20260615/data/
```

The complete final release package is currently:

```text
/Users/len/Downloads/final_release_20260615.tar.zst
size: 11,684,930,994 bytes
sha256: 1daed69947449dd852873aabbed1c8413581c6bfb8651944ddb0228d47828820
```

It contains the complete `final_release_20260615/` tree, including RF-DETR and
YOLO weights. The git repo keeps the release manifest, docs, checksums, and
metrics needed to identify those artifacts.

## Recommended Next Cleanup Passes

### Pass 1: Documentation And Navigation

Completed by this organization layer:

```text
systems/
docs/repo_organization_20260623.md
README.md
```

### Pass 2: Config Consolidation

Completed with compatibility links:

```text
systems/rfdetr/router/configs/
systems/rfdetr/recognition_models/*/configs/
systems/legacy_classification/configs/
configs/*.yaml -> systems/.../*.yaml
```

### Pass 3: Script Consolidation

Completed with compatibility links:

```text
systems/rfdetr/scripts/
systems/yolo/router/scripts/
systems/gemini/scripts/
systems/legacy_classification/scripts/
scripts/* -> systems/.../*
```

### Pass 4: Legacy Freeze

Mark these as historical reference unless active work resumes:

```text
router_crack_pipeline/
coarse_router_yolov9/
additional_data_2026-05-19/
handoff_20260519/
handoff_20260526/
```

Avoid deleting them until every referenced artifact has a documented external
location or checksum.
