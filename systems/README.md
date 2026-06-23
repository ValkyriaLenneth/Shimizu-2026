# Systems Index

This directory is the navigation layer for the organized repository.

The historical files are still kept in their original locations so existing
commands, imports, and handoff notes continue to work. Use this index to find
the active code and artifacts by model family and role.

## Model Families

| family | role in this repo | index |
|---|---|---|
| YOLO / YOLO9 | legacy router and earlier end-to-end pipeline baseline | [yolo](yolo/README.md) |
| RF-DETR | current production direction: router, downstream B/C/D models, final display pipeline | [rfdetr](rfdetr/README.md) |

## Common Organization

Each model family is organized conceptually into three blocks:

```text
router/
recognition_models/
pipeline/
```

`recognition_models` is split by business category:

```text
tenjo/
inner_wall/
rc_wall/
rc_column/
```

Large data, checkpoints, and compressed releases are not stored through this
index. See [Repository Organization](../docs/development_records/repo_organization_20260623.md).

Concrete navigation tree:

```text
systems/
  yolo/
    router/
    recognition_models/
      tenjo/
      inner_wall/
      rc_wall/
      rc_column/
    pipeline/
  rfdetr/
    router/
    recognition_models/
      tenjo/
      inner_wall/
      rc_wall/
      rc_column/
    pipeline/
```
