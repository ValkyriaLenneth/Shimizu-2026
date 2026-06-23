# E2E Pipeline CLI / JSON 契约

日期：2026-05-19

## CLI

单图或目录推理：

```bash
coarse_router_yolov9/.venv/bin/python -m router_crack_pipeline.pipeline.run_full_pipeline \
  --config router_crack_pipeline/configs/pipeline.default.yaml \
  --source <image-or-dir> \
  --output-dir outputs/pipeline/<run_id> \
  --device cpu
```

参数：

| 参数 | 说明 |
|---|---|
| `--config` | pipeline YAML 配置 |
| `--source` | 单张图片或图片目录 |
| `--output-dir` | 输出目录 |
| `--device` | `cpu`、`0`、`1` 等 YOLO device |
| `--mock-crack` | 使用 mock 下游判别模型做系统测试 |
| `--limit` | 目录输入时限制处理数量 |
| `--skip-visualization` | 不输出可视化图片 |

## 输出目录

```text
outputs/pipeline/<run_id>/
  results.jsonl
  summary.json
  visualizations/
```

## results.jsonl

每行对应一张图：

```json
{
  "image": "path/to/image.jpg",
  "image_shape": [1185, 1585, 3],
  "pipeline_version": "router3_crack_v1",
  "router": {
    "router_model": ".../best.pt",
    "classes": {"0": "天井", "1": "壁类", "2": "RC柱"},
    "detections": [
      {
        "bbox_xyxy": [192.0, 20.0, 1582.0, 816.0],
        "confidence": 0.70,
        "class_id": 1,
        "class_name": "壁类",
        "area_ratio": 0.59
      }
    ],
    "route_decision": {
      "status": "ok",
      "strategy": "keep_all_router_boxes",
      "primary_class": "壁类",
      "low_confidence_fallback_todo": false
    }
  },
  "crack_detections": [
    {
      "bbox_xyxy": [328.0, 279.0, 1476.0, 792.0],
      "confidence": 0.7468,
      "damage_grade": "耐震壁の損傷程度C",
      "source_model": "rc_wall",
      "source_router_class": "壁类",
      "coordinate_space": "original_image"
    }
  ],
  "warnings": [],
  "elapsed_ms": 1234.5
}
```

## summary.json

```json
{
  "run_id": "20260519_000000",
  "images": 3,
  "results_jsonl": ".../results.jsonl",
  "visualizations": ".../visualizations",
  "router_status_counts": {"ok": 3},
  "crack_detections": 5,
  "warning_counts": {},
  "error_count": 0
}
```

## 结果汇总

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/summarize_pipeline_results.py \
  outputs/pipeline/<run_id>/results.jsonl
```
