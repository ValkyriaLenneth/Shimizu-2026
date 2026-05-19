# 系统集成与 smoke test 报告

日期：2026-05-19

## 已实现范围

新增独立工程目录：

```text
router_crack_pipeline/
```

已实现端到端链路：

```text
真实图片
  -> 三类 YOLOv9 router
  -> router bbox 保留重叠框
  -> ndarray slice 生成 region view，不写临时 crop 文件
  -> 下游裂缝 detector registry
  -> 判别框映射回原图
  -> 下游判别框 NMS 合并
  -> JSONL / summary.json / 可视化图片
```

## 关键代码

```text
pipeline/yolov9_backend.py
pipeline/router_infer.py
pipeline/region_view.py
pipeline/crack_detector_registry.py
pipeline/result_merge.py
pipeline/run_full_pipeline.py
```

## 当前 detector 状态

router 已接入真实权重：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50/weights/best.pt
```

裂缝/损伤模型 registry 已实现：

- 配置真实权重时，走 YOLOv9 backend。
- 未配置真实权重时，走 no-op detector。
- smoke test 可使用 mock detector 验证完整数据流。

上一期 GPL 训练权重已解压到本地：

```text
downloads/previous_phase_gpl_model_unpacked/infer_models/
```

四个权重均已确认可被当前 YOLOv9 `DetectMultiBackend` 加载：

| 文件 | 类别 |
|---|---|
| `TIANJING.pt` | 天井 B/C/D |
| `NEIBI.pt` | 内壁 B/C/D |
| `RCBI.pt` | 耐震壁 B/C/D |
| `RCZHU.pt` | RC柱 B/C/D |

本地真实权重配置样例：

```text
router_crack_pipeline/configs/pipeline.previous_phase_gpl.local.yaml
```

## Smoke test

命令：

```bash
coarse_router_yolov9/.venv/bin/python -m router_crack_pipeline.pipeline.run_full_pipeline \
  --config router_crack_pipeline/configs/pipeline.default.yaml \
  --source data/unzip/3_RC壁/obj_train_data/c-10.jpg \
  --output-dir outputs/pipeline/smoke_router_crack \
  --device cpu \
  --mock-crack
```

结果：

```json
{
  "images": 1,
  "router_status_counts": {
    "ok": 1
  },
  "crack_detections": 2
}
```

输出：

```text
outputs/pipeline/smoke_router_crack/results.jsonl
outputs/pipeline/smoke_router_crack/summary.json
outputs/pipeline/smoke_router_crack/visualizations/c-10_pipeline.jpg
```

No-op detector 也已测试：

```text
router 正常输出，crack_detections=0
```

## 真实上一期裂缝模型集成测试

命令：

```bash
coarse_router_yolov9/.venv/bin/python -m router_crack_pipeline.pipeline.run_full_pipeline \
  --config router_crack_pipeline/configs/pipeline.previous_phase_gpl.local.yaml \
  --source data/unzip/3_RC壁/obj_train_data/c-10.jpg \
  --output-dir outputs/pipeline/smoke_previous_phase_real \
  --device cpu
```

结果：

```json
{
  "images": 1,
  "router_status_counts": {
    "ok": 1
  },
  "crack_detections": 1
}
```

该样例中 router 输出了 `壁类` 和 `天井` 两个重叠/相邻区域，pipeline 保留了两个 router 区域并分别进入下游策略。真实裂缝模型最终输出 1 个检测框：

```text
source_router_class=壁类
source_model=rc_wall
damage_grade=耐震壁の損傷程度C
confidence=0.7468
```

输出：

```text
outputs/pipeline/smoke_previous_phase_real/results.jsonl
outputs/pipeline/smoke_previous_phase_real/summary.json
outputs/pipeline/smoke_previous_phase_real/visualizations/c-10_pipeline.jpg
```

## 固化测试脚本

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/test_pipeline_smoke.py
```

该脚本验证：

- router 权重可加载。
- 单图 pipeline 可运行。
- router status 为 `ok`。
- mock 下游检测结果可完成坐标映射与输出。
- 可视化图片生成。

## 批量 smoke test

命令：

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/test_pipeline_batch_smoke.py
```

该脚本验证：

- 目录输入可运行。
- `--limit` 可限制批量处理张数。
- `--skip-visualization` 可跳过图片绘制以提升批处理速度。
- 单图异常不会破坏 JSONL/summary 输出结构。
- `summarize_pipeline_results.py` 可读取 pipeline JSONL 并输出聚合统计。

本次运行结果：

```json
{
  "images": 3,
  "router_status_counts": {
    "ok": 3
  },
  "crack_detections": 6,
  "warning_counts": {},
  "error_count": 0
}
```

结果汇总：

```json
{
  "images": 3,
  "errors": 0,
  "router_status": {
    "ok": 3
  },
  "router_classes": {
    "壁类": 4,
    "天井": 2
  },
  "crack_models": {
    "mock": 6
  },
  "crack_grades": {
    "C": 6
  },
  "warnings": {}
}
```

## 当前暂缓事项

以下内容需要真实裂缝标注、用户验收口径或完整业务数据，当前不作为系统集成阻塞项：

```text
docs/deferred_real_crack_data_tasks_2026-05-19.md
```

## 后续 TODO

1. 对真实裂缝模型输出做重叠合并策略验收：
   - 同等级 NMS。
   - 跨等级 conflict-aware NMS。
   - 是否优先保留更严重等级。
2. 确认 router 中低置信度时是否允许多模型 fallback。
3. 确认 GPL 权重在最终交付形态中的许可证处理方式。
