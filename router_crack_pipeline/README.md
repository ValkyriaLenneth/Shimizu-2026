# Router + Crack Detection Pipeline

本目录用于承载新的端到端工程，和历史粗筛训练工程 `coarse_router_yolov9/`、外部 YOLO repo、数据清洗输出目录分开。

## 当前定位

```text
输入图片
  -> 三类 router: 天井 / 壁类 / RC柱
  -> 通过 ndarray slice 生成内存区域视图
  -> 调用对应裂缝/损伤 B-D 判别模型
  -> 合并判别模型的重叠输出
  -> 输出 JSON、可视化、审计日志
```

## 关键决策

- router 的 bbox 重叠是正常现象，不在 router 阶段强制去重。
- router 输出的区域默认通过 NumPy slice / tensor slice 传递给下游模型，不写临时 crop 图片。
- 真正需要处理的是下游裂缝/损伤判别模型的重叠框。
- 低置信度 router 是否同时调用多个判别模型，先作为 TODO 保留，待业务验收口径确认。
- 大型数据、权重、训练结果不复制到本目录，通过配置文件引用。

## 目录

```text
configs/   端到端 pipeline 配置
docs/      当前阶段设计与数据说明副本
pipeline/  后续端到端推理代码
scripts/   当前阶段必要的数据/训练辅助脚本副本
```

## 当前已完成

- 三类 router 推理已接入默认权重。
- region 通过内存 slice 传递给下游 detector，不生成临时 crop 文件。
- detector registry 已支持 mock / no-op / 真实 YOLOv9 权重三种模式。
- 下游判别框已实现可配置 NMS 合并骨架。
- CLI 已支持单图/目录输入、`--limit`、`--skip-visualization`、异常隔离和 JSONL 结果输出。
- 批量 smoke test 和结果汇总脚本已固化。

## 暂缓事项

以下内容需要真实裂缝标注、用户验收口径或完整业务数据，当前不阻塞系统集成：

- 上一期裂缝模型的正式精度评估。
- 下游 B/C/D 重叠冲突策略的业务验收。
- `壁类 -> 内壁 / RC壁` 的正式拆分或并行策略。
- router 中低置信度时的多模型 fallback 策略。

详见：

```text
docs/deferred_real_crack_data_tasks_2026-05-19.md
```

## Smoke Test

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/test_pipeline_smoke.py
```

或手动运行：

```bash
coarse_router_yolov9/.venv/bin/python -m router_crack_pipeline.pipeline.run_full_pipeline \
  --config router_crack_pipeline/configs/pipeline.default.yaml \
  --source data/unzip/3_RC壁/obj_train_data/c-10.jpg \
  --output-dir outputs/pipeline/smoke_router_crack \
  --device cpu \
  --mock-crack
```

Batch smoke:

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/test_pipeline_batch_smoke.py
```

汇总任意一次 pipeline JSONL 输出：

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/summarize_pipeline_results.py \
  outputs/pipeline/batch_smoke_mock/results.jsonl
```
