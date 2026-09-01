# RF-DETR Production Pipeline

本目录用于承载新的端到端工程，和历史粗筛训练工程 `coarse_router_yolov9/`、外部 YOLO repo、数据清洗输出目录分开。当前默认配置已经切到 RF-DETR router + RF-DETR 下游 B/C/D detector，用作替换 YOLO9 生产链路的独立起点。

## 当前定位

```text
输入图片
  -> RF-DETR 三类 router: 天井 / 壁类 / RC柱
  -> 通过 ndarray slice 生成内存区域视图
  -> 调用对应 RF-DETR 裂缝/损伤 B-D 判别模型
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

- 三类 RF-DETR router 推理已接入默认候选权重。
- region 通过内存 slice 传递给下游 detector，不生成临时 crop 文件。
- detector registry 已支持 mock / no-op / YOLOv9 / RF-DETR 权重四种模式。
- 天井、RC壁、内壁 接入 2026-06-09 threshold-tuned RF-DETR 模型。
- RC柱 接入 2026-06-02 RF-DETR epoch47 候选模型。
- 壁类 router 输出会同时调用内壁与 RC壁 模型，但 PC 显示只输出一个 `壁-B/C/D`。
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
python rfdetr_prod_pipeline/scripts/test_pipeline_smoke.py
```

或手动运行：

```bash
python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline \
  --config rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml \
  --source data/rfdetr_rc_wall_all_non_legacy_test_v1/test/images/data_add100__3-B-00009.jpg \
  --output-dir outputs/rfdetr_prod_pipeline/smoke \
  --device cpu \
  --skip-visualization
```

Batch smoke:

```bash
python rfdetr_prod_pipeline/scripts/test_pipeline_batch_smoke.py
```

Five-class precision ensemble (two GPUs):

```bash
python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline \
  --config systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_precision_ensemble.yaml \
  --source <image-or-directory> \
  --device cuda:0
```

This configuration places the primary model on `cuda:0`, confirmation models
on `cuda:1`, and lazily runs brace confirmation only when required.

汇总任意一次 pipeline JSONL 输出：

```bash
python rfdetr_prod_pipeline/scripts/summarize_pipeline_results.py \
  outputs/rfdetr_prod_pipeline/smoke/results.jsonl
```
