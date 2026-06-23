# 三类路由接入裂缝检测全流程细化大纲

日期：2026-05-19

## 1. 当前目标

本阶段目标不是单独训练一个构件分类模型，而是把三类构件路由模型接入上一期裂缝/损伤 YOLO 生产流程，形成可解释、可调参、可回退的完整推理链路。

端到端目标：

```text
输入巡检图片
  -> 三类构件路由模型：天井 / 壁类 / RC柱
  -> 根据路由结果选择裂缝/损伤模型
  -> 运行对应类别的 B-D 裂缝/损伤检测
  -> 合并路由结果、裂缝框、等级、置信度和诊断信息
  -> 输出统一 JSON / 可视化图片 / 审计日志
```

## 2. 类别策略

路由模型只负责识别视觉上可稳定区分的大类：

| 路由类别 | 来源类别 | 后续处理 |
|---|---|---|
| 天井 | 天井 | 调用天井裂缝/损伤模型 |
| 壁类 | 内壁 + RC壁 | 先调用墙类策略，再处理内壁/RC壁差异 |
| RC柱 | RC柱 | 调用 RC柱裂缝/损伤模型 |

`内壁` 与 `RC壁` 在外观上可能完全一致，因此路由阶段不强行区分。后续如果业务必须恢复二者差异，应通过非视觉信息或流程信息处理，例如：

- 图片所在目录、项目元数据、拍摄位置、图纸信息。
- 用户在前端选择的构件上下文。
- 若无上下文，则输出 `壁类` 并标记 `wall_subtype=unknown`。

## 3. 模块划分

### 3.1 Router 模块

职责：

- 加载三类 YOLOv9 路由模型。
- 对输入整图推理，输出构件候选框。
- 过滤低置信度框。
- 保留通过阈值的 router 框；router 层不强制处理框重叠。
- 生成路由决策：单类别、多类别、未知、低置信度。

建议输出字段：

```json
{
  "router_model": "gelan_c_router_3class_cleaned_e50",
  "classes": ["天井", "壁类", "RC柱"],
  "detections": [
    {
      "class_id": 1,
      "class_name": "壁类",
      "confidence": 0.82,
      "bbox_xyxy": [120, 80, 1800, 1400],
      "area_ratio": 0.73
    }
  ],
  "route_decision": {
    "status": "ok",
    "primary_class": "壁类",
    "strategy": "single_primary"
  }
}
```

### 3.2 Region View / Context 模块

职责：

- 根据 router bbox 生成下游判别模型需要的图像区域视图。
- 默认不落盘生成切片图片，而是从原始 ndarray / tensor 通过 slice 得到 region view。
- 给 bbox 增加上下文 padding，避免边界太紧导致裂缝模型漏检。
- 记录 slice 坐标，后续把裂缝框映射回原图。

原则：

- router 框宁可偏大，不宜偏小。
- padding 建议先用 bbox 边长的 `5%-15%`，并限制在图片范围内。
- 如果 router 框接近整图，直接用整图送入后续模型。
- 只有在 debug、审计或可视化需要时，才把 slice 结果保存成图片文件。

轻量传递建议：

```python
image = cv2.imread(image_path)  # H x W x C
region = image[y1:y2, x1:x2]    # 不新建中间文件
meta = {
    "source_image": image_path,
    "slice_xyxy": [x1, y1, x2, y2],
    "padding_ratio": 0.10,
}
```

注意：NumPy slice 通常是 view，不一定连续；如果下游 YOLO 推理接口要求 contiguous array，则在模型入口处统一做 `np.ascontiguousarray(region)`，仍然不需要写临时图片。

### 3.3 Crack Detector 模块

职责：

- 加载上一期四个类别裂缝/损伤 YOLO 模型。
- 对 router 选择的区域运行对应模型。
- 输出 B-D 等级框和置信度。

初始接入策略：

```text
天井 -> ceiling crack model
RC柱 -> rc_column crack model
壁类 -> 暂定墙类策略
```

墙类策略建议分三档：

1. `wall_parallel_debug`：同时运行内壁模型和 RC壁模型，但只用于离线对比，不直接给用户混乱结果。
2. `wall_single_model`：如果后续训练出合并墙类裂缝模型，则只调用该模型。
3. `wall_context_select`：如果外部元数据能区分内壁/RC壁，则根据元数据选择对应模型。

### 3.4 Result Merge 模块

职责：

- 把裂缝检测结果从 crop 坐标映射回原图坐标。
- 合并多个 router 区域的裂缝结果。
- 处理判别模型产生的重复裂缝框。
- 保留路由链路信息，方便追查错误来源。

建议输出字段：

```json
{
  "image_id": "...",
  "pipeline_version": "router3_crack_v1",
  "router": {...},
  "crack_detections": [
    {
      "source_router_class": "壁类",
      "crack_model": "inner_wall_or_wall_strategy",
      "damage_grade": "C",
      "confidence": 0.76,
      "bbox_xyxy": [200, 300, 450, 360],
      "coordinate_space": "original_image"
    }
  ],
  "warnings": []
}
```

## 4. 路由决策规则

### 4.1 单一高置信度类别

条件：

```text
top_confidence >= router_conf_threshold
and top_area_ratio >= min_area_ratio
```

处理：

- 使用 top 类别调用对应裂缝模型。
- 如果框很大或接近整图，用整图作为后续输入。

### 4.2 多个构件区域

条件：

```text
多个 router 框通过阈值
且类别不同或区域分离
```

处理：

- 分别裁剪并分别调用对应裂缝模型。
- 输出保留每个裂缝框来自哪个 router 区域。
- router 框之间允许重叠，不在 router 阶段去重。
- 真正需要去重的是下游判别模型产生的裂缝/损伤框。

说明：

建筑构件受拍摄角度影响，实际区域可能是三角形或梯形，但 YOLO 标注只能用矩形 bbox 表达。因此 router bbox 重叠是正常现象。router 的职责是把可能相关的上下文交给下游模型，而不是在这个阶段做几何精裁。

### 4.3 低置信度或无检测

处理优先级：

1. 若业务允许，输出 `route_status=unknown`，不调用裂缝模型。
2. 若必须给出结果，可进入保守 fallback：
   - 运行最稳的通用/墙类策略。
   - 或在中低置信度区间同时调用多个候选判别模型，并把结果标记为 `fallback_multi_model`。

不建议默认并行跑所有旧模型并直接给用户展示，因为这正是上一期用户反馈“结果很乱”的来源。

TODO：

- 确认中低置信度区间是否允许同时调用多个判别模型。
- 确认多模型 fallback 的展示方式：正式输出、warning 输出，还是仅 debug 输出。
- 确认多模型 fallback 的代价上限，例如最多调用几个模型、是否允许异步。

## 4.4 判别模型重叠框处理

判别模型输出的是裂缝/损伤框，重叠会直接影响最终报告，因此需要在结果合并阶段处理。

初始策略：

1. 先把所有下游检测框映射回原图坐标。
2. 按 `damage_grade` 分组，优先在同等级内做去重。
3. 同等级高 IoU 框：保留置信度更高的框。
4. 不同等级高 IoU 框：保留更严重等级，或保留高置信度但输出 conflict warning。该策略需用旧项目验收口径确认。
5. 来自不同 router 区域但映射后高度重叠的裂缝框，同样进入下游 NMS / WBF。

建议先实现两种可切换策略：

```text
crack_merge_mode=nms
crack_merge_mode=weighted_box_fusion
```

默认建议：

```text
同等级: NMS, iou_threshold=0.5
不同等级: conflict-aware NMS, iou_threshold=0.6, prefer_higher_grade=true
```

## 5. 累积误差控制

风险：

```text
路由错 -> 调错裂缝模型 -> 裂缝检测再错 -> 用户看到错误等级
```

控制措施：

- router 输出必须进入最终结果，不能只保留裂缝结果。
- 每个裂缝结果必须记录 `source_router_class` 和 `crack_model`。
- 设置 router 低置信度保护区间：
  - 高置信度：正常调用。
  - 中置信度：调用但输出 warning。
  - 中低置信度：TODO，确认是否同时调用多个候选判别模型。
  - 低置信度：进入 unknown/fallback。
- 离线评估时单独统计：
  - router 准确率。
  - 正确路由下的裂缝检测精度。
  - 错误路由下的裂缝检测退化。

## 6. 数据与模型产物

当前三类路由训练数据：

```text
coarse_router_yolov9/datasets/coarse_router_3class_full
coarse_router_yolov9/datasets/coarse_router_3class_cleaned
```

当前训练输出：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_full_e50
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50
```

当前倾向：

- `cleaned` 模型作为优先候选。
- `full` 模型作为对照，判断清洗是否损失召回。

最终选择标准：

1. 优先看 `mAP50` 和 `mAP50-95`。
2. 同时看三类 per-class precision/recall，尤其是 `RC柱` 是否被墙类吞掉。
3. 检查测试集可视化，确认 bbox 是否足够覆盖构件区域。
4. 如果 cleaned precision 高但 recall 低，需要调低置信度阈值或扩大 crop fallback。

## 7. 工程实现顺序

### Step 1：固化路由模型选择

- 等待 `full` 与 `cleaned` 训练完成。
- 汇总 `results.csv`、best.pt、last.pt、PR 曲线、confusion matrix。
- 选择默认 router 权重。
- 文档记录选择原因。

### Step 2：实现 Router 推理封装

建议新增模块：

```text
pipeline/router_infer.py
```

职责：

- 输入图片路径或 ndarray。
- 加载 router 权重。
- 输出标准化 router JSON。
- 支持阈值、NMS、最大框数量配置。

### Step 3：实现 Region View 与坐标映射

建议新增模块：

```text
pipeline/region_view.py
```

职责：

- bbox padding。
- 通过 slice 生成内存区域视图。
- 仅在 debug 时保存 crop 可视化。
- slice 坐标到原图坐标的正反映射。

### Step 4：接入旧裂缝模型

建议新增模块：

```text
pipeline/crack_detector_registry.py
```

职责：

- 管理四个旧模型权重路径。
- 根据 router 类别返回模型。
- 支持墙类策略配置。

### Step 5：实现端到端 runner

建议新增入口：

```text
pipeline/run_full_pipeline.py
```

职责：

- 对单张图或目录批处理。
- 输出 JSON。
- 输出可视化图。
- 输出错误与 warning 汇总。

### Step 6：离线评估与可视化

建议新增：

```text
pipeline/evaluate_pipeline.py
pipeline/visualize_pipeline_result.py
```

评估内容：

- router 检测效果。
- downstream 裂缝检测效果。
- 端到端失败案例分类。
- 墙类策略对比。

## 8. 配置文件建议

新增统一配置：

```yaml
pipeline:
  router_weights: coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50/weights/best.pt
  router_conf_threshold: 0.25
  router_iou_threshold: 0.45
  crop_padding_ratio: 0.10
  use_full_image_if_area_ratio_ge: 0.85
  router_overlap_policy: keep_all
  region_transport: ndarray_slice

crack_models:
  ceiling: path/to/ceiling.pt
  inner_wall: path/to/inner_wall.pt
  rc_wall: path/to/rc_wall.pt
  rc_column: path/to/rc_column.pt

wall_strategy:
  mode: context_select
  fallback: wall_parallel_debug

crack_merge:
  mode: nms
  same_grade_iou_threshold: 0.50
  cross_grade_iou_threshold: 0.60
  prefer_higher_grade: true

todo:
  low_confidence_multi_model_fallback: true
```

## 9. 输出物

端到端流程至少输出三类文件：

```text
outputs/pipeline/<run_id>/results.jsonl
outputs/pipeline/<run_id>/visualizations/*.jpg
outputs/pipeline/<run_id>/summary.json
```

`summary.json` 应包含：

- 输入图片数量。
- router 各类别数量。
- unknown / low_confidence 数量。
- 各裂缝等级 B/C/D 数量。
- warning 统计。
- 各模型版本和权重路径。

## 10. 测试计划

### 单元测试

- bbox padding 不越界。
- crop 坐标映射可逆。
- router 输出 JSON schema 稳定。
- 空检测、低置信度、多框、多类别场景可处理。

### 集成测试

- 单图端到端推理。
- 目录批处理。
- 壁类图片进入墙类策略。
- 无 router 检测时不会输出混乱裂缝结果。

### 回归测试

- 使用上一期典型样例图片。
- 对比旧流程“多模型并行”的混乱输出与新流程路由输出。
- 统计因 router 引入导致的漏检/错检案例。

## 11. 今日后续优先级

1. 等当前 `full` / `cleaned` 训练完成。
2. 自动汇总训练结果，选择默认 router 权重。
3. 固化 JPEG 修复、数据清洗、训练脚本和文档。
4. 开始实现 `router_infer.py` 与标准 router JSON。
5. 在拿到上一期裂缝模型权重路径后，实现模型 registry 和端到端 runner。

## 12. 独立工程目录

为避免和当前粗筛训练工程、外部 YOLO repo、历史产物混在一起，端到端 pipeline 单独放入：

```text
router_crack_pipeline/
```

建议目录：

```text
router_crack_pipeline/
  README.md
  configs/
    pipeline.default.yaml
  docs/
    full_pipeline_detailed_outline_2026-05-19.md
    router_3class_data_cleaning_2026-05-19.md
    router_3class_training_preparation_2026-05-19.md
  pipeline/
    router_infer.py
    region_view.py
    crack_detector_registry.py
    result_merge.py
    run_full_pipeline.py
  scripts/
    summarize_router_training.py
```

该目录只放端到端接入所需的工程代码、配置和说明；大型数据、权重、训练结果仍放在原位置并通过配置引用。
