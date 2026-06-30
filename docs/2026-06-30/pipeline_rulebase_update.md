# Pipeline Rule-base 调整记录

日期：2026-06-30

## 背景

本次调整围绕 RF-DETR production pipeline 中的 rule-base 后处理逻辑展开，目标是让流程更符合当前预期：

- Router 只负责提供候选构件区域，不再提前按 dominant class 删除其他类别。
- 下游模型应对整张图片进行检测，再用 router region 对检测结果做筛选。
- 边界处检测框不能因为轻微跨出 router region 就被误删。
- 下游模型空输出时，不再只使用一次固定 fallback threshold，而是动态降阈值直到 region 内出现候选或达到最低阈值。
- 切换到 full-image-filter 主路径后，不再额外执行 wall full image rescue。
- 前端输出中，壁类只显示为 `壁類 / 壁-B/C/D`，不暴露 `inner_wall`、`rc_wall` 作为最终类别。
- 跨类别 ambiguity 需要保留全部候选，不再自动偏向 wall 收敛。

## 本次配置调整

文件：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.all_small_boxes.yaml
```

### Router region 过滤

删除：

```yaml
router_min_region_confidence: 0.08
```

原因：

当前 router 已经有：

```yaml
router_low_conf_threshold: 0.03
```

如果后续再用 `router_min_region_confidence=0.08` 过滤，则 `0.03 ~ 0.08` 的低置信保底候选会被再次删掉，逻辑上互相冲突。因此本次取消 `router_min_region_confidence`，保留 `0.03` 低阈值保底。

### Dominant class rule

调整为：

```yaml
dominant_router_class_policy:
  enabled: false
```

原因：

原逻辑会在天井 confidence 很高时只保留天井 region。这会提前删除其他类别候选，不符合当前召回优先和后端统一筛选的策略。

### Region transport

调整为：

```yaml
region_transport: full_image_filter
region_filter_mode: center_or_ioa
region_filter_ioa_threshold: 0.50
```

含义：

下游模型对整张图推理，然后用 router region 筛选检测结果。

边界框处理规则：

```text
检测框中心点在 router region 内
或
检测框自身至少 50% 面积落在 router region 内
```

满足任一条件则保留。这样可以处理检测框正好卡在 router region 边界处的情况。

### Full image rescue

调整为：

```yaml
full_image_rescue:
  enabled: false
```

原因：

主路径已经改成 full-image-filter，即下游模型本来就对整张图片推理。因此原先为 crop 推理补漏设计的 `full_image_rescue` 会变成重复逻辑，需要关闭。

### Downstream empty fallback

调整为：

```yaml
downstream_empty_fallback:
  enabled: true
  dynamic: true
  min_threshold: 0.05
  step: 0.05
  max_outputs_per_region: 1
```

含义：

当某个 detector 在某个 router region 内没有有效输出时，不再只用一组固定 fallback threshold，而是从模型原始阈值开始逐步下降。

示例：

```text
[0.30, 0.45, 0.35]
-> [0.25, 0.40, 0.30]
-> [0.20, 0.35, 0.25]
-> ...
-> 最低 0.05
```

停止条件：

```text
在对应 router region 内筛选出至少一个候选
或
阈值下降到 min_threshold
```

每个 region 最多补 `1` 个动态阈值候选。

### Ambiguity collapse

调整为：

```yaml
final_display_postprocess:
  collapse_ambiguity:
    enabled: false
```

原因：

跨类别 ambiguity 的含义是同一位置存在多个构件类别解释，例如 `壁類` 与 `RC柱`。当前要求是发送给前端的结果中保留全部候选，而不是因为其中存在 wall 就自动偏向 wall 收敛。

注意：前端候选类别只保留到以下层级：

```text
天井 / 壁類 / RC柱
```

不输出 `inner_wall` 或 `rc_wall` 作为前端类别。

### Wall display

调整为：

```yaml
wall_display:
  mode: rule_merged
  raw_append_if_uncovered: false
```

原因：

客户预期最终只有合并后的壁类输出。壁类检测可以来自 `inner_wall`、`rc_wall` 两个模型，但最终展示时都统一为：

```text
壁類 / 壁-B / 壁-C / 壁-D
```

未配对的单模型 wall 候选也会转换成壁类显示，不再把 raw `inner_wall` / `rc_wall` 候选追加到最终前端结果。

## 本次代码调整

### 动态阈值

文件：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/run_full_pipeline.py
```

新增逻辑：

- `_empty_fallback_threshold_schedule`
- `_dynamic_empty_detector_outputs`
- `_dynamic_empty_detector_outputs_for_region`

其中 full-image-filter 主路径使用 `_dynamic_empty_detector_outputs_for_region`，确保动态降阈值的停止条件是：

```text
router region 内存在候选
```

而不是全图任意位置存在候选。

### Region 边界筛选

沿用并显式配置：

```text
detection_in_router_region
```

当前规则：

```text
center_or_ioa
```

也就是检测框中心点在 region 内，或检测框自身与 region 的 IoA 达到 `0.50`。

### Wall 输出口径

文件：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/wall_candidate_display.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/ambiguity_display.py
```

调整后：

- `inner_wall` / `rc_wall` 作为模型来源保留在 `raw_source_model`。
- 前端显示用 `source_model` 统一为 `wall`。
- `structure_type` 统一为 `壁類`。
- `damage_grade` 统一为 `壁-B/C/D`。

## 关于 Crack merge / NMS

当前配置仍然是：

```yaml
crack_merge:
  mode: nms
```

这一步当前是“删除重复框”，不是坐标融合。

规则：

```text
同等级：IoU > 0.90 时删除低优先级候选
跨等级：IoU > 0.95 时删除低优先级候选
```

保留优先级：

```text
高 damage grade 优先
同等级下 confidence 高者优先
```

代码中确实存在另一个合并逻辑：

```text
prod_like_merge_detections
```

这套逻辑会做跨模型聚类和 confidence 加权坐标融合。但当前配置没有启用，仍使用 NMS。是否切换到 `prod_like` 需要单独确认，因为它会改变最终框坐标和评估口径。

## 当前最终流程概况

调整后主流程为：

```text
输入图片
  -> RF-DETR router
  -> 保留 router 候选，不再用 0.08 二次过滤
  -> 下游 detector 对整图推理
  -> 用 router region 筛选下游结果
  -> 边界框按 center_or_ioa 保留
  -> region 内无结果时动态降阈值
  -> raw detection NMS
  -> 跨类别 ambiguity 保留多候选
  -> inner_wall / rc_wall 统一转为壁類
  -> final same-family cluster
  -> 输出 display_crack_detections
```
