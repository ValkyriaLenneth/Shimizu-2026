# YOLO 类别路由接入开发方针

日期：2026-05-19

## 1. 背景

上一期生产项目中，我们已经按四个建筑构件类别分别训练了 YOLO 裂缝/损伤识别模型：

- 天井
- 内壁
- RC壁
- RC柱

这些模型的职责是：在对应构件类别的图像中检测裂缝或损伤区域，并输出 B-D 的损伤等级分类。

上一期流程的问题是，同一张图如果并行交给多个类别专用模型处理，输出会变得混乱，用户难以判断哪个结果可信。因此，本轮开发的类别识别 YOLO 不是独立 demo，而是用于解决“先自动判断构件类别，再调用对应裂缝模型”的前置路由问题。

目标流程如下：

```text
巡检图像
  -> 建筑构件类别路由器
  -> 将选中的整图或区域交给类别专用裂缝 YOLO
  -> 裂缝/损伤 B-D 检测
  -> 汇总统一输出
```

## 2. 当前类别路由模型状态

上一轮开发中，我们使用 Gemini 生成建筑构件粗框标注，然后训练了 YOLOv9 粗筛路由模型。

历史记录中的结果如下：

- 数据集：`coarse_router_yolov9/datasets/coarse_cross_fixed`
- 类别：`天井`、`内壁`、`RC壁`、`RC柱`
- 源图像：1935 张
- 总框数：3950
- 划分：train 1548、val 194、test 193
- 模型：YOLOv9 GELAN-C
- 训练轮数：50 epochs
- 最优验证 epoch：48
- precision：0.724
- recall：0.636
- mAP@0.5：0.712
- mAP@0.5:0.95：0.580

当前本地 checkout 里只有代码、脚本和文档，还没有恢复原始数据、Gemini 输出、YOLO 数据集、权重、训练结果、QA 页面和归档包。这些内容后续需要通过现场恢复方式补齐。

## 3. 本轮开发主目标

本轮的直接目标是：把建筑构件类别路由模型接入上一期生产版裂缝识别流程。

类别路由器应该成为生产 pipeline 的前置模块，用于决定后续应该调用哪个类别专用裂缝/损伤 YOLO，而不是只作为单独的模型结果展示。

期望接入逻辑如下：

```text
router 识别为 天井
  -> 调用天井裂缝/损伤 YOLO

router 识别为 RC柱
  -> 调用 RC柱裂缝/损伤 YOLO

router 识别为 墙类
  -> 进入 内壁 / RC壁 的墙类分流策略
```

生产 pipeline 的最终输出应该同时保留类别路由结果和裂缝检测结果。这样出现错误时，可以区分问题来自类别路由、裁剪策略、裂缝检测，还是结果合并。

## 4. RC壁 与 内壁 问题

和用户开会时已经确认：在当前巡检图像里，RC壁 和 内壁 在外观上可能完全一致。也就是说，仅依赖图像本身，VLM 或 YOLO 都不应该被期待稳定地区分这两个类别。

因此，类别路由阶段不应继续强行学习四个视觉类别，而应将任务调整为三个可观测的大类：

```text
天井
壁类 = 内壁 + RC壁
RC柱
```

原来的四类 router 可以保留为 baseline 或兼容版本，但主开发方向建议切换到三类 router。

墙体细分类应后置处理，可使用以下信息或策略：

- 项目位置、楼层、房间、图纸编号、源目录等非视觉元数据
- 默认规则，例如先使用某个墙类模型
- 同时调用 内壁 与 RC壁 两个裂缝模型，再做合并或择优
- 当必须区分墙体子类但缺少可靠依据时，标记为需要人工复核

## 5. 累积误差风险

集成系统存在明显的累积误差问题：

```text
类别路由判断错误
  -> 调用了错误的裂缝模型
  -> 裂缝模型本身还有检测/分类误差
  -> 最终结果更难解释，也更难让用户信任
```

因此，router 不应在低置信度情况下做不可逆的硬分类。

建议的规避策略：

- 在最终输出里保留 router 置信度。
- 支持 Top-N 候选路由。
- 对低置信度结果启用 fallback。
- 对墙类区域使用特殊策略，不强行区分 内壁 / RC壁。
- 生成 review 输出，把 router 错误和裂缝检测错误分开检查。

## 6. 框选与裁剪策略

在这个应用里，router 框太小比框太大更危险。框太小可能裁掉裂缝或必要上下文，导致后续裂缝模型失效。

建议裁剪策略：

- 将 router 框传给裂缝模型前先做外扩。
- 初始外扩比例建议为 15%-25%。
- 外扩后自动 clamp 到图像边界内。
- 如果 router 框已经覆盖图像大部分区域，则直接使用原图或接近原图的裁剪区域。
- 输出中同时保留原始 router 框和外扩后的裁剪框。

## 7. 建议的 Router 输出 Schema

router 应输出结构化 JSON，用于驱动后续裂缝检测 pipeline：

```json
{
  "image_path": "path/to/image.jpg",
  "router_model": "coarse-router-name-or-version",
  "router_mode": "three_class",
  "detections": [
    {
      "router_class": "壁类",
      "confidence": 0.82,
      "bbox_xyxy": [100, 80, 1200, 900],
      "expanded_bbox_xyxy": [0, 0, 1350, 1000],
      "routing_candidates": [
        {"target": "inner_wall_crack_model", "policy": "wall_fallback"},
        {"target": "rc_wall_crack_model", "policy": "wall_fallback"}
      ],
      "review_required": false
    }
  ]
}
```

最终 pipeline 输出应同时包含 router 阶段和裂缝检测阶段：

```json
{
  "image_path": "path/to/image.jpg",
  "router": {},
  "crack_detections": [],
  "warnings": []
}
```

## 8. 墙类分流策略选项

对于 `壁类`，有三种现实可行的策略。

### 方案 A：同时调用两个墙类裂缝模型

同时运行 `内壁` 和 `RC壁` 裂缝模型，然后合并或择优。

优点：

- 因墙体子类判断错误而漏检裂缝的风险最低。
- 避免要求图像 router 解决不可观测问题。

缺点：

- 推理成本更高。
- 需要设计结果合并规则，避免重复框或冲突结果。

### 方案 B：默认主模型 + fallback

先使用一个墙类模型作为主模型，仅在置信度或结果质量较差时调用另一个模型。

优点：

- 比始终运行两个模型成本更低。
- 仍保留回退路径。

缺点：

- 需要定义“结果质量较差”的判断规则。

### 方案 C：使用元数据选择墙体子类

使用项目元数据、源目录、图纸信息或巡检上下文来决定调用 内壁 还是 RC壁 模型。

优点：

- 如果元数据可靠，这是概念上最正确的方案。

缺点：

- 需要依赖图像以外的信息，而这些信息未必总是存在。

初始建议：先把接口设计成能支持三种策略，再根据运行成本和数据可用性选择默认策略。若没有可靠元数据，优先考虑方案 A 或方案 B。

## 9. 模型改善方向

首要改善方向：

- 重新构建三类 router 数据集：`天井`、`壁类`、`RC柱`。

其他改善方向：

- 恢复并检查上一轮数据、预测结果和 review 页面。
- 补充弱势类别数据，尤其是 RC柱。
- 增加多构件同图、构件交界、遮挡、斜角、近景等困难样本。
- 建立一个小型人工审核评估集。
- 评估时重点关注路由召回率和主要构件命中率，而不是只看 mAP。
- 为高召回路由调优置信度阈值。
- 对不确定样本使用 Top-2 或 fallback 路由。
- 通过可视化 QA 页面持续收集 hard cases，用于下一轮训练。

## 10. 推荐执行顺序

1. 恢复上一期生产 repo，确认四个裂缝 YOLO 的推理入口。
2. 定义 router 输出 schema 和路由表。
3. 将上一轮四类 router 封装为 baseline `coarse_router` 模块。
4. 将 router 输出接到类别专用裂缝 YOLO 调用逻辑。
5. 在 crop-based 裂缝检测前加入 bbox 外扩。
6. 加入低置信度 fallback 和墙类特殊分流策略。
7. 将 `内壁` 与 `RC壁` 合并为 `壁类`，重建 router 数据集。
8. 训练并评估三类 router。
9. 在完整 pipeline 里对比四类 router 和三类 router。
10. 按阶段记录错误来源：router 错误、裁剪/上下文错误、裂缝检测错误、合并/报告错误。

## 11. 工作原则

router 只应做视觉上有依据的判断。它的职责是识别可观测的建筑构件区域，并据此路由裂缝检测；不应在图像无法提供区分依据时强行判断 RC壁 或 内壁。

生产系统需要保留足够的中间信息，以解释最终错误来自类别路由、裁剪策略、裂缝检测，还是结果合并。

## 12. 生产 Repo 调查结果

上一期生产 repo 已 clone 到：

```text
/workspace/Shimizu-VLM-Crack-Detection-Prod
```

该 repo 是一个基于 YOLOv9 的 FastAPI 推理服务。当前服务已经封装了四个类别专用裂缝/损伤模型的调用逻辑。

核心文件如下：

- `api/config.py`：定义四个裂缝模型的默认路径和环境变量覆盖方式。
- `api/inference.py`：定义 `InferenceEngine`，启动时加载模型，并按 `types` 对输入图像逐个模型推理。
- `api/main.py`：定义 FastAPI 服务，其中 `/api/v1/analyze` 是当前主要推理接口。
- `api/schemas.py`：定义请求/响应 schema，并从 `type` 或 `types` 字段解析需要调用的模型类型。
- `utils/postprocess_plugin.py`：对同模型/跨模型的重叠检测框做后处理合并。

当前四个裂缝模型类型与默认权重路径：

```text
天井   -> infer_models/TIANJING.pt
内壁   -> infer_models/NEIBI.pt
耐震壁 -> infer_models/RCBI.pt
RC柱   -> infer_models/RCZHU.pt
```

注意：prod repo 使用 `耐震壁` 表示上一轮讨论中的 `RC壁`。因此两个项目结合时需要做名称归一：

```text
当前 router 文档/数据中的 RC壁 == prod repo 中的 耐震壁
```

当前生产 API 的调用方式是：

```text
前端传入 image + type/types
  -> 后端按 type/types 调用指定裂缝模型
  -> 返回标注图和 detections
```

这正是用户反馈“同一张图片被多个模型并行使用，结果很乱”的来源。router 的接入目标就是把前端手动传 `types` 改成由系统自动识别构件类别，再决定调用哪些裂缝模型。

当前 clone 下来的 prod repo 没有包含 `.pt` 权重文件，`infer_models/` 目录也不存在。后续需要通过模型恢复包或环境变量补齐权重路径。

## 13. Router 与 Prod Pipeline 的结合方案

建议不要破坏旧接口，先在 prod repo 中新增自动路由能力。旧接口继续保留，便于回归测试和人工指定类型。

整体结合方式：

```text
旧模式：
前端传 image + type/types
  -> 直接调用指定裂缝模型

新模式：
前端只传 image
  -> router 自动识别构件区域和类别
  -> 根据 router 结果决定裂缝模型调用
  -> 对 router 区域做外扩 crop
  -> 裂缝模型在 crop 上推理
  -> 裂缝 bbox 回写到原图坐标
  -> 汇总 router、routed regions、crack detections、warnings
```

推荐新增接口：

```text
POST /api/v1/analyze_auto
```

保留现有接口：

```text
POST /api/v1/analyze
```

这样可以降低改动风险，也方便对比：

- 手动指定类型的旧流程
- router 自动分流的新流程

## 14. 第一轮工程实现计划

第一轮目标是先把自动路由框架接入 prod pipeline，不急于重新训练模型。模型和数据恢复后，再插入实际 router 权重和裂缝模型权重做端到端验证。

### 14.1 新增 Router 配置

在 prod repo 的 `api/config.py` 中新增 router 相关环境变量：

```text
ROUTER_MODEL_PATH
ROUTER_MODE=four_class|three_class
ROUTER_IMAGE_SIZE
ROUTER_CONF_THRES
ROUTER_IOU_THRES
ROUTER_MAX_DET
ROUTER_EXPAND_RATIO
ROUTER_LOW_CONF_THRES
WALL_POLICY=run_both|primary_with_fallback|metadata
WALL_PRIMARY_TYPE=内壁|耐震壁
```

初始建议：

```text
ROUTER_MODE=three_class
ROUTER_EXPAND_RATIO=0.20
WALL_POLICY=run_both
```

### 14.2 新增 RouterEngine

新增文件建议：

```text
api/router.py
```

职责：

- 加载 coarse router 权重。
- 对输入图像进行构件区域检测。
- 输出 router detections。
- 支持四类和三类两种模式。
- 将四类结果中的 `内壁` 和 `耐震壁/RC壁` 归并为 `壁类`。
- 计算外扩 bbox。

router 输出字段建议：

```json
{
  "region_id": "r001",
  "router_class": "壁类",
  "raw_router_class": "内壁",
  "confidence": 0.82,
  "bbox_xyxy": [100, 80, 1200, 900],
  "expanded_bbox_xyxy": [0, 0, 1350, 1000],
  "review_required": false
}
```

### 14.3 重构裂缝推理入口

当前 `InferenceEngine.run_inference()` 接收 base64 图像，并在内部解码后对整图运行指定模型。

为了支持 router crop，需要在 `api/inference.py` 中增加内部复用方法：

```text
run_inference_on_image(crack_types, im0, ...)
run_inference_on_region(crack_types, im0, region_bbox, ...)
```

其中 `run_inference_on_region` 应做：

1. 从原图中按 `expanded_bbox_xyxy` 裁剪 crop。
2. 对 crop 调用类别专用裂缝模型。
3. 将裂缝检测 bbox 从 crop 坐标回写到原图坐标。
4. 给每个裂缝检测结果增加 `router_region_id` 和 `source_crack_type`。

### 14.4 路由表

三类 router 的路由表：

```text
router 天井
  -> 裂缝模型 天井

router RC柱
  -> 裂缝模型 RC柱

router 壁类
  -> 按 WALL_POLICY 处理
```

初始墙类策略使用：

```text
WALL_POLICY=run_both
壁类 -> 同时调用 内壁 + 耐震壁
```

原因：

- 用户已经确认 内壁 与 RC壁/耐震壁 在图像上可能无法区分。
- 强行二选一会把 router 错误叠加到裂缝检测上。
- 先运行两个墙类模型更保守，漏检风险较低。

后续可以再引入：

- metadata 规则
- primary model + fallback
- 双模型结果合并策略

### 14.5 新增自动路由 API

新增接口建议：

```text
POST /api/v1/analyze_auto
```

请求体初版：

```json
{
  "image": "data:image/jpeg;base64,...",
  "router_conf_thres": 0.25,
  "router_expand_ratio": 0.2,
  "wall_policy": "run_both",
  "postprocess": true
}
```

响应体初版：

```json
{
  "success": true,
  "image": "data:image/png;base64,...",
  "router": {
    "mode": "three_class",
    "model": "path-or-version",
    "detections": []
  },
  "routed_regions": [],
  "detections": [],
  "warnings": []
}
```

其中 `detections` 是裂缝/损伤检测结果，应继续包含：

```text
bbox
confidence
level
type
```

并新增：

```text
router_region_id
router_class
crop_bbox_xyxy
```

### 14.6 低置信度与 fallback

第一版建议实现以下规则：

- router 高置信度：按 Top-1 路由。
- router 识别为 `壁类`：不做墙体子类硬判，进入 `WALL_POLICY`。
- router 低于 `ROUTER_LOW_CONF_THRES`：标记 `review_required=true`。
- 若 router 没有检测结果：返回 warning，并可选择回退到旧多模型模式或要求人工复核。

建议 warning 类型：

```text
no_router_detection
low_router_confidence
wall_subtype_unresolved
fallback_to_manual_review
fallback_to_all_models
```

### 14.7 可视化与 QA

自动路由接入后，需要输出或保留以下可视化信息：

- router 原始框
- router 外扩框
- 裂缝检测框
- 每个裂缝框来自哪个 router region
- 每个裂缝框来自哪个类别专用模型

这一步很重要，因为后续需要区分错误来源：

```text
router 类别错
router 框太小或区域不完整
墙类分流策略导致结果冲突
裂缝模型自身漏检/误检
后处理合并错误
```

## 15. 后续模型与数据恢复后的验证计划

模型和数据恢复后，按以下顺序验证：

1. 确认 prod repo 四个裂缝模型权重可加载。
2. 确认当前四类 router 权重可加载。
3. 用少量样本跑通 `/api/v1/analyze` 旧接口，确认裂缝模型本身正常。
4. 用少量样本跑通 `/api/v1/analyze_auto` 新接口，确认 router + crop + 裂缝推理 + 坐标回写正常。
5. 对比整图裂缝推理与 crop 裂缝推理的差异。
6. 记录 router 框过小、类型错误、墙类冲突等问题。
7. 基于恢复的数据构建三类 router 数据集：`天井`、`壁类`、`RC柱`。
8. 训练三类 router。
9. 在完整 pipeline 中对比四类 router 与三类 router。

短期验证重点不是 mAP，而是：

- router 是否命中主要构件区域
- crop 是否覆盖裂缝和上下文
- 自动路由是否减少多模型并行造成的混乱
- 墙类 run-both 策略是否带来可接受的结果数量和重复框
- 最终输出是否能解释错误来源

## 16. MIT 版本 YOLO 调查与迁移成本评估

另一个候选 YOLO repo 已 clone 到：

```text
external/MultimediaTechLab-YOLO
```

该 repo 的 `LICENSE` 文件为 MIT License，README 也明确说明这是 MIT License 的官方 YOLO 实现。它覆盖 YOLOv7、YOLOv9 和 YOLO-RD。

### 16.1 项目结构差异

该 MIT repo 与当前 prod repo 使用的 WongKinYiu/yolov9 脚本式结构差异较大。

当前 prod repo：

```text
detect.py / train.py / val.py
models/common.py
utils/general.py
DetectMultiBackend
FastAPI 直接封装 DetectMultiBackend
```

MIT repo：

```text
yolo/lazy.py
Hydra config
LightningModule
yolo/model/yolo.py
yolo/tools/solver.py
yolo/tools/data_loader.py
yolo/utils/model_utils.py
```

MIT repo 的主要入口：

```text
python yolo/lazy.py task=train dataset=... model=v9-c ...
python yolo/lazy.py task=inference task.data.source=...
```

这意味着它不是现有 prod repo 的 drop-in replacement。迁移时需要重写模型加载、推理、后处理和 API 封装。

### 16.2 License 结论

MIT repo 的 license 状态清晰：

```text
MIT License
Copyright (c) 2024 Kin-Yiu, Wong and Hao-Tang, Tsui
```

如果客户或生产环境对 GPL/AGPL 等 license 有严格限制，则这个 repo 是更适合作为长期生产基线的方向。

但这只是工程 license 判断，不构成法律意见。正式对外发布前仍建议由项目负责人确认第三方依赖、预训练权重来源和部署方式是否满足客户要求。

### 16.3 数据兼容性

MIT repo 支持类似 YOLO 的目录结构：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
```

但它读取 txt label 的逻辑与常见 YOLO `class x_center y_center width height` 不完全一致。

在 `yolo/tools/data_loader.py` 中，txt label 会被读取为：

```text
class x1 y1 x2 y2 ...
```

然后将后续坐标 reshape 成点集，并用 min/max 得到 bbox。因此，如果直接把标准 YOLO box label：

```text
class x_center y_center width height
```

交给它，会被误解释为两个点：

```text
(x_center, y_center), (width, height)
```

这不是正确的 bbox。

因此，现有数据不能原样直接使用，需要做 label 转换：

```text
YOLO x_center y_center width height
  -> x1 y1 x2 y2
```

或者转成四点 polygon：

```text
class x1 y1 x2 y1 x2 y2 x1 y2
```

这个转换成本不高，适合用脚本完成。router 数据和裂缝数据都可以这样转换。

### 16.4 权重兼容性

MIT repo 的 `create_model()` 和 `save_load_weights()` 使用它自己的模型结构和 state_dict key 命名。

它包含一些旧权重转换逻辑，例如 `yolo/tools/format_converters.py`，但当前无法假设上一期 WongKinYiu/yolov9 训练出来的 `.pt` 权重可以直接加载。

主要风险：

- prod repo 的 `.pt` 可能是包含 `model` / `ema` / optimizer 等字段的训练 checkpoint。
- MIT repo 期望的是自身结构的 state_dict 或 Lightning `state_dict`。
- 即使是 YOLOv9-C/GELAN-C，模块命名和 head 结构也可能不同。
- 类别数不同会导致 detection head shape mismatch。

因此，旧权重迁移需要等模型文件恢复后实测。当前预估：

```text
直接加载旧权重：高风险，可能失败
写转换脚本迁移 backbone：中等到高成本
用 MIT repo 重新训练 router：中等成本
用 MIT repo 重新训练四个裂缝模型：高成本
```

### 16.5 推理 API 迁移成本

当前 prod API 依赖 `DetectMultiBackend`，核心推理在：

```text
api/inference.py
```

MIT repo 没有等价的 `DetectMultiBackend` API。它的推理走：

```text
InferenceModel
create_model
create_converter
PostProcess
StreamDataLoader
draw_bboxes
```

如果要迁移到 MIT repo，需要重写：

- 模型加载
- 图像预处理
- 推理调用
- NMS/PostProcess
- bbox 坐标还原
- base64 API 输入输出
- router crop 推理
- 可视化绘制

这部分是主要工程成本。

### 16.6 迁移策略建议

不建议立刻把当前 prod repo 全量替换成 MIT repo。建议分阶段做：

#### 阶段 A：保留现有 prod repo，完成 router 接入

目的：

- 先把业务链路跑通。
- 验证自动路由是否能解决用户反馈的多模型混乱问题。
- 利用已存在的四个裂缝模型和现有 API，最快形成端到端结果。

风险：

- license 问题仍需后续处理。

#### 阶段 B：用 MIT repo 训练三类 router POC

目的：

- 验证 MIT repo 能否稳定训练我们的自定义数据。
- 确认 label 转换、训练、推理、指标和导出流程。
- 优先只做三类 router：`天井`、`壁类`、`RC柱`。

理由：

- router 数据规模相对小。
- 类别少。
- 业务风险低于一次性迁移四个裂缝模型。
- 可以快速判断 MIT repo 是否适合作为长期基线。

#### 阶段 C：评估裂缝模型迁移

如果阶段 B 成功，再考虑：

- 将四个裂缝数据集转换到 MIT repo 格式。
- 重新训练四个裂缝模型。
- 或尝试只迁移旧权重 backbone，再 fine-tune detection head。
- 对比旧 prod 模型和 MIT 模型的 B-D 检测效果。

#### 阶段 D：替换生产推理内核

当 MIT repo 训练出的模型达到可接受效果后，再重写 prod API 的推理内核，使其不再依赖当前非 MIT YOLOv9 代码。

### 16.7 当前成本判断

综合判断：

```text
只确认 license 和跑通 demo：低成本
将 router 数据转成 MIT repo 可训练格式：低到中成本
用 MIT repo 重新训练三类 router：中成本
把现有 router 权重直接迁移到 MIT repo：中到高成本，需实测
把四个裂缝模型直接迁移到 MIT repo：高成本
重写 prod API 以完全替换推理内核：中到高成本
```

当前最务实的路线：

1. 继续先把 router 接入现有 prod pipeline，跑通业务闭环。
2. 并行准备 MIT repo 的三类 router POC。
3. 如果 MIT router POC 成功，再决定是否迁移四个裂缝模型和生产推理 API。

这样既不阻塞今天的主目标，也为 license 合规方向留出清晰路径。



## 17. Gemini 全量标注后的三类路由训练计划

### 17.1 当前目标

当前正在重新调用 `gemini-3.1-pro-preview` 对原始四类图片做全量构件粗框标注。

本阶段目标不是直接训练裂缝 B-D 检测模型，而是重新训练一个自动类别识别/路由模型，使输入图片可以先被自动分配到合适的后续裂缝检测流程。

新的路由类别不再沿用四类：

```text
天井
内壁
RC壁
RC柱
```

而是合并为三类：

```text
天井
壁类 = 内壁 + RC壁
RC柱
```

原因是用户会议中已经确认：`内壁` 和 `RC壁/耐震壁` 在外观上高度相似，VLM 和视觉检测模型都难以稳定区分。继续强行区分会把系统性误差引入流程最前端，并在后续裂缝检测阶段继续放大。

### 17.2 Gemini 结果 QA

Gemini 全量结果完成后，先做 QA，不直接训练。

必须检查：

- `ok/error` 数量。
- 每个原始类别的图片数、框数。
- 每个目标类别的框数。
- 空框、越界框、异常小框、异常大框。
- 一图多框情况。
- Gemini 输出标签与原始目录标签的一致性。
- `内壁` 与 `RC壁` 混淆情况。
- 抽样可视化页面和 contact sheet。

这一步的目标是判断 Gemini 粗标注是否足以作为 YOLO 路由训练信号。

### 17.3 三类标签转换规则

训练路由模型时使用如下转换：

```text
天井 -> 天井
内壁 -> 壁类
RC壁 -> 壁类
RC柱 -> RC柱
```

YOLO 类别建议固定为：

```yaml
names:
  0: 天井
  1: 壁类
  2: RC柱
```

注意命名归一：

```text
文档/原始数据: RC壁
prod repo: 耐震壁
业务含义: 同一类结构墙模型
```

### 17.4 构建新的 YOLO 路由数据集

基于 Gemini `results.jsonl` 构建新的三类 YOLO 数据集：

```text
coarse_router_yolov9/datasets/coarse_router_3class/
  images/
  labels/
  data.yaml
```

建议同时保留追踪信息：

```text
image_path
expected_label
gemini_labels
original_gemini_bbox
merged_class
source_results_jsonl
```

可以先构建两个版本：

```text
coarse_router_3class_raw
coarse_router_3class_filtered
```

`raw` 版本直接使用 Gemini 框，快速验证训练可行性。

`filtered` 版本再加入低置信度过滤、异常框过滤和必要的标签规则修正。

### 17.5 训练与评估重点

路由模型的目标是“把图片或区域送到正确后续模型”，不是精细损伤检测。

评估重点：

- 三类 mAP。
- `天井`、`壁类`、`RC柱` 的 recall。
- 混淆矩阵。
- `RC柱` 被误判为 `壁类` 的比例。
- 低置信度样本比例。
- bbox 是否足够覆盖主要构件区域。

路由阶段宁可框稍大，也不要漏掉主要构件。后续裂缝检测可以在扩展后的裁剪区域上运行。

### 17.6 接入裂缝检测流程

新的自动流程建议如下：

```text
输入图片
  -> 三类路由 YOLO 检测构件区域
  -> 天井区域送天井裂缝模型
  -> RC柱区域送 RC柱裂缝模型
  -> 壁类区域同时送内壁模型和耐震壁/RC壁模型
  -> 合并裂缝检测结果
  -> NMS/去重/置信度融合
  -> 输出最终结果和路由信息
```

短期内，`壁类` 不再尝试在路由阶段二分 `内壁/RC壁`。先并行运行两个墙类裂缝模型，再通过后处理解决冲突。

### 17.7 风险与缓解

主要风险：

- Gemini 框过小，导致后续裂缝检测漏检。
- 路由模型误判导致后续裂缝模型选择错误。
- `壁类` 双模型输出重复或冲突。
- 原始四类目录标签本身与视觉内容不完全一致。

缓解策略：

- 路由 bbox 推理时默认扩展 15%-25%。
- 低置信度或多类别冲突时触发 fallback，多跑一个或多个裂缝模型。
- 对 `壁类` 的内壁/耐震壁双模型结果做统一 NMS 和去重。
- 在响应中保留 router 信息，便于人工追踪错误来源。

### 17.8 近期执行顺序

1. 等待 `gemini-3.1-pro-preview` 全量标注完成。
2. 统计并可视化 Gemini 输出。
3. 改造 `build_coarse_yolo_dataset.py`，支持三类合并标签。
4. 构建 `coarse_router_3class_raw` 数据集。
5. 训练三类 YOLO 路由模型。
6. 根据混淆矩阵和 QA 结果决定是否构建 `filtered` 版本。
7. 在 prod repo 中新增自动路由 API，将三类 router 接入四个旧裂缝模型。
