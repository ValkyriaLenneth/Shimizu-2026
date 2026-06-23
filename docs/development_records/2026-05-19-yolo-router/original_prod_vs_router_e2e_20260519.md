# 原版 prod 推理 vs 当前 router-E2E 对比

日期：2026-05-19

## prod 原版推理方式确认

上一期 prod 仓库位置：

- `/workspace/Shimizu-VLM-Crack-Detection-Prod`

关键代码：

- `/workspace/Shimizu-VLM-Crack-Detection-Prod/api/main.py`
- `/workspace/Shimizu-VLM-Crack-Detection-Prod/api/schemas.py`
- `/workspace/Shimizu-VLM-Crack-Detection-Prod/api/inference.py`

结论：

- prod API 不是自动识别建筑类别。
- `/api/v1/analyze` 要求前端传 `type` 或 `types`。
- 后端根据传入的类型调用对应模型。
- `types` 实际支持多个值，所以后端能力上可以多模型推理；但是否多模型并行取决于用户/前端传参。
- 典型线上使用应理解为“用户选择类别后推理”。客户反馈“一个图片被多个模型并行使用，结果很乱”，应来自前端传了多个 `types` 或测试脚本中的多组合模式。

prod 默认推理参数：

| 参数 | 值 |
|---|---:|
| IMAGE_SIZE | 960 |
| CONF_THRES | 0.01 |
| IOU_THRES | 0.45 |
| MAX_DET | 1000 |
| POSTPROCESS_ENABLE | true |
| POSTPROCESS_SAME_MODEL_IOA | 0.7 |
| POSTPROCESS_CROSS_MODEL_IOU | 0.55 |

## 本次对比设置

对同一批 E2E 80 张抽样图片，模拟“用户正确选择原始类别”的原版流程：

| manifest class_key | prod 请求类型 |
|---|---|
| tenjo | 天井 |
| inner_wall | 内壁 |
| rc_wall | 耐震壁 |
| rc_column | RC柱 |

即：原版 prod 在全图上跑单个正确类别模型；当前流程为 router 切区域后调用下游模型。

对比脚本：

- `router_crack_pipeline/scripts/compare_original_prod_inference.py`

输出目录：

- `outputs/e2e_debug_sample_80_router_merged4219_ddp_20260519/original_prod_compare`

关键文件：

- `summary.json`
- `original_results.jsonl`
- `original_eval_by_image.csv`
- `comparison_index.csv`
- `contact_sheet.jpg`
- `visualizations/*.jpg`

## 指标对比

| 类别 | GT | 当前 router-E2E match | 当前 FN | 当前 FP | 原版 prod match | 原版 FN | 原版 FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 天井 | 25 | 24 | 1 | 5 | 24 | 1 | 18 |
| 内壁 | 22 | 15 | 7 | 10 | 16 | 6 | 23 |
| RC壁 | 23 | 15 | 8 | 5 | 17 | 6 | 31 |
| RC柱 | 22 | 9 | 13 | 6 | 15 | 7 | 40 |
| 合计 | 92 | 63 | 29 | 26 | 72 | 20 | 112 |

## 目视结论

### 1. 原版 prod 召回更高，但误检非常多

原版全图推理在同一批样本上从 `63/92` 提升到 `72/92`，主要收益来自 RC柱：

- 当前 RC柱：`9/22`
- 原版 RC柱：`15/22`

但原版 FP 从当前的 `26` 增加到 `112`。这说明原版全图 + 低阈值策略更偏召回，用户看到的结果会明显更乱。

### 2. 当前 router-E2E 更干净，但牺牲了召回

当前流程通过 router 限定区域，FP 明显减少：

- 当前 FP：26
- 原版 FP：112

但如果 router 没把正确区域送给对应模型，或者切片影响下游模型，主分支就会漏检。

### 3. RC柱是最明显的 trade-off

代表图：

- `visualizations/65_rc_column_d-173.jpg`
- `visualizations/77_rc_column_4-C-00022.jpg`
- `visualizations/64_rc_column_d-40004.jpg`

现象：

- 原版全图 RC柱模型能检出部分当前流程漏掉的柱裂缝。
- 但原版会给出更多低置信度大框或背景框。
- 当前流程失败常见原因是 router 把柱区域判成壁类，或 router 命中后下游 RC柱模型在切片上无输出。

### 4. 墙类原版也更容易多报

代表图：

- `visualizations/60_rc_wall_c-40773.jpg`
- `visualizations/59_rc_wall_c-40621.jpg`

现象：

- 原版全图有时能补到当前流程漏检。
- 但也会在同一张图里输出更多大框/边缘框。
- 当前流程更接近“干净结果”，但如果 router 框没有覆盖足够上下文，下游模型容易漏。

## 工程判断

当前 router-E2E 的方向是对的，因为它解决了用户反馈的“多个模型并行结果混乱”问题。但不能简单替换原版全图推理，因为：

- 原版全图单模型召回仍然更高；
- 当前流程在 RC柱上召回损失明显；
- 部分漏检不是下游模型能力问题，而是 router/切片策略带来的信息损失。

建议下一步：

1. 对 RC柱加入 fallback：当 router 输出壁类但存在低置信 RC柱候选，或图像存在竖向柱状结构时，同时调用 RC柱模型。
2. 对 router 命中的区域尝试扩大 padding：比较 `0.10 / 0.20 / 0.30`。
3. 对当前漏检但原版命中的样本，单独统计“router 没送到正确模型”和“送到了但切片后无输出”的比例。
4. 正式输出继续避免无条件四模型全图并行；可以把多模型推理作为低置信 fallback，而不是默认行为。

## 追加确认：固定 prod 参数后，router 切片是否减少 FP

用户提出的判断方向是正确的：如果固定同一个检测模型、同一套阈值、同一套后处理，那么 router 切掉无关区域后，理论上 FP 应该减少。

为确认这一点，重新跑了一组 prod 参数版 router-E2E：

- 配置：`router_crack_pipeline/configs/pipeline.router_merged4219_ddp.prodparams.local.yaml`
- 输出：`outputs/e2e_debug_sample_80_router_merged4219_ddp_prodparams_20260519`
- 下游裂缝模型参数对齐 prod：`imgsz=960`、`conf_threshold=0.01`、`iou_threshold=0.45`

### raw 主分支统计

| 类别 | 整图 prod FP | router 切片 FP | 整图 prod match | router 切片 match |
|---|---:|---:|---:|---:|
| 天井 | 18 | 46 | 24 | 25 |
| 内壁 | 23 | 87 | 16 | 18 |
| RC壁 | 31 | 90 | 17 | 17 |
| RC柱 | 40 | 76 | 15 | 13 |
| 合计 | 112 | 299 | 72 | 73 |

### 正式合并输出统计

额外用 `crack_detections` 而不是 `raw_crack_detections` 重算：

- 输出 CSV：`outputs/e2e_debug_sample_80_router_merged4219_ddp_prodparams_20260519/eval_by_image_merged_outputs.csv`

| 类别 | router 正式输出 match | FN | FP |
|---|---:|---:|---:|
| 天井 | 25/25 | 0 | 41 |
| 内壁 | 17/22 | 5 | 72 |
| RC壁 | 15/23 | 8 | 74 |
| RC柱 | 12/22 | 10 | 57 |
| 合计 | 69/92 | 23 | 244 |

### 结论修正

当前实现没有验证出“router 后 FP 减少”，反而 FP 增加。原因不是“同一区域同模型识别能力下降”，而是当前 router-E2E 实现有几个工程差异：

1. router 不是只输出一个干净区域，一张图平均会输出多个区域。
2. 墙类 router 区域会同时调用内壁和 RC壁两个模型，raw 候选天然变多。
3. prod 低阈值 `0.01` 在 crop 上会释放大量低置信框。
4. 当前后处理没有完全复刻 prod 的 `same_model_ioa=0.7`、`cross_model_iou=0.55` 聚合逻辑，而且多 router region 的重复框需要跨 region 合并。
5. router box 经常很大或重叠，实际没有充分去掉无关区域；有时还会把局部区域放大，使局部纹理更容易触发低置信检测。

因此下一步不是简单降低阈值或扩大切片，而是要先把 router-E2E 的后处理补齐：

1. 所有 router region 映射回原图后，先按 `source_model` 做跨 region 去重。
2. 对同模型同等级/跨等级使用 prod 的 IoA 合并逻辑。
3. 对墙类并行的内壁/RC壁输出，再做跨模型冲突合并与等级差异标记。
4. 只有在这套后处理后，再评估“router 是否减少 FP”。

## 已实现的修正实验

已在 pipeline 中增加两项能力：

1. `region_transport: full_image_filter`
   - 下游裂缝模型接收整张图，避免 crop 边界和尺度上下文变化。
   - 检测结果映射仍在原图坐标内，只保留中心点落入 router box 或 IoA 达标的结果。
   - 这样可以验证“只用 router 限定结果区域，而不是改变检测模型输入”。

2. `crack_merge.mode: prod_like`
   - 复刻 prod 的同模型 IoA 去重逻辑：高等级覆盖低等级。
   - 支持跨模型 IoU 聚合。
   - 对本项目而言，跨模型聚合会改变 `source_model`，所以单模型公平对比时也测试了 `cross_model_iou_threshold: 1.01`，即只做同模型去重。

相关代码：

- `router_crack_pipeline/pipeline/run_full_pipeline.py`
- `router_crack_pipeline/pipeline/result_merge.py`

实验配置：

- 整图推理 + prod-like 跨模型合并：`router_crack_pipeline/configs/pipeline.router_merged4219_ddp.fullimage_prodmerge.local.yaml`
- 整图推理 + 只做同模型去重：`router_crack_pipeline/configs/pipeline.router_merged4219_ddp.fullimage_samemodel.local.yaml`

### 同口径结果

| 流程 | 正式输出 match | FN | FP | 说明 |
|---|---:|---:|---:|---|
| 整图 prod 单模型 | 72/92 | 20 | 112 | 用户正确选择类别，全图推理 |
| router crop + prod 参数 + prod-like 合并 | 69/92 | 23 | 244 | crop 仍会产生大量重复/边界问题 |
| router full-image filter + prod-like 跨模型合并 | 60/92 | 32 | 73 | FP 降低，但跨模型合并影响主分支统计 |
| router full-image filter + 同模型去重 | 66/92 | 26 | 77 | 当前更公平的候选方向 |

### 当前确认结论

`full_image_filter + 同模型去重` 已经验证出预期方向：

- FP 从整图 prod 的 `112` 降到 `77`。
- 但 match 从 `72/92` 降到 `66/92`。

也就是说，router 限定结果区域确实可以减少无关区域 FP；剩余问题是 FN 增加，主要来自：

1. router 没输出正确类别，尤其 RC柱。
2. router box 过滤规则可能过严，部分整图检测结果被排除。
3. 墙类双模型并行后的跨模型合并需要作为输出策略，而不是混入主分支单模型评价。

下一步优化重点应从“是否切图”转为：

- 保持 `full_image_filter`，不再默认 crop 给下游模型。
- 调整 router 过滤规则，例如比较 `center_or_ioa`、`ioa`、不同 `region_filter_ioa_threshold`。
- 给 RC柱增加 fallback，使壁类/RC柱低置信混淆时能同时保留 RC柱候选。
