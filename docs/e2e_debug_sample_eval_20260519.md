# E2E 小样本评估调试记录

日期：2026-05-19

## 目的

用少量数据验证当前 E2E 评估方案是否可落地，重点检查：

- 主分支/次分支是否能从 E2E 输出中拆分。
- `内壁` 和 `RC壁` 合并为 router 的 `壁类` 后，是否还能按原始标签判断主模型。
- 同一 wall 区域同时经过 `inner_wall` 与 `rc_wall` 模型时，能否统计等级偏移。
- 当前输出结构是否足够支持后续扩大评估。

## 本次改动

pipeline 增加未合并下游输出：

```text
raw_crack_detections
```

新增调试脚本：

```text
router_crack_pipeline/scripts/debug_e2e_sample_eval.py
```

输出目录：

```text
outputs/e2e_debug_sample_20260519
```

主要输出：

```text
sampled_manifest.csv
results.jsonl
eval_by_image.csv
wall_grade_shift_pairs.csv
eval_summary.json
```

## 运行命令

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/debug_e2e_sample_eval.py \
  --per-class 2 \
  --device cpu \
  --output-dir outputs/e2e_debug_sample_20260519
```

本次每类抽 2 张，共 8 张。

## 主/次分支口径

| 原始类别 | 期望 router 类 | 主分支模型 | 次分支模型 |
|---|---|---|---|
| `tenjo` | `天井` | `ceiling` | 其他 |
| `inner_wall` | `壁类` | `inner_wall` | `rc_wall`、其他 |
| `rc_wall` | `壁类` | `rc_wall` | `inner_wall`、其他 |
| `rc_column` | `RC柱` | `rc_column` | 其他 |

## 本次结果

```json
{
  "images": 8,
  "wall_grade_shift_pairs": 1,
  "wall_grade_delta_rc_minus_inner": {
    "0": 1
  }
}
```

按类别汇总：

| 类别 | 图片 | router 命中 | GT框 | 主分支预测 | 次分支预测 | 主分支匹配 IoU50 | FN | FP | 等级正确 | 等级不一致 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| tenjo | 2 | 1 | 2 | 1 | 3 | 1 | 1 | 0 | 1 | 0 |
| inner_wall | 2 | 2 | 3 | 1 | 2 | 1 | 2 | 0 | 1 | 0 |
| rc_wall | 2 | 1 | 2 | 1 | 1 | 1 | 1 | 0 | 1 | 0 |
| rc_column | 2 | 2 | 2 | 2 | 2 | 1 | 1 | 1 | 1 | 0 |

## wall 等级偏移样例

本次只有 1 对 `inner_wall` 与 `rc_wall` 同区域检测满足重叠阈值：

```text
image: c-10001.JPG
gt_class: rc_wall
IoU: 0.9888
inner_wall_grade: B
rc_wall_grade: B
grade_delta_rc_minus_inner: 0
```

该样本没有出现客户担心的等级偏移，但样本数太少，不能说明风险不存在。

## 暴露的问题点

### 1. raw 输出需要 router region id

第一轮调试时发现 `raw_crack_detections` 能区分 `source_model`，但不能追溯到具体哪个 router bbox。

影响：

- 当 router 对同一类输出多个框时，同一个判别模型会被调用多次。
- 后续难以判断某个下游检测来自哪个区域，也难以分析重复检测是否由 router 多框引起。

已补充字段：

```text
router_region_index
router_bbox_xyxy
router_confidence
router_class_name
detector_input_shape
```

### 2. router 多框会增加次分支噪声

样例：

```text
tenjo/a-30095.jpg:
router = 天井 + 壁类 + 壁类
raw = ceiling 1 个 + inner_wall 1 个
```

主分支可正常评估，但次分支会增加 wall 输出。该输出不应计入主 FP，但需要在辅助统计里作为“额外模型调用/额外检测”记录。

### 3. wall 类主/次分支拆分可行，但必须依赖 manifest

`内壁` 和 `RC壁` 在 router 层都是 `壁类`，只有 `manifest.csv` 的 `class_key` 能决定主模型：

```text
inner_wall -> main_model=inner_wall
rc_wall -> main_model=rc_wall
```

这意味着正式评估脚本不能只读图片路径和 label txt，必须读 manifest。

### 4. router 漏路由会直接造成主分支漏检

样例：

```text
rc_wall/c-40463.jpg:
expected router = 壁类
router = RC柱, RC柱
raw = 0
warning = router_low_confidence_multi_model_fallback_todo
```

该类样本要单独统计为：

```text
router_miss_caused_main_fn
```

否则主分支 FN 会混合 router 错误和判别模型错误。

### 5. 主分支指标目前只能用于链路调试

上一期 GPL 判别模型未必使用本次 `final_crack_yolo_20260519` 数据训练，当前 8 张结果不能代表真实性能，只用于验证统计方法和输出结构。

## 下一步建议

1. 给 `raw_crack_detections` 增加 router region metadata。
2. 评估脚本增加 FN 来源拆分：
   - router miss。
   - router hit 但主模型无输出。
   - 主模型有输出但 IoU 不匹配。
   - IoU 匹配但等级错误。
3. 将样本扩大到每类 20 张，共 80 张。
4. 对 wall 样本单独扩大抽样，优先观察：
   - `inner_wall` 与 `rc_wall` 同区域检测对数。
   - `rc_wall_grade - inner_wall_grade` 的分布。
   - GT 为 `inner_wall` 和 GT 为 `rc_wall` 时偏移方向是否不同。

## 扩大到 80 张后的确认结果

运行命令：

```bash
coarse_router_yolov9/.venv/bin/python router_crack_pipeline/scripts/debug_e2e_sample_eval.py \
  --per-class 20 \
  --device 0 \
  --output-dir outputs/e2e_debug_sample_80_20260519
```

输出目录：

```text
outputs/e2e_debug_sample_80_20260519
```

抽样构成：

| 来源 | 数量 |
|---|---:|
| `labels_20251107` | 57 |
| `data_add100` | 23 |

按类别结果：

| 类别 | 图片 | router 命中 | GT框 | 主分支预测 | 主分支匹配 | Precision | Recall | 次分支预测 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tenjo | 20 | 18 | 25 | 25 | 23 | 92.00% | 92.00% | 9 |
| inner_wall | 20 | 20 | 22 | 24 | 15 | 62.50% | 68.18% | 14 |
| rc_wall | 20 | 19 | 23 | 21 | 16 | 76.19% | 69.57% | 14 |
| rc_column | 20 | 13 | 22 | 13 | 7 | 53.85% | 31.82% | 25 |

FN 来源拆分：

| 类别 | FN 总数 | router miss | router 命中但主模型无输出 | 主模型有输出但 IoU 不匹配 |
|---|---:|---:|---:|---:|
| tenjo | 2 | 2 | 0 | 0 |
| inner_wall | 7 | 0 | 1 | 6 |
| rc_wall | 7 | 1 | 3 | 3 |
| rc_column | 15 | 7 | 2 | 6 |

本轮确认的问题：

- `RC柱` 当前是最明显风险点：router 命中只有 13/20，且次分支预测 25 个，说明不少柱图被路由到 `壁类` 或 `天井`。
- `天井` 主分支效果较稳定，主要损失来自 router miss，而不是判别模型。
- `inner_wall` 和 `rc_wall` 的 router 命中较高，但主分支存在较多 IoU 不匹配，后续需要看可视化确认是框偏移、切片范围、还是旧模型/新数据分布差异。
- 次分支数量不可忽略，尤其 `RC柱` 的次分支预测超过主分支预测，正式评估必须继续保持主/次分支分离。

wall 等级偏移：

| 指标 | 数量 |
|---|---:|
| `inner_wall` vs `rc_wall` 同区域检测对 | 20 |
| 等级一致 | 19 |
| 等级不一致 | 1 |
| `rc_wall_grade - inner_wall_grade = -2` | 1 |

唯一强等级偏移样本：

```text
image: additional_data_2026-05-19/unpacked/data_add100/3_RC壁/train/images/c-10054.jpg
gt_class: rc_wall
IoU: 0.2709
IoA(min): 0.9428
inner_wall_grade: D
rc_wall_grade: B
grade_delta_rc_minus_inner: -2
```

该样本说明客户担心的“同一视觉区域在不同 wall 模型下等级不同”确实可能发生。虽然 80 张里只出现 1 例，但等级差达到 2 档，后续应优先扩大 wall 样本并生成可视化核对。
