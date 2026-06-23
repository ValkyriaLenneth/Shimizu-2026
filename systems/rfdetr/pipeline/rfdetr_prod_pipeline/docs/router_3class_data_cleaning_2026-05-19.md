# 三类路由数据检查与清洗阶段成果

日期：2026-05-19

## 1. 阶段目标

本阶段完成从 Gemini 3.1 全量构件粗标注到三类 YOLO 路由训练数据的转换与清洗。

核心目标：

1. 建立量化数据评价标准。
2. 对每张 Gemini 标注结果打分并标记可疑样本，供人工确认。
3. 合并 `内壁` 与 `RC壁` 为 `壁类`，生成 `full` 和 `cleaned` 两套三类 YOLO 数据集。
4. 生成可详细说明流程的文档、表格和例图。

## 2. 输入

- Gemini 结果：`outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19/results.jsonl`
- Gemini 模型：`gemini-3.1-pro-preview`
- 原始图片：`data/unzip/`
- 总样本：`3015`

原始类别数量：

| 原始类别 | 数量 |
|---|---:|
| 天井 | 742 |
| 内壁 | 857 |
| RC壁 | 981 |
| RC柱 | 435 |

## 3. 三类合并规则

| 原始标签 | 路由训练标签 |
|---|---|
| 天井 | 天井 |
| 内壁 | 壁类 |
| RC壁 | 壁类 |
| RC柱 | RC柱 |

YOLO 类别定义：

```yaml
names:
  0: 天井
  1: 壁类
  2: RC柱
```

## 4. 量化评价与清洗规则

每张图初始分为 100 分，发现问题后扣分。所有 issue 都保留在审计表与 QA 页面中。

`cleaned` 数据集保留规则：

```text
score >= 80
and no critical issue
and valid_boxes_count > 0
```

关键错误包括：

```text
not_ok
image_missing
image_unreadable
empty_elements
no_valid_boxes
invalid_label
invalid_bbox_shape
invalid_bbox_value
invalid_bbox_geometry
```

非关键 issue 仍进入人工确认清单，但不会自动从 `cleaned` 删除，例如：

```text
almost_full_image_box
multi_class_conflict
label_mismatch
too_large_box
thin_box
too_small_box
low_confidence
```

## 5. 审计结果

总体结果：

| 指标 | 数值 |
|---|---:|
| 总图片 | 3015 |
| Gemini OK | 3015 |
| 可疑样本 | 2530 |
| cleaned 保留 | 2710 |

主要 issue 计数：

```json
{
  "almost_full_image_box": 976,
  "too_large_box": 851,
  "label_mismatch": 162,
  "multi_class_conflict": 1668,
  "thin_box": 20,
  "too_small_box": 17,
  "low_confidence": 1,
  "empty_elements": 2,
  "no_valid_boxes": 2,
  "too_many_boxes": 1
}
```

Gemini 原始标签框数：

```json
{
  "内壁": 2284,
  "天井": 1752,
  "RC壁": 992,
  "RC柱": 762
}
```

合并后三类框数：

```json
{
  "壁类": 2587,
  "天井": 1708,
  "RC柱": 696
}
```

## 6. 按原始类别的清洗影响

| 原始类别 | 原始数量 | cleaned 保留 | clean 掉 | clean 掉比例 |
|---|---:|---:|---:|---:|
| 天井 | 742 | 690 | 52 | 7.01% |
| 内壁 | 857 | 743 | 114 | 13.30% |
| RC壁 | 981 | 915 | 66 | 6.73% |
| RC柱 | 435 | 362 | 73 | 16.78% |


总体：

| 指标 | 数值 |
|---|---:|
| 原始总数 | 3015 |
| cleaned 保留 | 2710 |
| clean 掉 | 305 |
| clean 掉比例 | 10.12% |

详细统计已固化到：

- `outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19/qa/class_cleaning_summary.csv`
- `outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19/qa/class_cleaning_summary.json`

## 7. 生成的数据集

### Full 数据集

路径：`coarse_router_yolov9/datasets/coarse_router_3class_full`

| 指标 | 数值 |
|---|---:|
| 图片 | 3013 |
| 框 | 5790 |
| train 图片 | 2411 |
| val 图片 | 302 |
| test 图片 | 300 |

类别框数：

```json
{
  "壁类": 3276,
  "天井": 1752,
  "RC柱": 762
}
```

### Cleaned 数据集

路径：`coarse_router_yolov9/datasets/coarse_router_3class_cleaned`

| 指标 | 数值 |
|---|---:|
| 图片 | 2710 |
| 框 | 5176 |
| train 图片 | 2168 |
| val 图片 | 271 |
| test 图片 | 271 |

类别框数：

```json
{
  "壁类": 2987,
  "天井": 1492,
  "RC柱": 697
}
```

## 8. QA 与人工确认产物

QA 目录：`outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19/qa/`

关键文件：

- `annotation_audit.csv`：逐图片评分、issue、标签与框统计。
- `annotation_audit.json`：审计摘要。
- `suspicious_samples.jsonl`：可疑样本清单，供人工确认。
- `bbox_stats.csv`：逐框面积、长宽比、issue。
- `label_confusion_summary.csv`：原始合并标签与 Gemini 合并标签的混淆统计。
- `class_cleaning_summary.csv`：按原始四类统计 clean 影响。
- `cleaning_report.md`：自动生成的详细清洗报告。
- `index.html`：QA 例图总览。
- `suspicious/index.html`：可疑样本例图。
- `label_conflicts/index.html`：标签冲突例图。
- `clean_examples/index.html`：高质量样本例图。

已生成 QA 例图：`240` 张。

## 9. 脚本

完整流程脚本：

```text
scripts/audit_and_build_router_3class.py
```

重跑命令：

```bash
python3 scripts/audit_and_build_router_3class.py
```

## 10. 当前判断

`full` 数据集适合先训练 baseline，最大化利用 Gemini 标注信号。

`cleaned` 数据集排除了关键错误和低分样本，仍保留带非关键 issue 的可训练样本，并通过 manifest/QA 标记供人工复查。它适合与 `full` 对比训练效果，重点观察三类 recall 和混淆矩阵。

下一步建议：先用 `coarse_router_3class_full` 训练三类 router baseline，再用 `coarse_router_3class_cleaned` 训练对照模型。
