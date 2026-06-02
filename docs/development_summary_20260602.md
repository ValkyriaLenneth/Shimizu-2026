# 2026-06-02 开发结果总结

## 总目标

今天的主要目标是推动 YOLO9 到 RF-DETR 的迁移验证，重点包括：

- 自动识别模型从 YOLO9 迁移到 RF-DETR，并确认是否达到客户提出的 Precision 0.90 目标。
- 下游单类别裂缝/损伤等级模型先从最弱的 RC柱 开始验证 RF-DETR 的有效性。
- 固定旧 `data_split.json` 的 official test 作为对齐基准，避免不同数据划分造成指标不可比。
- 为下周恢复现场准备模型、文档、数据划分说明和恢复说明。

## 重要安全事项

本轮对话中出现过 GitHub token。该 token 没有被写入任何文档、脚本或压缩包。建议在销毁服务器前或之后，直接到 GitHub 账户中撤销该 token 并重新生成。

## 自动识别模型 RF-DETR 迁移

自动识别模型使用与上周 YOLO 自动识别模型调优相同的数据和 test set。

RF-DETR 结果：

| model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv9 tuned baseline | 0.863 | 0.850 | - | 0.888 | 0.775 |
| RF-DETR epoch 23 | 0.905 | 0.852 | 0.877 | 0.904 | 0.782 |

结论：

- RF-DETR 自动识别模型达到客户目标 `Precision >= 0.90`。
- Recall 与 YOLO9 tuned baseline 基本持平。
- mAP50 和 mAP50-95 小幅提升。
- 自动识别模型可以先告一段落，后续重点转向下游单类别模型。

保留模型：

```text
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest/epoch_pth/checkpoint_epoch_023.pth
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest/checkpoint_23.ckpt
```

## 自动识别模型可视化

生成了 YOLO9 vs RF-DETR 左右对比图。左侧为 YOLO9，右侧为 RF-DETR；灰色框为 GT，彩色框为模型预测。

最终采用 case 3/4/5：

| case | image | YOLO9 问题 | RF-DETR 结果 |
|---:|---|---|---|
| 3 | `RC柱_d-40027_03307.jpg` | RC柱 -> 壁类 | RC column, conf 0.811 |
| 4 | `RC壁_c-199_03206.jpg` | RC柱 -> 壁类 | RC column, conf 0.336 |
| 5 | `RC壁_c-40616_03440.jpg` | 壁类 -> RC柱 | Wall, conf 0.633 |

素材：

```text
docs/report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_03_RC柱_d-40027_03307.jpg
docs/report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_04_RC壁_c-199_03206.jpg
docs/report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_05_RC壁_c-40616_03440.jpg
```

## RC柱 RF-DETR 下游模型

RC柱 是四个下游单类别裂缝/损伤等级模型中较弱的类别，因此今天优先从 RC柱 开始验证。

本轮使用与去年结果对齐的数据划分，没有使用半监督学习，只进行了基础监督学习。

RF-DETR RC柱 结果：

| 范围 | 去年报告目标 R | RF-DETR R |
|---|---:|---:|
| Overall | 0.742 | 0.826 |
| B | 0.700 | 0.750 |
| C | 0.706 | 0.727 |
| D | 0.807 | 1.000 |

整体指标：

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.661 | 0.826 | 0.725 | 0.726 | 0.299 |

结论：

- RC柱 RF-DETR 超过去年报告 recall 目标。
- B/C/D 三个等级均超过目标。
- 这是下游单类别模型迁移的第一个正向结果。

保留模型：

```text
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_047.pth
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/checkpoint_47.ckpt
```

## RC柱 可视化

生成了 YOLO9 未检出、RF-DETR 检出的对比图。左侧为 YOLO9，右侧为 RF-DETR；红色框表示 YOLO9 漏掉的 GT，右侧彩色框表示 RF-DETR 检出结果。

采用三个样例，分别覆盖 D/B/C：

| case | 等级 | YOLO9 结果 | RF-DETR 结果 |
|---:|---|---|---|
| 1 | D | 未检出 | D, conf 0.968 |
| 2 | B | 未检出 | B, conf 0.935 |
| 3 | C | 未检出 | C, conf 0.887 |

素材：

```text
docs/report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_01_data_add100__4-D-00168.jpg
docs/report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_02_data_add100__4-B-00118.jpg
docs/report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_03_data_add100__d-10.jpg
```

## RC壁 第一轮结果

RC壁 已完成 RF-DETR 第一轮训练和候选 sweep，但当前暂不作为汇报重点。

当前三个候选：

| epoch | Precision | Recall | F1 | mAP50 | mAP50-95 | 用途 |
|---:|---:|---:|---:|---:|---:|---|
| 22 | 0.731 | 0.607 | 0.657 | 0.648 | 0.241 | 综合最好 |
| 71 | 0.613 | 0.679 | 0.643 | 0.631 | 0.155 | recall 最高 |
| 19 | 0.759 | 0.565 | 0.645 | 0.624 | 0.247 | precision 最高 |

这三个候选已纳入交接压缩包，供下周继续分析。

## 今日新增/更新的关键文档

```text
docs/meeting_rfdetr_progress_story_20260602.md
docs/development_summary_20260602.md
docs/handoff_restore_20260602.md
docs/downstream_dataset_restore_20260602.md
```

## 今日新增关键素材

```text
docs/report_assets_20260602_rfdetr/
```

其中包含：

- 自动识别模型 YOLO9 vs RF-DETR 对比图。
- RC柱 YOLO9 未检出 vs RF-DETR 检出对比图。
- 指标对比图。
- 对比 case CSV。

## 下周优先级

1. 自动识别模型暂时不再作为主要优化对象，仅做必要验证。
2. 继续推进下游单类别模型 RF-DETR 迁移。
3. 优先推广到天井、内壁、RC壁。
4. 模型选择时以 recall 为主，但必须兼顾 precision、F1、mAP 和可视化结果。
5. 四个下游模型完成后，做完整 pipeline 的端到端评估。
