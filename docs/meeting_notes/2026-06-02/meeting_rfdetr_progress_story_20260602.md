# RF-DETR 迁移进展汇报草稿

日期：2026-06-02

本次汇报主要说明三件事：第一，为什么我们在 YOLO9 之后继续评估新的检测模型；第二，RF-DETR 在自动识别模型上的迁移结果；第三，RF-DETR 在单类别裂缝等级检测，尤其是 RC柱 上的初步效果，以及后续如何推广到其他类别。

## 1. 前情提要

当前系统采用两阶段流程。

第一阶段是自动识别模型。它负责判断图像中需要处理的是哪一类建筑构件区域，例如天井、壁类、RC柱。第二阶段是单类别裂缝/损伤等级检测模型。自动识别模型判断完成后，系统会把对应区域送入相应的后续模型，判断 B/C/D 等损伤等级。

上次会议中，客户对自动识别模型提出了一个明确目标：Precision 希望达到 0.90。这个目标是合理的。因为自动识别模型的错误会直接影响后续流程，如果把 RC柱 区域误送到壁类模型，后续的损伤等级判断就会建立在错误前提上。相比之下，自动识别模型的 recall 可以通过后续规则扩大检出范围来补偿，因此当前自动识别模型的核心目标是 precision 优先。

而对于后续的裂缝/损伤等级检测模型，目标略有不同。这里 recall 更重要，因为漏掉真实损伤比多检出一个候选框更危险。当然 precision、F1、mAP 和可视化结果仍然需要作为模型选择的约束，不能只看 recall。

## 2. YOLO9 之后的模型选择

YOLO9 在当前项目中已经证明是一个有效 baseline。它让我们确认了两阶段方案是可行的，也给出了可以对比的性能基准。

YOLO9 之后，目标检测模型的主要改进方向集中在两个方面。第一是 backbone 和特征提取能力更强，模型可以更好地理解复杂场景；第二是检测头和训练策略不断改进，使得模型在小目标、遮挡、复杂背景上的表现更稳定。

不过，从工程项目角度看，不能只看模型性能。YOLO9 之后的一些更先进 YOLO 系模型，在 license 上已经涉及收费或商业使用成本问题。对于后续正式部署和长期维护来说，如果继续沿用这一路线，模型本身的使用成本和授权不确定性都会增加。

因此，本周我们选择 RF-DETR 作为新的候选方向进行验证。目标不是单纯替换模型名称，而是在保持或提升精度的同时，降低后续工程和运用风险。

## 3. 为什么选择 RF-DETR

RF-DETR 是一个较新的目标检测模型。和 YOLO 系模型一样，它最终输出的也是类别和检测框，也就是判断图像中有什么目标，以及这些目标在哪里。

它的先进性主要来自 transformer 系列结构。这里可以简单理解为：Transformer 是最近大语言模型 LLM 的基础结构，而在视觉领域，ViT，也就是 Vision Transformer，也已经成为非常重要的视觉 backbone。RF-DETR 使用这类视觉 transformer 能力来做目标检测，因此在复杂图像理解上有比较好的基础。

这也引出另一个重要概念：预训练模型。RF-DETR 不是从零开始学建筑检查图片，而是先在大量通用视觉数据上学习过基本视觉特征，再用我们的项目数据进行微调。对我们这种数据量相对有限的工程项目来说，这一点很重要。因为小数据集直接从零训练通常很难稳定，而预训练模型可以把通用视觉能力迁移到当前任务上。

本次我们选择的是 RFDETRMedium。这个尺寸是性能和成本之间的平衡选择：它比小模型有更强表达能力，但训练和推理成本仍然可控。

从工程适配角度看，RF-DETR 可以适配我们现有的数据组织和训练代码。当前已有的部署基础设施，包括推理用的 NVIDIA T4 GPU，也可以覆盖 RF-DETR Medium 的推理需求。因此迁移成本主要集中在训练和评估本身，不需要大幅改变现有 pipeline。

## 4. 自动识别模型迁移结果

自动识别模型使用的数据与上周调优 YOLO 自动识别模型时使用的数据相同，评估也使用同一 test set。因此这里的对比是直接对齐的。

本次自动识别模型的核心结果如下：

| model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv9 tuned baseline | 0.863 | 0.850 | - | 0.888 | 0.775 |
| RF-DETR | 0.905 | 0.852 | 0.877 | 0.904 | 0.782 |

这里最重要的是 Precision。RF-DETR 达到了客户提出的 `Precision >= 0.90` 目标，同时 Recall 与调优后的 YOLO baseline 基本持平。mAP50 和 mAP50-95 也有小幅提升。

因此，自动识别模型的 RF-DETR 迁移目前可以认为已经达到阶段目标。

![Auto-recognition metrics](report_assets_20260602_rfdetr/router_yolo_vs_rfdetr_metrics.png)

## 5. 自动识别模型的可视化改善案例

除了整体指标，我们也检查了过去 YOLO 自动识别模型的 hard case。

之前 YOLO 自动识别模型最重要的问题之一，是 `壁类` 和 `RC柱` 之间容易混淆。这类错误对系统影响很大，因为它会导致后续模型选择错误。例如本来应该进入 RC柱 损伤等级模型的区域，如果被自动识别模型判断成壁类，后续结果就会偏离。

这次 RF-DETR 在多个相同测试样本上修正了这类错误。

| case | image | YOLO9 的问题 | RF-DETR 结果 |
|---:|---|---|---|
| 3 | `RC柱_d-40027_03307.jpg` | RC柱 -> 壁类 | RC column, conf 0.811 |
| 4 | `RC壁_c-199_03206.jpg` | RC柱 -> 壁类 | RC column, conf 0.336 |
| 5 | `RC壁_c-40616_03440.jpg` | 壁类 -> RC柱 | Wall, conf 0.633 |

下面这些图可以作为汇报时的重点 case。每张图左侧是 YOLO9 的预测结果，右侧是 RF-DETR 的预测结果；灰色框是真值标注，彩色框是模型预测。

![YOLO vs RF-DETR comparison case 03](report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_03_RC柱_d-40027_03307.jpg)

![YOLO vs RF-DETR comparison case 04](report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_04_RC壁_c-199_03206.jpg)

![YOLO vs RF-DETR comparison case 05](report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_05_RC壁_c-40616_03440.jpg)

这些 case 的意义是：RF-DETR 不只是整体数值上提升了 Precision，而且确实修正了之前 YOLO 自动识别模型中影响后续流程的典型错误。前三个 case 中既包含 `RC柱 -> 壁类` 的错误，也包含 `壁类 -> RC柱` 的反向错误，因此可以说明 RF-DETR 对两类之间的边界判断都有改善。

## 6. RF-DETR 在 RC柱 损伤等级检测上的结果

自动识别模型达到阶段目标后，我们进一步验证 RF-DETR 是否也能用于后续单类别裂缝/损伤等级检测。

这里我们先从 RC柱 入手。原因是，在之前四个单类别模型中，RC柱 的数据量最少，效果也相对最差。如果 RF-DETR 能先在这个最弱类别上取得提升，就更能说明该路线有推广价值。

这次 RC柱 使用的是和去年结果对齐的数据划分，但没有使用半监督学习，也没有做复杂调优，只进行了基础的监督学习。因此这个结果可以理解为 RF-DETR 的第一轮基础迁移效果。

与去年报告目标的对比如下：

| 范围 | 去年报告目标 R | RF-DETR R |
|---|---:|---:|
| Overall | 0.742 | 0.826 |
| B | 0.700 | 0.750 |
| C | 0.706 | 0.727 |
| D | 0.807 | 1.000 |

整体指标如下：

| Precision | Recall | F1 | mAP50 | mAP50-95 |
|---:|---:|---:|---:|---:|
| 0.661 | 0.826 | 0.725 | 0.726 | 0.299 |

这里的重点是 recall。RC柱 的 RF-DETR 候选模型超过了去年报告中的 overall recall 目标，同时 B/C/D 三个等级也都超过目标。

这个结果说明 RF-DETR 不仅可以用于自动识别模型，也有潜力替换后续的单类别损伤等级检测模型。考虑到这次只是基础监督学习，后续如果继续加入更细的训练策略和数据调整，还有进一步提升空间。

![RC column recall comparison](report_assets_20260602_rfdetr/rc_column_recall_target_comparison.png)

除了数值对比，我们也筛选了 YOLO9 未检出、但 RF-DETR 成功检出的样例。这里的 YOLO9 使用常规展示阈值 `conf=0.25` 进行推理；筛选条件是同一 GT 下，YOLO9 没有同类别匹配框，而 RF-DETR 有同类别匹配框。

| case | 等级 | YOLO9 结果 | RF-DETR 结果 |
|---:|---|---|---|
| 1 | D | 未检出 | D, conf 0.968 |
| 2 | B | 未检出 | B, conf 0.935 |
| 3 | C | 未检出 | C, conf 0.887 |

下面三张图左侧是 YOLO9 结果，右侧是 RF-DETR 结果。红色框表示 YOLO9 漏掉的 GT，右侧彩色框表示 RF-DETR 检出的结果。

![RC column YOLO missed RF-DETR detected case 01](report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_01_data_add100__4-D-00168.jpg)

![RC column YOLO missed RF-DETR detected case 02](report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_02_data_add100__4-B-00118.jpg)

![RC column YOLO missed RF-DETR detected case 03](report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_03_data_add100__d-10.jpg)

## 7. 接下来计划

基于目前结果，自动识别模型可以先告一段落。RF-DETR 已经达到 Precision 0.90 目标，并且在过去 YOLO 的典型 hard case 上有可视化改善案例。因此后续自动识别模型只需要保留必要验证，不再作为当前主要优化对象。

接下来重点应该放在后续单类别裂缝/损伤等级检测模型上。

第一，保留 RC柱 当前 RF-DETR 结果作为第一个成功候选。RC柱 是四个类别中相对最弱、数据最少的类别，能够先在这里超过去年报告 recall 目标，是一个比较有说服力的结果。

第二，用同样流程推广到剩余三个类别：天井、内壁、RC壁。每个类别都需要使用和去年对齐的 test protocol，保证指标可比。

第三，模型选择时仍然以业务目标为核心。对裂缝/损伤检测来说，recall 是优先指标，但不能只看 recall。最终候选还需要综合 precision、F1、mAP 和可视化案例，避免选择明显过度检出的模型。

第四，四个后续检测模型都完成后，再做完整 pipeline 的端到端评估。这样可以确认 RF-DETR 自动识别模型与 RF-DETR 单类别检测模型组合后，在实际流程中的整体收益。

## 8. 当前结论

本周最重要的结论有两个。

第一，RF-DETR 在自动识别模型上已经达到客户提出的 Precision 0.90 目标。相比 YOLO9 tuned baseline，RF-DETR 的 Precision 从 0.863 提升到 0.905，同时 Recall 基本持平。

第二，RF-DETR 在 RC柱 损伤等级检测上也取得了明确正向结果。RC柱 的 overall recall 达到 0.826，超过去年报告目标 0.742，并且 B/C/D 三个等级都超过目标。

因此，RF-DETR 可以作为 YOLO9 后续替换方向继续推进。下一步重点是把同样方法推广到其他三个单类别模型，并完成完整 pipeline 的端到端验证。
