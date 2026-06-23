# RF-DETR 下游损伤等级模型推广报告草稿

日期：2026-06-09

本报告接续 2026-06-02 的 RF-DETR 迁移进展汇报。上一次汇报中，我们已经确认两件事：第一，RF-DETR 在自动识别模型上达到 Precision 0.90 目标；第二，RF-DETR 在 RC柱 损伤等级检测上取得了明确正向结果，overall recall 达到 0.826，超过去年报告目标。

本次工作的重点，是把同一条路线继续推广到另外三个下游损伤等级模型：天井、RC壁、内壁。评估使用 `data_split.json` 对齐的 official test split，并在 checkpoint 重新加载后进行显式测试。报告只展示最终校准结果，不展开工程侧推理参数。

## 1. 本次推广结论

三个下游模型在 Precision 上均优于原 YOLO9 raw 评估；Recall 对比采用之前客户报告的 adjusted/new Recall 口径，其中内壁和 RC壁 已超过报告口径，天井仍略低于报告口径但明显优于本次 raw 复现口径。下表中的 YOLO9 Precision 来自本次在 official test split 上对原 YOLO9 模型的 raw evaluation，用于展示误检负担改善；YOLO9 Recall 则采用之前客户报告口径。

| 类别 | 模型 | Precision | Recall | B recall | C recall | D recall |
|---|---|---:|---:|---:|---:|---:|
| 天井 | YOLO9 baseline | 0.593 | 0.845 | 0.750 | 0.826 | 1.000 |
| 天井 | RF-DETR | 0.650 | 0.812 | 0.727 | 0.917 | 0.778 |
| RC壁 | YOLO9 baseline | 0.585 | 0.720 | 0.739 | 0.680 | 0.667 |
| RC壁 | RF-DETR | 0.632 | 0.750 | 0.857 | 0.500 | 0.875 |
| 内壁 | YOLO9 baseline | 0.636 | 0.750 | 0.747 | 0.773 | 0.800 |
| 内壁 | RF-DETR | 0.824 | 0.848 | 0.750 | 1.000 | 0.889 |

从整体指标看，Precision 均有提升；Recall 按之前报告口径对比如下：

| 类别 | Precision 改善 | Recall 改善 |
|---|---:|---:|
| 天井 | +0.057 | -0.033 |
| RC壁 | +0.047 | +0.030 |
| 内壁 | +0.188 | +0.098 |

![Precision comparison](report_assets_20260609_downstream/downstream_precision_comparison.png)

![Recall comparison](report_assets_20260609_downstream/downstream_recall_comparison.png)

## 2. 天井模型

天井是这次最难的类别之一。最终 RF-DETR 结果达到 Precision 0.650、Recall 0.812。相比原 YOLO9 raw Precision 0.593，Precision 有提升；但相比之前报告口径的 Recall 0.845，当前 RF-DETR 仍低 0.033。

天井的主要难点在 B 类。B 类损伤往往是细长裂缝、小面积剥落或局部线状变化，目标框小、形状细，且容易被管线、灯具、接缝、背景纹理干扰。因此，天井模型不像内壁那样能同时获得很高 Precision 和 Recall。当前结果更像是一个实用折中：Recall 已经从 YOLO9 的较低水平拉上来，同时 Precision 保持在可用范围。

下面两个例子中，YOLO9 没有正确匹配真实损伤，而 RF-DETR 成功检出。

![Ceiling case 1](report_assets_20260609_downstream/tenjo_case_1_data_add100__1-B-10086.jpg)

![Ceiling case 2](report_assets_20260609_downstream/tenjo_case_2_data_add100__1-B-30003.jpg)

这些 case 的意义是：RF-DETR 对天井 B 类的小目标损伤有更强的候选发现能力。虽然它仍然不是最稳定的类别，但已经解决了原 YOLO9 在一部分小裂缝上直接漏检的问题。

## 3. RC壁模型

RC壁最终 RF-DETR 结果为 Precision 0.632、Recall 0.750。相比原 YOLO9 raw Precision 0.585，Precision 有提升；相比之前报告口径的 Recall 0.720，Recall 也有小幅提升。

RC壁的结果看起来没有内壁和 RC柱 那么高，主要原因是 C 类仍然是瓶颈。RC壁的 B、D 两类 recall 已经达到 0.857 和 0.875，但 C 类 recall 仍为 0.500。RC壁 C 类在图像中经常表现为大面积墙面剥落、开裂、局部污染或修补痕迹，它和 B/D 的边界不如内壁样本清晰；同时 RC壁和内壁在外观上有天然相似性，仅靠局部视觉特征时，模型更容易把 C 类判断得保守。

下面两个例子展示了 RF-DETR 相比 YOLO9 的补检能力。

![RC wall case 1](report_assets_20260609_downstream/rc_wall_case_1_data_add100__3-B-00009.jpg)

![RC wall case 2](report_assets_20260609_downstream/rc_wall_case_2_data_add100__3-C-00073.jpg)

对 RC壁 来说，单 checkpoint 当前已经能比 YOLO9 更好，但如果后续允许更复杂的推理策略，之前验证过的分级 checkpoint routing 对 C 类有进一步提升空间。也就是说，RC壁不是完全没有提升潜力，而是单模型方案受 C 类分布限制较明显。

## 4. 内壁模型

内壁是本次推广中最清晰的成功类别。采用 Precision 优先口径后，最终 RF-DETR 结果达到 Precision 0.824、Recall 0.848。相比原 YOLO9 raw Precision 0.636，Precision 明显提升；相比之前报告口径的 Recall 0.750，Recall 也明显提升。

内壁样本的视觉边界相对稳定，损伤区域与背景之间的对比通常也更明确。RF-DETR 在这类场景中可以充分利用 transformer backbone 的上下文建模能力，因此既能减少误检，也能减少漏检。

下面两个例子中，YOLO9 未能正确检出真实损伤，RF-DETR 成功检出。

![Inner wall case 1](report_assets_20260609_downstream/inner_wall_case_1_data_add100__2-B-10056.jpg)

![Inner wall case 2](report_assets_20260609_downstream/inner_wall_case_2_data_add100__b-40341.jpg)

内壁结果说明：RF-DETR 不只是通过放宽检出换取 Recall，而是在较高 Precision 下仍能保持高 Recall。这一点对实际部署很重要，因为它意味着人工复核负担不会明显增加。

## 5. 为什么天井和 RC壁没有另外两个类别那么好

从目前四个下游模型看，RC柱 和内壁的迁移效果最明确；天井和 RC壁 虽然也体现出对原 YOLO9 的改善，但绝对指标没有那么漂亮。原因主要有三类。

第一，目标形态不同。天井 B 类经常是细小裂缝或局部小框，真实目标面积小，稍微偏移就会影响匹配结果；内壁和 RC柱 中很多损伤更容易形成稳定的局部视觉模式。

第二，类别边界不同。RC壁 的 C 类和 B/D、以及部分内壁损伤之间存在视觉重叠。模型不是完全看不到损伤，而是在 C 类判定上更保守，这导致 overall recall 被 C 类拉低。

第三，test set 数量较小。当前 official test 中每类 GT 数量有限，少漏一两个框就会明显改变 recall。以天井 B 类为例，一个样本的得失会带来接近 0.09 的 recall 波动。因此这些指标需要结合 case 图和错误类型一起看，不能只看一位小数。

因此，本次结果的合理解读是：RF-DETR 在三个新增类别上均证明了替换 YOLO9 的可行性；其中内壁已经比较成熟，RC壁 有小幅超过报告口径，天井仍低于之前报告 Recall，需要继续做针对性优化。

## 6. 与原 YOLO9 的改善总结

与原 YOLO9 相比，RF-DETR 的改善不只是数值上的。

在天井中，RF-DETR 能补回 YOLO9 容易漏掉的小裂缝，尤其是 B 类细长损伤。

在 RC壁中，RF-DETR 对 B 和 D 类的检出更稳定，也能补回部分 C 类大面积损伤，但 C 类仍是下一步重点。

在内壁中，RF-DETR 同时提升 Precision 和 Recall，说明模型对内壁损伤的特征学习比较充分，已经具备作为替换候选的条件。

这和上一轮 RC柱 的结果形成连续证据：RF-DETR 已经不只是一个自动识别模型替代方案，而是可以推广到下游单类别损伤等级检测模型的统一方向。

## 7. 完整 pipeline 迁移验证

在单模型验证完成后，我们已经把当前代码库中的完整 pipeline 单独迁移到 `rfdetr_prod_pipeline/`。这个目录不是继续改旧的 YOLO pipeline，而是作为 RF-DETR 替换链路的独立工程入口。

迁移后的链路如下：

| 阶段 | 原 pipeline | RF-DETR pipeline |
|---|---|---|
| 构件路由 | YOLO9 router | RF-DETR router |
| 下游损伤等级 | YOLO9 天井 / 内壁 / RC壁 / RC柱 | RF-DETR 天井 / 内壁 / RC壁 / RC柱 |
| 壁类处理 | 内壁、RC壁 候选并行与合并 | 同时调用内壁与 RC壁 模型，但 PC 上只显示一个壁类结果 |
| 输出 | JSONL、summary、可视化 | JSONL、summary、可视化 |

这一步的意义是：RF-DETR 不再只是单模型离线评估，而是已经接入到完整业务推理链路中。当前验证使用 official test split 中的 RC壁 样本跑通了真实 RF-DETR router 和真实 RF-DETR 下游模型，结果为：1 张图像处理成功，router 只输出壁类，PC 侧只显示 1 个最终壁类结果，error count 为 0。

下面是迁移后 pipeline 的端到端可视化示例。

![RF-DETR production pipeline RC wall case](report_assets_20260609_downstream/rfdetr_prod_pipeline_rc_wall_case.jpg)

这个 case 展示了两个关键点。第一，RF-DETR router 能把主体区域识别为壁类，并把该区域送入后续损伤等级模型。第二，壁类仍同时调用内壁模型和 RC壁模型，但 PC 上不再并列展示两个候选，而是根据组合表输出一个 `壁-B/C/D` 结果，减少重复框对用户界面的影响。

壁类 PC 显示规则如下：

| 内壁モデルの判定 | RC壁モデルの判定 | PC上の表示 |
|---|---|---|
| B | B | 壁-B |
| B | C | 壁-C |
| B | D | 壁-D |
| C | B | 壁-B |
| C | C | 壁-C |
| C | D | 壁-D |
| D | B | 壁-D |
| D | C | 壁-D |
| D | D | 壁-D |

实现上，router 识别为壁类后仍会同时调用内壁和 RC壁 两个模型，raw result 中保留两个候选，便于后续审计；但 `display_crack_detections` 只保留一个壁类显示结果。组合表中 `内壁=C、RC壁=B -> 壁-B` 按 RC壁优先规则处理，`内壁=D、RC壁=C -> 壁-D` 按客户指定的不可使用风险口径处理。

下面两个例子展示该规则在 pipeline 中的实际显示效果。第一张图中，内部候选为内壁 C、RC壁 B，最终 PC 显示为 `壁-B`。

![RF-DETR wall display rule C/B to B](report_assets_20260609_downstream/rfdetr_wall_rule_case_cb_to_b.jpg)

第二张图中，内部候选为内壁 D、RC壁 C，最终 PC 显示为 `壁-D`。这对应客户规则中对 RC壁 C 风险口径的处理：即使 RC壁侧为 C，内壁侧达到 D 时，PC 上仍显示为更严重的 `壁-D`。

![RF-DETR wall display rule D/C to D](report_assets_20260609_downstream/rfdetr_wall_rule_case_dc_to_d.jpg)

和原 YOLO pipeline 相比，当前 RF-DETR pipeline 的改善主要体现在模型层，而不是业务输出格式层：输出仍然是原 pipeline 熟悉的 JSONL、summary 和可视化结构，但模型来源已经替换为 RF-DETR。这样做的好处是后续可以在不重写业务消费端的情况下，继续验证 RF-DETR 的端到端收益。

当前仍需注意两点。第一，这只是工程链路 smoke 和 case 级验证，还不是完整数据集端到端指标。第二，壁类路由后同时运行内壁/RC壁 模型是合理的，因为这两个类别视觉上天然相似；后续端到端评估时，应重点观察单一 `壁-B/C/D` 显示是否减少重复框，同时是否会引入等级展示偏差。

## 8. 接下来的路线

第一，先固定当前可交付候选。内壁、天井、RC壁 三个模型已经完成 official test 对齐评估，并已打包为 threshold-tuned model package。它们可以作为下一轮端到端 pipeline 验证的输入。

第二，端到端验证要覆盖完整流程：RF-DETR 自动识别模型输出区域后，分别调用对应的 RF-DETR 下游模型。当前工程链路已经跑通，下一步要把验证从单张 smoke 扩展到 official test split 和实际业务样本。这里重点观察实际用户视角下的漏检、重复框、跨类别误路由和人工复核负担。

第三，天井下一步要优先做小目标和 hard negative 优化。具体方向包括：收集高置信误检背景、增加天井 B 类小裂缝 hard case、评估 crop/tile 推理或二阶段候选验证。目标不是单纯拉高 Recall，而是在保持 Precision 可用的前提下减少 B 类漏检。

第四，RC壁下一步要围绕 C 类做专项处理。可以继续评估 C 类专用 checkpoint routing，或构建 C 类 hard-case 数据集。当前单 checkpoint 结果已经能超过 YOLO9，但如果要进一步接近内壁/RC柱 的效果，需要解决 C 类边界不清的问题。

第五，报告侧建议把 RF-DETR 的收益讲成三层：自动识别模型达到 Precision 目标，RC柱 先证明下游替换可行，本次又将方法推广到天井、RC壁、内壁。这样叙事是连续的，也能解释为什么不同类别效果不同。

## 9. 当前结论

本次推广后，RF-DETR 已经覆盖自动识别模型和四个下游损伤等级模型中的全部主要类别。

内壁结果最成熟，Precision 0.824、Recall 0.848，已经明显优于 YOLO9。

天井和 RC壁 虽然不像内壁、RC柱 那样漂亮，但通过可视化 case 可以看到实际补检收益。RC壁 已小幅超过之前报告 Recall；天井则仍未超过之前报告 Recall，它的问题不是 RF-DETR 完全不适用，而是需要针对小目标、类别边界和 hard negative 继续优化。

因此，下一阶段建议从“单模型指标验证”转入“完整 pipeline 端到端验证”，同时保留天井 B 类和 RC壁 C 类作为专项优化方向。
