# E2E 问题样本可视化与初步目视分析

日期：2026-05-19

## 输入

- E2E 评估目录：`outputs/e2e_debug_sample_80_router_merged4219_ddp_20260519`
- Router 模型：`coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_ft_from_d900_imgw_rc_os900_e50_ddp/weights/best.pt`
- 抽样规模：每类 20 张，共 80 张
- 可视化脚本：`router_crack_pipeline/scripts/visualize_e2e_issues.py`
- 可视化输出：`outputs/e2e_debug_sample_80_router_merged4219_ddp_20260519/issue_visualizations`

## 可视化说明

每张图叠加以下信息：

- 白框：原始 YOLO 标注 GT
- 黄框：router 输出区域
- 绿框：主分支裂缝模型输出
- 蓝框：次分支裂缝模型输出
- 图上方：样本类别、问题标签、router 期望类别/实际类别、GT/预测/匹配/FN/FP 数量

汇总总览图：

- `outputs/e2e_debug_sample_80_router_merged4219_ddp_20260519/issue_visualizations/contact_sheet.jpg`

问题索引：

- `outputs/e2e_debug_sample_80_router_merged4219_ddp_20260519/issue_visualizations/issue_visualization_index.csv`

## 问题样本统计

本轮 80 张抽样中，可视化出 35 张问题样本。

按原始类别：

| 类别 | 问题样本数 |
|---|---:|
| RC柱 | 15 |
| RC壁 | 8 |
| 内壁 | 8 |
| 天井 | 4 |

按问题类型：

| 问题类型 | 数量 | 含义 |
|---|---:|---|
| router_miss | 5 | router 没输出期望大类，导致主分支没有被调用或主分支错误 |
| no_main | 7 | router 命中，但主分支裂缝模型没有输出 |
| iou_miss | 14 | 主分支有输出，但和 GT 的 IoU 未达到 0.5 |
| fp | 21 | 主分支多输出了未匹配框 |
| grade_mismatch | 3 | IoU 匹配成功，但 B/C/D 等级不同 |

## 目视结论

### 1. RC柱是当前最主要问题来源

RC柱占 15/35 个问题样本。目视后可以分成三类：

1. **router 误分为壁类或天井**
   - 代表样本：`01_rc_column_d-173_router_miss.jpg`、`02_rc_column_4-C-00022_router_miss.jpg`、`03_rc_column_d-40044_router_miss.jpg`
   - 图像上很多 RC柱并不是孤立柱体，而是贴近墙面、窗边、走廊边缘或局部立面。
   - 对人来说也更像“墙面中的柱状区域”，router 把它当成壁类并不完全离谱。
   - 这会直接导致 RC柱裂缝模型不被调用，E2E 主分支 FN 增加。

2. **router 命中 RC柱，但 RC柱裂缝模型无输出**
   - 代表样本：`05_rc_column_d-87_no_main.jpg`、`06_rc_column_d-40004_no_main.jpg`、`07_rc_column_d-40075_no_main.jpg`
   - 部分样本里次分支 wall 模型有输出，但 RC柱主模型没有输出。
   - 这说明 E2E 失败不只来自 router，上一期 RC柱裂缝模型本身或切片后的尺度/上下文也在限制结果。

3. **主模型有输出但与 GT 框口径不同**
   - 代表样本：`08_rc_column_4-B-00175_iou_miss_fp.jpg`、`09_rc_column_4-C-00111_iou_miss_fp.jpg`、`11_rc_column_d-36_iou_miss_fp.jpg`
   - 目视上绿色框经常覆盖裂缝主体或剥落主体，但和白色 GT 框偏移、过宽、过窄或只覆盖局部，因此 IoU50 失败。
   - 对这类样本，单纯看 IoU50 会低估肉眼效果。

### 2. 墙类的主要问题是下游模型和标注口径，不是 router

RC壁/内壁中，router 大多能输出壁类。失败更多来自：

- 主分支无输出：例如 `16_rc_wall_c-40621_no_main.jpg`，图像是远距离外墙小损伤，router 覆盖大墙面，但 RC壁裂缝模型没有检测到。
- IoU 口径不一致：例如 `18_rc_wall_c-40773_iou_miss_fp.jpg`，预测框覆盖了大面积破损，GT 只标一部分破损主体，IoU50 统计为 miss + FP。
- 内壁样本里也存在 GT 很窄、预测框跨越相邻结构的问题，例如 `25_inner_wall_b-30184_iou_miss_fp.jpg`。

这说明当前 E2E 评估不能只用 IoU50 解释用户肉眼感受。需要同时保留 IoA、中心点落入、或“预测是否覆盖 GT 损伤主体”的辅助指标。

### 3. 天井问题较少，但有类别/场景语义风险

天井问题只有 4 个。主要是：

- 预测框更大，覆盖了破损区域但和 GT 不完全重合，例如 `32_tenjo_1-D-00016_iou_miss_fp.jpg`。
- 有些样本视觉上更像室内走廊/吊顶破损，而不是纯天井，这会让 router 输出多个区域并增加 FP，例如 `33_tenjo_1-D-00032_fp_grade_mismatch.jpg`。

### 4. 当前 wall_parallel_debug 策略有价值

墙类中 RC壁和内壁外观不可区分的问题仍然存在，但从本轮样本看，很多重叠位置两个模型等级一致；`wall_grade_shift_pairs` 为 17，其中 16 组等级差为 0，只有 1 组 `RC - inner = -2`。

因此当前并行调试策略是合理的：

- 正式输出阶段可以继续把 RC壁/内壁作为 router 的同一大类。
- 下游可保留双模型输出，后续根据业务规则、建筑结构信息或用户输入决定采用 RC壁等级还是内壁等级。
- 对等级不一致的重叠输出，需要作为审核重点单独暴露。

## 当前建议

1. **E2E 评价指标需要增加“肉眼友好”的辅助指标**
   - 保留 IoU50 作为严格指标。
   - 增加 IoA 或 GT-center-in-pred 指标，专门识别“预测覆盖损伤但框口径不同”的样本。
   - 对大面积剥落/破损类样本，IoU50 不应作为唯一失败依据。

2. **RC柱建议引入候选多路由 fallback**
   - 当 router 输出壁类但图像中存在细长竖向结构，或 RC柱置信度处于低/中区间时，同时调用 RC柱和壁类模型。
   - 这能覆盖当前 `router_miss` 中一部分“看起来像壁的柱”。

3. **对 RC柱裂缝模型做单独复查**
   - 当前很多 RC柱样本 router 已命中，但主模型无输出或框偏移。
   - 下一步应固定 router，直接在 RC柱 GT 区域/全图上测试 RC柱裂缝模型，判断问题来自模型能力还是切片策略。

4. **切片 padding 可以做小实验**
   - 当前使用 `region_padding_ratio: 0.10`。
   - 对 RC柱和墙边缘样本，裂缝常贴近 router 框边界，建议比较 `0.10 / 0.20 / 0.30`。
   - 但 padding 增大会增加 FP，需要和辅助指标一起看。

5. **问题样本可作为下一轮人工确认集**
   - 优先人工复查 35 张问题图。
   - 尤其关注：GT 是否过窄、是否只标了局部损伤、是否存在类别标签与肉眼不一致。
