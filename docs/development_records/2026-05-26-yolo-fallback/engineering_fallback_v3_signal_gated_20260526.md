# 工程兜底 v3 - 信号触发器 + 跨类隔离 评估报告

**日期**: 2026-05-26
**目的**: 把"对所有壁类/RC柱图都兜底"的 v2 策略，改成只在 router 输出携带可疑信号时才兜底；同时彻底禁止跨类合并、把跨类重叠以二次意见候选展示。

## 一、修改清单（一次性合入）

### 改 1 - 合并器不再跨类聚类
`router_crack_pipeline/pipeline/result_merge.py::_merge_across_models`
- 增加同 `source_router_class` 才允许聚类的硬约束
- 任何跨类重叠都不再被静默合并成单个 box，留给 ambiguity 层处理

### 改 2 - 新增 ambiguous_class_candidate group
`router_crack_pipeline/pipeline/ambiguity_display.py`（新文件）
- 合并后扫描，对不同 `source_router_class` 且 IoU ≥ 0.5 的检测组成 group
- 每个 group 保留两/多个候选，标 `status="ambiguous_class_candidate"`，由审查人员人工确认
- `run_full_pipeline.py` 将被 ambiguity 占用的壁类检测从 `wall_candidate_display` 排除，避免双渲染

### 改 3 - 触发器化的 fallback
`router_crack_pipeline/pipeline/fallback_policy.py`（重写）
- **Trigger A (形态)**：router 壁类 box `area_ratio ≤ 0.05` 且 `confidence ≤ 0.50` → 触发 rc_column 兜底
- **Trigger B (主分支哑火)**：某 router region 内全部主 detector 最大 conf < 0.15 → 触发对应 sister detector
  - 壁类 region 哑火 → 跑 rc_column
  - RC柱 region 哑火 → 跑 inner_wall + rc_wall
- **Trigger C (并排相似壁)**：≥2 个壁类 box、conf 比值 ≤ 1.20、纵横比 ≥ 1.50 → 触发 rc_column 兜底
- **Low-confidence / empty router**：保留原有全图回填，作为最后兜底

`run_full_pipeline.py::_run_full_image_filter_with_fallback`
- 改为两阶段调度：先跑 main + 静态触发器（A/C/低置信/空 router），再根据每个 region 的主分支最大 conf 计算 Trigger B，按需追加 sister detector

### 改 5 - source_router_class 正确路由
兜底任务携带"目标类（缺失方）"作为 `source_router_class`，保证 `display_crack_detections` 把它走到对的分支（先前的 bug fix 已合入）。

### 配置文件
- `configs/pipeline.e2e_old_d900_fallback.local.yaml`
- `configs/pipeline.e2e_aug_v1_fallback.local.yaml`
- 替换为 `trigger_morphology / trigger_main_dropout / trigger_parallel_walls / trigger_low_confidence_router / trigger_empty_router` 五个开关
- 新增 `ambiguity_display.iou_threshold`

## 二、80 张 E2E 结果（old D900 router）

| 指标 | baseline (无兜底) | v2 (无脑 pair) | **v3 (信号触发)** |
|---|---|---|---|
| FN total | 19 | 14 | **16** |
| 　rc_column | 9 | 5 | 7 |
| 　rc_wall | 5 | 3 | 4 |
| 　inner_wall | 5 | 5 | 5 |
| 　tenjo | 0 | 1 | 0 |
| FP total | ≈201 | ≈200 | 202 |
| FP fallback share | 0 | ≈3 | **1** |
| fallback_rescued GT | 0 | 4 | 2 |
| 兜底覆盖图数 | 0 | ≈55 | **8 / 80 = 10%** |

### 触发器命中分布（old_v3）
- `morphology` (A): 1
- `main_dropout` (B): 8
- `parallel_walls` (C): 1
- 跨类合并次数: **0**（R1 生效）
- ambiguous_class_candidate group: 8 个（R2 生效）

## 三、关键判断

1. **R1 已彻底封堵跨类合并**：80 张图中 0 处同位置不同类被合并。Router 误判时不再"D 级被改读成 C 级"。
2. **R2 把不确定性显式化**：壁类与 RC柱 同位置重叠的 7 张图共 8 个 group，全部以两个候选并列方式呈现给审查员，安全语义保留。
3. **R3 兜底面收窄 7 倍**：v2 的兜底基本覆盖所有壁类/RC柱图；v3 只在 10% 图上触发。FP fallback share 从 ≈3 降到 1。
4. **救回案例打折但仍正向**：从 4 救 2，主要丢的是 case 4（router 输出无任何可疑信号、纯 router 错分）。这类案例不可能由工程兜底安全救回，只能靠改 router 本身。
5. **aug_v1 router 仍劣于 old**：FN 21 vs 16，router_miss 9 vs 4。强增强重训方向已确认无效，下一步走温和增强 v2 或 Gemini 合成。

## 四、未来工作

- 路线一 v2：温和增强重训（pattern 减半、移除 grid mask）
- 路线二：Gemini 合成数据 POC（高难度 RC柱 / 边界壁类）
- 触发器阈值复查：在更大样本上扫 trigger A/B/C 的 FP/recall 折线，必要时微调 0.05/0.15/1.20 三个数
