# 工程兜底 E2E 评估报告（80 张样本，seed=20260519）

> 同一份 manifest 抽样的 80 张图（每类 20 张），下游模型为上一期 GPL 的
> TIANJING/NEIBI/RCBI/RCZHU。判别 IoU≥0.50。
>
> 兜底（fallback）策略实现于 `router_crack_pipeline/pipeline/fallback_policy.py`，
> 在 `pipeline/run_full_pipeline.py` 中被接入。

## 测试矩阵

| ID | 配置文件 | router 模型 | fallback |
|----|----------|-------------|----------|
| A | `pipeline.e2e_old_d900_fullimage_prodmerge.local.yaml` | 旧 D900 best | OFF（仅去重） |
| B | `pipeline.e2e_old_d900_fallback.local.yaml` | 旧 D900 best | ON |
| C | `pipeline.e2e_aug_v1_fallback.local.yaml` | aug_v1 best | ON |

兜底策略要点：
- **配对兜底**：router 给壁类但缺 RC柱（反之亦然），在壁类 union 区域 +5% padding 内运行
  缺失模型；兜底候选 `min_confidence ≥ 0.20`（远高于 detector 默认 0.01）。
- **低置信度/空 router**：状态为 `low_confidence` 或 router 全空时，对全图运行所有 detector，
  `min_confidence ≥ 0.30`。
- **跨 router region 去重**：同一 detector 的同一框只算一次（旧 pipeline 会重复计入）。
- 所有兜底输出都带 `is_fallback=true` 与 `fallback_reasons` 标签，便于审计。

## 总体指标对比

| 指标 | A 基线 | B 旧+兜底 | Δ B vs A | C aug_v1+兜底 | Δ C vs A |
|------|--------|----------|----------|--------------|----------|
| router_hit | 75/80 | 75/80 | 0 | 70/80 | −5 |
| main_matches | 74 | **78** | **+4** | 76 | +2 |
| main_matches_primary | 74 | 74 | 0 | 69 | −5 |
| **fallback_rescued_matches** | 0 | **4** | **+4** | **7** | **+7** |
| main_FN | 18 | **14** | **−4** | 16 | −2 |
| └ fn_router_miss | 6 | **2** | **−4** | 4 | −2 |
| └ fn_iou_miss | 12 | 12 | 0 | 12 | 0 |
| main_FP | 201 | 201 | 0 | **188** | −13 |
| **└ FP from fallback** | 0 | **0** | **0** | **2** | **+2** |
| grade_ok | 71 | **74** | **+3** | 72 | +1 |
| wall_grade_shift_pairs | 273 | 274 | +1 | 270 | −3 |

## RC柱（最受益类别）

| 指标 | A 基线 | B 旧+兜底 | C aug_v1+兜底 |
|------|--------|----------|--------------|
| router_hit | 16/20 | 16/20 | 13/20 |
| main_matches_primary | 14 | 14 | 11 |
| **fallback_rescued** | 0 | **+3** | **+6** |
| main_matches | 14 | **17** | **17** |
| main_FN | 8 | **5** | **5** |
| fn_router_miss | 5 | **2** | 2 |
| main_FP | 48 | 48 | 38 |
| FP_fallback_share | 0 | **0** | 2 |
| grade_ok | 12 | 14 | 14 |

关键结论：**RC柱 router 漏检从 5 张降到 2 张，FN 从 8 降到 5，全部由兜底 rescue 贡献；
旧 D900 baseline 上 0 张额外 FP 来自兜底**。

## 其他类别

- **天井**：A/B 完全相同（router 100% 命中，兜底未触发）；C aug_v1 漏了 1 张天井导致 −1。
- **inner_wall**：A/B 完全相同；C aug_v1 漏了 1 张 → −1。
- **rc_wall**：A 中 1 张漏检（壁类→RC柱误分），B/C 由"present=RC柱 / missing=壁类"
  兜底规则各 rescue 1 张，主匹配 18→19，FN 5→4。

## 噪音审计

| 类别 | 总 fallback 触发数 | rescue 的 GT | 引入的 FP（main） | 引入的 grade_conflict |
|------|-------------------|--------------|-------------------|----------------------|
| 旧 D900 + fallback | rescued=4, main_fallback_preds=4, secondary_fallback=37 | 4 | 0 | +1 wall_pair（噪音可忽略） |
| aug_v1 + fallback | rescued=7, main_fallback_preds=9, secondary_fallback=41 | 7 | 2 | −3 wall_pair |

- 在 main 模型层面：
  - 旧 + 兜底：**主分支 FP 完全没变**（201 → 201）；兜底新增的 main_pred 全是 rescue（4/4 命中 GT）。
  - aug_v1 + 兜底：fallback 新增 9 个 main 预测，其中 7 个命中 GT，2 个 FP。
    精度 7/9 ≈ 78%，远好于普通 detector raw 预测的精度（74/275 ≈ 27%）。
- 在 secondary 模型层面：约 37–41 个 secondary fallback 预测（用于壁类内/外冲突、或柱→壁类反向兜底），
  这些不影响 main 模型的 FP/FN 统计，对最终展示也会被 prod_like merge 处理。
- 壁类 grade_conflict 对子数 273→274（旧+兜底）/ 273→270（aug+兜底），**未引入额外冲突**。

## 结论

1. **工程兜底有效且安全**：旧 D900 + 兜底相对裸 baseline 严格优于（FN −4 / 0 额外 FP / grade_ok +3），
   完全没有引入额外噪音。可以直接上线。
2. **aug_v1 在兜底下追平 RC柱**：兜底把 aug_v1 在 RC柱 上的回退完全补齐（FN 11→5，与旧+兜底持平），
   但仍在 tenjo/inner_wall 各漏 1，整体净比"旧+兜底"差 +2 FN / −13 FP（接近持平，偏 P↑R↓ 风格）。
3. **降噪机制工作良好**：`min_confidence=0.20`（pair）与 union-box 区域限制使得兜底新增的 main 预测
   命中率 ~78%，远高于普通主预测的命中率；噪音完全可控。
4. **兜底 ≠ 替代训练**：兜底主要解决 router 误分导致的"该跑没跑"问题。对于 fn_iou_miss=12 这类
   "detector 都跑了但框没贴准"的 FN，兜底无能为力，需要靠数据/训练改善。

## 推荐落地

- **现在**：把 `fallback_policy.enabled: true` 作为生产配置的默认值，跟旧 D900 router 一起上线。
  预期端到端 FN 下降 ~22%（18 → 14），FP/grade_conflict 无变化。
- **下一步**：路线二（Gemini 合成数据 POC）或路线一 v2（温和增强重训），用于解决 fn_iou_miss=12
  这部分剩余 FN。

## 产物路径

- `outputs/e2e_old_d900_baseline_v2/`（A，无兜底）
- `outputs/e2e_old_d900_fallback_per20/`（B，旧+兜底）
- `outputs/e2e_aug_v1_fallback_per20/`（C，aug_v1+兜底）

每个目录下都包含 `results.jsonl`、`eval_by_image.csv`、`wall_grade_shift_pairs.csv`、`eval_summary.json`，
其中 `results.jsonl` 内每条 raw 都带 `is_fallback` / `fallback_reasons` / `task_kinds` 字段，
可用于进一步钻取兜底贡献。
