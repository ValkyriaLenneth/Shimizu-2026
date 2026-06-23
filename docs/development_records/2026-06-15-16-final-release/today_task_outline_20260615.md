# 2026-06-15 今日任务大纲

## 目标

今天把两条工作合并推进：

1. 先把上传后的数据包和 RF-DETR 模型包整理成最终交付目录，保证后续测试、训练、pipeline 配置都引用同一套稳定路径。
2. 在整理后的最终版基础上，完成 pipeline 端到端测试、天井/RC壁专项优化、以及规则层 pipeline 优化。

当前 repo 状态显示，RF-DETR 版本已经具备独立入口 `rfdetr_prod_pipeline/`，默认配置为 RF-DETR router + RF-DETR 天井/内壁/RC壁/RC柱 下游模型。已有验证仍停留在 smoke / 小批量 case 级别：`docs/development_records/2026-06-08-09-rfdetr-downstream/rfdetr_work_summary_20260609.md` 记录了 1 张真实 smoke 和 12 张 wall-rule batch 成功，但还没有从整体数据重新拆分出的端到端测试集指标。

## 已确认基线

## 当前处理状态

更新：2026-06-15 23:25 UTC。

追加更新：2026-06-16。

Pipeline 部分当前以 `displaymerge_v1` 作为最新可接受结果；`wall_rc_sister_fallback_v1` 和后续 strict rescue 只作为实验记录保留，当前不进入正式 pipeline。

已完成：

- `final_download_20260526.tar.zst` 已上传并解压到：

```text
final_release_20260615/data/final_download_20260526/
```

- 已复制 `data_split.json` 到：

```text
final_release_20260615/data/data_split.json
```

- 已生成数据包文件清单、checksum 和四类下游 split 统计：

```text
final_release_20260615/docs/source_manifests/final_download_20260526.files.txt
final_release_20260615/docs/source_manifests/final_crack_yolo_split_summary.json
final_release_20260615/docs/checksums/SHA256SUMS_uploaded_archives.txt
```

- 已创建最终交付目录 manifest：

```text
final_release_20260615/MANIFEST.md
```

- 已从整体四类数据池构建固定 pipeline 测试集，不使用 `data_split.json`：

```text
data/pipeline_eval_20260615/
```

测试集策略：

```text
seed = 20260615
每类抽样 50 张
来源池 = final_crack_yolo_20260519/split 下四个类别的 train/valid/test 全部图片
总计 = 200 张
```

对应文件：

```text
data/pipeline_eval_20260615/images/
data/pipeline_eval_20260615/labels/
data/pipeline_eval_20260615/manifest.csv
data/pipeline_eval_20260615/split_summary.json
```

- 已按本次 pipeline 测试要求构建 official-plus 永久划分：覆盖 `data_split.json` official test 124 张，并额外补充同等数量 124 张非 official 样本，总计 248 张：

```text
data/pipeline_eval_official_plus_20260615/split.json
data/pipeline_eval_official_plus_20260615/manifest.csv
data/pipeline_eval_official_plus_20260615/split_summary.json
```

- 已完成 baseline pipeline 测试和性能/指标分析：

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/analysis_summary.json
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_baseline/per_image_analysis.csv
docs/development_records/2026-06-15-16-final-release/pipeline_eval_20260615.md
```

- 已完成用户展示层重复框合并与壁类展示修正，并重新跑完 248 张固定测试集：

```text
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/display_merge_summary.json
outputs/rfdetr_prod_pipeline/eval_official_plus_20260615_displaymerge_v1/human_review_static_large/large_index.html
```

最新展示层结果：

| output | precision | recall | F1 | TP | FP | FN | pred |
|---|---:|---:|---:|---:|---:|---:|---:|
| final display | 0.325 | 0.619 | 0.426 | 164 | 341 | 101 | 505 |
| internal pre-display | 0.280 | 0.740 | 0.407 | 196 | 503 | 69 | 699 |

展示层合并效果：

| item | value |
|---|---:|
| images | 248 |
| display before merge | 664 |
| display after merge | 505 |
| suppressed display detections | 236 |
| images with suppression | 133 |

按类别的 final display 指标：

| component | precision | recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| tenjo | 0.333 | 0.651 | 0.441 | 41 | 82 | 22 |
| inner_wall | 0.272 | 0.493 | 0.351 | 34 | 91 | 35 |
| rc_wall | 0.398 | 0.623 | 0.486 | 43 | 65 | 26 |
| rc_column | 0.309 | 0.719 | 0.432 | 46 | 103 | 18 |

本次保留到正式配置的 pipeline 规则：

- 最终输出层增加 `display_merge`，用于合并/抑制重叠框和部分包含的低等级重复框。
- 壁类展示从 `inner_wall` / `rc_wall` 下游结果合并为面向用户的 `壁-B/C/D` 输出。
- 壁类单候选展示数量从 1 放宽到 4，避免面积适中、接近 GT 的候选在展示前被过早丢弃。

本次明确不采用的 pipeline 规则：

- `wall_rc_sister_fallback_v1`：recall 从 0.619 提高到 0.645，但 FP 从 341 增加到 537，mean latency 从 121 ms 增加到 155 ms，140 张图 FP 上升，用户体验风险过高。
- strict rescue 系列：最严格版本基本没有 recall 收益，FP 仍略有上升；较宽松版本负优化更明显。

- 已完成 router 错误严重度分析：

```text
outputs/rfdetr_prod_pipeline/router_error_severity_20260616/summary.json
outputs/rfdetr_prod_pipeline/router_wall_rc_confusion_features_20260616/summary.json
outputs/rfdetr_prod_pipeline/router_per_query_ambiguity_20260616/
```

router 结论：

- Router 对体验有影响，但不是当前主要瓶颈。
- 265 个 GT 中，238 个有正确类别 router 覆盖，23 个只被错误类别 router 覆盖，4 个没有 router 覆盖。
- 101 个 final display 未匹配 GT 中，83 个其实已经有正确类别 router 覆盖；也就是说，多数漏检不是因为 router 把图分错，而是下游检测、展示合并、阈值或业务规则造成。
- wall/RC 混淆时 router 通常是高置信度错分，不是 top1/top2 分数接近的犹豫状态；单 query top2-close 策略在本测试集里几乎没有可用收益。

已核对的数据规模：

| component | train images | valid images | test images |
|---|---:|---:|---:|
| tenjo | 750 | 97 | 96 |
| inner_wall | 840 | 114 | 104 |
| rc_wall | 966 | 90 | 126 |
| rc_column | 498 | 71 | 67 |

模型整理状态：

- `rfdetr_model_candidates_20260602.tar.zst`、`rfdetr_threshold_tuned_models_20260609.tar.zst`、`rfdetr_inner_wall_rc_wall_single_models_20260608.tar.zst` 已完成上传并通过 `tar --zstd -tf` 校验。
- RF-DETR router、天井、内壁、RC壁、RC柱 推荐权重已整理到：

```text
final_release_20260615/models/rfdetr/
```

- final release 版本配置已生成：

```text
final_release_20260615/models/rfdetr/config/pipeline.rfdetr_prod.final_release.yaml
```

- source archive 清单和模型 checksum 已生成：

```text
final_release_20260615/docs/source_manifests/rfdetr_model_candidates_20260602.files.txt
final_release_20260615/docs/source_manifests/rfdetr_threshold_tuned_models_20260609.files.txt
final_release_20260615/docs/source_manifests/rfdetr_inner_wall_rc_wall_single_models_20260608.files.txt
final_release_20260615/docs/checksums/SHA256SUMS_rfdetr_models.txt
```

### 数据与模型包

数据来源包：

```text
final_download_20260526.tar.zst
```

RF-DETR router 与 RC柱候选包：

```text
handoff_20260602/rfdetr_model_candidates_20260602.tar.zst
```

2026-06-09 阈值校准后的下游模型包：

```text
rfdetr_threshold_tuned_models_20260609/
```

当前 pipeline 默认引用：

```text
rfdetr_model_candidates_20260602/router_epoch23/checkpoint_epoch_023.pth
rfdetr_threshold_tuned_models_20260609/checkpoints/tenjo_standard_orig_checkpoint_epoch_009.pth
rfdetr_threshold_tuned_models_20260609/checkpoints/inner_wall_checkpoint_epoch_026.pth
rfdetr_threshold_tuned_models_20260609/checkpoints/rc_wall_checkpoint_epoch_009.pth
rfdetr_model_candidates_20260602/rc_column_epoch47/checkpoint_epoch_047.pth
```

### 当前模型表现

已记录的 2026-06-09 下游结果：

| 类别 | P | R | B R | C R | D R | 状态 |
|---|---:|---:|---:|---:|---:|---|
| 天井 | 0.650 | 0.812 | 0.727 | 0.917 | 0.778 | 低于旧报告 recall 0.845 |
| RC壁 | 0.632 | 0.750 | 0.857 | 0.500 | 0.875 | 已超过旧报告 overall 0.720，但 C 类弱 |
| 内壁 | 0.824 | 0.848 | 0.750 | 1.000 | 0.889 | 当前较成熟 |
| RC柱 | 0.661 | 0.826 | 0.750 | 0.727 | 1.000 | 已超过旧报告目标 |

天井的主要问题是小目标 / 细长 B 类损伤的 score separation。`docs/development_records/2026-06-08-09-rfdetr-downstream/tenjo_rfdetr_failure_analysis_20260608.md` 显示低阈值下模型经常有候选框，但可用阈值下真阳性排序不够高，盲目 oversampling 和长时间 fine-tune 收益有限。

RC壁的主要问题是 C 类。已有记录显示单 checkpoint e009 overall 可用，e063 对 C 类更好但 D 类下降；之前的 class-specific checkpoint routing 已经证明 B/D 用 e009、C 用 e063 可提升 recall，但尚未产品化到 pipeline。

## 任务 0：最终交付目录整理

上传完成后先做只读核对，再整理目录。

建议最终结构：

```text
final_release_20260615/
  data/
    final_download_20260526/
    data_split.json
    README.md
  models/
    rfdetr/
      router/
        checkpoint_epoch_023.pth
        checkpoint_23.ckpt
      downstream/
        tenjo/
          tenjo_standard_orig_checkpoint_epoch_009.pth
          references/
        inner_wall/
          inner_wall_checkpoint_epoch_026.pth
        rc_wall/
          rc_wall_checkpoint_epoch_009.pth
          references/
        rc_column/
          checkpoint_epoch_047.pth
          checkpoint_47.ckpt
      config/
        pipeline.rfdetr_prod.local.yaml
        thresholds.yaml
  docs/
    source_manifests/
    checksums/
```

整理原则：

- 如果上传包中已经有完整数据目录，不强行拆散；在 `final_release_20260615/data/` 下单独保留，并写 README 说明来源。
- 模型目录必须按 router / downstream / 类别分层，避免把不同日期的候选 checkpoint 混放。
- 保留 2026-06-09 包中的 reference checkpoints，但推荐模型必须有清晰标记。
- 生成 `SHA256SUMS.txt` 和 `MANIFEST.md`，记录每个模型文件的来源包、用途、当前 pipeline 配置路径。

## 任务 1：Pipeline 整体测试

### 测试集策略

本轮不使用 `data_split.json`。目标是从整体数据中重新拆一个 pipeline 测试集，用来模拟完整业务输入，而不是复现旧报告指标。

建议：

- 从 `final_crack_yolo_20260519/split/{tenjo,inner_wall,rc_wall,rc_column}` 汇总整体图片池。
- 按类别分层抽样，先做一个可快速迭代的固定测试集，例如每类 50-100 张。
- 抽样输出到新目录，例如：

```text
data/pipeline_eval_20260615/
  images/
  labels/
  manifest.csv
  split_summary.json
```

`manifest.csv` 至少记录：

```text
image_path,component,source_split,label_path,canonical_stem
```

### 指标

需要记录两类指标。

Pipeline 工程指标：

- 总图像数、成功数、异常数。
- router status 分布。
- router 输出类别分布。
- 下游模型调用次数。
- warning 类型分布。
- 平均耗时、P50/P90/P95/P99 耗时。
- 单图最大耗时和对应文件。
- GPU / CPU 环境、batch 策略、是否保存可视化。

模型效果指标：

- router 是否命中图片所属大类，壁类中内壁/RC壁 合并为 `壁类` 口径。
- 最终 `crack_detections` 对 B/C/D 标签的 precision / recall / F1。
- 按 component 拆分的 B/C/D recall。
- 漏检 case、误检 case、重复显示 case、跨类误路由 case。

现有 `rfdetr_prod_pipeline/pipeline/run_full_pipeline.py` 已经写入每张图 `elapsed_ms`，`rfdetr_prod_pipeline/scripts/summarize_pipeline_results.py` 可以汇总平均耗时、router 状态、warning 和 detection 分布。今天需要补齐：

- 固定抽样测试集构建脚本。
- pipeline 输出与 YOLO 标签的匹配评估脚本。
- 更完整的速度统计，至少加入 P50/P90/P95/P99。

### 初始运行命令

整理完模型路径后，先用最终配置跑：

```bash
python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline \
  --config rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.local.yaml \
  --source data/pipeline_eval_20260615/images \
  --output-dir outputs/rfdetr_prod_pipeline/eval_20260615_baseline \
  --device cuda:0 \
  --skip-visualization
```

然后汇总：

```bash
python rfdetr_prod_pipeline/scripts/summarize_pipeline_results.py \
  outputs/rfdetr_prod_pipeline/eval_20260615_baseline/results.jsonl
```

## 任务 2：天井和 RC壁 快速优化

### 总原则

因为当前 RF-DETR 是预训练模型微调，历史记录已经显示很多 run 在 10 epoch 左右就进入过拟合或收益下降。因此今天优先做短周期、多策略验证：

- 每个策略 5-12 epoch。
- 每 epoch 保存 `.pth`。
- 每 epoch 用固定测试集或固定 holdout 做 threshold sweep。
- 选模型时以 recall 超过旧报告值为主，但必须同时记录 precision 和 FP 数量。

目标：

- 天井 overall recall 超过旧报告值 `0.845`。
- RC壁 recall 在保持当前优势基础上进一步提升，重点拉升 C 类 recall，旧报告 C recall 目标为 `0.680`。

### 天井候选策略

优先验证以下策略：

1. class-specific threshold calibration：B 类降低阈值，C/D 保持较高阈值，先确认是否能在可接受 FP 下超过 0.845。
2. high-recall proposal + filtering：低阈值保留 B 候选，再加规则过滤极端背景框、过小/过大框、低 IoU 形态异常框。
3. small-object / crop inference：对天井区域做 tile 或放大 crop 推理，重点看 B 类 recall。
4. hard-negative fine-tune：从高置信 false positives 中构造背景负例，不再只重复 B 正样本。
5. 短 epoch 微调：从 e009 或低 epoch reference checkpoint 出发，使用较低 lr、强 early stop，不做长训。

现有可用工具：

```text
scripts/evaluate_rfdetr_threshold_sweep.py
scripts/evaluate_rfdetr_class_threshold_grid.py
scripts/analyze_rfdetr_hard_cases.py
scripts/train_rfdetr_router.py
```

### RC壁候选策略

优先验证以下策略：

1. class-specific checkpoint routing：B/D 使用 e009，C 使用 e063 或 reference checkpoint，把之前离线验证过的策略接入可复用评估脚本。
2. class-specific thresholds：C 类阈值单独下降，B/D 保持当前配置，观察 C recall 和 FP 增量。
3. C 类 hard-case 短训：围绕 C 类 FN 和混淆样本做 5-10 epoch fine-tune，不做大规模 oversampling。
4. wall-context rule：当 router 为 `壁类` 时，同时利用内壁和 RC壁 输出，评估是否能用组合规则减少 RC壁 C 类漏显。

RC壁优化产物要求：

- 每个 run 记录训练命令、checkpoint、threshold、P/R/F1、B/C/D recall、FP/FN。
- 任何 routing 策略都必须明确是否是“单 checkpoint 模型”还是“多 checkpoint 推理策略”，避免报告口径混淆。

## 任务 3：Pipeline 规则层优化

规则目标不是替代模型，而是减少用户可见的问题：

- 重复框太多。
- 壁类同时跑内壁/RC壁 后显示混乱。
- router 低置信度导致漏调用。
- 大面积区域和小损伤区域的 crop 上下文不稳定。
- 高风险 D 类被合并或过滤掉。

优先规则：

1. router 低置信度 fallback：低于主阈值但高于低阈值时，按候选类别并行调用下游模型，并在 audit 中标记。
2. no-router fallback：router 无输出时，按轻量规则或全模型 fallback 跑一次，避免直接空结果。
3. wall display rule 扩展：保留当前 `内壁/RC壁 -> 壁-B/C/D` 组合表，同时记录 raw candidates，评估是否减少重复显示。
4. class-safe merge：跨 grade 合并时优先保留更高风险等级，尤其 D 类不得被低等级框吞掉。
5. region padding sweep：比较 `region_padding_ratio` 0.10 / 0.20 / full_image_filter 对 recall 和速度的影响。
6. 输出审计增强：每个最终显示框都能追踪到 router region、source_model、raw candidate、merge reason。

当前代码里已有相关基础：

```text
rfdetr_prod_pipeline/pipeline/fallback_policy.py
rfdetr_prod_pipeline/pipeline/result_merge.py
rfdetr_prod_pipeline/pipeline/wall_candidate_display.py
rfdetr_prod_pipeline/pipeline/run_full_pipeline.py
```

今天的重点是把规则变成可开关的配置项，并在 pipeline 测试集上量化对漏检、重复显示、耗时的影响。

## 今日交付物

最低交付：

- `final_release_20260615/` 目录结构和 manifest。
- pipeline 固定测试集目录与 manifest。
- baseline pipeline 测试结果：

```text
outputs/rfdetr_prod_pipeline/eval_20260615_baseline/
```

- pipeline 指标汇总文档：

```text
docs/development_records/2026-06-15-16-final-release/pipeline_eval_20260615.md
```

- 天井/RC壁 优化实验记录：

```text
docs/rfdetr_tenjo_rcwall_optimization_20260615.md
```

- 如果规则层有改动，记录配置、效果和回退方式：

```text
docs/rfdetr_pipeline_rule_optimization_20260615.md
```

## 推荐执行顺序

1. 等上传完成，校验并整理 `final_release_20260615/`。
2. 更新或生成最终 pipeline config，使其引用整理后的模型目录。
3. 从整体数据抽样构建 `data/pipeline_eval_20260615/`。
4. 跑 baseline pipeline，记录速度和错误类型。
5. 补 pipeline GT 评估脚本，输出 component / grade 级指标。
6. 并行做天井 threshold / crop / hard-negative 快速验证。
7. 并行做 RC壁 threshold / checkpoint routing 快速验证。
8. 将有效策略接入 pipeline 配置，跑同一测试集对比。
9. 写最终测试与优化记录，固定可复现命令。
