# 2026-05-26 开发现场保存与恢复说明

本文是销毁当前服务器实例前的最终交接文档，覆盖本次代码和文档改动、当前最佳自动识别模型、2026-05-26 新增评估结果、Gemini 合成数据保留范围，以及下次恢复开发现场的步骤。

## 1. 当前结论

本轮主要完成了四件事：

1. 壁类结果从“可能重复出两条”改成了“单图单结果，必要时在同图内并列候补”
2. 在自动识别与下游判定之间增加了工程化兜底机制
3. 继续做了数据增强训练，并把自动识别模型提升到了本轮当前最好版本
4. 跑通了 Gemini 合成数据生成 pipeline，并保留了最终 `4 x 300` 版本

当前最佳自动识别模型的独立测试集指标：

| 指标 | 数值 |
|---|---:|
| Precision | 0.863 |
| Recall | 0.850 |
| mAP50 | 0.888 |
| mAP50-95 | 0.775 |

相较于上周汇报时使用的版本：

- Precision: `0.843 -> 0.863`
- Recall: `0.841 -> 0.850`
- mAP50: `0.888 -> 0.888`

## 2. 本次保留的内容

### 2.1 代码与文档

代码和文档以 Git 提交为准，重点包括：

- 壁类候补展示逻辑
- fallback 策略
- 2026-05-26 汇报文档
- 2026-05-26 汇报图片与 fallback 对比图
- Gemini 合成数据 pipeline 脚本

### 2.2 当前最佳 router 训练结果

本次只保留当前最好的自动识别模型相关结果，不保留所有中间训练轮次。

保留内容包括：

- `coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv3_recall_ft_b48_e15`
  - `results.csv`
  - `opt.yaml`
  - `hyp.yaml`
  - 曲线图与样例图
  - `weights/epoch14.pt`
  - `weights/best_striped.pt`

以及与它直接相关的评估结果：

- `outputs/router_eval/augv3_epoch14_test`
- `outputs/router_eval/augv3_epoch14_test.log`
- `outputs/router_eval/augv3_epoch14_test_saved_preds`
- `outputs/router_eval/augv3_epoch14_test_saved_preds.log`
- `outputs/router_eval/d900_test`
- `outputs/e2e_old_d900_baseline_v2`
- `outputs/e2e_old_d900_fallback_per20`

### 2.3 Gemini 合成数据

本次只保留最终版本：

```text
outputs/synthetic_router_pipeline_nb2_promptgen_300x4_c10
```

这就是最后定稿的 `4 x 300` 数据。

不保留的中间版本包括：

- `20x4`
- `100x4`
- `400x4`
- dryrun
- concurrency benchmark
- 其他中途尝试版本

## 3. 本次新增的重要文档与资产

重点文档：

- `docs/meeting_outline_20260526_router_pipeline_and_synthetic.md`
- `docs/engineering_fallback_eval_20260526.md`
- `docs/engineering_fallback_v3_signal_gated_20260526.md`
- `docs/router_synthetic_mix_training_plan_20260526.md`

重点图片资产：

- `docs/report_assets_20260526/fallback_compare_png`

## 4. 恢复步骤

### 4.1 先恢复 Git 代码

```bash
git clone https://github.com/ValkyriaLenneth/Shimizu-2026.git
cd Shimizu-2026
```

### 4.2 解压本次交接包

将以下文件放到仓库根目录的 `handoff_20260526/` 下：

```text
handoff_20260526/shimizu_20260526_router_best_and_eval.tar.zst
handoff_20260526/shimizu_20260526_synthetic_router_final_300x4.tar.zst
handoff_20260526/SHA256SUMS.txt
```

然后执行：

```bash
tar --zstd -xf handoff_20260526/shimizu_20260526_router_best_and_eval.tar.zst -C .
tar --zstd -xf handoff_20260526/shimizu_20260526_synthetic_router_final_300x4.tar.zst -C .
sha256sum -c handoff_20260526/SHA256SUMS.txt
```

### 4.3 准备环境

仓库根目录：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

YOLOv9 环境：

```bash
cd coarse_router_yolov9
bash scripts/setup_yolov9_env.sh
source .venv/bin/activate
```

## 5. 恢复后优先确认的路径

恢复后应优先确认以下内容存在：

```text
docs/meeting_outline_20260526_router_pipeline_and_synthetic.md
docs/report_assets_20260526/fallback_compare_png/00_contact_sheet.png
outputs/router_eval/augv3_epoch14_test.log
outputs/e2e_old_d900_fallback_per20/eval_summary.json
outputs/synthetic_router_pipeline_nb2_promptgen_300x4_c10/pipeline_summary.json
coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_augv3_recall_ft_b48_e15/weights/epoch14.pt
```

## 6. 下次继续开发时的建议入口

建议按下面顺序继续：

1. 先阅读 `docs/meeting_outline_20260526_router_pipeline_and_synthetic.md`
2. 看 `docs/report_assets_20260526/fallback_compare_png`，确认壁类展示与 fallback 的客户展示口径
3. 用 `outputs/router_eval/augv3_epoch14_test.log` 和 `outputs/e2e_old_d900_fallback_per20` 重新对齐当前最佳 router 与工程兜底结果
4. 如果继续做合成数据训练，只从 `outputs/synthetic_router_pipeline_nb2_promptgen_300x4_c10` 出发，不再回头使用中间实验版本
