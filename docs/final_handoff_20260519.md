# 2026-05-19 开发现场保存与恢复说明

本文是销毁当前服务器实例前的最终交接文档，覆盖数据构成、下载与归档、训练环境、今日工作总结、恢复步骤和后续继续开发入口。

## 1. 当前结论

本轮已经完成三类自动识别模型的训练、数据清洗、端到端推理流程接入和汇报材料整理。

当前最好自动识别模型：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_ft_from_d900_imgw_rc_os900_e50_ddp/weights/best.pt
```

独立测试集指标：

| 类别 | Precision | Recall | mAP50 |
|---|---:|---:|---:|
| all | 0.843 | 0.841 | 0.888 |
| 天井 | 0.854 | 0.880 | 0.923 |
| 壁类 | 0.786 | 0.791 | 0.836 |
| RC柱 | 0.888 | 0.853 | 0.905 |

端到端抽样结论：

| 流程 | 主分支匹配 | FN | FP |
|---|---:|---:|---:|
| 上一期单模型整图方式 | 72/92 | 20 | 112 |
| 自动识别 + 整图识别 + 区域过滤 + 同模型合并 | 66/92 | 26 | 77 |

说明：自动识别接入后，无关误检明显减少；剩余损失主要来自 RC柱 与壁类混淆，以及少量区域覆盖不足。

## 2. 数据集构成和来源

### 2.1 原始图片数据

当前统合后的原始图片来源在：

```text
data/unzip
additional_data_2026-05-19/unpacked
```

其中：

- `data/unzip`：detect_dataset-cvat 对应图片。
- `additional_data_2026-05-19/unpacked/data_add100`：每类约 300 张，图片 + label。
- `additional_data_2026-05-19/unpacked/labels_20251107`：最后一次追加的四类 label，图片来自 `data/unzip`。

### 2.2 裂缝标注数据集

整理后的最终裂缝检测训练数据在：

```text
data/final_crack_yolo_20260519
```

规模：

| 类别 | 图片数 | 裂缝框数 |
|---|---:|---:|
| 天井 | 943 | 1023 |
| 内壁 | 1058 | 1218 |
| RC壁 | 1182 | 1480 |
| RC柱 | 636 | 679 |
| 合计 | 3819 | 4400 |

说明文档：

```text
docs/final_crack_dataset_20260519.md
data/final_crack_yolo_20260519/README.md
```

### 2.3 Gemini 标注的自动识别数据

Gemini 3.1 Pro 标注输出：

```text
outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19
outputs/gemini_merged_4219_3_1_pro_preview_2026-05-19
outputs/gemini_data_add100_3_1_pro_preview_2026-05-19
```

三类 YOLO 自动识别数据集：

```text
coarse_router_yolov9/datasets/coarse_router_3class_full_merged_4219
coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219
coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900
```

类别定义：

```text
0: 天井
1: 壁类 = 内壁 + RC壁
2: RC柱
```

清洗后三类数据规模：

| 自动识别类别 | 标注框数 |
|---|---:|
| 天井 | 1902 |
| 壁类 | 3785 |
| RC柱 | 948 |
| 合计 | 6635 |

## 3. 训练环境准备

### 3.1 基础 Python 环境

```bash
cd /workspace/Shimizu-2026
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需重新调用 Gemini：

```bash
export GEMINI_API_KEY=...
```

不要把 API key 写入 Git 或归档文档。

### 3.2 YOLOv9 环境

当前训练代码在：

```text
coarse_router_yolov9
```

环境准备脚本：

```bash
cd /workspace/Shimizu-2026/coarse_router_yolov9
bash scripts/setup_yolov9_env.sh
source .venv/bin/activate
```

本次机器使用双 GPU 训练，训练脚本已经保留：

```text
coarse_router_yolov9/scripts/train_router_3class_parallel.sh
coarse_router_yolov9/scripts/train_router_3class_tuning_b_c.sh
coarse_router_yolov9/scripts/train_router_3class_tuning_d_parallel.sh
```

## 4. 今日工作总结

### 4.1 数据侧

- 下载并整理了 `data_add100`、`20251107` 四类追加 label 和 `detect_dataset-cvat` 配套图片。
- 构建了最终裂缝检测数据集 `data/final_crack_yolo_20260519`。
- 重新调用 Gemini 对建筑区域进行标注。
- 建立了自动识别模型的 `full` 与 `cleaned` 两套数据集。
- 按会议共识将 `RC壁` 与 `内壁` 在自动识别阶段合并为 `壁类`。

### 4.2 训练侧

- 完成 full/cleaned baseline 对照。
- 针对 RC柱 数据少、容易与壁类混淆的问题，尝试了 image-weights、RC柱 oversampling、D800/D900 策略。
- 在 D900 策略基础上并入追加数据，完成当前最好模型训练。
- 保存了训练结果、验证结果和汇报用指标。

### 4.3 系统集成侧

系统集成代码在：

```text
router_crack_pipeline
```

已完成：

- 自动识别模型推理封装。
- 裂缝模型 registry。
- 端到端 pipeline runner。
- 初版局部图片方式。
- 改进后的整图推理 + 区域过滤方式。
- 同模型去重合并逻辑。
- 抽样端到端评估脚本和可视化脚本。

### 4.4 汇报材料

汇报故事线和图片材料：

```text
docs/report_story_router_e2e_20260519.md
docs/report_assets_20260519
```

## 5. 下次如何恢复开发现场

### 5.1 从 Git 恢复代码

```bash
git clone https://github.com/ValkyriaLenneth/Shimizu-2026.git
cd Shimizu-2026
```

如果 GitHub 上的提交不可用，也可以从交接目录中的 git bundle 恢复。

### 5.2 解压归档数据

将交接目录中的 zip 放到仓库根目录后执行：

```bash
unzip handoff_20260519/shimizu_20260519_data_package.zip -d .
unzip handoff_20260519/shimizu_20260519_router_models_and_results.zip -d .
```

恢复后应存在：

```text
data/unzip
data/final_crack_yolo_20260519
outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19
outputs/gemini_merged_4219_3_1_pro_preview_2026-05-19
coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219
coarse_router_yolov9/runs/train/gelan_c_router_3class_merged4219_ft_from_d900_imgw_rc_os900_e50_ddp/weights/best.pt
```

### 5.3 校验归档

```bash
cd /workspace/Shimizu-2026
sha256sum -c handoff_20260519/SHA256SUMS.txt
```

### 5.4 重新训练自动识别模型

```bash
cd /workspace/Shimizu-2026/coarse_router_yolov9
source .venv/bin/activate
bash scripts/train_router_3class_parallel.sh
```

RC柱 调优：

```bash
bash scripts/train_router_3class_tuning_b_c.sh
bash scripts/train_router_3class_tuning_d_parallel.sh
```

### 5.5 运行端到端抽样评估

优先使用当前最好配置：

```bash
cd /workspace/Shimizu-2026
python router_crack_pipeline/scripts/debug_e2e_sample_eval.py \
  --config router_crack_pipeline/configs/pipeline.router_merged4219_ddp.fullimage_samemodel.local.yaml \
  --sample-size 80 \
  --output-dir outputs/e2e_debug_sample_80_router_merged4219_ddp_fullimage_samemodel_20260519
```

## 6. 关键文档入口

数据构建与清洗：

```text
docs/router_3class_data_cleaning_2026-05-19.md
router_crack_pipeline/docs/router_3class_data_cleaning_2026-05-19.md
```

训练准备与结果：

```text
docs/router_3class_training_preparation_2026-05-19.md
docs/router_tuning_b_c_results_20260519.md
router_crack_pipeline/docs/router_3class_training_preparation_2026-05-19.md
```

端到端流程：

```text
router_crack_pipeline/README.md
router_crack_pipeline/docs/full_pipeline_detailed_outline_2026-05-19.md
router_crack_pipeline/docs/integration_test_report_2026-05-19.md
docs/original_prod_vs_router_e2e_20260519.md
```

汇报材料：

```text
docs/report_story_router_e2e_20260519.md
docs/report_assets_20260519
```

## 7. 需要注意的遗留问题

- 自动识别模型不是 100%，RC柱 与壁类混淆仍会影响端到端结果。
- 壁类区域同时调用内壁模型和 RC壁模型，部分样本会出现等级差异，需要后续业务规则或 UI 确认。
- 当前推荐使用“整图推理 + 区域过滤 + 同模型合并”，不要优先回到局部图片切片方式。
- 后续可考虑 AI 合成数据补充 RC柱 与壁类边界场景。
- UI 侧建议增加“自动识别区域预览与用户确认”步骤。

