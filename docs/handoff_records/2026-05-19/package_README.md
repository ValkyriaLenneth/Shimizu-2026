# Shimizu-2026 交接包说明

生成日期：2026-05-19

本目录用于销毁当前服务器实例前保存可下载内容。完整恢复说明见：

```text
docs/handoff_records/2026-05-19/final_handoff_20260519.md
```

## 归档包

### 1. 数据包

```text
shimizu_20260519_data_package.zip
```

包含：

- `data/unzip`：统合后的原始图片数据。
- `additional_data_2026-05-19/unpacked`：追加数据解压结果。
- `data/final_crack_yolo_20260519`：整理好的裂缝标注数据集。
- `outputs/gemini_full_all_4classes_3_1_pro_preview_2026-05-19`：Gemini 全量标注与 QA。
- `outputs/gemini_merged_4219_3_1_pro_preview_2026-05-19`：追加后 Gemini 标注。
- `outputs/gemini_data_add100_3_1_pro_preview_2026-05-19`：data_add100 Gemini 标注。
- `coarse_router_yolov9/datasets/coarse_router_3class_full_merged_4219`：三类 full 自动识别数据。
- `coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219`：三类 cleaned 自动识别数据。
- `coarse_router_yolov9/datasets/coarse_router_3class_cleaned_merged_4219_rc_os900`：RC柱 oversample 训练数据。

### 2. 模型与训练结果包

```text
shimizu_20260519_router_models_and_results.zip
```

包含：

- 当前最好模型 `best.pt`。
- 各阶段 router 训练的 `best.pt`、`results.csv`、`results.png`。
- val/test 评估输出。
- E2E 抽样评估输出。
- 训练配置和超参。

### 3. Git bundle

```text
shimizu_20260519_git_bundle.bundle
```

用于在 GitHub 不可用时恢复代码提交。

## 恢复命令

```bash
git clone https://github.com/ValkyriaLenneth/Shimizu-2026.git
cd Shimizu-2026
unzip handoff_20260519/shimizu_20260519_data_package.zip -d .
unzip handoff_20260519/shimizu_20260519_router_models_and_results.zip -d .
sha256sum -c handoff_20260519/SHA256SUMS.txt
```

从 bundle 恢复：

```bash
git clone handoff_20260519/shimizu_20260519_git_bundle.bundle restored-shimizu-2026
```

## 注意事项

- zip 包不进入 Git；需要单独下载保存。
- 归档不包含 Python 虚拟环境 `.venv`，下次按文档重新安装。
- 归档不保存任何 API key 或 GitHub token。
