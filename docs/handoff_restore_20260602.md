# 2026-06-02 现场恢复交接文档

## 目的

用于下周在新实例上恢复今天的 RF-DETR 迁移现场，包括：

- 旧 YOLO 数据划分 JSON。
- YOLO9 四个下游模型的下载链接。
- 今天训练/筛选出的 RF-DETR 模型候选。
- 自动识别模型和 RC柱 的汇报素材。
- 下游数据集恢复方式。

## 安全注意

不要把 GitHub token 写入任何脚本、文档或压缩包。今天对话中出现过 token，但本交接文档和压缩包不会保存该 token。建议在 GitHub 中撤销旧 token 并重新生成。

## 关键输入文件

旧 YOLO 训练/评估划分 JSON：

```text
data_split.json
```

该文件今天已复制进模型交接压缩包：

```text
handoff_20260602/rfdetr_model_candidates_20260602/data/data_split.json
```

用途：

- 重建去年 YOLO 结果对应的 official test。
- 对齐 RF-DETR 下游模型评估协议。
- 避免新旧数据划分不一致导致指标不可比。

## YOLO9 模型下载链接

四个旧 YOLO9 下游模型来自同一个 Google Drive 文件：

```text
https://drive.google.com/file/d/1hMzZDjCh6QJB3kk2pCHxYC84istb_PAh/view?usp=drive_link
```

下载后应包含：

```text
TIANJING.pt
NEIBI.pt
RCBI.pt
RCZHU.pt
```

今天本机路径：

```text
downloads/previous_phase_gpl_model_unpacked/infer_models/TIANJING.pt
downloads/previous_phase_gpl_model_unpacked/infer_models/NEIBI.pt
downloads/previous_phase_gpl_model_unpacked/infer_models/RCBI.pt
downloads/previous_phase_gpl_model_unpacked/infer_models/RCZHU.pt
```

其中 `RCZHU.pt` 今天用于生成 RC柱 YOLO9 未检出 vs RF-DETR 检出的对比图。

## 今天的 RF-DETR 模型交接包

压缩包：

```text
handoff_20260602/rfdetr_model_candidates_20260602.tar.zst
```

解压命令：

```bash
tar --zstd -xf handoff_20260602/rfdetr_model_candidates_20260602.tar.zst
```

包内结构：

```text
rfdetr_model_candidates_20260602/
  README.md
  data/
    data_split.json
  docs/
    development_summary_20260602.md
    handoff_restore_20260602.md
    downstream_dataset_restore_20260602.md
    meeting_rfdetr_progress_story_20260602.md
    report_assets_20260602_rfdetr/
  router_epoch23/
  rc_column_epoch47/
  rc_wall_candidates_epoch19_22_71/
```

## RF-DETR 自动识别模型

候选：

```text
router_epoch23/checkpoint_epoch_023.pth
router_epoch23/checkpoint_23.ckpt
```

本机源路径：

```text
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest/epoch_pth/checkpoint_epoch_023.pth
outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest/checkpoint_23.ckpt
```

核心结果：

| model | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| YOLOv9 tuned baseline | 0.863 | 0.850 | - | 0.888 | 0.775 |
| RF-DETR epoch 23 | 0.905 | 0.852 | 0.877 | 0.904 | 0.782 |

结论：

- 达到客户提出的 Precision 0.90 目标。
- 下周自动识别模型可以先告一段落，重点转向下游单类别模型。

## RF-DETR RC柱 下游模型

候选：

```text
rc_column_epoch47/checkpoint_epoch_047.pth
rc_column_epoch47/checkpoint_47.ckpt
```

本机源路径：

```text
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/epoch_pth/checkpoint_epoch_047.pth
outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1/checkpoint_47.ckpt
```

结果：

| 范围 | 去年报告目标 R | RF-DETR R |
|---|---:|---:|
| Overall | 0.742 | 0.826 |
| B | 0.700 | 0.750 |
| C | 0.706 | 0.727 |
| D | 0.807 | 1.000 |

## RF-DETR RC壁 候选模型

RC壁 当前不作为汇报重点，但保留三个候选用于下周恢复分析：

| epoch | Precision | Recall | F1 | mAP50 | mAP50-95 | 用途 |
|---:|---:|---:|---:|---:|---:|---|
| 22 | 0.731 | 0.607 | 0.657 | 0.648 | 0.241 | 综合最好 |
| 71 | 0.613 | 0.679 | 0.643 | 0.631 | 0.155 | recall 最高 |
| 19 | 0.759 | 0.565 | 0.645 | 0.624 | 0.247 | precision 最高 |

包内文件：

```text
rc_wall_candidates_epoch19_22_71/checkpoint_epoch_019.pth
rc_wall_candidates_epoch19_22_71/checkpoint_19.ckpt
rc_wall_candidates_epoch19_22_71/checkpoint_epoch_022.pth
rc_wall_candidates_epoch19_22_71/checkpoint_22.ckpt
rc_wall_candidates_epoch19_22_71/checkpoint_epoch_071.pth
rc_wall_candidates_epoch19_22_71/checkpoint_71.ckpt
```

## 关键汇报文档

中文 Markdown 汇报稿：

```text
docs/meeting_rfdetr_progress_story_20260602.md
```

该文档已嵌入今天生成的图片，可直接用于 Markdown 滚动汇报。

## 关键可视化素材

自动识别模型：YOLO9 vs RF-DETR 对比图，最终采用 case 3/4/5。

```text
docs/report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_03_RC柱_d-40027_03307.jpg
docs/report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_04_RC壁_c-199_03206.jpg
docs/report_assets_20260602_rfdetr/comparison_yolo_vs_rfdetr_05_RC壁_c-40616_03440.jpg
```

RC柱：YOLO9 未检出，RF-DETR 检出。

```text
docs/report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_01_data_add100__4-D-00168.jpg
docs/report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_02_data_add100__4-B-00118.jpg
docs/report_assets_20260602_rfdetr/rc_column_yolo_missed_rfdetr_detected_03_data_add100__d-10.jpg
```

## 新实例恢复建议顺序

1. Clone repo。
2. 安装 RF-DETR 依赖：

```bash
python -m pip install -r requirements-rfdetr.txt
```

3. 下载并解压旧 YOLO9 模型包。
4. 放回 `data_split.json`。
5. 按 `docs/downstream_dataset_restore_20260602.md` 重建下游 RF-DETR 数据集 view。
6. 解压今天的 RF-DETR 模型候选包。
7. 使用 `scripts/sweep_rfdetr_router_test.py` 或现有推理脚本验证 checkpoint 可加载。

## 销毁当前实例前检查

必须确认已保存：

- `handoff_20260602/rfdetr_model_candidates_20260602.tar.zst`
- `data_split.json`
- `docs/meeting_rfdetr_progress_story_20260602.md`
- `docs/report_assets_20260602_rfdetr/`
- 本文档和 `docs/downstream_dataset_restore_20260602.md`

建议另外保存：

- `final_download_20260526.tar.zst`，如果本地已有则不需要从实例保存。
- 旧 YOLO9 模型 Google Drive 链接。

不要保存：

- GitHub token。
- 未筛选的 RC壁 全量 80 epoch checkpoint，除非确实需要占用约 51GB 的完整训练现场。
