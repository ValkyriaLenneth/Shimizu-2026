# 最终裂缝检测数据集整理记录

日期：2026-05-19

## 目标

根据本次补充信息，将当前可直接使用的两部分数据汇总为一个最终版本目录：

1. `data_add100`
   - 每类 301 张图片 + label。
   - 本地路径：`additional_data_2026-05-19/unpacked/data_add100`
2. `xx_20251107` 四类 label 包
   - 仅包含 label。
   - 对应图片来自 `detect_dataset-cvat`，当前本地解压路径为 `data/unzip`。

最终输出目录：

```text
data/final_crack_yolo_20260519
```

构建脚本：

```text
scripts/build_final_crack_dataset_20260519.py
```

## 下载的 20251107 label 包

```text
downloads/labels_20251107/tenjo_20251107.zip
downloads/labels_20251107/inner_wall_20251107.zip
downloads/labels_20251107/rc_wall_20251107.zip
downloads/labels_20251107/rc_column_20251107.zip
```

解压路径：

```text
additional_data_2026-05-19/unpacked/labels_20251107
```

## 核对结果

`data_add100` 与 20251107 labels 按文件 stem 对比，四类均无重叠：

| 类别 | data_add100 labels | 20251107 labels | stem 重叠 |
|---|---:|---:|---:|
| 天井 | 301 | 642 | 0 |
| 内壁 | 301 | 757 | 0 |
| RC壁 | 301 | 881 | 0 |
| RC柱 | 301 | 335 | 0 |

20251107 labels 与 `data/unzip/*/obj_train_data` 图片匹配结果：

| 类别 | labels | 匹配图片 | 缺失图片 |
|---|---:|---:|---:|
| 天井 | 642 | 642 | 0 |
| 内壁 | 757 | 757 | 0 |
| RC壁 | 881 | 881 | 0 |
| RC柱 | 335 | 335 | 0 |

## 最终目录结构

```text
data/final_crack_yolo_20260519/
  all/<class>/images
  all/<class>/labels
  all/<class>/data.yaml
  split/<class>/train/images
  split/<class>/train/labels
  split/<class>/valid/images
  split/<class>/valid/labels
  split/<class>/test/images
  split/<class>/test/labels
  split/<class>/data.yaml
  raw_sources/<class>
  manifest.csv
  summary.json
  README.md
```

说明：

- `all` 保留每类全部样本。
- `split` 对全部样本按稳定 hash 重新切分为 train/valid/test。
- 文件名加入来源前缀，例如 `data_add100__...`、`labels_20251107__...`，避免同名覆盖。
- `manifest.csv` 记录每个样本的原始路径、来源、原始 split、最终 split、box 数和 label 校验结果。
- `raw_sources` 保存每类来源侧 metadata 副本。

## 汇总统计

| 类别 | 总样本 | box 数 | data_add100 | labels_20251107 | train | valid | test | 无效 label 文件 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 天井 | 943 | 1023 | 301 | 642 | 750 | 97 | 96 | 0 |
| 内壁 | 1058 | 1218 | 301 | 757 | 840 | 114 | 104 | 0 |
| RC壁 | 1182 | 1480 | 301 | 881 | 966 | 90 | 126 | 0 |
| RC柱 | 636 | 679 | 301 | 335 | 498 | 71 | 67 | 0 |
| 合计 | 3819 | 4400 | 1204 | 2615 | 3054 | 372 | 393 | 0 |

## 可直接使用的 data.yaml

```text
data/final_crack_yolo_20260519/split/tenjo/data.yaml
data/final_crack_yolo_20260519/split/inner_wall/data.yaml
data/final_crack_yolo_20260519/split/rc_wall/data.yaml
data/final_crack_yolo_20260519/split/rc_column/data.yaml
```

这四个 `data.yaml` 对应四个独立裂缝/损伤等级模型，类别均为 B/C/D 三类。

## 当前未做事项

- 没有合并四个构件类为一个裂缝等级模型；当前仍按上一期四模型路线保留四套数据。
- 没有基于真实业务效果重新清洗或剔除样本。
- 没有对 B/C/D 标注质量做人工复核。
