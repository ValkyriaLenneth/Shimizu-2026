# 下游 RF-DETR 数据集恢复说明 2026-06-02

## 目的

今天不会额外保存一份完整下游训练数据。本文件记录如何在下周新实例上恢复今天用于下游 RF-DETR 训练/测试的数据集 view。

关键原则：

- 使用当前 20260519 下游裂缝/损伤数据作为训练来源。
- 使用 `data_split.json` 中的旧 test split 作为 official test。
- 训练集最大化，但不能把 official test 图片放进 train。
- `valid = test = official test`，用于固定协议下的 checkpoint 选择。

## 必需输入

### 1. 原始数据包

今天数据来源为此前上传并解压的：

```text
final_download_20260526.tar.zst
```

解压后关键路径：

```text
final_download_20260526/handoff_20260519/shimizu_20260519_minimal_repro_package/data/final_crack_yolo_20260519/split
```

该路径下包含四个下游类别：

```text
tenjo
inner_wall
rc_wall
rc_column
```

### 2. 旧 YOLO 划分 JSON

```text
data_split.json
```

用途：

- 重建旧 official test。
- 固定下游模型比较协议。
- 避免使用当前随机划分导致和去年指标不可比。

## 今天已构建的数据集 view

### RC柱

```text
data/rfdetr_rc_column_all_non_legacy_test_v1
```

策略：

```text
train = 当前 RC柱 全量图片 - data_split.json 中 RC柱 official test stems
valid = data_split.json 中 RC柱 official test
test  = data_split.json 中 RC柱 official test
```

数据量：

| split | images | B boxes | C boxes | D boxes |
|---|---:|---:|---:|---:|
| train | 605 | 317 | 186 | 145 |
| valid | 31 | 12 | 11 | 8 |
| test | 31 | 12 | 11 | 8 |

元数据：

```text
data/rfdetr_rc_column_all_non_legacy_test_v1/data.yaml
data/rfdetr_rc_column_all_non_legacy_test_v1/split_summary.json
data/rfdetr_rc_column_all_non_legacy_test_v1/preflight.json
```

### RC壁

```text
data/rfdetr_rc_wall_all_non_legacy_test_v1
```

策略：

```text
train = 当前 RC壁 全量图片 - data_split.json 中 RC壁 official test stems
valid = data_split.json 中 RC壁 official test
test  = data_split.json 中 RC壁 official test
```

数据量：

| split | images | B boxes | C boxes | D boxes |
|---|---:|---:|---:|---:|
| train | 1151 | 1024 | 233 | 191 |
| valid | 31 | 14 | 10 | 8 |
| test | 31 | 14 | 10 | 8 |

元数据：

```text
data/rfdetr_rc_wall_all_non_legacy_test_v1/data.yaml
data/rfdetr_rc_wall_all_non_legacy_test_v1/split_summary.json
data/rfdetr_rc_wall_all_non_legacy_test_v1/preflight.json
```

## 自动识别模型数据

自动识别模型使用与上周 YOLO 自动识别模型调优相同的数据：

```text
data/rfdetr_router_base_aug_v2
```

该目录是 RF-DETR 兼容 view，指向原始 coarse router 数据。

数据量：

| split | images | 天井 boxes | 壁类 boxes | RC柱 boxes |
|---|---:|---:|---:|---:|
| train | 4052 | 2128 | 4172 | 1800 |
| valid | 351 | 188 | 399 | 85 |
| test | 348 | 183 | 391 | 102 |

元数据：

```text
data/rfdetr_router_base_aug_v2/data.yaml
outputs/rfdetr_router/base_aug_v2_dataset_preflight.json
```

## 恢复方法

下周恢复时，不需要保存今天完整数据目录。需要做的是：

1. 解压 `final_download_20260526.tar.zst`。
2. 放回 `data_split.json` 到 repo 根目录。
3. 重建 `data/yolo9_legacy_split_eval`，用于 official test view。
4. 重建 RF-DETR 下游数据集 view：

```text
data/rfdetr_rc_column_all_non_legacy_test_v1
data/rfdetr_rc_wall_all_non_legacy_test_v1
```

今天这些目录本质上是根据当前数据和 `data_split.json` 生成的兼容数据集目录，不需要长期单独保存完整图片副本。

## 需要恢复的生成逻辑

如果下周要完全复现今天的数据集构建逻辑，应遵循以下规则：

### official test stems

从 `data_split.json` 中读取对应部位的 test 图片 stem。

映射关系：

| JSON key | 下游数据目录 |
|---|---|
| `ceiling` | `tenjo` |
| `interior` | `inner_wall` |
| `rc_wall` | `rc_wall` |
| `rc_column` | `rc_column` |

### train

从当前 20260519 split 的该部位所有图片中，排除 official test stems。

### valid/test

使用 official test stems 对应的图片和标签。

### 标签格式

保持 YOLO txt 标签格式：

```text
class_id x_center y_center width height
```

class 顺序保持：

```text
0 = B
1 = C
2 = D
```

不同部位只改变类别名称，不改变 class id。

## 可用于核对的目标统计

恢复后必须核对统计是否与今天一致。

RC柱：

```text
train images = 605
valid images = 31
test images = 31
train boxes B/C/D = 317/186/145
test boxes B/C/D = 12/11/8
```

RC壁：

```text
train images = 1151
valid images = 31
test images = 31
train boxes B/C/D = 1024/233/191
test boxes B/C/D = 14/10/8
```

自动识别模型：

```text
train images = 4052
valid images = 351
test images = 348
test boxes 天井/壁类/RC柱 = 183/391/102
```

## 训练命令参考

RC柱：

```bash
python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_rc_column_baseline.yaml \
  --experiment medium \
  --dataset-dir data/rfdetr_rc_column_all_non_legacy_test_v1 \
  --output-dir outputs/rfdetr_single_crack/rc_column_medium_all_non_legacy_test_v1 \
  --epochs 80 \
  --batch-size 28 \
  --device cuda:0 \
  --trainer-precision 16-mixed
```

RC壁：

```bash
python scripts/train_rfdetr_router.py \
  --config configs/rfdetr_rc_wall_baseline.yaml \
  --experiment medium \
  --dataset-dir data/rfdetr_rc_wall_all_non_legacy_test_v1 \
  --output-dir outputs/rfdetr_single_crack/rc_wall_medium_all_non_legacy_test_v1 \
  --epochs 80 \
  --batch-size 28 \
  --device cuda:0 \
  --trainer-precision 16-mixed
```

自动识别模型：

```bash
python scripts/train_rfdetr_router.py \
  --experiment medium \
  --output-dir outputs/rfdetr_router/medium_base_aug_v2_fp16_noepochtest
```

## 注意事项

- 不要把 official test 放入 train。
- 下游模型评估时要固定使用 `data_split.json` 对齐的 official test。
- RF-DETR 自带 best checkpoint 可能按 mAP 选择，不一定符合业务的 recall-first 策略。
- 下游模型候选需要重新 force-evaluate 到 official test，再综合 recall、precision、F1、mAP 和可视化选择。
