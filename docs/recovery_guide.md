# 开发现场恢复指南

本文用于在新机器或重新 clone 后恢复当前开发现场，覆盖数据集划分、环境准备、训练产物恢复和常用命令。

## 1. Git 代码恢复

```bash
git clone https://github.com/ValkyriaLenneth/Shimizu-2026.git
cd Shimizu-2026
```

Git 仓库只保存代码、配置、脚本和汇报文档。以下内容不进 GitHub，需要从归档包恢复：

- 原始与处理后数据：`data/zip`、`data/unzip`、`data/shimizu-split`
- Gemini 输出：`outputs`
- YOLO 粗筛数据和训练产物：`coarse_router_yolov9/datasets`、`runs`、`qa`、`weights`

## 2. 数据集划分说明

### 原始分类数据

原始数据约定在：

```text
data/unzip
```

类别映射：

```text
a.天井  -> 天井
b.内壁  -> 内壁
c.RC壁  -> RC壁
d.RC柱  -> RC柱
```

分类任务的固定划分曾记录在 `RESULTS_SUMMARY.md`：

```text
Raw valid images: 3015
Excluded cross-class duplicate files: 20
Final fixed dataset size: 2995
Train: 2094
Val: 451
Test: 450
```

测试集类别分布：

```text
RC壁: 146
RC柱: 65
内壁: 128
天井: 111
```

### 旧 YOLO 数据

旧 YOLO 数据在：

```text
data/shimizu-split
```

它按构件类型分成 `ceiling`、`inner_wall`、`rc_wall`、`rc_column`，每个子数据集的标签是损伤等级 B/C/D，不是构件类型粗筛标签。因此它用于各构件专用损伤模型，不直接用于粗筛路由模型。

### Gemini 粗标注数据

Gemini 输出在：

```text
outputs/gemini_balanced_300x4_3_1_pro
outputs/gemini_additional_200_each_no_overlap_3_1_pro
outputs/gemini_wall_label_fixed_3_1_pro
```

关键结果：

```text
初始平衡批次: 1200 张，四类各 300 张
修正合并后唯一图像: 1935 张
预期来源类别: 天井 500, 内壁 500, RC壁 500, RC柱 435
```

可视化入口：

```text
outputs/gemini_coarse_3_1_pro_50x4/index.html
outputs/gemini_coarse_3_1_pro_50x4/contact_sheet.jpg
```

### YOLO 粗筛数据

粗筛数据集在：

```text
coarse_router_yolov9/datasets/coarse_cross_fixed
```

类别：

```text
0: 天井
1: 内壁
2: RC壁
3: RC柱
```

划分与规模：

```text
source images: 1935
train images: 1548
val images: 194
test images: 193
total boxes: 3950
box counts: 天井 1198, 内壁 1311, RC壁 822, RC柱 619
```

构建命令：

```bash
python coarse_router_yolov9/scripts/build_coarse_yolo_dataset.py \
  --input outputs/gemini_wall_label_fixed_3_1_pro/results.jsonl \
  --output-dir coarse_router_yolov9/datasets/coarse_cross_fixed
```

标注检查页生成命令：

```bash
python coarse_router_yolov9/scripts/visualize_yolo_dataset.py \
  --dataset coarse_router_yolov9/datasets/coarse_cross_fixed \
  --output coarse_router_yolov9/qa/coarse_cross_fixed_labels
```

## 3. 环境准备

### 分类/Gemini 脚本环境

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如需调用 Gemini，需要设置：

```bash
export GEMINI_API_KEY=...
```

### YOLOv9 环境

```bash
cd coarse_router_yolov9
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
python -m pip install -r yolov9/requirements.txt
cd ..
```

当前验证环境曾使用 NVIDIA RTX 4090。若 CUDA/PyTorch 版本不同，优先按目标机器 CUDA 版本安装匹配的 `torch`/`torchvision`。

## 4. 训练结果恢复

将归档包解压到仓库根目录后，应恢复以下路径：

```text
data/
outputs/
coarse_router_yolov9/datasets/
coarse_router_yolov9/runs/
coarse_router_yolov9/qa/
coarse_router_yolov9/weights/
```

恢复后检查：

```bash
test -f coarse_router_yolov9/datasets/coarse_cross_fixed/data.yaml
test -f coarse_router_yolov9/runs/train/gelan_c_cross_fixed_e50/results.csv
test -f coarse_router_yolov9/runs/train/gelan_c_cross_fixed_e50/weights/best.pt
test -f coarse_router_yolov9/qa/model_review_conf025/index.html
```

YOLO 粗筛已完成的关键指标：

```text
model: YOLOv9 GELAN-C
epochs: 50
best validation epoch by mAP@0.5: 48
precision: 0.724
recall: 0.636
mAP@0.5: 0.712
mAP@0.5:0.95: 0.580
```

结果查看入口：

```text
coarse_router_yolov9/qa/model_review_conf025/index.html
coarse_router_yolov9/qa/model_review_conf025/val.html
coarse_router_yolov9/qa/model_review_conf025/test.html
coarse_router_yolov9/runs/val/gelan_c_cross_fixed_e50_best_test/confusion_matrix.png
coarse_router_yolov9/runs/val/gelan_c_cross_fixed_e50_best_test/PR_curve.png
coarse_router_yolov9/runs/val/gelan_c_cross_fixed_e50_best_test/F1_curve.png
```

## 5. 常用命令

重新运行 Gemini 可视化：

```bash
python scripts/visualize_gemini_coarse.py \
  --results outputs/gemini_coarse_3_1_pro_50x4/results.jsonl \
  --out-dir outputs/gemini_coarse_3_1_pro_50x4
```

重新生成 YOLO 预测 review 页面：

```bash
python coarse_router_yolov9/scripts/make_prediction_review.py \
  --dataset coarse_router_yolov9/datasets/coarse_cross_fixed_copy \
  --pred-root coarse_router_yolov9/runs/detect \
  --output coarse_router_yolov9/qa/model_review_conf025
```

查看汇报文档：

```text
docs/client_report_yolo_coarse_router.md
```

## 6. 归档包建议

本次现场建议保存两个归档：

```text
artifacts/shimizu_2026_datasets_and_results_YYYYMMDD.tar.gz
artifacts/shimizu_2026_code_snapshot_YYYYMMDD.tar.gz
```

第一个用于恢复数据、训练结果、权重、预测 review。第二个是代码现场快照，便于在 GitHub 不可用时恢复脚本和文档。
