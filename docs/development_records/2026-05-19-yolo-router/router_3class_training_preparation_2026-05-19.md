# 三类路由模型训练准备

日期：2026-05-19

## 1. 目标

基于 Gemini 3.1 清洗后的三类路由数据，准备两组并行训练：

- `full`：最大化使用全部合法 Gemini 框，作为 baseline。
- `cleaned`：排除关键错误和低分样本，作为对照模型。

两张 GPU 并行使用：

```text
GPU 0 -> coarse_router_3class_full
GPU 1 -> coarse_router_3class_cleaned
```

## 2. 数据集

```text
coarse_router_yolov9/datasets/coarse_router_3class_full
coarse_router_yolov9/datasets/coarse_router_3class_cleaned
```

类别定义：

```yaml
names:
  0: 天井
  1: 壁类
  2: RC柱
```

数据校验结果：

```text
full:
  images: 3013
  boxes: 5790
  train/val/test: 2411 / 302 / 300

cleaned:
  images: 2533
  boxes: 4758
  train/val/test: 2026 / 254 / 253
```

说明：训练前发现原始数据中存在可恢复但不规范的 JPEG，以及少量截断风险。数据构建脚本已改为使用 OpenCV 完整解码并重编码训练图片；增强后的 preflight 会检查图片可解码和 JPEG EOI 标记。清洗后 `cleaned` 样本数从早期的 2710 调整为 2533。

## 3. 新增脚本

预检脚本：

```text
coarse_router_yolov9/scripts/check_router_3class_training_ready.py
```

双卡并行训练脚本：

```text
coarse_router_yolov9/scripts/train_router_3class_parallel.sh
```

## 4. 默认训练参数

沿用上一轮粗筛路由模型设置：

```text
model: YOLOv9 GELAN-C
cfg: coarse_router_yolov9/yolov9/models/detect/gelan-c.yaml
hyp: coarse_router_yolov9/yolov9/data/hyps/hyp.scratch-high.yaml
epochs: 50
imgsz: 640
batch-size: 32 per process
workers: 8
close-mosaic: 10
optimizer: SGD, YOLOv9 默认
weights: 空字符串，默认从 cfg 冷启动
```

如果后续提供 GELAN-C 预训练权重，可以通过环境变量覆盖：

```bash
WEIGHTS=/path/to/gelan-c.pt coarse_router_yolov9/scripts/train_router_3class_parallel.sh
```

## 5. 环境要求

当前机器 GPU 状态已确认：

```text
GPU 0: NVIDIA GeForce RTX 4090, 24GB
GPU 1: NVIDIA GeForce RTX 4090, 24GB
```

当前已通过 `coarse_router_yolov9/scripts/setup_yolov9_env.sh` 准备 YOLO 环境，Torch 能识别两张 RTX 4090。重新准备环境可执行：

```bash
cd coarse_router_yolov9
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision
python -m pip install -r yolov9/requirements.txt
cd ..
```

如果目标机器 CUDA/PyTorch 版本需要精确匹配，应按 PyTorch 官方 CUDA wheel 选择安装命令。

## 6. 预检命令

环境装好后执行：

```bash
python3 coarse_router_yolov9/scripts/check_router_3class_training_ready.py
```

当前没有 torch 时可只验证数据：

```bash
python3 coarse_router_yolov9/scripts/check_router_3class_training_ready.py --skip-torch
```

## 7. 并行训练命令

```bash
coarse_router_yolov9/scripts/train_router_3class_parallel.sh
```

脚本内部使用物理 GPU 编号：

```text
python yolov9/train.py ... full ... --device 0
python yolov9/train.py ... cleaned ... --device 1
```

早期尝试过 `CUDA_VISIBLE_DEVICES=0/1`，但实际会让两个进程都落到物理 GPU0，因此已改为直接传物理 `--device 0/1`。

## 8. 可覆盖参数

```bash
EPOCHS=80 BATCH_SIZE=24 IMGSZ=640 WORKERS=8 EXTRA_ARGS="--cos-lr" coarse_router_yolov9/scripts/train_router_3class_parallel.sh
```

常用覆盖项：

```text
EPOCHS
BATCH_SIZE
IMGSZ
WORKERS
WEIGHTS
PROJECT
FULL_NAME
CLEANED_NAME
EXTRA_ARGS
```

## 9. 输出

默认输出：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_full_e50
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50
coarse_router_yolov9/runs/train_parallel_logs/router_3class_<timestamp>/
```

日志文件：

```text
full.log
cleaned.log
launcher.log
summary.txt
nvidia_smi_before.txt
nvidia_smi_after.txt
```

## 10. 训练后比较重点

对比 full 与 cleaned：

- precision
- recall
- mAP@0.5
- mAP@0.5:0.95
- 三类 confusion matrix
- `RC柱` 与 `壁类` 的互相误判
- `天井` recall
- bbox 是否偏小或偏大

第一轮建议以 recall 和混淆矩阵为主，不只看 mAP。路由模型的业务目标是减少后续裂缝模型调用错误。

## 11. 本轮训练结果

训练已完成，最终 epoch 49 指标如下：

| 数据集 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| full | 0.71981 | 0.59097 | 0.69483 | 0.54370 |
| cleaned | 0.75379 | 0.58129 | 0.69721 | 0.52449 |

初步结论：

- `cleaned` 的 precision 和 mAP@0.5 略高，作为默认 router 候选更稳。
- `full` 的 mAP@0.5:0.95 略高，可作为对照模型保留。
- 最终默认权重建议先使用：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50/weights/best.pt
```

## 12. 端到端集成目录

系统集成开发已单独放入：

```text
router_crack_pipeline/
```

该目录包含 router 推理封装、内存 slice 区域传递、下游 detector registry、判别框合并、端到端 runner 和 smoke test。
