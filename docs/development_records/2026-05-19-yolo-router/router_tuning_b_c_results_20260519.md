# Router B/C/D 调优实验结果

日期：2026-05-19

## 实验设置

目标：改善 router 中 `RC柱` 识别弱的问题。

基线模型：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_e50/weights/best.pt
```

并行实验：

| 实验 | GPU | 方法 | 权重 |
|---|---:|---|---|
| B | 0 | 从基线 `best.pt` fine-tune，开启 `--image-weights` | `coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_ft_imgw_e50/weights/best.pt` |
| C | 1 | 从基线 `best.pt` fine-tune，训练集对 `RC柱` oversample | `coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_ft_rc_os_e50/weights/best.pt` |

共同设置：

```text
epochs=50
batch=32
imgsz=640
lr0=0.003
data val/test 均使用原 cleaned split
```

C 的 oversample 数据集：

```text
coarse_router_yolov9/datasets/coarse_router_3class_cleaned_rc_column_oversample
```

训练集 `RC柱` 框从 492 增加到 1096；val/test 不变。

## Router val 对比

| 模型 | all P | all R | all mAP50 | 天井 P/R | 壁类 P/R | RC柱 P/R | RC柱 mAP50 |
|---|---:|---:|---:|---|---|---|---:|
| baseline cleaned | 0.697 | 0.683 | 0.723 | 0.847 / 0.763 | 0.661 / 0.745 | 0.584 / 0.543 | 0.548 |
| B image_weights | 0.792 | 0.699 | 0.776 | 0.862 / 0.793 | 0.772 / 0.719 | 0.740 / 0.586 | 0.676 |
| C RC oversample | 0.753 | 0.718 | 0.774 | 0.834 / 0.818 | 0.747 / 0.734 | 0.678 / 0.601 | 0.665 |

## Router test 对比

| 模型 | all P | all R | all mAP50 | 天井 P/R | 壁类 P/R | RC柱 P/R | RC柱 mAP50 |
|---|---:|---:|---:|---|---|---|---:|
| baseline cleaned | 0.627 | 0.653 | 0.694 | 0.757 / 0.760 | 0.583 / 0.726 | 0.542 / 0.474 | 0.547 |
| B image_weights | 0.680 | 0.708 | 0.745 | 0.758 / 0.813 | 0.663 / 0.749 | 0.618 / 0.561 | 0.623 |
| C RC oversample | 0.641 | 0.684 | 0.733 | 0.748 / 0.813 | 0.661 / 0.695 | 0.514 / 0.544 | 0.609 |

结论：

- B 在 val/test 上整体更稳。
- C 在 val 的 `RC柱` recall 略高，但 test 上 `RC柱` precision 明显低于 B。
- 两者都明显优于 baseline 的 `RC柱`。

## 80 张 E2E 抽样对比

抽样：

```text
每类 20 张，共 80 张
```

基线输出：

```text
outputs/e2e_debug_sample_80_20260519
```

B 输出：

```text
outputs/e2e_debug_sample_80_router_imgw_20260519
```

C 输出：

```text
outputs/e2e_debug_sample_80_router_rc_os_20260519
```

### Router hit

| 类别 | baseline | B image_weights | C RC oversample |
|---|---:|---:|---:|
| 天井 | 18/20 | 18/20 | 19/20 |
| 内壁 | 20/20 | 20/20 | 19/20 |
| RC壁 | 19/20 | 20/20 | 20/20 |
| RC柱 | 13/20 | 14/20 | 14/20 |

### 主分支匹配

| 类别 | baseline | B image_weights | C RC oversample |
|---|---:|---:|---:|
| 天井 | 23/25 | 23/25 | 24/25 |
| 内壁 | 15/22 | 17/22 | 13/22 |
| RC壁 | 16/23 | 16/23 | 16/23 |
| RC柱 | 7/22 | 8/22 | 10/22 |

### 次分支预测

| 类别 | baseline | B image_weights | C RC oversample |
|---|---:|---:|---:|
| 天井 | 9 | 9 | 7 |
| 内壁 | 14 | 13 | 16 |
| RC壁 | 14 | 12 | 13 |
| RC柱 | 25 | 25 | 33 |

E2E 结论：

- B 对 `内壁` 和 `RC壁` 更稳，次分支输出略少。
- C 对 `RC柱` 主分支匹配更好，但 `RC柱` 次分支预测从 25 增加到 33，说明误路由/多路由噪声也更明显。
- 如果当前优先目标是整体 router 稳定和减少次分支混乱，B 更适合作为默认候选。
- 如果当前优先目标是尽可能提升 `RC柱` E2E recall，C 值得继续专项分析，但需要配合阈值/后处理控制次分支噪声。

## 当前建议

短期默认候选：

```text
B image_weights
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_ft_imgw_e50/weights/best.pt
```

理由：

- val/test 的 `RC柱` P/R 均比 baseline 有明显提升。
- test 上比 C 更稳。
- E2E 中 `RC柱` 有小幅改善，同时没有明显增加 wall 噪声。

后续可继续：

1. 对 B 模型做 router confidence threshold sweep。
2. 对 `RC柱` 做专项可视化，确认仍 miss 的样本是否是标注框/构图问题。
3. 等 `data_add100` Gemini 粗标注完成后，将新增 router 标注并入训练集，再跑一轮完整 router 训练。

## D 组合实验

用户提出是否可以结合 B 和 C。为避免 C 强 oversample 带来的误判噪声，新增两个组合实验：

| 实验 | 方法 | RC柱 train boxes |
|---|---|---:|
| D800 | `--image-weights` + RC柱 mild oversample | 800 |
| D900 | `--image-weights` + RC柱 mild oversample | 900 |

训练集构建脚本：

```text
coarse_router_yolov9/scripts/build_router_3class_rc_column_oversample_target.py
```

并行训练脚本：

```text
coarse_router_yolov9/scripts/train_router_3class_tuning_d_parallel.sh
```

权重：

```text
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_ft_imgw_rc_os800_e50/weights/best.pt
coarse_router_yolov9/runs/train/gelan_c_router_3class_cleaned_ft_imgw_rc_os900_e50/weights/best.pt
```

### Router val 对比：加入 D

| 模型 | all P | all R | all mAP50 | 天井 P/R | 壁类 P/R | RC柱 P/R | RC柱 mAP50 |
|---|---:|---:|---:|---|---|---|---:|
| baseline cleaned | 0.697 | 0.683 | 0.723 | 0.847 / 0.763 | 0.661 / 0.745 | 0.584 / 0.543 | 0.548 |
| B image_weights | 0.792 | 0.699 | 0.776 | 0.862 / 0.793 | 0.772 / 0.719 | 0.740 / 0.586 | 0.676 |
| C RC oversample | 0.753 | 0.718 | 0.774 | 0.834 / 0.818 | 0.747 / 0.734 | 0.678 / 0.601 | 0.665 |
| D800 image_weights + os800 | 0.744 | 0.710 | 0.779 | 0.842 / 0.790 | 0.702 / 0.741 | 0.688 / 0.599 | 0.685 |
| D900 image_weights + os900 | 0.733 | 0.720 | 0.761 | 0.840 / 0.816 | 0.712 / 0.712 | 0.648 / 0.630 | 0.650 |

### Router test 对比：加入 D

| 模型 | all P | all R | all mAP50 | 天井 P/R | 壁类 P/R | RC柱 P/R | RC柱 mAP50 |
|---|---:|---:|---:|---|---|---|---:|
| baseline cleaned | 0.627 | 0.653 | 0.694 | 0.757 / 0.760 | 0.583 / 0.726 | 0.542 / 0.474 | 0.547 |
| B image_weights | 0.680 | 0.708 | 0.745 | 0.758 / 0.813 | 0.663 / 0.749 | 0.618 / 0.561 | 0.623 |
| C RC oversample | 0.641 | 0.684 | 0.733 | 0.748 / 0.813 | 0.661 / 0.695 | 0.514 / 0.544 | 0.609 |
| D800 image_weights + os800 | 0.711 | 0.670 | 0.731 | 0.775 / 0.740 | 0.710 / 0.691 | 0.648 / 0.579 | 0.620 |
| D900 image_weights + os900 | 0.753 | 0.673 | 0.761 | 0.832 / 0.748 | 0.759 / 0.676 | 0.666 / 0.596 | 0.664 |

### E2E 80 张抽样：加入 D

Router hit：

| 类别 | baseline | B | C | D800 | D900 |
|---|---:|---:|---:|---:|---:|
| 天井 | 18/20 | 18/20 | 19/20 | 19/20 | 19/20 |
| 内壁 | 20/20 | 20/20 | 19/20 | 20/20 | 20/20 |
| RC壁 | 19/20 | 20/20 | 20/20 | 20/20 | 19/20 |
| RC柱 | 13/20 | 14/20 | 14/20 | 16/20 | 16/20 |

主分支匹配：

| 类别 | baseline | B | C | D800 | D900 |
|---|---:|---:|---:|---:|---:|
| 天井 | 23/25 | 23/25 | 24/25 | 21/25 | 23/25 |
| 内壁 | 15/22 | 17/22 | 13/22 | 14/22 | 15/22 |
| RC壁 | 16/23 | 16/23 | 16/23 | 16/23 | 15/23 |
| RC柱 | 7/22 | 8/22 | 10/22 | 9/22 | 10/22 |

次分支预测：

| 类别 | baseline | B | C | D800 | D900 |
|---|---:|---:|---:|---:|---:|
| 天井 | 9 | 9 | 7 | 7 | 9 |
| 内壁 | 14 | 13 | 16 | 17 | 16 |
| RC壁 | 14 | 12 | 13 | 15 | 12 |
| RC柱 | 25 | 25 | 33 | 27 | 28 |

## D 实验结论

- D900 是当前标准 router test 指标最好的模型：
  - all mAP50：`0.761`
  - RC柱 P/R：`0.666 / 0.596`
  - RC柱 mAP50：`0.664`
- D900 在 E2E 中将 `RC柱 router_hit` 从 baseline `13/20` 提到 `16/20`，主分支匹配从 `7/22` 提到 `10/22`。
- D900 的 `RC柱` 次分支预测为 `28`，低于 C 的 `33`，但仍高于 B/baseline 的 `25`。
- D800 的 router test 不如 D900，且 E2E 对天井、内壁有回退，不建议作为默认。

当前候选优先级：

1. **D900**：如果接受少量次分支噪声上升，优先使用。它是当前标准 test 和 RC柱 E2E 综合最强。
2. **B image_weights**：如果优先保守和稳定，仍是安全候选。
3. **C / D800**：暂不建议作为默认。

推荐下一步：

- 用 D900 做默认 router 候选。
- 对 D900 做 router threshold sweep，重点看 `RC柱` 的 `secondary_predictions` 能否压回接近 B。
- 等 `data_add100` Gemini 粗标注并入后，再训练一版“扩容数据 + D900 策略”的最终 router。
