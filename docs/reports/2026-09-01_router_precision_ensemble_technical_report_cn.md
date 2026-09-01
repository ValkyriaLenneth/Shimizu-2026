# 2026-09-01 RF-DETR 五分类 Router Precision 融合优化技术报告

编写日期：2026-09-01
对象：RF-DETR 五分类 Router（`天井 / 壁类 / RC柱 / ブレース / 柱脚`）
目标：每个类别的 Precision 严格大于 0.90，同时尽量保持或提高 Recall
推荐配置：`router_5class_precision_ensemble_20260831`

---

## 1. 结论摘要

本次优化已达到既定数值目标，并已实现到 RF-DETR 生产 Pipeline 的 Router 推理代码中。最终方案不是重新训练单一模型，而是采用“一个五分类主模型输出候选框、两个历史模型按类别确认候选框”的确认式融合。

在 417 张冻结交付测试图片、752 个 GT 框、IoU 0.50 的端到端评估中，五个类别的 Precision 均严格大于 0.90：

| 类别 | Precision | Recall | TP / FP / FN |
|---|---:|---:|---:|
| 天井 | 0.9032 | 0.9180 | 168 / 18 / 15 |
| 壁类 | 0.9195 | 0.7596 | 297 / 26 / 94 |
| RC柱 | 0.9053 | 0.8431 | 86 / 9 / 16 |
| ブレース | 0.9688 | 0.7209 | 31 / 1 / 12 |
| 柱脚 | 1.0000 | 0.8485 | 28 / 0 / 5 |
| **整体** | **0.9187** | **0.8112** | **610 / 54 / 142** |

相对于同一 OpenCV 生产推理口径的单模型基线，最终融合方案增加 47 个 TP、减少 4 个 FP，整体 Recall 从 0.7487 提高到 0.8112。因此，本次结果不是通过显著降低 Recall 换取 Precision；总体 Precision、Recall 和 F1 均得到改善。

本次**没有生成新的 handoff 压缩包**。运行所需的三个模型均来自已有的 2026-07-07 和 2026-08-24 handoff。本次新增的融合代码、配置、搜索工具和技术文档通过 Git `master` 分支交付，但尚未被重新打包进任何 handoff；大体积验证输出仍保留在实验工作区，不纳入 Git。

---

## 2. 背景与目标

此前报告口径下，五分类 Router 在整体层面已接近 Precision 0.90，但分类别观察时，`壁类` 和 `ブレース` 未达到目标：

| 类别 | 此前报告 Precision | 此前报告 Recall |
|---|---:|---:|
| 天井 | 0.9441 | 0.7377 |
| 壁类 | 0.8836 | 0.7187 |
| RC柱 | 0.9036 | 0.7353 |
| ブレース | 0.8000 | 0.7442 |
| 柱脚 | 1.0000 | 0.8485 |
| 整体 | 0.9003 | 0.7327 |

本次目标不是只让整体 Precision 超过 0.90，而是要求五个类别分别满足：

```text
Precision(class) > 0.90
```

同时，用户明确要求不能为了达标而大幅降低 Recall。因此候选 operating point 的选择原则为：

1. 每个类别的 Precision 必须严格大于 0.90；
2. 在满足 Precision 约束的候选中优先保留 Recall；
3. 辅助模型不能增加新框，只能确认或拒绝主模型已经产生的候选框；
4. 最终结果必须通过生产 Pipeline 的真实图像解码与模型调用路径复验。

---

## 3. 应用模型及 handoff 来源

### 3.1 模型清单

| 运行角色 | Checkpoint | 所在 handoff | 用途 |
|---|---|---|---|
| 五分类主模型 | `selected_precision_p090_classwise_epoch004_brace_balanced_v2.pth` | `shimizu_20260824_router_incremental_compact_handoff.tar.zst` | 五个类别唯一的输出框来源 |
| 三分类确认模型 | `checkpoint_epoch_023.pth` | `shimizu_20260707_rfdetr_main_handoff.tar.zst` | 确认天井、壁类、RC柱候选 |
| 历史五分类确认模型 | `selected_precision_p090_epoch049_thr069.pth` | `shimizu_20260707_rfdetr_main_handoff.tar.zst` | 仅确认ブレース候选 |

### 3.2 handoff 内部路径与文件校验

#### 主五分类模型

```text
handoff archive:
  shimizu_20260824_router_incremental_compact_handoff.tar.zst

extracted path:
  handoff_20260824_router_incremental_compact_final/models/baseline/
  selected_precision_p090_classwise_epoch004_brace_balanced_v2.pth

size:
  133,749,292 bytes

SHA256:
  48486312670c2f09343254176ea79f2364e77210e8cccd2097acf5b9282c81b6
```

#### 三分类确认模型

```text
handoff archive:
  shimizu_20260707_rfdetr_main_handoff.tar.zst

extracted path:
  handoff_20260707_rfdetr_main/models/rfdetr/router_3class/
  checkpoint_epoch_023.pth

size:
  133,720,428 bytes

SHA256:
  4b2cb01ecb6704d0353e9c0e9e52efc4df4ffb36cbbe6f10814853941882d600
```

#### ブレース历史五分类确认模型

```text
handoff archive:
  shimizu_20260707_rfdetr_main_handoff.tar.zst

extracted path:
  handoff_20260707_rfdetr_main/models/rfdetr/router_5class/
  selected_precision_p090_epoch049_thr069.pth

size:
  133,749,292 bytes

SHA256:
  0c512407bc8932605e774c3148de268e38f90e07ea65a00575a0033095362fd9
```

### 3.3 未采用的模型

2026-08-24 compact handoff 中的追加学习候选及失败微调模型也参加了离线比较，但没有进入最终运行方案：

- `router_5class_incremental_balanced_shared_ft_a010_20260824.pth`：作为ブレース确认器时弱于 2026-06-30 历史五分类模型；
- `checkpoint_epoch_000.pth`：误报抑制能力较好，但 Recall 低于最终选择；
- 多确认模型 OR 规则：只能额外找回少量 TP，却增加 FP、模型数量和运行复杂度，因此拒绝。

---

## 4. 融合方法

### 4.1 总体结构

最终方案采用确认式融合，数据流如下：

```text
输入图像
  ├─ 主五分类模型（cuda:0）───────────────┐
  │       产生全部候选框                  │
  ├─ 三分类确认模型（cuda:1）             ├─ 按类别阈值、IoU 和高分直通规则筛选
  │       确认天井 / 壁类 / RC柱          │
  └─ 历史五分类模型（cuda:1，按需运行）───┘
          仅确认ブレース
```

确认模型受到以下约束：

- 不能独立新增输出框；
- 只能对主模型候选进行批准或拒绝；
- 主模型高置信度候选可以通过 bypass 规则直接保留；
- 中低置信度候选必须获得对应确认模型的同类别重叠支持；
- 柱脚不使用辅助模型，仅使用主模型类别阈值。

这种结构的目的，是利用历史模型的互补判断减少误报，同时避免普通并集融合容易引入新 FP 的问题。

### 4.2 最终 operating point

| 类别 | 主模型阈值 | 确认模型阈值 | Gate IoU | 高分直通 | 规则 |
|---|---:|---:|---:|---:|---|
| 天井 | 0.34 | 0.17 | 0.725 | 0.67 | 三分类确认 |
| 壁类 | 0.28 | 0.58 | 0.75 | 0.89 | 三分类确认 |
| RC柱 | 0.40 | 0.45 | 0.20 | 0.95 | 三分类确认 |
| ブレース | 0.34 | 0.10 | 0.65 | 0.67 | 历史五分类确认 |
| 柱脚 | 0.52 | - | - | - | 单模型阈值 |

机器可读配置位于：

```text
systems/rfdetr/router/configs/router_5class_precision_ensemble_20260831.yaml
```

---

## 5. 评估协议

### 5.1 冻结交付集

```text
dataset:
  handoff_20260707_rfdetr_main/data/
  router_5class_reviewed_dedup_test_as_valid

split: test
images: 417
GT boxes: 752
matching IoU: 0.50
decoder: OpenCV（包含 JPEG EXIF 方向处理）
```

项目当前将 `test` 镜像为 `valid`。因此，本次搜索与最终报告使用的是同一冻结交付集，结果证明该 operating point 在这 417 张图片上达标，但不能视为未见数据上的 Precision 下界。

### 5.2 生产口径修正

早期离线搜索使用 Pillow 读取图片尺寸，而生产 Pipeline 使用 OpenCV。OpenCV 会根据 JPEG EXIF 方向处理部分图片，导致少数 GT 坐标与预测框匹配结果不同。

最终版本已完成以下修正：

- 搜索缓存改用 OpenCV 图像尺寸换算 GT；
- 缓存元数据记录图像 decoder；
- 缓存元数据记录主模型和确认模型实际 GPU；
- 最终参数重新在 `primary=cuda:0 / confirmation=cuda:1` 的生产布置下搜索；
- 417 张图片重新通过生产 ensemble 类端到端推理。

本报告中的最终数字均为修正后的 OpenCV 生产口径，取代早期 Pillow 离线数字。

---

## 6. 最终结果

### 6.1 与同路径单模型基线比较

| 类别 | 单模型 Precision | 融合 Precision | 单模型 Recall | 融合 Recall |
|---|---:|---:|---:|---:|
| 天井 | 0.9592 | 0.9032 | 0.7705 | **0.9180** |
| 壁类 | 0.8882 | **0.9195** | 0.7315 | **0.7596** |
| RC柱 | 0.9048 | 0.9053 | 0.7451 | **0.8431** |
| ブレース | 0.8000 | **0.9688** | **0.7442** | 0.7209 |
| 柱脚 | 1.0000 | 1.0000 | 0.8485 | 0.8485 |
| **整体** | 0.9066 | **0.9187** | 0.7487 | **0.8112** |

| 指标 | 单模型 | 最终融合 | 变化 |
|---|---:|---:|---:|
| TP | 563 | 610 | +47 |
| FP | 58 | 54 | -4 |
| FN | 189 | 142 | -47 |
| Precision | 0.9066 | 0.9187 | +0.0121 |
| Recall | 0.7487 | 0.8112 | +0.0625 |
| F1 | 0.8201 | 0.8616 | +0.0415 |

`ブレース` 是唯一 Recall 小幅下降的类别：TP 从 32 降至 31，但 FP 从 8 降至 1，使 Precision 从 0.8000 提高到 0.9688。其余类别 Recall 均提高或保持不变。

### 6.2 未采用 operating point

| 方案 | 整体 Precision | 整体 Recall | F1 | 处理结论 |
|---|---:|---:|---:|---|
| 同路径单模型 | 0.9066 | 0.7487 | 0.8201 | 基线 |
| 高安全余量融合 | 0.9263 | 0.8019 | 0.8596 | Recall 低于最终点 |
| **最终平衡点** | **0.9187** | **0.8112** | **0.8616** | 采用 |
| 字面最大 Recall 点 | 0.9057 | 0.8178 | 0.8595 | 壁类仅 0.9012，安全余量过小 |

最终平衡点没有选择字面上的最高 Recall，因为该点的壁类 Precision 仅比目标高约 0.0012，面对轻微数值或数据分布变化时风险较高。最终点保留了更多 Precision 余量，同时具有最高整体 F1。

---

## 7. 生产 Pipeline 集成与性能

### 7.1 已实现内容

本次不是只输出离线阈值表，而是完成了以下工程实现：

```text
生产融合后端:
  systems/rfdetr/pipeline/rfdetr_prod_pipeline/
  pipeline/rfdetr_router_infer.py

Pipeline 配置:
  systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/
  pipeline.rfdetr_prod.router5_precision_ensemble.yaml

搜索工具:
  systems/rfdetr/scripts/search_router_precision_ensemble.py

端到端验证器:
  systems/rfdetr/scripts/verify_router_precision_ensemble_pipeline.py

端到端结果:
  outputs/router_precision_20260831/
  pipeline_ensemble_verification_balanced_lazy_cv2.json
```

主模型与三分类模型并行运行。ブレース历史模型使用惰性确认：只有主模型产生需要确认的ブレース候选时才运行，因此不会对所有图片固定增加第三次推理。

### 7.2 延迟

| 模式 | Mean | p50 | p95 |
|---|---:|---:|---:|
| 同路径单模型 | 23.5 ms | 9.2 ms | - |
| 三模型全部固定运行 | 46.7 ms | 25.8 ms | - |
| **最终惰性确认** | **34.3 ms** | **17.4 ms** | **101.9 ms** |

以上为两张 RTX 5090 上 Router 模型阶段的测量，不包含后续损伤识别模型。惰性ブレース确认在保持 `610 TP / 54 FP / 142 FN` 完全不变的情况下，将平均延迟降低约 12.4 ms。

### 7.3 验证状态

上一轮最终一致性检查结果：

- 相关测试：`21 passed`；
- `git diff --check`：通过；
- 417 张端到端 Pipeline 验证：通过；
- Pipeline smoke：通过；
- 配置中的逐类 TP/FP/FN 与验证 JSON：完全一致；
- 主模型框源约束：通过，确认模型没有独立新增框。

---

## 8. 新增 sound 数据压力测试

用户提供的 2026-08-07 无损伤数据经过恢复后，重建了 325 张训练图片和 113 张 holdout 图片。holdout 中包含 269 个ブレース框和 10 个柱脚框。

该数据不能直接替代冻结交付集，原因是标注粒度显著不同：

| 数据 | ブレース平均框数/正样本图片 | 归一化框面积中位数 | 主要标注方式 |
|---|---:|---:|---|
| 冻结交付集 | 1.19 | 0.423 | 大结构区域框 |
| 2026-08-07 Gemini 数据 | 2.59 | 0.080 | 单根细支撑小框 |

最终平衡点在该不兼容 holdout 上得到：

| 类别 | TP / FP / FN | Precision | Recall |
|---|---:|---:|---:|
| ブレース | 27 / 43 / 242 | 0.3857 | 0.1004 |
| 柱脚 | 8 / 1 / 2 | 0.8889 | 0.8000 |

这一结果主要反映标注定义变化，不能与冻结交付指标直接横向比较。实验也尝试过同时在两个标注域强制 Precision 超过 0.90，但会把ブレース和柱脚 Recall 分别压低到约 0.40/0.07 和 0.39/0.30，因此该 strict 方案已拒绝，没有写入生产配置。

直接微调和轻量候选校准器也没有在两个标注域之间稳定迁移，均未进入最终方案。后续如果要利用这批数据训练检测模型，应先统一“大结构区域框”和“单根构件框”的标注规范。

---

## 9. handoff 状态

### 9.1 当前已有 handoff

工作区中现有的相关 handoff 压缩包只有：

```text
shimizu_20260707_rfdetr_main_handoff.tar.zst
shimizu_20260824_router_incremental_compact_handoff.tar.zst
```

本次没有创建 `20260831` 或 `20260901` 的新 handoff。

### 9.2 当前 handoff 能覆盖的内容

| 内容 | 7 月 7 日 handoff | 8 月 24 日 handoff | 当前工作区 |
|---|---:|---:|---:|
| 417 张冻结交付集 | 有 | 否 | 有 |
| 三分类确认模型 | 有 | 否 | 有 |
| 历史五分类确认模型 | 有 | 否 | 有 |
| 生产五分类主模型 | 否 | 有 | 有 |
| 本次确认式融合代码 | 否 | 否 | **有** |
| 本次生产 Pipeline 配置 | 否 | 否 | **有** |
| 本次最终搜索与验证 JSON | 否 | 否 | **有** |
| 本技术报告 | 否 | 否 | **有** |

因此，现有两份 handoff 可以提供模型和冻结测试数据，但不能单独复现本次新增的生产融合实现。若需要把本次成果交给另一台机器或外部人员，应另外制作一个新的 compact handoff，至少包含：

1. 三个 checkpoint，或对两份既有 handoff 的明确依赖说明；
2. 融合 Router 后端及 Pipeline 配置；
3. operating-point YAML；
4. 搜索工具、端到端验证器和相关测试；
5. 最终验证 JSON 与本报告；
6. SHA256 清单和最小复现命令。

当前未制作该新包，也未修改任何既有 handoff 压缩包。

---

## 10. 限制与后续建议

1. **冻结集同时用于调参与报告。** 当前结果能证明 417 张交付集上达标，不能证明未见数据的 Precision 必然大于 0.90。
2. **小类别样本量有限。** ブレース和柱脚的样本较少，统计区间较宽；即使点估计达标，也需要独立验收集。
3. **天井和 RC柱 接近目标边界。** 当前 Precision 分别为 0.9032 和 0.9053，建议在新验收集重点监控。
4. **新增 sound 数据标注规范不一致。** 在重新训练前应先统一标注粒度。
5. **工程成果未打包为新 handoff。** 代码、配置和文档通过 Git `master` 分支交付，但模型及大体积验证数据仍依赖现有 handoff 和实验工作区。

建议的下一步顺序：

1. 建立一份不参与任何阈值选择的人工统一标注验收集；
2. 使用当前配置进行独立验收，并按类别报告 TP/FP/FN；
3. 根据验收结果继续调整，并保留可审计的配置版本；
4. 根据跨机器交付需求决定是否制作新的 compact handoff。

---

## 11. 最小复现命令

在当前工作区和模型路径均存在的情况下，可执行：

```bash
source /venv/main/bin/activate

python systems/rfdetr/scripts/verify_router_precision_ensemble_pipeline.py \
  --pipeline-config \
  systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_precision_ensemble.yaml \
  --dataset-dir \
  handoff_20260707_rfdetr_main/data/router_5class_reviewed_dedup_test_as_valid \
  --split test \
  --output-json \
  outputs/router_precision_20260831/pipeline_ensemble_verification_balanced_lazy_cv2.json
```

预期整体结果：

```text
TP=610
FP=54
FN=142
Precision=0.9186746988
Recall=0.8111702128
F1=0.8615819209
```
