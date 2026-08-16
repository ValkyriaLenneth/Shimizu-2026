# ブレース recall 冻结记录 2026-08-15

冻结对象:2026-08-04 测出「全体 recall 0.723」的那套推理配置,以及客户新要求的
B/C/D 分解结果。**本文档的每一个数字都可以用 `scripts/verify_freeze.py` 重新算出来**,
不要凭记忆引用。

前置文档:

| 文档 | 内容 |
|---|---|
| `docs/development_records/2026-08-04-label-noise-finding-and-sixteen-negative-results.md` | 0.723 的来源(§4 推理集成、§5 BRL) |
| `docs/handoff_records/2026-08-04/handoff_20260804.md` | 当日总结 |
| `systems/rfdetr/recognition_models/brace/configs/rfdetr_brace_baseline.yaml` | 类别映射、测试集规模、选择协议 |

## 0. 一行

0.723 是 **BRL + WBF 多模型融合 + 水平翻转 TTA** 在阈值 (B,C,D) = (0.30, 0.15, 0.12)
处的结果。**B 类只有 0.636,没达到 0.7**;把阈值改成 (0.12, 0.15, 0.18) 可以让四项
全部达标(B 0.727 / C 0.733 / D 0.750 / 全体 0.735),代价是 precision 从 0.408 降到 0.359。
**B 的 recall 天花板就是 0.727,没有余量。**

## 1. 模型现状:两个权重在,交付基线不在

盘点了本机全部 `*.pth` / `*.ckpt`,与 ブレース 相关的只有两个,都已复制进本包:

| 文件 | sha256 | 训练来源 |
|---|---|---|
| `checkpoints/brace_brl_ignore035_epoch_032.pth` | `d155b15f487cc2390de41ebec205a54e1102f800b4758b92596a39932763997c` | `outputs/rfdetr_new_classes/brl/brace_brl035` |
| `checkpoints/brace_cpsym33_epoch_058.pth` | `85f739ff53115d785ab33bbd9a6711590754c1f72e40de75a250129cb452e266` | `outputs/rfdetr_new_classes/cpsym/brace_cpsym33` |

**`brace_neg2x_epoch_067.pth`(交付基线)不在本机任何位置。** 它是
`br_tta2.csv`(单模型+TTA)和 `br_ens2*.csv`(2 模型融合)的成员,所以那两条
对照线目前无法复现。

两个权重的训练参数(从 ckpt 的 `args` 读出,非文档转述):

```text
model_name              RFDETRMedium          rfdetr_version   1.7.1
batch_size 28           grad_accum_steps 1    epochs 80
lr 1e-4                 lr_encoder 1.5e-4     weight_decay 1e-4
drop_path 0.0           ema_decay 0.993       use_ema True
ia_bce_loss True        group_detr 13         num_select 300
multi_scale True        expanded_scales True  square_resize_div_64 True
seed 20260602           分辨率 RFDETRMedium 默认(576)
BRL 训练数据集          data/rfdetr_brace_bcd_20260725_neg2x_test_as_valid
cpsym33 训练数据集      data/rfdetr_brace_bcd_20260725_cpsym33_test_as_valid
```

BRL 侧另加 `brl_patch.py --brl-threshold 0.35 --brl-mode ignore`(把高置信未匹配
query 的背景权重置零)。

### 1.1 两个必须先解决的不确定性

**(a) 融合成员未被记录。** `ensemble_wbf_eval.py` 只接受 `--output-csv`,不写
provenance,所以 `br_brl_ens_tta.csv` 用了哪几个 checkpoint 无法从产物反推。
最可能是 `{brace_brl_ignore035_*, brace_cpsym33_epoch_058}` —— 依据是 cpsym33
本身是负结果(copy-paste 对称 −0.072),交接包却单独打包了它,除了作为融合成员
没有别的理由。**这是推断,不是记录。**

**(b) BRL 的 epoch 存疑。** 08-04 文档 §5 的单模型 BRL 表(P>=0.55 → 0.590、
P>=0.50 → 0.639、P>=0.45 → 0.663、P>=0.40 → 0.675)逐位对应
`brace_brl035_ep038.csv`,而打包的权重是 **epoch_032**:

| ブレース BRL-ignore 0.35 | P>=0.55 | P>=0.50 | P>=0.45 | P>=0.40 |
|---|---:|---:|---:|---:|
| ep038(= 文档数字) | 0.590 | 0.639 | 0.663 | 0.675 |
| **ep032(= 包内权重)** | 不可达 | **0.530** | 0.602 | 0.651 |

ep032 在 P>=0.50 处 **低于交付基线 0.590**。融合是用 ep032 还是 ep038 未知,
因此 §2 的运行点在拿到缺失资产前**只是"已记录",不是"已复现"**。

同一类错误在 08-04 文档里已经出现过一次(交付基线写 `epoch_050`,数字却是
`epoch_067` 的),那一处至今未在仓库中修正。

## 2. 冻结的推理配置与运行点

三个点全部来自 `results/br_brl_ens_tta.csv`(3375 行 = 15³ 阈值网格)。

### 2.1 推理参数(`ensemble_wbf_eval.py` / `tta_wbf_eval.py` 的默认值)

```text
match IoU            0.229    ← 与交付四类同协议,换值即不可比
WBF 融合 IoU         0.55
per-model 检测下限    0.10
conf_type            avg
TTA views            orig,hflip
num_classes          3
阈值网格             0.05 0.07 0.10 0.12 0.15 0.18 0.20 0.22 0.25
                     0.30 0.32 0.35 0.40 0.45 0.50   (每类独立,共 15³)
```

### 2.2 类别映射与测试集(取自 `rfdetr_brace_baseline.yaml`)

```text
class 0 = ブレースの損傷程度B     22 框
class 1 = ブレースの損傷程度C     45 框
class 2 = ブレースの損傷程度D     16 框
test = 58 张图 / 83 框,valid 镜像 test,train/test 按 scene group 8:2 不交叉
```

分辨率上的量化步长必须随数字一起报:B 一个框 = 0.045,C = 0.022,D = 0.0625,
全体 = 0.012。

### 2.3 运行点

| 编号 | 阈值 (B,C,D) | B | C | D | 全体 R | P | 说明 |
|---|---|---:|---:|---:|---:|---:|---|
| **FP-1** | 0.30 / 0.15 / 0.12 | **0.636** (14/22) | 0.733 (33/45) | 0.812 (13/16) | **0.723** | 0.408 | 上次汇报的点。B 未达标 |
| **FP-2** | 0.12 / 0.15 / 0.18 | **0.727** (16/22) | 0.733 (33/45) | 0.750 (12/16) | **0.735** | 0.359 | **四项全部 ≥0.70,precision 最高的点** |
| **FP-3** | 0.10 / 0.15 / 0.07 | 0.727 (16/22) | 0.733 (33/45) | 0.875 (14/16) | 0.759 | 0.300 | 再让一档 precision 换 D 和全体 |
| REF | 0.40 / 0.20 / 0.30 | 0.545 (12/22) | 0.644 (29/45) | 0.562 (9/16) | 0.602 | 0.610 | 不带 BRL 的 2模型+TTA,对照 |

网格内满足「B/C/D/全体 四项均 ≥0.70」的点共 **140 个**,FP-2 是其中 precision 最高的。

## 3. B 是硬约束

在整个 15³ 网格里,把三类阈值全部压到 0.05,各等级 recall 的上限是:

| | 上限 recall | 命中 | 距目标 0.70 |
|---|---:|---|---|
| **B** | **0.727** | 16 / 22 | **+0.027,一个框的余量都没有** |
| C | 0.800 | 36 / 45 | +0.100 |
| D | 0.875 | 14 / 16 | +0.175 |
| 全体 | 0.795 | 66 / 83 | +0.095 |

**22 个 B 框里有 6 个在任何阈值组合下都检不出来。** 0.70 换算成 15.4/22,只能取
16/22 = 0.727;少命中一个就是 15/22 = 0.682,直接掉线。所以 FP-2 的「B 达标」是
踩线达成,不具备工程冗余——test 只有 58 张图,换一批测试数据就可能翻掉。

**BRL 是达标的必要条件**:不带 BRL 的 2模型+TTA(`br_ens2tta.csv`)B 的上限只有
0.682,整个网格找不到四项都 ≥0.70 的点。

## 4. 与客户目标的关系

* 去年定的 0.7 是**全体** recall,FP-1 的 0.723 达成,这一条不变。
* 新要求「B/C/D 四个都 ≥0.7」在现有模型上可以达成,但 precision 要降到 **0.359**
  ——大约每 3 个报警里 2 个是误报。交付的四类模型 precision 在 0.596–0.824,
  这是一个数量级上的差别,建议在汇报时明确写出、书面确认,而不是只说「优先 recall」。
* `rfdetr_brace_baseline.yaml` 里登记的官方目标是 `overall_recall 0.800` /
  `precision_floor 0.600`。**按那份配置,FP-1 和 FP-2 都不达标**(precision 分别
  0.408 / 0.359)。客户口头同意降 precision 后,该 config 的 `official_targets`
  需要同步更新,否则下一个人会按 0.600 的下限重新筛选,得到完全不同的结论。

## 5. 复现本冻结所缺的资产

选择逻辑现在就能验(`python3 scripts/verify_freeze.py`,无需 GPU/数据集)。
但**重新打分**需要以下四项,都不在本机:

1. `data/rfdetr_brace_bcd_20260725_test_as_valid`(冻结 test split,58 图 / 83 框)
2. `brace_neg2x_epoch_067.pth`(交付基线,REF 与 `br_tta2` / `br_ens2` 的成员)
3. `brace_brl035` 训练目录下的 **epoch_038** 权重(用于消除 §1.1(b) 的歧义)
4. `data/frozen/new_classes_20260725.lock.json` 对应的原始数据,用于
   `freeze_new_class_datasets.py --check`

拿到后的验证顺序:

```bash
# 1. 数据完整性
python3 systems/rfdetr/scripts/freeze_new_class_datasets.py --check

# 2. 复现 FP-1 / FP-2(先用 ep032,再用 ep038,看哪一个对上 br_brl_ens_tta.csv)
python3 scripts/ensemble_wbf_eval.py \
  --checkpoint checkpoints/brace_brl_ignore035_epoch_032.pth \
  --checkpoint checkpoints/brace_cpsym33_epoch_058.pth \
  --dataset-dir data/rfdetr_brace_bcd_20260725_test_as_valid \
  --split test --tta-hflip \
  --iou-threshold 0.229 --wbf-iou 0.55 --floor 0.10 --conf-type avg \
  --output-csv repro_br_brl_ens_tta.csv

# 3. 与冻结网格逐行比对
python3 - <<'PY'
import csv
a={r['thresholds']:r for r in csv.DictReader(open('results/br_brl_ens_tta.csv'))}
b={r['thresholds']:r for r in csv.DictReader(open('repro_br_brl_ens_tta.csv'))}
bad=[k for k in a if k in b and a[k]['recall']!=b[k]['recall']]
print(f"{len(a)} rows frozen, {len(bad)} mismatched")
PY
```

第 2 步必须两个 epoch 都跑。如果都对不上,说明融合成员的推断(§1.1(a))是错的,
需要向原主机索取运行命令历史。

## 6. 未解决

1. **§1.1 的两个不确定性**,在缺失资产到位前无法关闭。
2. **B 的 6 个不可检框**要单独看图,判断是标注问题还是尺度/形态问题。08-04 的
   全数审计显示 ブレース 的漏标率只有 5%,所以这 6 个更可能是真·难例;
   如果确认是「构面整体变形」类型,那就与 Part-aware Sampling 的假设一致。
3. **仓库文档里 `brace_neg2x_epoch_050.pth` 的笔误未修**
   (`docs/development_records/2026-07-26-new-classes-final-state.md:21`、
   `2026-07-26-new-classes-negatives-results.md:107`),数字实为 epoch_067 的。
