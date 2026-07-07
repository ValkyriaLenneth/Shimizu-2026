# 2026-07-07 Pipeline API / Backend 交接记录

## 范围

本记录补充 2026-07-07 当天后续开发成果，覆盖以下内容：

- 5 类 router 基于人工审核数据的再训练和最终推荐 checkpoint。
- `ブレース` 表现优化结果。
- 最终展示层合并规则调整：类内取最重、类间不合并。
- 面向生产部署的 FastAPI 后端骨架。
- 中文 API 文档。
- 新增两类 `ブレース` / `柱脚` 下游模型未就位时的部署占位方案。

## 1. Router5 最终推荐模型

最终推荐 checkpoint：

```text
outputs/rfdetr_router/medium_5class_20260707_brace_crop4_ctx4_ft_crop_e009_lr1e5/selected_precision_p090_classwise_epoch004_brace_balanced_v2.pth
```

对应 manifest：

```text
outputs/rfdetr_router/medium_5class_20260707_brace_crop4_ctx4_ft_crop_e009_lr1e5/selected_precision_p090_classwise_epoch004_brace_balanced_v2_manifest.json
```

显式 class-wise threshold sweep：

```text
outputs/rfdetr_router/threshold_classwise_brace_crop4_ctx4_e004_fine.csv
```

推荐阈值：

| 类别 | threshold |
|---|---:|
| 天井 | 0.90 |
| 壁类 | 0.66 |
| RC柱 | 0.76 |
| ブレース | 0.34 |
| 柱脚 | 0.52 |

整体指标：

| metric | value |
|---|---:|
| precision | 0.9003 |
| recall | 0.7327 |
| f1 | 0.8079 |
| min class recall | 0.7187 |

各类结果：

| 类别 | precision | recall | f1 |
|---|---:|---:|---:|
| 天井 | 0.9441 | 0.7377 | 0.8282 |
| 壁类 | 0.8836 | 0.7187 | 0.7927 |
| RC柱 | 0.9036 | 0.7353 | 0.8108 |
| ブレース | 0.8000 | 0.7442 | 0.7711 |
| 柱脚 | 1.0000 | 0.8485 | 0.9180 |

相对上一版均衡候选，`ブレース` recall 从 `0.6977` 提升到 `0.7442`，最低类 recall 从 `0.6977` 提升到 `0.7187`，同时整体 precision 保持在 `0.90` 以上。

## 2. 最终展示层合并逻辑

用户会议结论：

```text
类内取最重、类间不合并
```

已修改：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/pipeline/display_merge.py
```

规则：

- 同一大类内部：如果框高度重叠，允许合并，保留最严重等级。
- 不同大类之间：即使 IOU/IOA 超过阈值，也不能互相 suppression。
- 典型场景：`RC柱` 和 `壁类` 真实重叠时，最终展示必须保留两条结果。

相关测试：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_display_merge.py
```

验证结果：

```text
python3 -m pytest systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_display_merge.py -q
4 passed

python3 -m pytest systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_wall_display_rule.py systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_ambiguity_display.py -q
7 passed
```

## 3. Pipeline API 后端

新增 FastAPI 后端入口：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/api/main.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/api/service.py
```

服务对象：

```text
rfdetr_prod_pipeline.api.main:app
```

接口：

```text
GET  /api/v1/health
GET  /api/v1/pipeline/info
POST /api/v1/pipeline/predict
POST /api/v1/analyze_auto
```

`/api/v1/analyze_auto` 对齐上一期 prod repo 的迁移方向：前端不再手动传 `type/types`，而是上传图片，由 router 自动识别部材并调用对应下游模型。

## 4. 5 类假设部署配置

新增配置：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml
```

配置要点：

- 使用当前推荐 5 类 router checkpoint。
- `天井` / `壁类` / `RC柱` 接已有下游 BCD 模型。
- `ブレース` / `柱脚` 因为下游 BCD 数据和模型未就位，暂时配置为 `noop`。
- API 会在 `model_readiness_warnings` 中提示：
  - `downstream_model_pending:ブレース`
  - `downstream_model_pending:柱脚`

当两类下游模型训练完成后，仅需替换 YAML 中：

```yaml
brace:
  backend: noop
column_base:
  backend: noop
```

为真实 RF-DETR checkpoint 配置即可，API shape 不需要变。

## 5. 中文 API 文档

中文 API 文档：

```text
docs/development_records/2026-07-07-prod-pipeline-api-doc.md
```

内容包括：

- 服务启动方式。
- 环境变量。
- `/api/v1/health`
- `/api/v1/pipeline/info`
- `/api/v1/pipeline/predict`
- `/api/v1/analyze_auto`
- 请求字段和响应字段。
- 前端展示字段建议。
- 新类模型未就位时的响应约定。
- 最终展示层合并规则。
- 后续替换真实新类模型的方式。

## 6. 依赖变更

`requirements.txt` 新增：

```text
fastapi
uvicorn[standard]
python-multipart
```

## 7. 后端验证

新增测试：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_api_backend.py
```

验证结果：

```text
python3 -m pytest systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_api_backend.py -q
3 passed
```

相关展示层测试：

```text
python3 -m pytest systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_display_merge.py systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_wall_display_rule.py systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_ambiguity_display.py -q
11 passed
```

## 8. 启动命令

从 repo 根目录：

```bash
export PYTHONPATH="$PWD"
export SHIMIZU_PIPELINE_CONFIG="rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml"
export SHIMIZU_DEVICE="cuda:0"
uvicorn rfdetr_prod_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

CPU/mock smoke mode：

```bash
export PYTHONPATH="$PWD"
export SHIMIZU_MOCK_CRACK=1
export SHIMIZU_DEVICE="cpu"
uvicorn rfdetr_prod_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

## 9. 新完整交接包

新的完整交接包在原 RF-DETR 主包基础上追加今天的部署成果：

```text
.local_artifacts/shimizu_20260707_rfdetr_main_pipeline_api_handoff.tar.zst
```

包内新增目录约定：

```text
handoff_20260707_rfdetr_main/docs/
handoff_20260707_rfdetr_main/source/rfdetr_prod_pipeline/
handoff_20260707_rfdetr_main/models/rfdetr/router_5class/
handoff_20260707_rfdetr_main/eval/router5_threshold_sweeps/
```

其中 `source/rfdetr_prod_pipeline/` 包含 API 后端、配置、pipeline 合并逻辑和测试文件，可用于把当前成果迁移到生产 repo。
