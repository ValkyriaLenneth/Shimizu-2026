# Shimizu Pipeline 生产 API 文档

日期：2026-07-07

## 1. 文档目的

本文档定义当前 Shimizu 裂缝/损伤检测 pipeline 的可部署 HTTP API。

设计原则参考上一期生产 repo 的接口方向：

- 保留上一期手动指定类型接口 `/api/v1/analyze` 的思路，不强行破坏旧系统。
- 新增自动路由接口 `/api/v1/analyze_auto`，由部材识别模型自动判断调用哪些下游模型。
- 当前 repo 内部仍保留原来的 pipeline JSON 结果结构，方便继续复用 CLI、JSONL、summary 和可视化流程。

当前已实现的 FastAPI 服务入口：

```text
rfdetr_prod_pipeline.api.main:app
```

当前已实现接口：

```text
GET  /api/v1/health
GET  /api/v1/pipeline/info
POST /api/v1/pipeline/predict
POST /api/v1/analyze_auto
```

## 2. 当前模型假设

当前 pipeline 预先按 5 类部材识别设计：

| class id | 部材类别 | 下游 BCD/等级判定模型状态 |
|---:|---|---|
| 0 | 天井 | 已有模型 |
| 1 | 壁类 | 已有模型，当前通过 `inner_wall` + `rc_wall` 并行处理 |
| 2 | RC柱 | 已有模型 |
| 3 | ブレース | 数据未就位，暂时没有下游 BCD 模型 |
| 4 | 柱脚 | 数据未就位，暂时没有下游 BCD 模型 |

因此当前部署配置采用如下策略：

- `天井` / `壁类` / `RC柱`：正常调用已有下游损伤等级模型。
- `ブレース` / `柱脚`：router 会返回部材检测结果，但下游 BCD 模型暂时配置为 `noop`。
- 如果图中识别到 `ブレース` 或 `柱脚`，API 会在 `model_readiness_warnings` 中提示对应下游模型尚未就位。

当前假设版部署配置文件：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml
```

后续当 `ブレース` / `柱脚` 的下游模型训练完成后，只需要替换配置文件中的 `brace` / `column_base` 模型配置，API 请求和响应结构不需要改变。

## 3. 启动服务

从仓库根目录启动：

```bash
export PYTHONPATH="$PWD"
export SHIMIZU_PIPELINE_CONFIG="rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml"
export SHIMIZU_DEVICE="cuda:0"
uvicorn rfdetr_prod_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

如果只想做 CPU/mock smoke test：

```bash
export PYTHONPATH="$PWD"
export SHIMIZU_MOCK_CRACK=1
export SHIMIZU_DEVICE="cpu"
uvicorn rfdetr_prod_pipeline.api.main:app --host 0.0.0.0 --port 8000
```

环境变量说明：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `SHIMIZU_PIPELINE_CONFIG` | `rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml` | pipeline 配置文件 |
| `SHIMIZU_DEVICE` | `cpu` | 推理设备，例如 `cpu`、`cuda:0` |
| `SHIMIZU_MOCK_CRACK` | 空 | 设为 `1` / `true` 时使用 mock 下游模型 |
| `SHIMIZU_REPO_ROOT` | 当前工作目录 | repo 根目录，通常不需要设置 |

## 4. 接口：健康检查

```http
GET /api/v1/health
```

也兼容：

```http
GET /health
```

### 响应示例

```json
{
  "status": "ok",
  "service": "shimizu-rfdetr-pipeline",
  "api_version": "v1"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `status` | string | 服务状态 |
| `service` | string | 服务名称 |
| `api_version` | string | API 版本 |

## 5. 接口：pipeline 信息

```http
GET /api/v1/pipeline/info
```

用于确认当前服务加载的是哪个配置、哪个 router、有哪些类别、哪些下游模型已就绪。

### 响应示例

```json
{
  "api_version": "v1",
  "service": "shimizu-rfdetr-pipeline",
  "pipeline_version": "router5_rfdetr_crack_v1_assumed",
  "config": ".../pipeline.rfdetr_prod.router5_assumed.yaml",
  "device": "cuda:0",
  "mock_crack": false,
  "router": {
    "backend": "rfdetr",
    "checkpoint": ".../selected_precision_p090_classwise_epoch004_brace_balanced_v2.pth",
    "classes": {
      "0": "天井",
      "1": "壁类",
      "2": "RC柱",
      "3": "ブレース",
      "4": "柱脚"
    }
  },
  "model_readiness": {
    "downstream": {
      "天井": "ready",
      "壁类": "ready_via_inner_wall_and_rc_wall_parallel",
      "RC柱": "ready",
      "ブレース": "pending_training",
      "柱脚": "pending_training"
    }
  },
  "detectors": {
    "天井": ["ceiling"],
    "壁类": ["inner_wall", "rc_wall"],
    "RC柱": ["rc_column"],
    "ブレース": ["noop_brace"],
    "柱脚": ["noop_column_base"]
  }
}
```

重点字段说明：

| 字段 | 说明 |
|---|---|
| `router.classes` | 当前部材识别模型支持的类别 |
| `model_readiness.downstream` | 各部材类别的下游等级判定模型状态 |
| `detectors` | 每个 router 类别实际会调用的下游 detector |
| `noop_brace` | 表示 `ブレース` 下游模型尚未训练，当前不会输出 B/C/D 结果 |
| `noop_column_base` | 表示 `柱脚` 下游模型尚未训练，当前不会输出 B/C/D 结果 |

## 6. 接口：单图 pipeline 推理

```http
POST /api/v1/pipeline/predict
Content-Type: multipart/form-data
```

这是当前推荐的新 pipeline 标准接口。它返回完整 pipeline 结果，适合后端、评估脚本和前端调试使用。

### 请求字段

| 字段 | 必填 | 类型 | 默认值 | 说明 |
|---|---:|---|---|---|
| `image` | 是 | file | - | 输入图片 |
| `request_id` | 否 | string | 自动生成 UUID | 调用方传入的追踪 ID |
| `include_visualization` | 否 | bool | `false` | 是否保存可视化图片 |
| `include_raw` | 否 | bool | `false` | 是否返回原始下游检测结果 |
| `include_debug` | 否 | bool | `false` | 是否返回 suppression、wall candidate、ambiguity 等调试字段 |

### 请求示例

```bash
curl -X POST http://localhost:8000/api/v1/pipeline/predict \
  -F image=@sample.jpg \
  -F request_id=demo-001 \
  -F include_visualization=true
```

### 响应示例

```json
{
  "request_id": "demo-001",
  "status": "succeeded",
  "api_version": "v1",
  "result": {
    "image": "/tmp/shimizu_api_xxx.jpg",
    "image_shape": [1185, 1585, 3],
    "pipeline_version": "router3_crack_v2_class_safe",
    "router": {
      "router_model": "...",
      "classes": {
        "0": "天井",
        "1": "壁类",
        "2": "RC柱",
        "3": "ブレース",
        "4": "柱脚"
      },
      "detections": [
        {
          "bbox_xyxy": [100.0, 80.0, 1200.0, 900.0],
          "confidence": 0.91,
          "class_id": 1,
          "class_name": "壁类",
          "area_ratio": 0.54
        }
      ],
      "route_decision": {
        "status": "ok",
        "strategy": "keep_all_router_boxes",
        "primary_class": "壁类"
      }
    },
    "crack_detections": [],
    "display_crack_detections": [],
    "warnings": [],
    "model_readiness_warnings": [],
    "elapsed_ms": 1234.5
  },
  "artifacts": {
    "visualization_path": ".../outputs/rfdetr_prod_pipeline_api/api_requests/demo-001/sample_pipeline.jpg"
  }
}
```

### 顶层响应字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `request_id` | string | 本次请求 ID |
| `status` | string | `succeeded` 或 `failed` |
| `api_version` | string | API 版本 |
| `result` | object | pipeline 单图结果 |
| `artifacts` | object | 可视化等产物路径 |

### `result` 字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `image` | string | 服务端临时图片路径 |
| `image_shape` | array | `[height, width, channels]` |
| `pipeline_version` | string | pipeline 内部版本 |
| `router` | object | 部材识别结果 |
| `crack_detections` | array | 合并后的机器检测结果 |
| `display_crack_detections` | array | 推荐给前端展示的最终结果 |
| `warnings` | array | pipeline 运行警告 |
| `model_readiness_warnings` | array | 模型未就位相关警告 |
| `elapsed_ms` | number | 单图处理耗时，单位毫秒 |

## 7. 接口：上一期 prod 兼容自动分析

```http
POST /api/v1/analyze_auto
Content-Type: multipart/form-data
```

这个接口面向上一期 prod API 的迁移。上一期生产接口的核心是 `/api/v1/analyze`，由前端传入图片和 `type/types`，后端按指定类型调用模型。

新的 `/api/v1/analyze_auto` 不要求前端指定类型，而是：

```text
输入图片
  -> 部材识别 router 自动识别天井 / 壁类 / RC柱 / ブレース / 柱脚
  -> 根据 router 结果调用对应下游模型
  -> 汇总最终展示结果
```

### 请求字段

| 字段 | 必填 | 类型 | 默认值 | 说明 |
|---|---:|---|---|---|
| `image` | 是 | file | - | 输入图片 |
| `request_id` | 否 | string | 自动生成 UUID | 调用方追踪 ID |
| `include_visualization` | 否 | bool | `true` | 是否保存可视化图 |

### 请求示例

```bash
curl -X POST http://localhost:8000/api/v1/analyze_auto \
  -F image=@sample.jpg \
  -F request_id=demo-auto-001
```

### 响应示例

```json
{
  "success": true,
  "request_id": "demo-auto-001",
  "api_version": "v1",
  "pipeline_version": "router3_crack_v2_class_safe",
  "router": {
    "detections": []
  },
  "detections": [],
  "raw_result": {},
  "artifacts": {
    "visualization_path": "..."
  }
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `success` | bool | 是否成功 |
| `request_id` | string | 请求 ID |
| `pipeline_version` | string | pipeline 内部版本 |
| `router` | object | 部材识别结果 |
| `detections` | array | 前端展示用检测结果，来自 `display_crack_detections` |
| `raw_result` | object | 完整 pipeline 结果 |
| `artifacts` | object | 可视化产物 |

## 8. 部材识别结果格式

`router.detections` 中每个元素格式：

```json
{
  "bbox_xyxy": [100.0, 80.0, 1200.0, 900.0],
  "confidence": 0.91,
  "class_id": 1,
  "class_name": "壁类",
  "area_ratio": 0.54
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `bbox_xyxy` | array | 原图坐标 `[x1, y1, x2, y2]` |
| `confidence` | number | router 置信度 |
| `class_id` | int | 类别 ID |
| `class_name` | string | 类别名 |
| `area_ratio` | number | bbox 面积占整图比例 |

## 9. 损伤检测结果格式

`crack_detections` / `display_crack_detections` 中每个元素格式：

```json
{
  "bbox_xyxy": [328.0, 279.0, 1476.0, 792.0],
  "confidence": 0.7468,
  "damage_grade": "壁-C",
  "source_model": "rc_wall",
  "source_router_class": "壁类",
  "coordinate_space": "original_image"
}
```

字段说明：

| 字段 | 类型 | 说明 |
|---|---|---|
| `bbox_xyxy` | array | 原图坐标 `[x1, y1, x2, y2]` |
| `confidence` | number | 下游模型置信度 |
| `damage_grade` | string | B/C/D 等级或显示标签 |
| `source_model` | string | 产生该结果的下游模型 |
| `source_router_class` | string | 来源部材类别 |
| `coordinate_space` | string | 当前固定为 `original_image` |

## 10. 推荐前端展示字段

生产前端建议优先使用：

```text
result.display_crack_detections
```

原因：

- 这是经过最终展示层规则处理后的结果。
- 已应用“类内取最重、类间不合并”的规则。
- 更接近最终用户看到的画面。

调试或审核时可以同时展示：

```text
result.router.detections
result.model_readiness_warnings
```

特别是当前 `ブレース` / `柱脚` 下游模型未就位，所以这两类可能只有 router 框，没有 B/C/D 损伤等级框。

## 11. 调试字段

默认情况下，`/api/v1/pipeline/predict` 不返回过多调试字段。

如果请求中设置：

```text
include_raw=true
include_debug=true
```

则可能返回：

| 字段 | 说明 |
|---|---|
| `raw_crack_detections` | 下游模型原始检测结果 |
| `suppressed_display_crack_detections` | 最终展示层被 suppression 的结果 |
| `wall_candidate_display` | 壁类候选展示逻辑中间结果 |
| `ambiguity_candidate_groups` | 跨类 ambiguity 候选组 |

这些字段主要用于开发、评估和错误分析，不建议直接作为普通用户界面的主结果。

## 12. 最终展示层合并规则

当前最终展示层规则按会议结论实现：

```text
类内取最重、类间不合并
```

具体含义：

### 12.1 同一大类内部

例如都是 `壁类`，或者都是 `RC柱`：

- 如果多个框重叠度较高，可以合并。
- 展示时保留最严重等级。
- 例如同一个 RC柱 区域同时出现 B 和 D，最终展示 D。

### 12.2 不同大类之间

例如 `RC柱` vs `壁类`：

- 即使 IOU/IOA 很高，也不能互相 suppression。
- 两条结果都保留。
- 这样可以避免真实场景中前景柱子和背景墙重叠时，其中一个类别被吞掉。

## 13. 模型未就位时的响应约定

如果 router 识别到 `ブレース` 或 `柱脚`，但下游模型尚未就位，响应中会出现：

```json
{
  "model_readiness_warnings": [
    "downstream_model_pending:ブレース"
  ]
}
```

或者：

```json
{
  "model_readiness_warnings": [
    "downstream_model_pending:柱脚"
  ]
}
```

前端建议：

- 可以展示 router 框。
- 不要展示假的 B/C/D 等级。
- 可以在审核模式中提示“该部材类别的等级判定模型尚未接入”。

## 14. 与上一期 prod API 的关系

上一期 prod repo 的主要接口是：

```text
POST /api/v1/analyze
```

上一期逻辑：

```text
前端传 image + type/types
  -> 后端按 type/types 调用指定裂缝模型
```

当前新增逻辑：

```text
前端只传 image
  -> router 自动识别部材区域
  -> 后端自动决定调用哪些下游模型
  -> 输出最终展示结果
```

推荐迁移方式：

1. 旧 `/api/v1/analyze` 暂时保留。
2. 新增 `/api/v1/analyze_auto` 给前端灰度接入。
3. 前端优先展示 `detections` 或 `raw_result.display_crack_detections`。
4. 开发/审核页面展示 `raw_result.router.detections`。
5. 等 5 类 pipeline 稳定后，再考虑将 `/api/v1/analyze_auto` 升级为默认入口。

## 15. 后续替换真实新类模型

当 `ブレース` / `柱脚` 的下游 BCD 数据和模型就位后，替换配置即可。

当前占位配置：

```yaml
brace:
  backend: noop
  note: ブレース BCD model is not trained yet.

column_base:
  backend: noop
  note: 柱脚 BCD model is not trained yet.
```

未来替换为：

```yaml
brace:
  backend: rfdetr
  checkpoint: path/to/brace_checkpoint.pth
  thresholds: [0.25, 0.25, 0.25]

column_base:
  backend: rfdetr
  checkpoint: path/to/column_base_checkpoint.pth
  thresholds: [0.25, 0.25, 0.25]
```

API 层不需要修改。

## 16. 错误处理建议

当前服务如果图片为空，会返回 HTTP 400。

建议生产化时统一错误格式：

```json
{
  "status": "failed",
  "api_version": "v1",
  "error": {
    "code": "image_unreadable",
    "message": "图片无法读取或解码",
    "details": {}
  }
}
```

建议错误码：

| code | 说明 |
|---|---|
| `invalid_request` | 请求字段缺失或格式错误 |
| `unsupported_media_type` | 不支持的文件类型 |
| `image_unreadable` | 图片无法读取 |
| `pipeline_exception` | pipeline 内部异常 |
| `model_unavailable` | 模型文件不存在或加载失败 |
| `timeout` | 推理超时 |

## 17. 当前实现文件

API 后端：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/api/main.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/api/service.py
```

5 类假设部署配置：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml
```

相关测试：

```text
systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_api_backend.py
systems/rfdetr/pipeline/rfdetr_prod_pipeline/tests/test_display_merge.py
```
