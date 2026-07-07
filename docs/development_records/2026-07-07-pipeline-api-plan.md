# Pipeline API Plan

Date: 2026-07-07

## Scope

Plan a production-facing HTTP API for the Shimizu crack-detection pipeline, using the previous CLI / JSON contract as the compatibility baseline and the current RF-DETR pipeline output as the internal result model.

GitHub repository check:

- Target: `Generative-AI-Tokyo/Shimizu-VLM-Crack-Detection-Prod`
- Result: access failed with `Repository not found / Authentication failed`.
- Impact: this plan is based on the local previous-phase API contract and local RF-DETR pipeline code. Re-check the target repository once a token with repository access is available.

## Baseline From Previous API Contract

Previous contract:

- CLI entrypoint: `python -m rfdetr_prod_pipeline.pipeline.run_full_pipeline`
- Inputs: `--config`, `--source`, `--output-dir`, `--device`, `--mock-crack`, `--limit`, `--skip-visualization`
- Output directory:
  - `results.jsonl`
  - `summary.json`
  - `visualizations/`
- Per-image result core fields:
  - `image`
  - `image_shape`
  - `pipeline_version`
  - `router`
  - `crack_detections`
  - `warnings`
  - `elapsed_ms`

Current RF-DETR pipeline adds useful fields that should be preserved:

- `raw_crack_detections`
- `display_crack_detections`
- `suppressed_display_crack_detections`
- `wall_candidate_display`
- `ambiguity_candidate_groups`

## API Principles

1. Keep the previous per-image JSON shape as the canonical response body.
2. Add API wrapper metadata around the existing result instead of rewriting the result schema.
3. Support both synchronous single-image inference and asynchronous batch jobs.
4. Separate machine result (`crack_detections`) from UI/product display result (`display_crack_detections`).
5. Make model/config version explicit in every response.
6. Treat visualizations as optional artifacts, not mandatory response payload.

## Versioning

Base path:

```text
/api/v1
```

Pipeline version should be returned independently:

```json
{
  "api_version": "v1",
  "pipeline_version": "router5_rfdetr_crack_v1"
}
```

Recommended router classes for the current 5-class router:

```json
{
  "0": "天井",
  "1": "壁类",
  "2": "RC柱",
  "3": "ブレース",
  "4": "柱脚"
}
```

## Endpoints

### `GET /api/v1/health`

Purpose: lightweight service readiness.

Response:

```json
{
  "status": "ok",
  "service": "shimizu-crack-pipeline",
  "api_version": "v1"
}
```

### `GET /api/v1/pipeline/info`

Purpose: expose active model/config metadata without running inference.

Response:

```json
{
  "api_version": "v1",
  "pipeline_version": "router5_rfdetr_crack_v1",
  "router": {
    "backend": "rfdetr",
    "classes": {
      "0": "天井",
      "1": "壁类",
      "2": "RC柱",
      "3": "ブレース",
      "4": "柱脚"
    },
    "threshold_mode": "classwise"
  },
  "outputs": {
    "supports_visualization": true,
    "supports_raw_detections": true,
    "supports_display_detections": true
  }
}
```

### `POST /api/v1/pipeline/predict`

Purpose: synchronous single-image inference.

Request: `multipart/form-data`

| Field | Required | Description |
|---|---:|---|
| `image` | yes | Input image file |
| `request_id` | no | Client-side idempotency / trace id |
| `config_profile` | no | Named config profile, default `production` |
| `include_visualization` | no | Boolean, default `false` |
| `include_raw` | no | Boolean, default `false`; include `raw_crack_detections` |
| `include_debug` | no | Boolean, default `false`; include suppressed / ambiguity / wall candidate debug |

Response:

```json
{
  "request_id": "client-optional-id",
  "status": "succeeded",
  "api_version": "v1",
  "result": {
    "image": "uploaded://image.jpg",
    "image_shape": [1185, 1585, 3],
    "pipeline_version": "router5_rfdetr_crack_v1",
    "router": {
      "router_model": "router5_rfdetr",
      "classes": {
        "0": "天井",
        "1": "壁类",
        "2": "RC柱",
        "3": "ブレース",
        "4": "柱脚"
      },
      "detections": [
        {
          "bbox_xyxy": [192.0, 20.0, 1582.0, 816.0],
          "confidence": 0.91,
          "class_id": 1,
          "class_name": "壁类",
          "area_ratio": 0.59
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
    "elapsed_ms": 1234.5
  },
  "artifacts": {
    "visualization_url": null
  }
}
```

### `POST /api/v1/pipeline/jobs`

Purpose: asynchronous batch inference for multiple images or archive uploads.

Request: `multipart/form-data`

| Field | Required | Description |
|---|---:|---|
| `files[]` | yes | Multiple image files, or one `.zip` |
| `request_id` | no | Client-side trace id |
| `config_profile` | no | Named config profile |
| `include_visualizations` | no | Boolean |
| `include_raw` | no | Boolean |
| `include_debug` | no | Boolean |

Response:

```json
{
  "job_id": "job_20260707_000001",
  "status": "queued",
  "api_version": "v1",
  "submitted_images": 42
}
```

### `GET /api/v1/pipeline/jobs/{job_id}`

Purpose: job status and summary.

Response:

```json
{
  "job_id": "job_20260707_000001",
  "status": "running",
  "api_version": "v1",
  "summary": {
    "images": 42,
    "completed": 17,
    "failed": 0,
    "crack_detections": 23,
    "warning_counts": {}
  },
  "links": {
    "results": "/api/v1/pipeline/jobs/job_20260707_000001/results",
    "artifacts": "/api/v1/pipeline/jobs/job_20260707_000001/artifacts"
  }
}
```

### `GET /api/v1/pipeline/jobs/{job_id}/results`

Purpose: retrieve batch results.

Response should mirror the previous `results.jsonl`, but wrapped in JSON:

```json
{
  "job_id": "job_20260707_000001",
  "status": "succeeded",
  "results": [
    {
      "image": "input/image001.jpg",
      "image_shape": [1185, 1585, 3],
      "pipeline_version": "router5_rfdetr_crack_v1",
      "router": {},
      "crack_detections": [],
      "display_crack_detections": [],
      "warnings": [],
      "elapsed_ms": 1234.5
    }
  ]
}
```

Optional streaming compatibility:

```text
GET /api/v1/pipeline/jobs/{job_id}/results.jsonl
```

This should return newline-delimited per-image result records exactly like the previous CLI output.

### `GET /api/v1/pipeline/jobs/{job_id}/artifacts`

Purpose: list generated visualizations and downloadable files.

Response:

```json
{
  "job_id": "job_20260707_000001",
  "artifacts": [
    {
      "type": "results_jsonl",
      "name": "results.jsonl",
      "url": "/api/v1/pipeline/jobs/job_20260707_000001/artifacts/results.jsonl"
    },
    {
      "type": "summary_json",
      "name": "summary.json",
      "url": "/api/v1/pipeline/jobs/job_20260707_000001/artifacts/summary.json"
    },
    {
      "type": "visualization",
      "name": "image001_pipeline.jpg",
      "url": "/api/v1/pipeline/jobs/job_20260707_000001/artifacts/visualizations/image001_pipeline.jpg"
    }
  ]
}
```

## Result Schema Notes

### Router Detection

```json
{
  "bbox_xyxy": [0.0, 0.0, 100.0, 100.0],
  "confidence": 0.9,
  "class_id": 3,
  "class_name": "ブレース",
  "area_ratio": 0.12
}
```

### Crack Detection

Use this for `crack_detections`, `raw_crack_detections`, and `display_crack_detections`.

```json
{
  "bbox_xyxy": [328.0, 279.0, 1476.0, 792.0],
  "confidence": 0.7468,
  "damage_grade": "耐震壁の損傷程度C",
  "source_model": "rc_wall",
  "source_router_class": "壁类",
  "coordinate_space": "original_image"
}
```

`raw_crack_detections` may include extra routing/debug fields:

```json
{
  "router_region_index": 0,
  "router_bbox_xyxy": [192.0, 20.0, 1582.0, 816.0],
  "router_confidence": 0.91,
  "router_class_name": "壁类",
  "detector_input_shape": [900, 1200, 3],
  "region_transport": "ndarray_slice"
}
```

## Error Contract

Use stable machine-readable error codes.

```json
{
  "request_id": "client-optional-id",
  "status": "failed",
  "api_version": "v1",
  "error": {
    "code": "image_unreadable",
    "message": "Input image could not be decoded.",
    "details": {}
  }
}
```

Recommended codes:

- `invalid_request`
- `unsupported_media_type`
- `image_unreadable`
- `pipeline_exception`
- `model_unavailable`
- `job_not_found`
- `artifact_not_found`
- `timeout`

## Compatibility Mapping

| Previous CLI Output | API v1 |
|---|---|
| `results.jsonl` | `GET /jobs/{job_id}/results.jsonl` |
| `summary.json` | `GET /jobs/{job_id}` |
| `visualizations/` | `GET /jobs/{job_id}/artifacts` |
| per-image JSON line | `result` in `POST /predict`, item in `/jobs/{job_id}/results` |
| `--skip-visualization` | `include_visualization=false` |
| `--mock-crack` | non-production `config_profile=mock` only |
| `--limit` | server-side batch option, admin/debug only |

## Implementation Plan

1. Extract the current CLI `run_one_safe()` path into a service-level callable:
   - input: image bytes/path + request options
   - output: current per-image result dict
2. Add a FastAPI service layer:
   - `GET /health`
   - `GET /pipeline/info`
   - `POST /pipeline/predict`
   - async job endpoints
3. Keep CLI behavior intact by making CLI and API call the same service function.
4. Add response filtering:
   - default: no raw/debug fields
   - `include_raw=true`: include `raw_crack_detections`
   - `include_debug=true`: include suppressed, wall candidate, ambiguity fields
5. Add artifact storage layout compatible with previous output directory:

```text
outputs/api_jobs/<job_id>/
  inputs/
  results.jsonl
  summary.json
  visualizations/
  request.json
```

6. Add tests:
   - single-image mock request
   - batch mock request
   - JSONL compatibility check
   - error response for unreadable image
   - `include_raw` / `include_debug` filtering

## Open Items

- Confirm target GitHub repository layout once access works.
- Confirm whether production API should accept only file uploads or also signed image URLs.
- Confirm deployment queue backend: in-process worker, Celery/RQ, or cloud task queue.
- Confirm artifact retention and authentication requirements.
- Confirm final naming: `ブレース` in model output vs any product/UI label normalization.
