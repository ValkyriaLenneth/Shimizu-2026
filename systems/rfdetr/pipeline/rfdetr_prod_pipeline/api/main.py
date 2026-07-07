"""FastAPI entrypoint for the RF-DETR production pipeline."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .service import PipelineService, PredictOptions


app = FastAPI(title="Shimizu RF-DETR Crack Detection Pipeline", version="1.0.0")


@lru_cache(maxsize=1)
def get_service() -> PipelineService:
    return PipelineService()


@app.get("/health")
@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "shimizu-rfdetr-pipeline", "api_version": "v1"}


@app.get("/api/v1/pipeline/info")
def pipeline_info() -> dict[str, object]:
    return get_service().info()


@app.post("/api/v1/pipeline/predict")
async def predict(
    image: Annotated[UploadFile, File()],
    request_id: Annotated[str | None, Form()] = None,
    include_visualization: Annotated[bool, Form()] = False,
    include_raw: Annotated[bool, Form()] = False,
    include_debug: Annotated[bool, Form()] = False,
) -> dict[str, object]:
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail={"code": "invalid_request", "message": "image is empty"})
    return get_service().predict_bytes(
        data,
        image.filename or "upload.jpg",
        PredictOptions(
            request_id=request_id,
            include_visualization=include_visualization,
            include_raw=include_raw,
            include_debug=include_debug,
        ),
    )


@app.post("/api/v1/analyze_auto")
async def analyze_auto(
    image: Annotated[UploadFile, File()],
    request_id: Annotated[str | None, Form()] = None,
    include_visualization: Annotated[bool, Form()] = True,
) -> dict[str, object]:
    response = await predict(
        image=image,
        request_id=request_id,
        include_visualization=include_visualization,
        include_raw=True,
        include_debug=True,
    )
    result = response["result"]
    return {
        "success": response["status"] == "succeeded",
        "request_id": response["request_id"],
        "api_version": "v1",
        "pipeline_version": result.get("pipeline_version") if isinstance(result, dict) else None,
        "router": result.get("router") if isinstance(result, dict) else None,
        "detections": result.get("display_crack_detections", []) if isinstance(result, dict) else [],
        "raw_result": result,
        "artifacts": response["artifacts"],
    }
