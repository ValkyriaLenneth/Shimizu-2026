"""Service layer shared by the FastAPI app and future deployment adapters."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from rfdetr_prod_pipeline.pipeline.crack_detector_registry import build_detector_registry
from rfdetr_prod_pipeline.pipeline.run_full_pipeline import (
    build_router,
    load_config,
    resolve_path,
    run_one_safe,
    save_visualization,
)


DEFAULT_CONFIG = "rfdetr_prod_pipeline/configs/pipeline.rfdetr_prod.router5_assumed.yaml"


@dataclass(frozen=True)
class PredictOptions:
    request_id: str | None = None
    include_visualization: bool = False
    include_raw: bool = False
    include_debug: bool = False


class PipelineService:
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        device: str | None = None,
        mock_crack: bool = False,
        repo_root: str | Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root or os.getenv("SHIMIZU_REPO_ROOT") or Path.cwd()).resolve()
        config_value = config_path or os.getenv("SHIMIZU_PIPELINE_CONFIG") or DEFAULT_CONFIG
        self.config_path = resolve_path(config_value, self.repo_root)
        self.config = load_config(self.config_path)
        self.device = device or os.getenv("SHIMIZU_DEVICE") or "cpu"
        self.mock_crack = mock_crack or os.getenv("SHIMIZU_MOCK_CRACK", "").lower() in {"1", "true", "yes"}
        self.router = build_router(self.config["pipeline"], self.config, self.config_path.parent, self.device)
        detector_cfg = {**self.config, "device": self.device}
        self.registry = build_detector_registry(detector_cfg, self.config_path.parent, mock=self.mock_crack)
        self.output_root = resolve_path(
            self.config.get("outputs", {}).get("root", "outputs/rfdetr_prod_pipeline_api"),
            self.config_path.parent,
        )

    def info(self) -> dict[str, Any]:
        pipeline_cfg = self.config.get("pipeline", {})
        model_readiness = self.config.get("model_readiness", {})
        router_models = pipeline_cfg.get("router_ensemble_checkpoint_overrides")
        if not router_models:
            checkpoint = pipeline_cfg.get("router_checkpoint")
            router_models = {"primary": checkpoint} if checkpoint else {}
        return {
            "api_version": "v1",
            "service": "shimizu-rfdetr-pipeline",
            "pipeline_version": "router5_rfdetr_crack_v1_assumed",
            "config": str(self.config_path),
            "device": self.device,
            "mock_crack": self.mock_crack,
            "router": {
                "backend": pipeline_cfg.get("router_backend"),
                "checkpoint": (
                    str(resolve_path(pipeline_cfg["router_checkpoint"], self.config_path.parent))
                    if pipeline_cfg.get("router_checkpoint")
                    else None
                ),
                "models": {
                    str(name): str(resolve_path(value, self.config_path.parent))
                    for name, value in router_models.items()
                },
                "classes": self.config.get("classes", {}).get("router", {}),
            },
            "model_readiness": model_readiness,
            "detectors": {key: [getattr(detector, "name", "unknown") for detector in value] for key, value in self.registry.items()},
        }

    def predict_file(self, image_path: Path, options: PredictOptions | None = None) -> dict[str, Any]:
        options = options or PredictOptions()
        request_id = options.request_id or uuid4().hex
        request_dir = self.output_root / "api_requests" / request_id
        request_dir.mkdir(parents=True, exist_ok=True)
        result = run_one_safe(image_path, self.router, self.registry, self.config)
        artifacts: dict[str, Any] = {"visualization_path": None}
        if options.include_visualization and not result.get("error"):
            vis_path = request_dir / f"{image_path.stem}_pipeline.jpg"
            save_visualization(image_path, result, vis_path)
            artifacts["visualization_path"] = str(vis_path)
        (request_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return {
            "request_id": request_id,
            "status": "failed" if result.get("error") else "succeeded",
            "api_version": "v1",
            "result": filter_result(result, include_raw=options.include_raw, include_debug=options.include_debug),
            "artifacts": artifacts,
        }

    def predict_bytes(self, data: bytes, filename: str, options: PredictOptions | None = None) -> dict[str, Any]:
        suffix = Path(filename).suffix or ".jpg"
        with tempfile.NamedTemporaryFile(prefix="shimizu_api_", suffix=suffix, delete=False) as handle:
            handle.write(data)
            temp_path = Path(handle.name)
        try:
            return self.predict_file(temp_path, options)
        finally:
            temp_path.unlink(missing_ok=True)


def filter_result(result: dict[str, Any], *, include_raw: bool, include_debug: bool) -> dict[str, Any]:
    filtered = dict(result)
    if not include_raw:
        filtered.pop("raw_crack_detections", None)
    if not include_debug:
        for key in [
            "suppressed_display_crack_detections",
            "wall_candidate_display",
            "ambiguity_candidate_groups",
        ]:
            filtered.pop(key, None)
    filtered["model_readiness_warnings"] = pending_class_warnings(result)
    return filtered


def pending_class_warnings(result: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    router = result.get("router") or {}
    for det in router.get("detections") or []:
        class_name = str(det.get("class_name") or "")
        if class_name in {"ブレース", "柱脚"}:
            warnings.append(f"downstream_model_pending:{class_name}")
    return warnings
