from rfdetr_prod_pipeline.api.main import app
from rfdetr_prod_pipeline.api.service import filter_result
from rfdetr_prod_pipeline.pipeline.crack_detector_registry import NoOpCrackDetector, build_detector_registry


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}

    assert "/api/v1/health" in paths
    assert "/api/v1/pipeline/info" in paths
    assert "/api/v1/pipeline/predict" in paths
    assert "/api/v1/analyze_auto" in paths


def test_filter_result_reports_pending_new_class_models():
    result = {
        "router": {
            "detections": [
                {"class_name": "ブレース"},
                {"class_name": "柱脚"},
            ]
        },
        "raw_crack_detections": [{"debug": True}],
        "suppressed_display_crack_detections": [{"debug": True}],
        "wall_candidate_display": {"debug": True},
        "ambiguity_candidate_groups": [{"debug": True}],
    }

    filtered = filter_result(result, include_raw=False, include_debug=False)

    assert "raw_crack_detections" not in filtered
    assert "suppressed_display_crack_detections" not in filtered
    assert filtered["model_readiness_warnings"] == [
        "downstream_model_pending:ブレース",
        "downstream_model_pending:柱脚",
    ]


def test_detector_registry_adds_noop_for_pending_router_classes(tmp_path):
    config = {
        "classes": {"router": {0: "天井", 1: "壁类", 2: "RC柱", 3: "ブレース", 4: "柱脚"}},
        "crack_models": {
            "ceiling": {"backend": "noop"},
            "inner_wall": {"backend": "noop"},
            "rc_wall": {"backend": "noop"},
            "rc_column": {"backend": "noop"},
            "brace": {"backend": "noop"},
            "column_base": {"backend": "noop"},
        },
    }

    registry = build_detector_registry(config, tmp_path)

    assert set(registry) == {"天井", "壁类", "RC柱", "ブレース", "柱脚"}
    assert isinstance(registry["ブレース"][0], NoOpCrackDetector)
    assert isinstance(registry["柱脚"][0], NoOpCrackDetector)
