from rfdetr_prod_pipeline.pipeline.result_merge import Detection
from rfdetr_prod_pipeline.pipeline.run_full_pipeline import ensure_minimum_display_outputs
from rfdetr_prod_pipeline.pipeline.wall_candidate_display import (
    build_wall_candidate_display,
    wall_display_grade,
)


def test_wall_display_grade_matrix() -> None:
    expected = {
        ("B", "B"): "B",
        ("B", "C"): "C",
        ("B", "D"): "D",
        ("C", "B"): "B",
        ("C", "C"): "C",
        ("C", "D"): "D",
        ("D", "B"): "D",
        ("D", "C"): "D",
        ("D", "D"): "D",
    }
    for pair, grade in expected.items():
        assert wall_display_grade(*pair) == grade


def test_wall_pair_bbox_always_uses_merged_geometry() -> None:
    records = [
        {
            "source_router_class": "壁类",
            "source_model": "inner_wall",
            "damage_grade": "C",
            "confidence": 0.80,
            "bbox_xyxy": [10, 10, 40, 40],
        },
        {
            "source_router_class": "壁类",
            "source_model": "rc_wall",
            "damage_grade": "D",
            "confidence": 0.70,
            "bbox_xyxy": [20, 20, 50, 50],
        },
    ]

    default_result = build_wall_candidate_display(records, iou_threshold=0.10)
    legacy_flag_result = build_wall_candidate_display(
        records,
        iou_threshold=0.10,
        use_union_bbox_for_pairs=False,
    )

    assert default_result["display_detections"][0]["bbox_xyxy"] == [10.0, 10.0, 50.0, 50.0]
    assert legacy_flag_result["display_detections"][0]["bbox_xyxy"] == [10.0, 10.0, 50.0, 50.0]
    assert legacy_flag_result["display_detections"][0]["display_bbox_source"] == "paired_wall_union"


def test_wall_pair_same_grade_small_rc_box_does_not_replace_larger_inner_box() -> None:
    records = [
        {
            "source_router_class": "壁类",
            "source_model": "inner_wall",
            "damage_grade": "D",
            "confidence": 0.3334,
            "bbox_xyxy": [509.091, 170.357, 803.725, 399.108],
        },
        {
            "source_router_class": "壁类",
            "source_model": "rc_wall",
            "damage_grade": "D",
            "confidence": 0.2995,
            "bbox_xyxy": [586.507, 248.375, 722.615, 341.282],
        },
    ]

    result = build_wall_candidate_display(
        records,
        iou_threshold=0.60,
        ioa_threshold=0.98,
        use_union_bbox_for_pairs=False,
    )

    assert len(result["display_detections"]) == 1
    assert result["display_detections"][0]["damage_grade"] == "壁-D"
    assert result["display_detections"][0]["bbox_xyxy"] == [509.091, 170.357, 803.725, 399.108]
    assert result["display_detections"][0]["display_bbox_source"] == "paired_wall_union"


def test_model_specific_single_threshold_keeps_second_rc_wall_candidate() -> None:
    records = [
        {
            "source_router_class": "壁类",
            "source_model": "inner_wall",
            "damage_grade": "D",
            "confidence": 0.7036,
            "bbox_xyxy": [22.302, 7.353, 1517.74, 946.713],
        },
        {
            "source_router_class": "壁类",
            "source_model": "rc_wall",
            "damage_grade": "D",
            "confidence": 0.5039,
            "bbox_xyxy": [505.77, 425.243, 598.026, 537.011],
        },
        {
            "source_router_class": "壁类",
            "source_model": "rc_wall",
            "damage_grade": "D",
            "confidence": 0.4507,
            "bbox_xyxy": [1136.418, 433.587, 1226.518, 538.356],
        },
    ]

    result = build_wall_candidate_display(
        records,
        iou_threshold=0.55,
        ioa_threshold=0.75,
        min_single_confidence=0.50,
        min_single_confidence_by_model={"inner_wall": 0.50, "rc_wall": 0.45},
        max_single_groups_per_model=1,
        use_union_bbox_for_pairs=False,
    )

    assert len(result["display_detections"]) == 2
    assert [det["status"] for det in result["display_detections"]] == ["wall_rule_merged", "single_model"]
    assert result["display_detections"][1]["bbox_xyxy"] == [1136.418, 433.587, 1226.518, 538.356]


def test_final_output_fallback_rebuilds_wall_display_when_empty() -> None:
    wall_records = [
        {
            "source_router_class": "壁类",
            "source_model": "rc_wall",
            "damage_grade": "D",
            "confidence": 0.18,
            "bbox_xyxy": [100, 100, 160, 180],
        }
    ]

    display_items, warnings = ensure_minimum_display_outputs(
        display_items=[],
        suppressed_display_items=[],
        wall_records=wall_records,
        merged=[],
        raw_records=wall_records,
        wall_display_cfg={"min_single_confidence": 0.50},
        fallback_cfg={
            "enabled": True,
            "rebuild_wall_display_if_empty": True,
            "relaxed_min_single_confidence": 0.10,
            "relaxed_min_single_confidence_by_model": {"rc_wall": 0.10},
            "include_merged_candidates": False,
            "include_raw_candidates": False,
            "max_outputs": 4,
        },
    )

    assert len(display_items) == 1
    assert display_items[0]["damage_grade"] == "壁-D"
    assert warnings == ["final_output_fallback_used:1"]


def test_final_output_fallback_uses_merged_candidate_when_needed() -> None:
    merged = [
        Detection(
            xyxy=(10, 20, 50, 60),
            confidence=0.42,
            grade="C",
            source_model="ceiling",
            source_router_class="天井",
        )
    ]

    display_items, warnings = ensure_minimum_display_outputs(
        display_items=[],
        suppressed_display_items=[],
        wall_records=[],
        merged=merged,
        raw_records=[],
        wall_display_cfg={},
        fallback_cfg={
            "enabled": True,
            "rebuild_wall_display_if_empty": False,
            "include_merged_candidates": True,
            "include_raw_candidates": False,
            "max_outputs": 4,
        },
    )

    assert len(display_items) == 1
    assert display_items[0]["damage_grade"] == "C"
    assert display_items[0]["status"] == "final_output_fallback"
    assert warnings == ["final_output_fallback_used:1"]
