from rfdetr_prod_pipeline.pipeline.display_merge import suppress_overlapping_display_detections


def test_cross_class_overlap_is_kept_even_when_contained():
    detections = [
        {
            "structure_type": "壁類",
            "damage_grade": "壁-D",
            "confidence": 0.85,
            "bbox_xyxy": [493, 187, 964, 828],
            "source_model": "rc_wall",
            "source_router_class": "壁类",
        },
        {
            "damage_grade": "B",
            "confidence": 0.28,
            "bbox_xyxy": [765, 169, 968, 252],
            "source_model": "ceiling",
            "source_router_class": "天井",
        },
    ]

    kept, suppressed = suppress_overlapping_display_detections(detections)

    assert len(kept) == 2
    assert suppressed == []
    assert {item["source_router_class"] for item in kept} == {"壁类", "天井"}


def test_cross_class_rc_column_and_wall_overlap_is_kept():
    detections = [
        {
            "structure_type": "壁類",
            "damage_grade": "壁-C",
            "confidence": 0.81,
            "bbox_xyxy": [100, 100, 520, 780],
            "source_model": "rc_wall",
            "source_router_class": "壁类",
        },
        {
            "structure_type": "RC柱",
            "damage_grade": "D",
            "confidence": 0.76,
            "bbox_xyxy": [180, 120, 500, 760],
            "source_model": "rc_column",
            "source_router_class": "RC柱",
        },
    ]

    kept, suppressed = suppress_overlapping_display_detections(detections, iou_threshold=0.35, ioa_threshold=0.70)

    assert len(kept) == 2
    assert suppressed == []
    assert {item["source_router_class"] for item in kept} == {"壁类", "RC柱"}


def test_same_class_overlap_keeps_more_severe_grade():
    detections = [
        {
            "structure_type": "RC柱",
            "damage_grade": "B",
            "confidence": 0.95,
            "bbox_xyxy": [100, 100, 520, 780],
            "source_model": "rc_column",
            "source_router_class": "RC柱",
        },
        {
            "structure_type": "RC柱",
            "damage_grade": "D",
            "confidence": 0.62,
            "bbox_xyxy": [120, 120, 500, 760],
            "source_model": "rc_column",
            "source_router_class": "RC柱",
        },
    ]

    kept, suppressed = suppress_overlapping_display_detections(detections, iou_threshold=0.35, ioa_threshold=0.70)

    assert len(kept) == 1
    assert len(suppressed) == 1
    assert kept[0]["source_router_class"] == "RC柱"
    assert kept[0]["damage_grade"] == "D"
    assert kept[0]["display_suppressed_count"] == 1


def test_separate_same_grade_wall_boxes_are_kept():
    detections = [
        {
            "structure_type": "壁類",
            "damage_grade": "壁-D",
            "confidence": 0.85,
            "bbox_xyxy": [493, 187, 964, 828],
            "source_model": "rc_wall",
            "source_router_class": "壁类",
        },
        {
            "structure_type": "壁類",
            "damage_grade": "壁-D",
            "confidence": 0.59,
            "bbox_xyxy": [68, 510, 230, 832],
            "source_model": "rc_wall",
            "source_router_class": "壁类",
        },
    ]

    kept, suppressed = suppress_overlapping_display_detections(detections)

    assert len(kept) == 2
    assert suppressed == []
