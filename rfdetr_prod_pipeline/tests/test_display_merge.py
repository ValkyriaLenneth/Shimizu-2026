from rfdetr_prod_pipeline.pipeline.display_merge import suppress_overlapping_display_detections


def test_cross_class_contained_lower_grade_fragment_is_suppressed():
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

    assert len(kept) == 1
    assert len(suppressed) == 1
    assert kept[0]["damage_grade"] == "壁-D"
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
