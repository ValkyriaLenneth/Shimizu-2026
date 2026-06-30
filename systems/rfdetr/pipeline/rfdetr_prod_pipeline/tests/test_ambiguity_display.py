from rfdetr_prod_pipeline.pipeline.ambiguity_display import build_ambiguity_candidate_groups
from rfdetr_prod_pipeline.pipeline.result_merge import Detection


def test_cross_class_ambiguity_uses_one_union_display_box_with_candidates() -> None:
    merged = [
        Detection(
            xyxy=(813.352, 93.748, 1052.521, 1188.872),
            confidence=0.4569,
            grade="B",
            source_model="rc_column",
            source_router_class="RC柱",
        ),
        Detection(
            xyxy=(857.781, 101.735, 1079.189, 1188.225),
            confidence=0.2975,
            grade="B",
            source_model="rc_wall",
            source_router_class="壁类",
        ),
    ]

    groups, used = build_ambiguity_candidate_groups(merged, iou_threshold=0.50)

    assert used == {0, 1}
    assert len(groups) == 1
    assert len(groups[0]["display_detections"]) == 1

    display = groups[0]["display_detections"][0]
    assert display["bbox_xyxy"] == [813.352, 93.748, 1079.189, 1188.872]
    assert display["display_bbox_source"] == "ambiguous_class_union"
    assert display["display_labels"] == ["RC柱-B", "壁类-壁-B"]
    assert len(display["candidates"]) == 2
