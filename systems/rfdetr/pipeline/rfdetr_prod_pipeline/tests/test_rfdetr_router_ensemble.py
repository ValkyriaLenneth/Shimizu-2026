from pathlib import Path

import numpy as np
import pytest

from rfdetr_prod_pipeline.pipeline.rfdetr_backend import RfdetrDetection
from rfdetr_prod_pipeline.pipeline.rfdetr_router_infer import (
    PrecisionGateConfig,
    RfdetrPrecisionEnsembleConfig,
    RfdetrPrecisionEnsembleInfer,
)


class FakeBackend:
    def __init__(self, detections: list[RfdetrDetection]) -> None:
        self.detections = detections
        self.calls: list[tuple[list[float], int]] = []

    def predict(self, image_bgr, thresholds, max_det=1000):
        thresholds = list(thresholds)
        self.calls.append((thresholds, max_det))
        return [
            det
            for det in self.detections
            if det.class_id < len(thresholds) and det.confidence >= thresholds[det.class_id]
        ][:max_det]


def detection(class_id: int, confidence: float, box=(0.0, 0.0, 10.0, 10.0)) -> RfdetrDetection:
    return RfdetrDetection(
        xyxy=box,
        confidence=confidence,
        class_id=class_id,
        class_name={0: "天井", 1: "壁类"}[class_id],
    )


def test_precision_ensemble_only_emits_primary_boxes() -> None:
    primary = FakeBackend(
        [
            detection(0, 0.50),
            detection(0, 0.95, (20.0, 20.0, 30.0, 30.0)),
            detection(0, 0.55, (40.0, 40.0, 50.0, 50.0)),
            detection(1, 0.60),
        ]
    )
    confirmation = FakeBackend(
        [
            detection(0, 0.70),
            detection(0, 0.99, (60.0, 60.0, 70.0, 70.0)),
        ]
    )
    config = RfdetrPrecisionEnsembleConfig(
        primary_checkpoint=Path("primary.pth"),
        confirmation_checkpoints={"old": Path("old.pth")},
        class_names={0: "天井", 1: "壁类"},
        operating_points={
            0: PrecisionGateConfig(0.40, "old", 0.60, 0.50, 0.90),
            1: PrecisionGateConfig(0.50),
        },
        devices={"primary": "cpu", "old": "cpu"},
        parallel=False,
    )
    router = RfdetrPrecisionEnsembleInfer(
        config,
        backends={"primary": primary, "old": confirmation},
    )

    result = router.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [(row["class_id"], row["confidence"]) for row in result["detections"]] == [
        (0, 0.95),
        (1, 0.60),
        (0, 0.50),
    ]
    assert result["route_decision"]["strategy"] == "primary_boxes_with_specialized_confirmation"
    assert primary.calls == [([0.40, 0.50], 300)]
    assert confirmation.calls == [([0.60], 300)]


def test_precision_ensemble_rejects_incomplete_gate() -> None:
    config = RfdetrPrecisionEnsembleConfig(
        primary_checkpoint=Path("primary.pth"),
        confirmation_checkpoints={"old": Path("old.pth")},
        class_names={0: "天井"},
        operating_points={0: PrecisionGateConfig(0.40, confirmation_model="old")},
        devices={"primary": "cpu", "old": "cpu"},
    )

    with pytest.raises(ValueError, match="incomplete confirmation gate"):
        RfdetrPrecisionEnsembleInfer(
            config,
            backends={"primary": FakeBackend([]), "old": FakeBackend([])},
        )


def test_lazy_confirmation_is_skipped_without_a_candidate() -> None:
    primary = FakeBackend([detection(1, 0.60)])
    confirmation = FakeBackend([detection(0, 0.90)])
    config = RfdetrPrecisionEnsembleConfig(
        primary_checkpoint=Path("primary.pth"),
        confirmation_checkpoints={"old": Path("old.pth")},
        class_names={0: "天井", 1: "壁类"},
        operating_points={
            0: PrecisionGateConfig(0.40, "old", 0.60, 0.50, 0.90),
            1: PrecisionGateConfig(0.50),
        },
        devices={"primary": "cpu", "old": "cpu"},
        parallel=False,
        lazy_confirmation_models=frozenset({"old"}),
    )
    router = RfdetrPrecisionEnsembleInfer(
        config,
        backends={"primary": primary, "old": confirmation},
    )

    result = router.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [row["class_id"] for row in result["detections"]] == [1]
    assert confirmation.calls == []
