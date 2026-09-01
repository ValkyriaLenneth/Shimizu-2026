from __future__ import annotations

from evaluate_rfdetr_threshold_sweep import Prediction
from search_router_precision_ensemble import blend_selector, gate_selector


def prediction(cls: int, score: float, box=(0.0, 0.0, 10.0, 10.0)) -> Prediction:
    return Prediction(cls=cls, conf=score, xyxy=box)


def test_gate_accepts_supported_candidate() -> None:
    row = {
        "predictions_5class": [prediction(1, 0.40)],
        "predictions_confirmation": [prediction(1, 0.80)],
    }
    assert gate_selector(row, 1, 0.30, 0.70, 0.50, 0.90) == row["predictions_5class"]


def test_gate_rejects_wrong_class_and_low_iou_support() -> None:
    row = {
        "predictions_5class": [prediction(1, 0.40)],
        "predictions_confirmation": [
            prediction(0, 0.95),
            prediction(1, 0.95, (20.0, 20.0, 30.0, 30.0)),
        ],
    }
    assert gate_selector(row, 1, 0.30, 0.70, 0.50, 0.90) == []


def test_gate_high_primary_score_bypasses_confirmation() -> None:
    row = {
        "predictions_5class": [prediction(2, 0.96)],
        "predictions_confirmation": [],
    }
    assert gate_selector(row, 2, 0.30, 0.70, 0.50, 0.95) == row["predictions_5class"]


def test_blend_penalizes_unsupported_candidate() -> None:
    row = {
        "predictions_5class": [prediction(1, 0.90)],
        "predictions_confirmation": [],
    }
    assert blend_selector(row, 1, 0.50, 0.50, 0.50) == []
