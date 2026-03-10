"""
Tests for YOLO Object Detection Framework.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from models.detector import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    clip_box,
    box_area,
    compute_iou,
    AnchorBoxGenerator,
    non_max_suppression,
    ObjectDetector,
    DetectionEvaluator,
)


class TestBoundingBoxUtils:
    def test_xyxy_to_xywh(self):
        result = xyxy_to_xywh([10, 20, 30, 40])
        assert result == [20.0, 30.0, 20.0, 20.0]

    def test_xywh_to_xyxy(self):
        result = xywh_to_xyxy([20, 30, 20, 20])
        assert result == [10.0, 20.0, 30.0, 40.0]

    def test_roundtrip_conversion(self):
        original = [10.0, 20.0, 50.0, 80.0]
        converted = xywh_to_xyxy(xyxy_to_xywh(original))
        for a, b in zip(original, converted):
            assert abs(a - b) < 1e-6

    def test_clip_box_within_bounds(self):
        result = clip_box([10, 20, 30, 40], 640, 480)
        assert result == [10, 20, 30, 40]

    def test_clip_box_exceeding_bounds(self):
        result = clip_box([-5, -10, 700, 500], 640, 480)
        assert result == [0, 0, 640, 480]

    def test_box_area(self):
        assert box_area([0, 0, 10, 20]) == 200

    def test_box_area_degenerate(self):
        assert box_area([10, 10, 5, 5]) == 0


class TestIoU:
    def test_perfect_overlap(self):
        box = [0, 0, 100, 100]
        assert compute_iou(box, box) == 1.0

    def test_no_overlap(self):
        assert compute_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0

    def test_partial_overlap(self):
        iou = compute_iou([0, 0, 10, 10], [5, 5, 15, 15])
        assert 0.0 < iou < 1.0

    def test_containing_box(self):
        iou = compute_iou([0, 0, 100, 100], [25, 25, 75, 75])
        expected = 2500 / 10000  # inner area / outer area
        assert abs(iou - expected) < 1e-6


class TestAnchorBoxGenerator:
    def test_default_count(self):
        gen = AnchorBoxGenerator()
        assert gen.count() == (13 * 13 + 26 * 26 + 52 * 52) * 3

    def test_generate_returns_list(self):
        gen = AnchorBoxGenerator(grid_sizes=[2], anchor_ratios=[(10, 10), (20, 20), (30, 30)])
        anchors = gen.generate()
        assert len(anchors) == 2 * 2 * 3

    def test_anchor_fields(self):
        gen = AnchorBoxGenerator(grid_sizes=[1], anchor_ratios=[(10, 10), (20, 20), (30, 30)])
        anchors = gen.generate()
        for a in anchors:
            assert "cx" in a
            assert "cy" in a
            assert "width" in a
            assert "height" in a


class TestNonMaxSuppression:
    def test_empty_input(self):
        assert non_max_suppression([]) == []

    def test_single_detection(self):
        dets = [{"confidence": 0.9, "bbox": [0, 0, 10, 10]}]
        assert len(non_max_suppression(dets)) == 1

    def test_overlapping_detections(self):
        dets = [
            {"confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"confidence": 0.8, "bbox": [1, 1, 11, 11]},
        ]
        result = non_max_suppression(dets, iou_threshold=0.3)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.9

    def test_non_overlapping_detections(self):
        dets = [
            {"confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"confidence": 0.8, "bbox": [100, 100, 110, 110]},
        ]
        result = non_max_suppression(dets, iou_threshold=0.5)
        assert len(result) == 2


class TestObjectDetector:
    def test_initialization(self):
        det = ObjectDetector(model_name="yolov8s", confidence=0.5)
        assert det.model_name == "yolov8s"
        assert det.confidence == 0.5

    def test_detect_returns_list(self):
        det = ObjectDetector(confidence=0.3)
        results = det.detect((640, 640, 3))
        assert isinstance(results, list)

    def test_detect_respects_confidence(self):
        det = ObjectDetector(confidence=0.9)
        results = det.detect((640, 640, 3))
        for r in results:
            assert r["confidence"] >= 0.9

    def test_detect_sorted_by_confidence(self):
        det = ObjectDetector(confidence=0.1)
        results = det.detect((640, 640, 3))
        confs = [r["confidence"] for r in results]
        assert confs == sorted(confs, reverse=True)

    def test_process(self):
        det = ObjectDetector(confidence=0.3)
        results = det.process("dummy.jpg")
        assert isinstance(results, list)


class TestDetectionEvaluator:
    def setup_method(self):
        self.evaluator = DetectionEvaluator(iou_threshold=0.5)

    def test_perfect_detection(self):
        gt = [{"class": "cat", "bbox": [0, 0, 100, 100]}]
        pred = [{"class": "cat", "bbox": [0, 0, 100, 100], "confidence": 0.9}]
        result = self.evaluator.compute_precision_recall(pred, gt)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0

    def test_no_predictions(self):
        gt = [{"class": "cat", "bbox": [0, 0, 100, 100]}]
        result = self.evaluator.compute_precision_recall([], gt)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0

    def test_false_positive(self):
        pred = [{"class": "cat", "bbox": [0, 0, 10, 10], "confidence": 0.9}]
        result = self.evaluator.compute_precision_recall(pred, [])
        assert result["precision"] == 0.0

    def test_class_mismatch(self):
        gt = [{"class": "cat", "bbox": [0, 0, 100, 100]}]
        pred = [{"class": "dog", "bbox": [0, 0, 100, 100], "confidence": 0.9}]
        result = self.evaluator.compute_precision_recall(pred, gt)
        assert result["true_positives"] == 0

    def test_map_computation(self):
        gt = [
            {"class": "cat", "bbox": [0, 0, 100, 100]},
            {"class": "dog", "bbox": [200, 200, 300, 300]},
        ]
        pred = [
            {"class": "cat", "bbox": [5, 5, 95, 95], "confidence": 0.9},
            {"class": "dog", "bbox": [205, 205, 295, 295], "confidence": 0.8},
        ]
        result = self.evaluator.compute_map(pred, gt)
        assert "mAP" in result
        assert result["mAP"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
