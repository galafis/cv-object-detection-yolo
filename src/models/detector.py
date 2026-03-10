"""
YOLO-like Object Detection Framework

Anchor box generation, Non-Max Suppression, IoU calculation,
bounding box utilities, detection pipeline, and evaluation metrics (mAP).

Author: Gabriel Demetrios Lafis
"""

import math
import random
from typing import Dict, List, Optional, Tuple


# ── Bounding Box Utilities ────────────────────────────────────────────

def xyxy_to_xywh(box: List[float]) -> List[float]:
    """Convert [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = box
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """Convert [cx, cy, w, h] to [x1, y1, x2, y2]."""
    cx, cy, w, h = box
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def clip_box(box: List[float], img_w: int, img_h: int) -> List[float]:
    """Clip bounding box to image boundaries."""
    return [
        max(0, min(box[0], img_w)),
        max(0, min(box[1], img_h)),
        max(0, min(box[2], img_w)),
        max(0, min(box[3], img_h)),
    ]


def box_area(box: List[float]) -> float:
    """Compute area of an [x1, y1, x2, y2] box."""
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


# ── IoU ───────────────────────────────────────────────────────────────

def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    inter_x1 = max(box_a[0], box_b[0])
    inter_y1 = max(box_a[1], box_b[1])
    inter_x2 = min(box_a[2], box_b[2])
    inter_y2 = min(box_a[3], box_b[3])

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = box_area(box_a)
    area_b = box_area(box_b)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


# ── Anchor Box Generation ────────────────────────────────────────────

class AnchorBoxGenerator:
    """Generate anchor boxes for a grid-based detection model."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (416, 416),
        grid_sizes: List[int] = None,
        anchor_ratios: List[Tuple[float, float]] = None,
    ):
        self.image_size = image_size
        self.grid_sizes = grid_sizes or [13, 26, 52]
        self.anchor_ratios = anchor_ratios or [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326),
        ]

    def generate(self) -> List[Dict]:
        """Generate anchor boxes across all grid cells."""
        anchors = []
        img_w, img_h = self.image_size
        ratios_per_scale = len(self.anchor_ratios) // len(self.grid_sizes)

        for scale_idx, grid_size in enumerate(self.grid_sizes):
            cell_w = img_w / grid_size
            cell_h = img_h / grid_size
            start = scale_idx * ratios_per_scale
            scale_ratios = self.anchor_ratios[start: start + ratios_per_scale]

            for row in range(grid_size):
                for col in range(grid_size):
                    cx = (col + 0.5) * cell_w
                    cy = (row + 0.5) * cell_h
                    for aw, ah in scale_ratios:
                        anchors.append({
                            "cx": cx, "cy": cy,
                            "width": aw, "height": ah,
                            "grid_size": grid_size,
                            "scale_idx": scale_idx,
                        })
        return anchors

    def count(self) -> int:
        ratios_per_scale = len(self.anchor_ratios) // len(self.grid_sizes)
        return sum(gs * gs * ratios_per_scale for gs in self.grid_sizes)


# ── Non-Maximum Suppression ──────────────────────────────────────────

def non_max_suppression(
    detections: List[Dict],
    iou_threshold: float = 0.5,
    score_key: str = "confidence",
    box_key: str = "bbox",
) -> List[Dict]:
    """Apply NMS to a list of detections.

    Each detection must have *score_key* (float) and *box_key* ([x1,y1,x2,y2]).
    """
    if not detections:
        return []

    sorted_dets = sorted(detections, key=lambda d: d[score_key], reverse=True)
    keep: List[Dict] = []

    while sorted_dets:
        best = sorted_dets.pop(0)
        keep.append(best)
        remaining = []
        for det in sorted_dets:
            if compute_iou(best[box_key], det[box_key]) < iou_threshold:
                remaining.append(det)
        sorted_dets = remaining

    return keep


# ── Detection Pipeline ────────────────────────────────────────────────

class ObjectDetector:
    """Full YOLO-like detection pipeline."""

    COCO_CLASSES = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
        "truck", "boat", "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
        "cow",
    ]

    def __init__(
        self,
        model_name: str = "yolov8n",
        confidence: float = 0.25,
        nms_threshold: float = 0.45,
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.device = device
        self.is_loaded = False

    def load_model(self) -> None:
        """Simulate model loading."""
        self.is_loaded = True

    def detect(self, image_shape: Tuple[int, int, int], seed: int = 42) -> List[Dict]:
        """Simulate detections on an image.

        Args:
            image_shape: (H, W, C) of the input image.
            seed: Random seed for reproducibility.

        Returns:
            List of filtered and NMS-applied detections.
        """
        if not self.is_loaded:
            self.load_model()

        rng = random.Random(seed)
        h, w, _ = image_shape
        raw_dets = []
        num = rng.randint(5, 15)

        for _ in range(num):
            cls = rng.choice(self.COCO_CLASSES)
            conf = rng.uniform(0.1, 0.99)
            x1 = rng.uniform(0, w * 0.7)
            y1 = rng.uniform(0, h * 0.7)
            x2 = x1 + rng.uniform(20, w * 0.3)
            y2 = y1 + rng.uniform(20, h * 0.3)
            raw_dets.append({
                "class": cls,
                "confidence": round(conf, 4),
                "bbox": clip_box([x1, y1, x2, y2], w, h),
            })

        # Filter by confidence
        filtered = [d for d in raw_dets if d["confidence"] >= self.confidence]

        # Per-class NMS
        classes = set(d["class"] for d in filtered)
        results = []
        for cls in classes:
            cls_dets = [d for d in filtered if d["class"] == cls]
            results.extend(non_max_suppression(cls_dets, self.nms_threshold))

        return sorted(results, key=lambda d: d["confidence"], reverse=True)

    def process(self, image_path: str) -> List[Dict]:
        """Process an image file (simulated)."""
        return self.detect((640, 640, 3))


# ── Evaluation Metrics ────────────────────────────────────────────────

class DetectionEvaluator:
    """Compute object detection evaluation metrics."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    def compute_precision_recall(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict[str, float]:
        """Compute precision, recall, and F1 from detections vs ground truth.

        Both lists contain dicts with 'class' and 'bbox' keys.
        """
        if not predictions and not ground_truths:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
        if not predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        if not ground_truths:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        tp = 0
        matched_gt = set()

        sorted_preds = sorted(predictions, key=lambda d: d.get("confidence", 1.0), reverse=True)
        for pred in sorted_preds:
            best_iou = 0
            best_idx = -1
            for idx, gt in enumerate(ground_truths):
                if idx in matched_gt:
                    continue
                if gt["class"] != pred["class"]:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= self.iou_threshold and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)

        precision = tp / len(predictions) if predictions else 0
        recall = tp / len(ground_truths) if ground_truths else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "true_positives": tp,
            "false_positives": len(predictions) - tp,
            "false_negatives": len(ground_truths) - tp,
        }

    def compute_ap(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
        class_name: str,
    ) -> float:
        """Compute Average Precision for a single class using 11-point interpolation."""
        cls_preds = sorted(
            [p for p in predictions if p["class"] == class_name],
            key=lambda d: d.get("confidence", 0),
            reverse=True,
        )
        cls_gts = [g for g in ground_truths if g["class"] == class_name]
        if not cls_gts:
            return 0.0

        tp_list = []
        matched = set()
        for pred in cls_preds:
            best_iou = 0
            best_idx = -1
            for idx, gt in enumerate(cls_gts):
                if idx in matched:
                    continue
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= self.iou_threshold and best_idx >= 0:
                tp_list.append(1)
                matched.add(best_idx)
            else:
                tp_list.append(0)

        # Cumulative TP / FP
        cum_tp = 0
        precisions = []
        recalls = []
        for i, tp in enumerate(tp_list):
            cum_tp += tp
            precisions.append(cum_tp / (i + 1))
            recalls.append(cum_tp / len(cls_gts))

        # 11-point interpolation
        ap = 0.0
        for t in [i / 10 for i in range(11)]:
            prec_at_recall = [p for p, r in zip(precisions, recalls) if r >= t]
            ap += max(prec_at_recall) / 11 if prec_at_recall else 0
        return round(ap, 4)

    def compute_map(
        self,
        predictions: List[Dict],
        ground_truths: List[Dict],
    ) -> Dict[str, float]:
        """Compute mean Average Precision across all classes."""
        classes = set(g["class"] for g in ground_truths)
        aps = {}
        for cls in classes:
            aps[cls] = self.compute_ap(predictions, ground_truths, cls)
        mean_ap = sum(aps.values()) / len(aps) if aps else 0.0
        return {"mAP": round(mean_ap, 4), "per_class_ap": aps}
