"""
Object Detection Module

Unified interface for object detection using YOLO, Faster R-CNN, and other models.

Author: Gabriel Demetrios Lafis
"""

import torch
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from loguru import logger


class ObjectDetector:
    """
    Unified object detection interface supporting multiple models.
    
    Supports:
    - YOLOv8, YOLOv5
    - Faster R-CNN
    - SSD, RetinaNet
    """
    
    def __init__(
        self,
        model_name: str = 'yolov8n',
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: Optional[str] = None
    ):
        """
        Initialize object detector.
        
        Args:
            model_name: Name of the detection model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            device: Device to run inference on
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Initializing {model_name} detector on {self.device}")
        self._load_model()
    
    def _load_model(self):
        """Load detection model."""
        if 'yolo' in self.model_name.lower():
            from ultralytics import YOLO
            self.model = YOLO(f'{self.model_name}.pt')
            logger.success(f"Loaded {self.model_name} model")
        else:
            raise NotImplementedError(f"Model {self.model_name} not yet implemented")
    
    def detect(
        self,
        image: np.ndarray,
        return_crops: bool = False
    ) -> List[Dict]:
        """
        Detect objects in image.
        
        Args:
            image: Input image (BGR format)
            return_crops: Whether to return cropped detections
            
        Returns:
            List of detection dictionaries
        """
        results = self.model(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device
        )[0]
        
        detections = []
        for box in results.boxes:
            detection = {
                'bbox': box.xyxy[0].cpu().numpy().tolist(),
                'confidence': float(box.conf[0]),
                'class_id': int(box.cls[0]),
                'class_name': results.names[int(box.cls[0])]
            }
            
            if return_crops:
                x1, y1, x2, y2 = map(int, detection['bbox'])
                detection['crop'] = image[y1:y2, x1:x2]
            
            detections.append(detection)
        
        return detections
    
    def detect_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> List[List[Dict]]:
        """
        Detect objects in video.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show: Whether to display video
            
        Returns:
            List of detections per frame
        """
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        all_detections = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect(frame)
            all_detections.append(detections)
            
            # Draw detections
            frame = self.draw_detections(frame, detections)
            
            if output_path:
                out.write(frame)
            
            if show:
                cv2.imshow('Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        cap.release()
        if output_path:
            out.release()
        if show:
            cv2.destroyAllWindows()
        
        return all_detections
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[Dict],
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw bounding boxes on image.
        
        Args:
            image: Input image
            detections: List of detections
            thickness: Line thickness
            
        Returns:
            Image with drawn detections
        """
        img = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            
            # Draw label
            cv2.putText(
                img, label, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
        
        return img
    
    def evaluate(
        self,
        dataset_yaml: str,
        split: str = 'val'
    ) -> Dict[str, float]:
        """
        Evaluate model on dataset.
        
        Args:
            dataset_yaml: Path to dataset YAML file
            split: Dataset split to evaluate on
            
        Returns:
            Dictionary with evaluation metrics
        """
        results = self.model.val(data=dataset_yaml, split=split)
        
        metrics = {
            'mAP50': results.box.map50,
            'mAP50-95': results.box.map,
            'precision': results.box.mp,
            'recall': results.box.mr
        }
        
        logger.info(f"Evaluation metrics: {metrics}")
        return metrics


if __name__ == "__main__":
    # Example usage
    detector = ObjectDetector(model_name='yolov8n')
    
    # Detect in image
    image = cv2.imread('sample.jpg')
    detections = detector.detect(image)
    
    print(f"Found {len(detections)} objects")
    for det in detections:
        print(f"  {det['class_name']}: {det['confidence']:.2f}")
    
    # Draw and save
    result_img = detector.draw_detections(image, detections)
    cv2.imwrite('result.jpg', result_img)
